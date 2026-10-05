"""Primal Heuristics for MILP Branch-and-Bound.

This module provides an optional heuristic framework that can inspect the
current LP relaxation solution and search state, attempt to construct a
feasible integer solution, validate it rigorously, and provide it to the
MILP solver as a candidate incumbent.

The heuristic is OPTIONAL and does not change default solver behavior when
not configured.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from optiforge.milp.fingerprint import ProblemFingerprint
from optiforge.milp.search_state import SearchState
from optiforge.model import Constraint, ConstraintSense, Model, ObjectiveSense, Variable, VariableType


# Default tolerance values used when tolerances=None
DEFAULT_INTEGRATOLERANCE = 1e-6
DEFAULT_FEASIBILITY_TOLERANCE = 1e-7


@dataclass(frozen=True)
class HeuristicResult:
    """Result of a primal heuristic attempt.

    Attributes
    ----------
    success : bool
        Whether a feasible integer candidate was found.
    variable_values : dict[str, float] | None
        The candidate integer solution (variable name -> value).
    objective_value : float | None
        The objective value of the candidate solution.
    message : str
        Human-readable description of the outcome.
    """

    success: bool
    variable_values: dict[str, float] | None = None
    objective_value: float | None = None
    message: str = ""

    @classmethod
    def failure(cls, message: str = "Heuristic did not find a feasible candidate") -> HeuristicResult:
        """Create a failure result."""
        return cls(success=False, message=message)

    @classmethod
    def success_result(
        cls,
        variable_values: dict[str, float],
        objective_value: float,
        message: str = "Heuristic found feasible integer candidate",
    ) -> HeuristicResult:
        """Create a success result."""
        return cls(
            success=True,
            variable_values=variable_values,
            objective_value=objective_value,
            message=message,
        )


class PrimalHeuristic(ABC):
    """Abstract base class for primal heuristics.

    A primal heuristic attempts to construct a feasible integer solution
    from the current LP relaxation solution and search state.
    """

    @abstractmethod
    def run(
        self,
        lp_solution: dict[str, float],
        lp_objective: float,
        model: Model,
        search_state: SearchState | None,
        fingerprint: ProblemFingerprint | None,
        tolerances: Any,
    ) -> HeuristicResult:
        """Run the heuristic.

        Parameters
        ----------
        lp_solution : dict[str, float]
            Variable values from the current LP relaxation solution.
        lp_objective : float
            Objective value of the LP relaxation.
        model : Model
            The original MILP model (read-only).
        search_state : SearchState | None
            Current search state (may be None if not available).
        fingerprint : ProblemFingerprint | None
            Problem fingerprint (may be None if not available).
        tolerances : Any
            MILP tolerances for validation.

        Returns
        -------
        HeuristicResult
            Result containing the candidate solution if found.
        """
        ...

    def should_run(
        self,
        search_state: SearchState | None,
        fingerprint: ProblemFingerprint | None,
    ) -> bool:
        """Determine whether the heuristic should run given the current state.

        This allows search-state-aware activation. Override to customize.

        Parameters
        ----------
        search_state : SearchState | None
            Current search state.
        fingerprint : ProblemFingerprint | None
            Problem fingerprint.

        Returns
        -------
        bool
            True if the heuristic should run, False to skip.
        """
        # Default: run when no incumbent exists or gap is large
        if search_state is None:
            return True

        # Don't run if we already have a good incumbent and gap is small
        if search_state.incumbent_objective is not None:
            if search_state.gap is not None and search_state.gap < 0.01:
                return False

        return True


class LPRoundingHeuristic(PrimalHeuristic):
    """LP Rounding heuristic with bounded feasibility repair.

    This heuristic:
    1. Starts from the LP relaxation solution.
    2. Identifies fractional integer/binary variables.
    3. Constructs an integer candidate using deterministic rounding.
    4. Attempts bounded feasibility repair where necessary.
    5. Validates the final candidate against ALL constraints and bounds.

    The heuristic is deterministic and bounded - no uncontrolled loops.
    """

    def __init__(
        self,
        max_repair_iterations: int = 10,
        max_rounding_attempts: int = 3,
    ) -> None:
        """Initialize the LP rounding heuristic.

        Parameters
        ----------
        max_repair_iterations : int
            Maximum number of repair iterations per attempt.
        max_rounding_attempts : int
            Maximum number of different rounding strategies to try.
        """
        self.max_repair_iterations = max_repair_iterations
        self.max_rounding_attempts = max_rounding_attempts

    def should_run(
        self,
        search_state: SearchState | None,
        fingerprint: ProblemFingerprint | None,
    ) -> bool:
        """Run when no incumbent exists or gap is large enough to warrant effort."""
        if search_state is None:
            return True

        # Run if no incumbent yet
        if search_state.incumbent_objective is None:
            return True

        # Don't run if we already have a good incumbent and both gaps are small
        gap = search_state.gap
        abs_gap = search_state.absolute_gap
        if gap is not None and abs_gap is not None and gap <= 0.01 and abs_gap <= 0.01:
            return False

        # Run if gap is large (more than 1%)
        if gap is not None and gap > 0.01:
            return True

        # Run if frontier is small (few nodes remaining - worth trying)
        if search_state.nodes_remaining <= 10:
            return True

        return False

    def run(
        self,
        lp_solution: dict[str, float],
        lp_objective: float,
        model: Model,
        search_state: SearchState | None,
        fingerprint: ProblemFingerprint | None,
        tolerances: Any,
    ) -> HeuristicResult:
        """Run the LP rounding heuristic with bounded repair attempts."""
        # Use default tolerances if None provided
        if tolerances is None:
            integrality_tol = DEFAULT_INTEGRATOLERANCE
            feasibility_tol = DEFAULT_FEASIBILITY_TOLERANCE
        else:
            integrality_tol = getattr(tolerances, "integrality_tolerance", DEFAULT_INTEGRATOLERANCE)
            feasibility_tol = getattr(tolerances, "feasibility_tolerance", DEFAULT_FEASIBILITY_TOLERANCE)

        # Check if there are any integer/binary variables that are fractional
        fractional_vars = self._get_fractional_integer_vars(
            lp_solution, model, integrality_tol
        )

        if not fractional_vars:
            # LP solution is already integer-feasible
            return HeuristicResult.failure("LP solution already integer-feasible")

        # Try different rounding strategies
        for attempt in range(self.max_rounding_attempts):
            candidate = self._round_solution(
                lp_solution, model, fractional_vars, attempt
            )

            # Attempt bounded repair
            repaired = self._repair_candidate(candidate, model, feasibility_tol)

            if repaired is not None:
                # Validate the repaired candidate
                # Create a simple tolerances object with both attributes
                class _Tol:
                    integrality_tolerance = integrality_tol
                    feasibility_tolerance = feasibility_tol
                if self._validate_candidate(repaired, model, _Tol()):
                    obj_value = self._compute_objective(repaired, model)
                    return HeuristicResult.success_result(
                        variable_values=repaired,
                        objective_value=obj_value,
                        message=f"LP rounding heuristic found feasible candidate (attempt {attempt + 1})",
                    )

        return HeuristicResult.failure(
            f"LP rounding heuristic failed after {self.max_rounding_attempts} attempts"
        )

    def _get_fractional_integer_vars(
        self,
        solution: dict[str, float],
        model: Model,
        tol: float,
    ) -> dict[str, float]:
        """Get integer/binary variables that are fractional in the solution."""
        fractional = {}

        for var_name, var in model.variables.items():
            if not var.is_integer():
                continue

            value = solution.get(var_name, 0.0)
            nearest_int = round(value)
            deviation = abs(value - nearest_int)

            if deviation > tol:
                fractional[var_name] = value

        return fractional

    def _round_solution(
        self,
        lp_solution: dict[str, float],
        model: Model,
        fractional_vars: dict[str, float],
        attempt: int,
    ) -> dict[str, float]:
        """Apply a rounding strategy to create an integer candidate."""
        candidate = dict(lp_solution)

        for var_name, value in fractional_vars.items():
            var = model.variables[var_name]

            if var.is_binary():
                # Binary: round to 0 or 1
                if attempt == 0:
                    # Round to nearest
                    candidate[var_name] = 1.0 if value >= 0.5 else 0.0
                elif attempt == 1:
                    # Round down
                    candidate[var_name] = 0.0
                else:
                    # Round up
                    candidate[var_name] = 1.0
            else:
                # Integer: try different rounding strategies
                if attempt == 0:
                    # Round to nearest
                    candidate[var_name] = float(round(value))
                elif attempt == 1:
                    # Round down (floor)
                    candidate[var_name] = float(int(value))
                else:
                    # Round up (ceil)
                    import math
                    candidate[var_name] = float(math.ceil(value))

        return candidate

    def _repair_candidate(
        self,
        candidate: dict[str, float],
        model: Model,
        tolerances: Any,
    ) -> dict[str, float] | None:
        """Attempt to repair a candidate to satisfy bounds and constraints.

        Uses a simple bounded iterative repair:
        - Check bounds violations and clamp
        - Check constraint violations and adjust
        - Repeat up to max_repair_iterations
        """
        current = dict(candidate)
        tol = tolerances.feasibility_tolerance if hasattr(tolerances, 'feasibility_tolerance') else 1e-7

        for _ in range(self.max_repair_iterations):
            # Check and fix bound violations
            bounds_fixed = self._fix_bounds(current, model)

            # Check and fix constraint violations
            constraints_fixed = self._fix_constraints(current, model, tol)

            if not bounds_fixed and not constraints_fixed:
                # No violations found
                return current

        # Could not repair within iteration limit
        return None

    def _fix_bounds(
        self,
        candidate: dict[str, float],
        model: Model,
    ) -> bool:
        """Fix variable bound violations by clamping. Returns True if any fixed."""
        fixed = False

        for var_name, var in model.variables.items():
            value = candidate.get(var_name, 0.0)

            if value < var.lower_bound:
                candidate[var_name] = var.lower_bound
                fixed = True
            elif value > var.upper_bound:
                candidate[var_name] = var.upper_bound
                fixed = True

        return fixed

    def _fix_constraints(
        self,
        candidate: dict[str, float],
        model: Model,
        tol: float,
    ) -> bool:
        """Attempt to fix constraint violations by adjusting variables.

        Simple approach: for each violated constraint, try to adjust
        a single variable to satisfy it while staying within bounds.
        Returns True if any adjustment was made.
        """
        fixed = False

        for con in model.constraints.values():
            lhs = sum(
                coeff * candidate.get(var, 0.0)
                for var, coeff in con.coefficients.items()
            )

            violated = False
            if con.sense == ConstraintSense.LESS_EQUAL:
                if lhs > con.rhs + tol:
                    violated = True
            elif con.sense == ConstraintSense.GREATER_EQUAL:
                if lhs < con.rhs - tol:
                    violated = True
            else:  # EQUAL
                if abs(lhs - con.rhs) > tol:
                    violated = True

            if violated:
                # Try to fix by adjusting one variable
                if self._adjust_for_constraint(candidate, con, model, tol):
                    fixed = True

        return fixed

    def _adjust_for_constraint(
        self,
        candidate: dict[str, float],
        con: Constraint,
        model: Model,
        tol: float,
    ) -> bool:
        """Try to adjust a single variable to satisfy a constraint."""
        # Find a variable we can adjust (prefer continuous, then integer)
        for var_name, coeff in con.coefficients.items():
            if coeff == 0:
                continue

            var = model.variables[var_name]
            current_val = candidate.get(var_name, 0.0)

            # Compute required value to satisfy constraint
            other_sum = sum(
                c * candidate.get(v, 0.0)
                for v, c in con.coefficients.items()
                if v != var_name
            )

            if con.sense == ConstraintSense.LESS_EQUAL:
                required = (con.rhs - other_sum) / coeff
                if coeff > 0:
                    new_val = min(required, var.upper_bound)
                else:
                    new_val = max(required, var.lower_bound)
            elif con.sense == ConstraintSense.GREATER_EQUAL:
                required = (con.rhs - other_sum) / coeff
                if coeff > 0:
                    new_val = max(required, var.lower_bound)
                else:
                    new_val = min(required, var.upper_bound)
            else:  # EQUAL
                required = (con.rhs - other_sum) / coeff
                new_val = required

            # Check if new value is within bounds
            if var.lower_bound - tol <= new_val <= var.upper_bound + tol:
                # For integer/binary variables, round to nearest integer
                if var.is_integer():
                    new_val = float(round(new_val))
                    # Re-check bounds after rounding
                    if not (var.lower_bound - tol <= new_val <= var.upper_bound + tol):
                        continue

                candidate[var_name] = new_val
                return True

        return False

    def _validate_candidate(
        self,
        candidate: dict[str, float],
        model: Model,
        tolerances: Any,
    ) -> bool:
        """Validate a candidate solution against all requirements."""
        tol = tolerances.integrality_tolerance
        feas_tol = tolerances.feasibility_tolerance if hasattr(tolerances, 'feasibility_tolerance') else 1e-7

        # 1. Check all variables present
        for var_name in model.variables:
            if var_name not in candidate:
                return False

        # 2. Check bounds
        for var_name, var in model.variables.items():
            value = candidate[var_name]
            if value < var.lower_bound - feas_tol or value > var.upper_bound + feas_tol:
                return False

        # 3. Check integrality
        for var_name, var in model.variables.items():
            if var.is_integer():
                value = candidate[var_name]
                nearest_int = round(value)
                if abs(value - nearest_int) > tol:
                    return False

        # 4. Check binary validity
        for var_name, var in model.variables.items():
            if var.is_binary():
                value = candidate[var_name]
                if abs(value - 0.0) > tol and abs(value - 1.0) > tol:
                    return False

        # 5. Check all constraints
        for con in model.constraints.values():
            lhs = sum(
                coeff * candidate.get(var, 0.0)
                for var, coeff in con.coefficients.items()
            )

            if con.sense == ConstraintSense.LESS_EQUAL:
                if lhs > con.rhs + feas_tol:
                    return False
            elif con.sense == ConstraintSense.GREATER_EQUAL:
                if lhs < con.rhs - feas_tol:
                    return False
            else:  # EQUAL
                if abs(lhs - con.rhs) > feas_tol:
                    return False

        return True

    def _compute_objective(self, candidate: dict[str, float], model: Model) -> float:
        """Compute objective value for a candidate solution."""
        if model.objective is None:
            return 0.0

        return sum(
            coeff * candidate.get(var, 0.0)
            for var, coeff in model.objective.coefficients.items()
        )


# Import at the end to avoid circular imports
from typing import Any