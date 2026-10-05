"""Conservative presolve layer for OptiForge LPs.

This module implements a minimal Stage 5A presolve pipeline designed to be
reversible and mathematically safe:

- bound validation and fixed-variable elimination
- zero-row and zero-column handling
- singleton-constraint bound extraction
- activity-bound checks and redundancy detection
- bound propagation / fixed-point iteration
- objective constant tracking for postsolve reconstruction

The implementation is intentionally conservative: it only applies reductions that
preserve feasibility and can be undone exactly from the recorded metadata.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Collection, Sequence

from optiforge.core.solution import SolutionStatus
from optiforge.model.constraint import Constraint, ConstraintSense
from optiforge.model.model import Model
from optiforge.model.objective import Objective, ObjectiveSense
from optiforge.model.variable import Variable


class PresolveOperation(Enum):
    """Enumeration of Stage 5A presolve operations."""

    BOUND_VALIDATION = "bound_validation"
    """Validate bounds and detect fixed variables (lb == ub)."""

    ZERO_ROW_REMOVAL = "zero_row_removal"
    """Remove redundant zero rows and detect infeasible zero rows."""

    ZERO_COLUMN_ELIMINATION = "zero_column_elimination"
    """Fix variables that don't appear in any constraint based on objective sense."""

    SINGLETON_CONSTRAINTS = "singleton_constraints"
    """Derive bounds from constraints with only one variable, then remove the row."""

    ACTIVITY_CHECKS = "activity_checks"
    """Use variable bounds to check if constraints are redundant or infeasible."""

    BOUND_PROPAGATION = "bound_propagation"
    """Propagate bounds through constraints iteratively."""

    FIXED_POINT_ITERATION = "fixed_point_iteration"
    """Repeat all operations until no changes occur (fixed point)."""


STAGE_5A_OPERATION_ORDER: tuple[PresolveOperation, ...] = (
    PresolveOperation.BOUND_VALIDATION,
    PresolveOperation.ZERO_ROW_REMOVAL,
    PresolveOperation.ZERO_COLUMN_ELIMINATION,
    PresolveOperation.SINGLETON_CONSTRAINTS,
    PresolveOperation.ACTIVITY_CHECKS,
    PresolveOperation.BOUND_PROPAGATION,
    PresolveOperation.FIXED_POINT_ITERATION,
)


class _PresolveInfeasible(Exception):
    """Raised internally when presolve detects mathematical infeasibility."""


@dataclass
class PresolveResult:
    """Outcome of a single presolve pass sequence.

    Attributes
    ----------
    reduced_model : Model | None
        Reduced model after all deterministic reductions. None if the problem is
        infeasible or unbounded before entering the simplex stage.
    status : SolutionStatus
        Presolve status (typically FEASIBLE, INFEASIBLE, or UNBOUNDED).
    original_variable_order : list[str]
        Original variable sequence as stored in the input model.
    reduced_variable_mapping : dict[str, str]
        Maps a reduced variable name to the original variable name. Since the
        current presolve is conservative and does not aggregate variables, this is
        usually the identity map for surviving variables.
    fixed_variables : dict[str, float]
        Variables fixed during presolve and the values assigned to them.
    removed_constraints : list[dict[str, Any]]
        Records of constraints removed during reduction.
    objective_constant : float
        Constant shift added to the reduced objective so that the original
        objective value is recovered exactly as:
            original_obj = reduced_obj + objective_constant.
    transformation_metadata : list[str]
        Human-readable log of reductions applied.
    statistics : dict[str, int]
        Summary counters for presolve operations.
    original_to_reduced : dict[str, str]
        Mapping of original variable names to their reduced names if preserved.
    reduced_to_original : dict[str, str]
        Reverse mapping for postsolve reconstruction.
    """

    reduced_model: Model | None
    status: SolutionStatus
    original_variable_order: list[str] = field(default_factory=list)
    reduced_variable_mapping: dict[str, str] = field(default_factory=dict)
    fixed_variables: dict[str, float] = field(default_factory=dict)
    removed_constraints: list[dict[str, Any]] = field(default_factory=list)
    objective_constant: float = 0.0
    transformation_metadata: list[str] = field(default_factory=list)
    statistics: dict[str, int] = field(default_factory=dict)
    original_to_reduced: dict[str, str] = field(default_factory=dict)
    reduced_to_original: dict[str, str] = field(default_factory=dict)


def _as_float(value: float) -> float:
    return float(value)


def _copy_variable(var: Variable) -> Variable:
    return Variable(
        name=var.name,
        lower_bound=float(var.lower_bound),
        upper_bound=float(var.upper_bound),
        variable_type=var.variable_type,
    )


def _copy_constraint(constraint: Constraint) -> Constraint:
    return Constraint(
        name=constraint.name,
        coefficients=dict(constraint.coefficients),
        sense=constraint.sense,
        rhs=float(constraint.rhs),
    )


def _copy_objective(objective: Objective) -> Objective:
    return Objective(
        coefficients=dict(objective.coefficients),
        sense=objective.sense,
    )


def _bnd_tighten(var: Variable, *, lower: float | None = None, upper: float | None = None, tol: float) -> bool:
    """Tighten bounds without ever loosening an existing bound."""
    changed = False
    if lower is not None:
        new_lower = float(lower)
        if new_lower > var.lower_bound + tol:
            if new_lower > var.upper_bound + tol:
                raise ValueError(f"Variable '{var.name}' becomes infeasible after tightening lower bound to {new_lower}.")
            var.lower_bound = new_lower
            changed = True
    if upper is not None:
        new_upper = float(upper)
        if new_upper < var.upper_bound - tol:
            if new_upper < var.lower_bound - tol:
                raise ValueError(f"Variable '{var.name}' becomes infeasible after tightening upper bound to {new_upper}.")
            var.upper_bound = new_upper
            changed = True
    if math.isfinite(var.lower_bound) and math.isfinite(var.upper_bound) and abs(var.lower_bound - var.upper_bound) <= tol:
        var.lower_bound = var.upper_bound
    return changed


def _term_bounds_for_variable(value: float, lb: float, ub: float) -> tuple[float, float]:
    """Return (min_term, max_term) for a_i * x_i with current bounds."""
    if math.isinf(lb) and math.isinf(ub):
        if value > 0.0:
            return float("-inf"), float("inf")
        if value < 0.0:
            return float("-inf"), float("inf")
        return 0.0, 0.0

    if value > 0.0:
        return value * lb, value * ub
    if value < 0.0:
        return value * ub, value * lb
    return 0.0, 0.0


def _activity_bounds(constraint: Constraint, variables: dict[str, Variable], tol: float) -> tuple[float, float]:
    """Compute the minimum and maximum possible value of a^T x under variable bounds."""
    lo = 0.0
    hi = 0.0
    for name, coeff in constraint.coefficients.items():
        if abs(coeff) <= tol:
            continue
        var = variables.get(name)
        if var is None:
            raise ValueError(f"Constraint '{constraint.name}' references unknown variable '{name}'.")
        term_lo, term_hi = _term_bounds_for_variable(coeff, var.lower_bound, var.upper_bound)
        lo += term_lo
        hi += term_hi
    return lo, hi


def _evaluate_zero_row_values(sense: ConstraintSense, rhs: float, tol: float) -> tuple[bool, str]:
    """Return (is_redundant, message) for a zero coefficient row.

    Works on raw sense/rhs values so it can be applied to rows that have
    become empty after fixed-variable substitution (which cannot be
    represented as a :class:`Constraint` because empty coefficient
    dictionaries are rejected by the model layer).
    """
    if sense is ConstraintSense.LESS_EQUAL:
        if rhs >= -tol:
            return True, "redundant <= 0 row"
        return False, f"infeasible <= row: {rhs}"
    if sense is ConstraintSense.GREATER_EQUAL:
        if rhs <= tol:
            return True, "redundant >= 0 row"
        return False, f"infeasible >= row: {rhs}"
    if abs(rhs) <= tol:
        return True, "redundant == 0 row"
    return False, f"infeasible == row: {rhs}"


def _evaluate_zero_row(constraint: Constraint, tol: float) -> tuple[bool, str]:
    """Return (is_redundant, message) for a zero coefficient row."""
    if any(abs(coeff) > tol for coeff in constraint.coefficients.values()):
        return False, "not-zero-row"
    return _evaluate_zero_row_values(constraint.sense, float(constraint.rhs), tol)


def _preferred_zero_column_bound(var: Variable, coeff: float, sense: ObjectiveSense, tol: float = 1e-12) -> tuple[float | None, str | None]:
    """Return the preferred finite bound for a variable that has no constraints."""
    if abs(coeff) <= tol:
        return None, None
    if sense is ObjectiveSense.MINIMIZE:
        if coeff > 0.0:
            return float(var.lower_bound), "lower"
        if coeff < 0.0:
            return float(var.upper_bound), "upper"
    else:
        if coeff > 0.0:
            return float(var.upper_bound), "upper"
        if coeff < 0.0:
            return float(var.lower_bound), "lower"
    return None, None


def _apply_fixed_variable(
    *,
    variables: dict[str, Variable],
    constraints: list[Constraint],
    objective_coeffs: dict[str, float],
    fixed_variables: dict[str, float],
    objective_constant: float,
    var_name: str,
    value: float,
    tol: float,
    transformation_metadata: list[str],
    removed_constraints: list[dict[str, Any]],
) -> tuple[float, bool]:
    """Fix a variable by substituting its value into constraints and objective."""
    if var_name in fixed_variables:
        return objective_constant, False

    fixed_variables[var_name] = float(value)
    transformation_metadata.append(f"fixed {var_name} = {value:.12g}")

    if var_name in variables:
        variables.pop(var_name, None)

    new_constraints: list[Constraint] = []
    for constraint in constraints:
        coeff = constraint.coefficients.pop(var_name, 0.0)
        if abs(coeff) <= tol:
            new_constraints.append(constraint)
            continue
        new_rhs = float(constraint.rhs) - coeff * value
        new_coeffs = {k: v for k, v in constraint.coefficients.items() if abs(v) > tol}
        if not new_coeffs:
            # The fixed variable was the only coefficient: the row is now a
            # zero-variable constraint. Evaluate it through the zero-row logic.
            redundant, message = _evaluate_zero_row_values(constraint.sense, new_rhs, tol)
            if redundant:
                removed_constraints.append({"name": constraint.name, "reason": "empty-after-fix"})
                transformation_metadata.append(
                    f"removed zero row {constraint.name} after fixing {var_name}: {message}"
                )
                continue
            raise _PresolveInfeasible(
                f"Constraint '{constraint.name}' infeasible after fixing {var_name}: {message}"
            )
        updated_constraint = Constraint(
            name=constraint.name,
            coefficients=new_coeffs,
            sense=constraint.sense,
            rhs=new_rhs,
        )
        new_constraints.append(updated_constraint)
    constraints[:] = new_constraints

    if var_name in objective_coeffs:
        objective_constant += float(objective_coeffs[var_name]) * value
        objective_coeffs.pop(var_name, None)

    return objective_constant, True


def _update_single_bound(var: Variable, *, lower: float | None = None, upper: float | None = None, tol: float) -> bool:
    """Apply a safe bound tightening and return True when any change occurred.

    Raises
    ------
    _PresolveInfeasible
        If the tightening contradicts the existing bounds (mathematical
        infeasibility, not a programming error).
    """
    changed = False
    if lower is not None and lower > var.lower_bound + tol:
        if lower > var.upper_bound + tol:
            raise _PresolveInfeasible(
                f"Variable '{var.name}' infeasible after tightening lower bound to {lower}."
            )
        var.lower_bound = float(lower)
        changed = True
    if upper is not None and upper < var.upper_bound - tol:
        if upper < var.lower_bound - tol:
            raise _PresolveInfeasible(
                f"Variable '{var.name}' infeasible after tightening upper bound to {upper}."
            )
        var.upper_bound = float(upper)
        changed = True
    return changed


def _tighten_from_constraint(constraint: Constraint, variables: dict[str, Variable], tol: float) -> tuple[bool, int]:
    """Tighten bounds implied by a single row using current variable bounds."""
    nonzero = [(name, coeff) for name, coeff in constraint.coefficients.items() if abs(coeff) > tol]
    if len(nonzero) != 1:
        return False, 0

    var_name, coeff = nonzero[0]
    var = variables.get(var_name)
    if var is None:
        raise ValueError(f"Constraint '{constraint.name}' references unknown variable '{var_name}'.")

    rhs = float(constraint.rhs)
    changed = False
    if constraint.sense is ConstraintSense.LESS_EQUAL:
        if coeff > 0.0:
            bound = rhs / coeff
            changed = _update_single_bound(var, upper=bound, tol=tol)
        elif coeff < 0.0:
            bound = rhs / coeff
            changed = _update_single_bound(var, lower=bound, tol=tol)
    elif constraint.sense is ConstraintSense.GREATER_EQUAL:
        if coeff > 0.0:
            bound = rhs / coeff
            changed = _update_single_bound(var, lower=bound, tol=tol)
        elif coeff < 0.0:
            bound = rhs / coeff
            changed = _update_single_bound(var, upper=bound, tol=tol)
    elif constraint.sense is ConstraintSense.EQUAL:
        bound = rhs / coeff
        lower_changed = _update_single_bound(var, lower=bound, tol=tol)
        upper_changed = _update_single_bound(var, upper=bound, tol=tol)
        changed = lower_changed or upper_changed

    if abs(var.lower_bound - var.upper_bound) <= tol:
        return True, 1
    return changed, 1


def _compute_min_other(constraint: Constraint, variables: dict[str, Variable], exclude_name: str, tol: float) -> float:
    total = 0.0
    for name, coeff in constraint.coefficients.items():
        if name == exclude_name or abs(coeff) <= tol:
            continue
        var = variables.get(name)
        if var is None:
            continue
        a = float(coeff)
        lo = a * var.lower_bound
        hi = a * var.upper_bound
        total += min(lo, hi)
    return float(total)


def _compute_max_other(constraint: Constraint, variables: dict[str, Variable], exclude_name: str, tol: float) -> float:
    total = 0.0
    for name, coeff in constraint.coefficients.items():
        if name == exclude_name or abs(coeff) <= tol:
            continue
        var = variables.get(name)
        if var is None:
            continue
        a = float(coeff)
        lo = a * var.lower_bound
        hi = a * var.upper_bound
        total += max(lo, hi)
    return float(total)


def _propagate_constraint_bounds(constraint: Constraint, variables: dict[str, Variable], tol: float) -> bool:
    """Apply bound propagation on a single row."""
    changed = False
    for var_name, coeff in list(constraint.coefficients.items()):
        if abs(coeff) <= tol:
            continue
        var = variables.get(var_name)
        if var is None:
            continue
        rhs = float(constraint.rhs)

        if constraint.sense is ConstraintSense.LESS_EQUAL:
            min_other = _compute_min_other(constraint, variables, var_name, tol)
            bound_value = (rhs - min_other) / float(coeff)
            if coeff > 0.0 and math.isfinite(bound_value):
                changed = _update_single_bound(var, upper=bound_value, tol=tol) or changed
            elif coeff < 0.0 and math.isfinite(bound_value):
                changed = _update_single_bound(var, lower=bound_value, tol=tol) or changed
        elif constraint.sense is ConstraintSense.GREATER_EQUAL:
            max_other = _compute_max_other(constraint, variables, var_name, tol)
            bound_value = (rhs - max_other) / float(coeff)
            if coeff > 0.0 and math.isfinite(bound_value):
                changed = _update_single_bound(var, lower=bound_value, tol=tol) or changed
            elif coeff < 0.0 and math.isfinite(bound_value):
                changed = _update_single_bound(var, upper=bound_value, tol=tol) or changed
        elif constraint.sense is ConstraintSense.EQUAL:
            min_other = _compute_min_other(constraint, variables, var_name, tol)
            max_other = _compute_max_other(constraint, variables, var_name, tol)
            if coeff > 0.0:
                lo = (rhs - max_other) / float(coeff)
                hi = (rhs - min_other) / float(coeff)
                if math.isfinite(lo):
                    changed = _update_single_bound(var, lower=lo, tol=tol) or changed
                if math.isfinite(hi):
                    changed = _update_single_bound(var, upper=hi, tol=tol) or changed
            elif coeff < 0.0:
                lo = (rhs - min_other) / float(coeff)
                hi = (rhs - max_other) / float(coeff)
                if math.isfinite(lo):
                    changed = _update_single_bound(var, lower=lo, tol=tol) or changed
                if math.isfinite(hi):
                    changed = _update_single_bound(var, upper=hi, tol=tol) or changed

    # A constraint with no nonzero coefficients carries no propagation
    # information; zero rows are handled by the zero-row evaluation step.
    return changed


def _build_reduced_model(
    *,
    name: str,
    variables: dict[str, Variable],
    constraints: list[Constraint],
    objective_coeffs: dict[str, float],
    objective_sense: ObjectiveSense,
) -> Model:
    model = Model(name=name)
    for var in variables.values():
        model.add_variable(_copy_variable(var))
    for constraint in constraints:
        model.add_constraint(_copy_constraint(constraint))
    if objective_coeffs:
        model.set_objective(Objective(coefficients=dict(objective_coeffs), sense=objective_sense))
    return model


def _presolve_model_impl(
    model: Model,
    feasibility_tol: float = 1e-7,
    enabled_operations: set[PresolveOperation] | None = None,
) -> PresolveResult:
    """Run the conservative Stage 5A presolve pipeline.

    The routine is intentionally conservative and deterministic: it only performs
    safe reductions that preserve feasibility and can be reversed using the
    recorded fixed-variable and constraint metadata.

    Parameters
    ----------
    model : Model
        The model to presolve.
    feasibility_tol : float
        Feasibility tolerance for numerical comparisons.
    enabled_operations : set[PresolveOperation] | None
        Optional set of operations to enable. If None, all operations are enabled
        (STAGE_5A_OPERATION_ORDER). This allows adaptive presolve to selectively
        apply operations.

    Raises
    ------
    _PresolveInfeasible
        If the problem is detected to be mathematically infeasible.
    """
    tol = float(feasibility_tol)
    original_vars = list(model.variables.keys())
    original_constraints = list(model.constraints.keys())
    variables: dict[str, Variable] = {
        name: _copy_variable(var) for name, var in model.variables.items()
    }
    constraints: list[Constraint] = [
        _copy_constraint(constraint) for constraint in model.constraints.values()
    ]
    objective_coeffs: dict[str, float] = dict(model.objective.coefficients) if model.objective is not None else {}
    objective_sense = model.objective.sense if model.objective is not None else ObjectiveSense.MINIMIZE
    objective_constant = 0.0
    fixed_variables: dict[str, float] = {}
    removed_constraints: list[dict[str, Any]] = []
    transformation_metadata: list[str] = []
    original_to_reduced: dict[str, str] = {}
    reduced_to_original: dict[str, str] = {}
    total_tightened = 0
    passes = 0

    # Determine which operations are enabled
    if enabled_operations is None:
        enabled_operations = set(STAGE_5A_OPERATION_ORDER)

    # Helper to check if an operation is enabled
    def is_enabled(op: PresolveOperation) -> bool:
        return op in enabled_operations

    # Initial bound validation and fixed-value detection (always runs if enabled)
    if is_enabled(PresolveOperation.BOUND_VALIDATION):
        for var in list(variables.values()):
            if var.lower_bound > var.upper_bound + tol:
                return PresolveResult(
                    reduced_model=None,
                    status=SolutionStatus.INFEASIBLE,
                    original_variable_order=original_vars,
                    reduced_variable_mapping={},
                    fixed_variables={},
                    removed_constraints=[],
                    objective_constant=0.0,
                    transformation_metadata=[f"contradictory bounds for {var.name}"],
                    statistics={
                        "original_vars": len(original_vars),
                        "original_constraints": len(original_constraints),
                        "reduced_vars": 0,
                        "reduced_constraints": 0,
                        "fixed_vars": 0,
                        "removed_rows": 0,
                        "tightened_bounds": 0,
                        "passes": 0,
                    },
                    original_to_reduced={},
                    reduced_to_original={},
                )
            if abs(var.lower_bound - var.upper_bound) <= tol:
                value = (var.lower_bound + var.upper_bound) / 2.0
                objective_constant, _ = _apply_fixed_variable(
                    variables=variables,
                    constraints=constraints,
                    objective_coeffs=objective_coeffs,
                    fixed_variables=fixed_variables,
                    objective_constant=objective_constant,
                    var_name=var.name,
                    value=value,
                    tol=tol,
                    transformation_metadata=transformation_metadata,
                    removed_constraints=removed_constraints,
                )
                total_tightened += 1

    while True:
        passes += 1
        changed = False

        # 1. Validate and canonicalize bounds (always runs if bound validation enabled)
        if is_enabled(PresolveOperation.BOUND_VALIDATION):
            for var_name in list(variables):
                var = variables[var_name]
                if var.lower_bound > var.upper_bound + tol:
                    return PresolveResult(
                        reduced_model=None,
                        status=SolutionStatus.INFEASIBLE,
                        original_variable_order=original_vars,
                        reduced_variable_mapping={},
                        fixed_variables=fixed_variables,
                        removed_constraints=removed_constraints,
                        objective_constant=objective_constant,
                        transformation_metadata=transformation_metadata,
                        statistics={
                            "original_vars": len(original_vars),
                            "original_constraints": len(original_constraints),
                            "reduced_vars": len(variables),
                            "reduced_constraints": len(constraints),
                            "fixed_vars": len(fixed_variables),
                            "removed_rows": len(removed_constraints),
                            "tightened_bounds": total_tightened,
                            "passes": passes,
                        },
                        original_to_reduced=original_to_reduced,
                        reduced_to_original=reduced_to_original,
                    )
                if abs(var.lower_bound - var.upper_bound) <= tol:
                    value = (var.lower_bound + var.upper_bound) / 2.0
                    objective_constant, fixed_now = _apply_fixed_variable(
                        variables=variables,
                        constraints=constraints,
                        objective_coeffs=objective_coeffs,
                        fixed_variables=fixed_variables,
                        objective_constant=objective_constant,
                        var_name=var_name,
                        value=value,
                        tol=tol,
                        transformation_metadata=transformation_metadata,
                        removed_constraints=removed_constraints,
                    )
                    if fixed_now:
                        changed = True
                        total_tightened += 1

        # 2. Zero row removal / infeasibility detection
        if is_enabled(PresolveOperation.ZERO_ROW_REMOVAL):
            for constraint in list(constraints):
                redundant, message = _evaluate_zero_row(constraint, tol)
                if redundant:
                    removed_constraints.append({"name": constraint.name, "reason": message})
                    constraints.remove(constraint)
                    transformation_metadata.append(f"removed zero row {constraint.name}: {message}")
                    changed = True
                else:
                    if "infeasible" in message:
                        return PresolveResult(
                            reduced_model=None,
                            status=SolutionStatus.INFEASIBLE,
                            original_variable_order=original_vars,
                            reduced_variable_mapping={},
                            fixed_variables=fixed_variables,
                            removed_constraints=removed_constraints,
                            objective_constant=objective_constant,
                            transformation_metadata=transformation_metadata + [f"zero-row infeasible: {constraint.name} ({message})"],
                            statistics={
                                "original_vars": len(original_vars),
                                "original_constraints": len(original_constraints),
                                "reduced_vars": len(variables),
                                "reduced_constraints": len(constraints),
                                "fixed_vars": len(fixed_variables),
                                "removed_rows": len(removed_constraints),
                                "tightened_bounds": total_tightened,
                                "passes": passes,
                            },
                            original_to_reduced=original_to_reduced,
                            reduced_to_original=reduced_to_original,
                        )

        # 3. Zero-column elimination
        if is_enabled(PresolveOperation.ZERO_COLUMN_ELIMINATION):
            for name in list(variables):
                if any(name in constraint.coefficients for constraint in constraints):
                    continue
                coeff = objective_coeffs.get(name, 0.0)
                if abs(coeff) <= tol:
                    continue
                var = variables[name]
                preferred = None
                if objective_sense is ObjectiveSense.MINIMIZE:
                    if coeff > 0.0:
                        preferred = var.lower_bound
                    elif coeff < 0.0:
                        preferred = var.upper_bound
                else:
                    if coeff > 0.0:
                        preferred = var.upper_bound
                    elif coeff < 0.0:
                        preferred = var.lower_bound
                if preferred is None:
                    continue
                if math.isinf(preferred):
                    return PresolveResult(
                        reduced_model=None,
                        status=SolutionStatus.UNBOUNDED,
                        original_variable_order=original_vars,
                        reduced_variable_mapping={},
                        fixed_variables=fixed_variables,
                        removed_constraints=removed_constraints,
                        objective_constant=objective_constant,
                        transformation_metadata=transformation_metadata + [f"zero-column unbounded {name}"],
                        statistics={
                            "original_vars": len(original_vars),
                            "original_constraints": len(original_constraints),
                            "reduced_vars": len(variables),
                            "reduced_constraints": len(constraints),
                            "fixed_vars": len(fixed_variables),
                            "removed_rows": len(removed_constraints),
                            "tightened_bounds": total_tightened,
                            "passes": passes,
                        },
                        original_to_reduced=original_to_reduced,
                        reduced_to_original=reduced_to_original,
                    )
                objective_constant, _ = _apply_fixed_variable(
                    variables=variables,
                    constraints=constraints,
                    objective_coeffs=objective_coeffs,
                    fixed_variables=fixed_variables,
                    objective_constant=objective_constant,
                    var_name=name,
                    value=float(preferred),
                    tol=tol,
                    transformation_metadata=transformation_metadata,
                    removed_constraints=removed_constraints,
                )
                changed = True
                total_tightened += 1

        # 4. Singleton constraints: derive bounds and then drop row if safe
        if is_enabled(PresolveOperation.SINGLETON_CONSTRAINTS):
            for constraint in list(constraints):
                nonzero = [(name, coeff) for name, coeff in constraint.coefficients.items() if abs(coeff) > tol]
                if len(nonzero) != 1:
                    continue
                var_name, coeff = nonzero[0]
                if var_name not in variables:
                    continue
                var = variables[var_name]
                if constraint.sense is ConstraintSense.LESS_EQUAL:
                    if coeff > 0.0:
                        bound = float(constraint.rhs) / float(coeff)
                        if _update_single_bound(var, upper=bound, tol=tol):
                            changed = True
                            total_tightened += 1
                            transformation_metadata.append(f"singleton <= {constraint.name}: {var_name} <= {bound}")
                    else:
                        bound = float(constraint.rhs) / float(coeff)
                        if _update_single_bound(var, lower=bound, tol=tol):
                            changed = True
                            total_tightened += 1
                            transformation_metadata.append(f"singleton <= {constraint.name}: {var_name} >= {bound}")
                elif constraint.sense is ConstraintSense.GREATER_EQUAL:
                    if coeff > 0.0:
                        bound = float(constraint.rhs) / float(coeff)
                        if _update_single_bound(var, lower=bound, tol=tol):
                            changed = True
                            total_tightened += 1
                            transformation_metadata.append(f"singleton >= {constraint.name}: {var_name} >= {bound}")
                    else:
                        bound = float(constraint.rhs) / float(coeff)
                        if _update_single_bound(var, upper=bound, tol=tol):
                            changed = True
                            total_tightened += 1
                            transformation_metadata.append(f"singleton >= {constraint.name}: {var_name} <= {bound}")
                else:
                    bound = float(constraint.rhs) / float(coeff)
                    lower_changed = _update_single_bound(var, lower=bound, tol=tol)
                    upper_changed = _update_single_bound(var, upper=bound, tol=tol)
                    if lower_changed or upper_changed:
                        changed = True
                        total_tightened += 1
                        transformation_metadata.append(f"singleton == {constraint.name}: {var_name} = {bound}")
                if abs(var.lower_bound - var.upper_bound) <= tol:
                    objective_constant, _ = _apply_fixed_variable(
                        variables=variables,
                        constraints=constraints,
                        objective_coeffs=objective_coeffs,
                        fixed_variables=fixed_variables,
                        objective_constant=objective_constant,
                        var_name=var_name,
                        value=(var.lower_bound + var.upper_bound) / 2.0,
                        tol=tol,
                        transformation_metadata=transformation_metadata,
                        removed_constraints=removed_constraints,
                    )
                    changed = True
                if var.lower_bound > var.upper_bound + tol:
                    return PresolveResult(
                        reduced_model=None,
                        status=SolutionStatus.INFEASIBLE,
                        original_variable_order=original_vars,
                        reduced_variable_mapping={},
                        fixed_variables=fixed_variables,
                        removed_constraints=removed_constraints,
                        objective_constant=objective_constant,
                        transformation_metadata=transformation_metadata,
                        statistics={
                            "original_vars": len(original_vars),
                            "original_constraints": len(original_constraints),
                            "reduced_vars": len(variables),
                            "reduced_constraints": len(constraints),
                            "fixed_vars": len(fixed_variables),
                            "removed_rows": len(removed_constraints),
                            "tightened_bounds": total_tightened,
                            "passes": passes,
                        },
                        original_to_reduced=original_to_reduced,
                        reduced_to_original=reduced_to_original,
                    )
                # Remove singleton rows once their information has been transferred.
                # Guard against stale references: when the variable was fixed above,
                # _apply_fixed_variable rebuilt `constraints` with new objects and
                # already removed this row (as "empty-after-fix"), so the snapshot
                # reference is no longer in the list.
                if len(nonzero) == 1 and constraint in constraints:
                    removed_constraints.append({"name": constraint.name, "reason": "singleton-transferred"})
                    constraints.remove(constraint)
                    transformation_metadata.append(f"removed singleton row {constraint.name}")
                    changed = True

        # 5. Activity-based redundancy and infeasibility checks
        if is_enabled(PresolveOperation.ACTIVITY_CHECKS):
            for constraint in list(constraints):
                lo, hi = _activity_bounds(constraint, variables, tol)
                if constraint.sense is ConstraintSense.LESS_EQUAL:
                    if lo > float(constraint.rhs) + tol:
                        return PresolveResult(
                            reduced_model=None,
                            status=SolutionStatus.INFEASIBLE,
                            original_variable_order=original_vars,
                            reduced_variable_mapping={},
                            fixed_variables=fixed_variables,
                            removed_constraints=removed_constraints,
                            objective_constant=objective_constant,
                            transformation_metadata=transformation_metadata + [f"activity infeasible <= {constraint.name}"] ,
                            statistics={
                                "original_vars": len(original_vars),
                                "original_constraints": len(original_constraints),
                                "reduced_vars": len(variables),
                                "reduced_constraints": len(constraints),
                                "fixed_vars": len(fixed_variables),
                                "removed_rows": len(removed_constraints),
                                "tightened_bounds": total_tightened,
                                "passes": passes,
                            },
                            original_to_reduced=original_to_reduced,
                            reduced_to_original=reduced_to_original,
                        )
                    if hi <= float(constraint.rhs) + tol:
                        removed_constraints.append({"name": constraint.name, "reason": "activity-redundant"})
                        constraints.remove(constraint)
                        transformation_metadata.append(f"removed redundant <= row {constraint.name}")
                        changed = True
                elif constraint.sense is ConstraintSense.GREATER_EQUAL:
                    if hi < float(constraint.rhs) - tol:
                        return PresolveResult(
                            reduced_model=None,
                            status=SolutionStatus.INFEASIBLE,
                            original_variable_order=original_vars,
                            reduced_variable_mapping={},
                            fixed_variables=fixed_variables,
                            removed_constraints=removed_constraints,
                            objective_constant=objective_constant,
                            transformation_metadata=transformation_metadata + [f"activity infeasible >= {constraint.name}"] ,
                            statistics={
                                "original_vars": len(original_vars),
                                "original_constraints": len(original_constraints),
                                "reduced_vars": len(variables),
                                "reduced_constraints": len(constraints),
                                "fixed_vars": len(fixed_variables),
                                "removed_rows": len(removed_constraints),
                                "tightened_bounds": total_tightened,
                                "passes": passes,
                            },
                            original_to_reduced=original_to_reduced,
                            reduced_to_original=reduced_to_original,
                        )
                    if lo >= float(constraint.rhs) - tol:
                        removed_constraints.append({"name": constraint.name, "reason": "activity-redundant"})
                        constraints.remove(constraint)
                        transformation_metadata.append(f"removed redundant >= row {constraint.name}")
                        changed = True
                else:
                    if lo > float(constraint.rhs) + tol or hi < float(constraint.rhs) - tol:
                        return PresolveResult(
                            reduced_model=None,
                            status=SolutionStatus.INFEASIBLE,
                            original_variable_order=original_vars,
                            reduced_variable_mapping={},
                            fixed_variables=fixed_variables,
                            removed_constraints=removed_constraints,
                            objective_constant=objective_constant,
                            transformation_metadata=transformation_metadata + [f"activity infeasible == {constraint.name}"] ,
                            statistics={
                                "original_vars": len(original_vars),
                                "original_constraints": len(original_constraints),
                                "reduced_vars": len(variables),
                                "reduced_constraints": len(constraints),
                                "fixed_vars": len(fixed_variables),
                                "removed_rows": len(removed_constraints),
                                "tightened_bounds": total_tightened,
                                "passes": passes,
                            },
                            original_to_reduced=original_to_reduced,
                            reduced_to_original=reduced_to_original,
                        )
                    if abs(lo - float(constraint.rhs)) <= tol and abs(hi - float(constraint.rhs)) <= tol:
                        removed_constraints.append({"name": constraint.name, "reason": "activity-redundant"})
                        constraints.remove(constraint)
                        transformation_metadata.append(f"removed redundant == row {constraint.name}")
                        changed = True

        # 6. Bound propagation with current bounds
        if is_enabled(PresolveOperation.BOUND_PROPAGATION):
            for constraint in list(constraints):
                if _propagate_constraint_bounds(constraint, variables, tol):
                    changed = True
                    total_tightened += 1
                    transformation_metadata.append(f"bound propagation on {constraint.name}")

        # 7. If a variable is now fixed by current bounds, eliminate it
        if is_enabled(PresolveOperation.BOUND_VALIDATION):
            for name in list(variables):
                var = variables[name]
                if abs(var.lower_bound - var.upper_bound) <= tol:
                    value = (var.lower_bound + var.upper_bound) / 2.0
                    objective_constant, fixed_now = _apply_fixed_variable(
                        variables=variables,
                        constraints=constraints,
                        objective_coeffs=objective_coeffs,
                        fixed_variables=fixed_variables,
                        objective_constant=objective_constant,
                        var_name=name,
                        value=value,
                        tol=tol,
                        transformation_metadata=transformation_metadata,
                        removed_constraints=removed_constraints,
                    )
                    if fixed_now:
                        changed = True
                        total_tightened += 1

        # Check if we should continue iterating (fixed point)
        if not is_enabled(PresolveOperation.FIXED_POINT_ITERATION):
            break

        if not changed:
            break

    final_variables = {name: variables[name] for name in sorted(variables)}
    # Only surviving (non-fixed) variables participate in the reduced model.
    # Fixed variables are recorded in `fixed_variables` and restored during
    # postsolve; they must not appear as ordinary reduced variables.
    for name in original_vars:
        if name in final_variables:
            original_to_reduced[name] = name
            reduced_to_original[name] = name

    reduced_model = _build_reduced_model(
        name=model.name,
        variables=final_variables,
        constraints=constraints,
        objective_coeffs=objective_coeffs,
        objective_sense=objective_sense,
    )

    statistics = {
        "original_vars": len(original_vars),
        "original_constraints": len(original_constraints),
        "reduced_vars": len(final_variables),
        "reduced_constraints": len(constraints),
        "fixed_vars": len(fixed_variables),
        "removed_rows": len(removed_constraints),
        "tightened_bounds": total_tightened,
        "passes": passes,
    }

    return PresolveResult(
        reduced_model=reduced_model,
        status=SolutionStatus.FEASIBLE,
        original_variable_order=original_vars,
        reduced_variable_mapping={name: name for name in final_variables},
        fixed_variables=fixed_variables,
        removed_constraints=removed_constraints,
        objective_constant=objective_constant,
        transformation_metadata=transformation_metadata,
        statistics=statistics,
        original_to_reduced=original_to_reduced,
        reduced_to_original=reduced_to_original,
    )


def presolve_model(model: Model, feasibility_tol: float = 1e-7) -> PresolveResult:
    """Run the conservative Stage 5A presolve pipeline.

    The routine is intentionally conservative and deterministic: it only performs
    safe reductions that preserve feasibility and can be reversed using the
    recorded fixed-variable and constraint metadata.

    Mathematical infeasibility detected during bound tightening or propagation
    is reported through the returned ``PresolveResult.status`` (INFEASIBLE)
    rather than raised as an exception.
    """
    try:
        return _presolve_model_impl(model, feasibility_tol=feasibility_tol)
    except _PresolveInfeasible as exc:
        return PresolveResult(
            reduced_model=None,
            status=SolutionStatus.INFEASIBLE,
            original_variable_order=list(model.variables.keys()),
            reduced_variable_mapping={},
            fixed_variables={},
            removed_constraints=[],
            objective_constant=0.0,
            transformation_metadata=[str(exc)],
            statistics={
                "original_vars": len(model.variables),
                "original_constraints": len(model.constraints),
                "reduced_vars": 0,
                "reduced_constraints": 0,
                "fixed_vars": 0,
                "removed_rows": 0,
                "tightened_bounds": 0,
                "passes": 0,
            },
            original_to_reduced={},
            reduced_to_original={},
        )


def postsolve_solution(reduced_solution: dict[str, float], presolve_result: PresolveResult) -> dict[str, float]:
    """Reconstruct the original variable values from the reduced solution."""
    restored: dict[str, float] = dict(presolve_result.fixed_variables)
    reduced_model = presolve_result.reduced_model
    if reduced_model is None:
        return restored

    for original_name in presolve_result.original_variable_order:
        if original_name in restored:
            continue
        if original_name in presolve_result.original_to_reduced:
            red_name = presolve_result.original_to_reduced[original_name]
            if red_name in reduced_solution:
                restored[original_name] = float(reduced_solution[red_name])
            elif original_name in reduced_solution:
                restored[original_name] = float(reduced_solution[original_name])
        elif original_name in reduced_solution:
            restored[original_name] = float(reduced_solution[original_name])

    for name, value in reduced_solution.items():
        if name not in restored:
            restored[name] = float(value)

    return restored
