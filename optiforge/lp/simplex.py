"""Baseline Revised Simplex solver for Linear Programming (LP).

Implements the two-phase Revised Simplex method from first mathematical
principles, using NumPy dense arrays and linear system solves:

    Phase I:
        Find an initial basic feasible solution (BFS) by minimizing the
        sum of artificial variables:
            min  w = sum(a_i)
        If w* > 0, the LP is infeasible.
        If w* = 0, drive out any remaining basic artificial variables and
        transition to Phase II.

    Phase II:
        Optimize the original objective function (min or max) starting
        from the feasible basis obtained in Phase I.

Anti-Cycling & Pricing Strategies:
    - Bland's rule (default): Smallest-index selection for both entering
      and leaving variables. Mathematically proven to prevent cycling and
      guarantee finite termination under all conditions.
    - Dantzig pricing: Maximum absolute reduced cost entering variable
      selection. Yields significant pivot count reductions (often >70%) on
      standard LPs, but is prone to cycling on degenerate problems (e.g.
      Beale 1955).
    - Anti-cycling safeguards: When Dantzig pricing is active, two independent
      safeguards protect against cycling:
        1. Safeguard A: Consecutive degenerate pivot counter. If degenerate
           pivots (theta <= degeneracy_tolerance) reach threshold, permanently
           switches to Bland's rule for the remainder of the solve.
        2. Safeguard B: Basis repetition detection. If a basis signature repeats,
           declares cycle risk and permanently switches to Bland's rule.
"""

from __future__ import annotations

import bisect
import math
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from optiforge.core.solution import Solution, SolutionStatus
from optiforge.lp.profiling import SimplexProfileStats, SimplexProfiler
from optiforge.lp.standard_form import (
    SingularBasisError,
    StandardFormModel,
    to_standard_form,
)
from optiforge.model.model import Model
from optiforge.model.objective import ObjectiveSense
from optiforge.numerics import (
    BasisFactorization,
    IncrementalBasisFactorization,
    PivotUpdateFactorization,
    get_provider_stats,
)
from optiforge.numerics.numerical_monitor import (
    NumericalEventType,
    NumericalStabilityMonitor,
    NumericalThresholds,
)
import time


# ---------------------------------------------------------------------------
# Tolerances Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimplexTolerances:
    """Numerical tolerances for the Revised Simplex algorithm.

    Attributes
    ----------
    feasibility_tolerance : float
        Tolerance for checking basic variable non-negativity (x_B >= -tol).
        Default is 1e-7.
    reduced_cost_tolerance : float
        Tolerance for reduced cost optimality condition and candidate
        entering variable identification. Default is 1e-7.
    pivot_tolerance : float
        Threshold for positive pivot elements (d_i > tol) in the ratio test.
        Default is 1e-7.
    phase1_feasibility_tolerance : float
        Threshold for the Phase-I auxiliary objective value (|w*| <= tol).
        Default is 1e-6.
    max_iterations : int
        Maximum number of simplex pivots allowed before declaring iteration
        limit. Default is 10_000.
    use_incremental_reduced_costs : bool | None, optional
        Tri-state configuration for incremental reduced-cost and dual multiplier
        updates. Note: This applies specifically to the Dual Revised Simplex algorithm
        (`DualRevisedSimplexSolver`) where dual multipliers are maintained incrementally
        via Δy = -θ_D v and Δr = θ_D α. Primal Revised Simplex continues to evaluate
        B^T y = c_B directly for pricing.
        - None (default): AUTO. Automatically activates if and only if
          PivotUpdateFactorization (PFI) is actually active for the problem dimension m.
          If DenseLU is active, resolves to OFF.
        - False: Explicit OFF. Strictly forces full BTRAN solves on every pivot.
        - True: Explicit ON. Forces incremental reduced-cost updates (with safe full-solve
          fallback when DenseLU factorizes from scratch).
        
        Behavioral Compatibility Contract:
        - API Compatibility: 100% backward compatible (accepts bool, None, or default).
        - Unconfigured Solves: Default OptiForge runs DenseLU, resolving AUTO to OFF (unchanged).
        - Explicit False / True: Unchanged.
        - PFI-enabled Solves with Unspecified RC: OLD behavior was OFF (full BTRAN solves);
          NEW behavior is AUTO -> ON when PFI is active. This is an intentional behavioral
          enhancement delivering 1.15x-1.35x speedup without changing final solutions.
    incremental_rc_refresh_interval : int, optional
        Periodic full re-evaluation interval (default 25 pivots) to eliminate roundoff drift.
    """

    feasibility_tolerance: float = 1e-7
    reduced_cost_tolerance: float = 1e-7
    pivot_tolerance: float = 1e-7
    phase1_feasibility_tolerance: float = 1e-6
    max_iterations: int = 10_000
    use_incremental_basis: bool = False
    use_pivot_factorization: bool = False
    use_adaptive_pivot_factorization: bool = False
    adaptive_pfi_min_dimension: int = 6
    max_basis_updates: int = 30
    incremental_residual_tolerance: float = 1e-7
    use_fast_pivot_update: bool = False
    use_scipy_triangular_solve: bool = False
    scipy_triangular_min_dimension: int = 10
    pricing: str = "bland"
    degeneracy_tolerance: float = 1e-11
    max_consecutive_degenerate_pivots: int = 30
    enable_basis_repetition_detection: bool = True
    enable_degenerate_pivot_fallback: bool = True
    use_incremental_reduced_costs: bool | None = None
    incremental_rc_refresh_interval: int = 25
    use_optimized_triangular_solves: bool = False
    triangular_solve_provider: str = "python"
    use_scipy_lu_factor: bool = False
    scipy_lu_min_dimension: int = 10
    lu_factor_provider: str = "python"
    enable_profiling: bool = False
    zero_tolerance: float = 1e-12
    basis_singularity_tolerance: float = 1e-12
    complementarity_tolerance: float = 1e-7
    optimality_tolerance: float = 1e-7
    condition_limit: float = 1e12
    min_safe_pivot: float = 1e-8
    max_safe_residual: float = 1e-4
    numerical_verification: str = "none"
    condition_estimation_method: str = "auto"
    condition_estimation_max_iterations: int = 10
    condition_estimation_tolerance: float = 1e-7
    scaling_enabled: bool = False
    scaling_max_iterations: int = 15
    scaling_min_factor: float = 1e-5
    scaling_max_factor: float = 1e5
    scaling_balance_tolerance: float = 1e-3

    def __post_init__(self) -> None:
        if self.optimality_tolerance != self.reduced_cost_tolerance:
            if self.reduced_cost_tolerance != 1e-7:
                self.optimality_tolerance = self.reduced_cost_tolerance
            elif self.optimality_tolerance != 1e-7:
                self.reduced_cost_tolerance = self.optimality_tolerance

        if self.use_optimized_triangular_solves:
            self.use_scipy_triangular_solve = True
            self.triangular_solve_provider = "scipy"
        elif self.use_scipy_triangular_solve:
            self.triangular_solve_provider = "scipy"
        elif self.triangular_solve_provider == "scipy":
            self.use_scipy_triangular_solve = True

        if self.use_scipy_lu_factor:
            self.lu_factor_provider = "scipy"
        elif self.lu_factor_provider == "scipy":
            self.use_scipy_lu_factor = True

        if self.pricing not in ("bland", "dantzig"):
            raise ValueError(
                f"Unsupported pricing strategy '{self.pricing}'. Supported options are 'bland' and 'dantzig'."
            )

    def to_numerical_policy(self) -> Any:
        """Convert this SimplexTolerances instance into an immutable NumericalPolicy."""
        from optiforge.numerics.policy import NumericalPolicy

        return NumericalPolicy(
            feasibility_tolerance=self.feasibility_tolerance,
            optimality_tolerance=self.optimality_tolerance,
            pivot_tolerance=self.pivot_tolerance,
            zero_tolerance=self.zero_tolerance,
            basis_singularity_tolerance=self.basis_singularity_tolerance,
            complementarity_tolerance=self.complementarity_tolerance,
            condition_limit=self.condition_limit,
            min_safe_pivot=self.min_safe_pivot,
            max_safe_residual=self.max_safe_residual,
            verification_mode=self.numerical_verification,
            condition_estimation_method=self.condition_estimation_method,
            condition_estimation_max_iterations=self.condition_estimation_max_iterations,
            condition_estimation_tolerance=self.condition_estimation_tolerance,
            scaling_enabled=self.scaling_enabled,
            scaling_max_iterations=self.scaling_max_iterations,
            scaling_min_factor=self.scaling_min_factor,
            scaling_max_factor=self.scaling_max_factor,
            scaling_balance_tolerance=self.scaling_balance_tolerance,
        )



# ---------------------------------------------------------------------------
# Internal Simplex Iteration Result
# ---------------------------------------------------------------------------

@dataclass
class SimplexIterationResult:
    """Outcome of an iteration loop."""

    status: SolutionStatus
    basis_indices: list[int]
    nonbasic_indices: list[int]
    x: np.ndarray
    objective_value: float
    iterations: int
    message: str = ""
    basis_factorization: BasisFactorization | None = None
    degenerate_pivots: int = 0
    cycle_fallback_triggered: bool = False
    cycle_fallback_reason: str | None = None
    profile_stats: Optional[SimplexProfileStats] = None
    y: Optional[np.ndarray] = None
    reduced_costs: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Revised Simplex Solver
# ---------------------------------------------------------------------------

class RevisedSimplexSolver:
    """A mathematically transparent baseline Revised Simplex solver for LPs.

    Parameters
    ----------
    tolerances : SimplexTolerances | None
        Numerical tolerances. If None, default tolerances are used.
    numerical_monitor : NumericalStabilityMonitor | None
        Optional numerical stability monitor. If provided, numerical
        diagnostics will be recorded during the solve. Default is None
        (monitoring disabled).
    numerical_thresholds : NumericalThresholds | None
        Optional custom thresholds for the numerical monitor. Only used
        if numerical_monitor is provided but has default thresholds.
    """

    def __init__(
        self,
        tolerances: SimplexTolerances | None = None,
        numerical_monitor: NumericalStabilityMonitor | None = None,
        numerical_thresholds: NumericalThresholds | None = None,
        stop_event: Optional[threading.Event] = None,
        use_incremental_basis: bool | None = None,
        pricing: str | None = None,
    ) -> None:
        self.tolerances: SimplexTolerances = tolerances or SimplexTolerances()
        if use_incremental_basis is not None:
            self.tolerances.use_incremental_basis = use_incremental_basis
        if pricing is not None:
            if pricing not in ("bland", "dantzig"):
                raise ValueError(
                    f"Unsupported pricing strategy: '{pricing}'. Supported options are 'bland' and 'dantzig'."
                )
            self.tolerances.pricing = pricing
        self.numerical_monitor: NumericalStabilityMonitor | None = numerical_monitor
        self._numerical_thresholds = numerical_thresholds
        self.stop_event: Optional[threading.Event] = stop_event
        self.fallback_triggered: bool = False
        self.fallback_reason: str | None = None
        self.degenerate_pivots_count: int = 0
        self.last_iteration_result: SimplexIterationResult | None = None
        self.last_standard_form_model: StandardFormModel | None = None
        self.last_standard_form_basis_indices: list[int] | None = None

    def solve(
        self,
        model: Model | StandardFormModel,
        tolerances: SimplexTolerances | None = None,
        numerical_monitor: NumericalStabilityMonitor | None = None,
        stop_event: Optional[threading.Event] = None,
        pricing: str | None = None,
        initial_basis: list[int] | None = None,
    ) -> Solution:
        """Solve a linear programming problem.

        Parameters
        ----------
        model : Model | StandardFormModel
            The LP model to solve.
        tolerances : SimplexTolerances | None
            Optional tolerances overriding instance configuration.
        numerical_monitor : NumericalStabilityMonitor | None
            Optional numerical stability monitor for this solve. Overrides
            instance-level monitor if provided.
        stop_event : Optional[threading.Event]
            Optional cancellation event.
        pricing : str | None
            Optional pricing rule ('bland' or 'dantzig'). Overrides tolerances.

        Returns
        -------
        Solution
            Result container with solve status, variable values, objective
            value, and iteration count.
        """
        tol = tolerances or self.tolerances
        monitor = numerical_monitor if numerical_monitor is not None else self.numerical_monitor
        # Cooperative cancellation: prefer call-site event, fall back to instance event
        stop_ev = stop_event if stop_event is not None else self.stop_event

        self.fallback_triggered = False
        self.fallback_reason = None
        self.degenerate_pivots_count = 0
        self.last_iteration_result = None

        if pricing is not None:
            if pricing not in ("bland", "dantzig"):
                raise ValueError(
                    f"Unsupported pricing strategy: '{pricing}'. Supported options are 'bland' and 'dantzig'."
                )
            active_pricing = pricing
        else:
            active_pricing = tol.pricing
            if active_pricing not in ("bland", "dantzig"):
                raise ValueError(
                    f"Unsupported pricing strategy: '{active_pricing}'. Supported options are 'bland' and 'dantzig'."
                )


        if stop_ev is not None and stop_ev.is_set():
            return Solution(
                status=SolutionStatus.ITERATION_LIMIT,
                variable_values={},
                objective_value=None,
                message="Solver cancelled (stop signal received).",
                iterations=0,
            )

        # Initialize monitor with custom thresholds if provided
        if monitor is not None and self._numerical_thresholds is not None:
            monitor._thresholds = self._numerical_thresholds

        # Convert to StandardFormModel if a high-level Model was passed
        if isinstance(model, Model):
            model_sf = to_standard_form(model)
        elif isinstance(model, StandardFormModel):
            model_sf = model
        else:
            raise TypeError(
                f"Expected Model or StandardFormModel, got {type(model).__name__}."
            )

        total_iterations = 0

        # ── WARM-START CHECK ────────────────────────────────────────────────
        use_warm_start = False
        bf_warm = None
        if initial_basis is not None:
            m_sf, n_sf = model_sf.A.shape
            if len(initial_basis) != m_sf or len(set(initial_basis)) != m_sf or any(idx < 0 or idx >= n_sf for idx in initial_basis):
                raise ValueError(
                    f"Incompatible initial basis of length {len(initial_basis)} for model with {m_sf} rows and {n_sf} columns."
                )
            try:
                B_test = model_sf.A[:, initial_basis]
                bf_test = BasisFactorization(
                    B_test,
                    pivot_tolerance=tol.pivot_tolerance,
                    use_scipy_triangular_solve=tol.use_scipy_triangular_solve,
                    scipy_triangular_min_dimension=tol.scipy_triangular_min_dimension,
                    triangular_solve_provider=tol.triangular_solve_provider,
                    use_scipy_lu_factor=tol.use_scipy_lu_factor,
                    scipy_lu_min_dimension=tol.scipy_lu_min_dimension,
                    lu_factor_provider=tol.lu_factor_provider,
                )
                xB_test = bf_test.solve(model_sf.b)
                if np.all(xB_test >= -tol.feasibility_tolerance):
                    use_warm_start = True
                    bf_warm = bf_test
            except (SingularBasisError, ArithmeticError):
                pass

        if use_warm_start:
            all_cols = set(range(model_sf.A.shape[1]))
            nonbasic_ws = sorted(list(all_cols - set(initial_basis)))
            phase2_res = self._simplex_loop(
                A=model_sf.A,
                b=model_sf.b,
                c=model_sf.c,
                basis_indices=list(initial_basis),
                nonbasic_indices=nonbasic_ws,
                sense=model_sf.objective_sense,
                tolerances=tol,
                initial_iterations=total_iterations,
                numerical_monitor=monitor,
                stop_event=stop_ev,
                initial_basis_factorization=bf_warm,
                pricing=active_pricing,
            )
            total_iterations = phase2_res.iterations
            self.degenerate_pivots_count += phase2_res.degenerate_pivots
            if phase2_res.cycle_fallback_triggered:
                self.fallback_triggered = True
                self.fallback_reason = phase2_res.cycle_fallback_reason
            self.last_iteration_result = phase2_res

            if phase2_res.status != SolutionStatus.OPTIMAL:
                return Solution(
                    status=phase2_res.status,
                    variable_values={},
                    objective_value=None,
                    message=phase2_res.message,
                    iterations=total_iterations,
                )

            var_values = self._extract_original_variables(
                model_sf=model_sf,
                phase2_x=phase2_res.x,
                col_mapping=None,
                tolerances=tol,
            )

            original_obj = phase2_res.objective_value + model_sf.objective_constant
            self.last_standard_form_model = model_sf
            self.last_standard_form_basis_indices = list(phase2_res.basis_indices)
            return Solution(
                status=SolutionStatus.OPTIMAL,
                variable_values=var_values,
                objective_value=original_obj,
                message="Optimal solution found.",
                iterations=total_iterations,
            )

        # ── PHASE I (if artificial variables are present) ────────────────────
        if model_sf.needs_phase1:
            phase1_res = self._simplex_loop(
                A=model_sf.A,
                b=model_sf.b,
                c=model_sf.phase1_c,
                basis_indices=list(model_sf.phase1_basis_indices),
                nonbasic_indices=list(model_sf.phase1_nonbasic_indices),
                sense=ObjectiveSense.MINIMIZE,
                tolerances=tol,
                initial_iterations=total_iterations,
                numerical_monitor=monitor,
                stop_event=stop_ev,
                pricing=active_pricing,
            )
            total_iterations = phase1_res.iterations
            self.degenerate_pivots_count += phase1_res.degenerate_pivots
            if phase1_res.cycle_fallback_triggered:
                self.fallback_triggered = True
                self.fallback_reason = phase1_res.cycle_fallback_reason
                phase2_pricing = "bland"
            else:
                phase2_pricing = active_pricing
            self.last_iteration_result = phase1_res

            if phase1_res.status == SolutionStatus.ITERATION_LIMIT:
                return Solution(
                    status=SolutionStatus.ITERATION_LIMIT,
                    variable_values={},
                    objective_value=None,
                    message=phase1_res.message,
                    iterations=total_iterations,
                )

            if phase1_res.status == SolutionStatus.UNBOUNDED:
                # Phase-I objective sum(a_i) is bounded below by 0, but handle defensively
                return Solution(
                    status=SolutionStatus.UNBOUNDED,
                    variable_values={},
                    objective_value=None,
                    message="Phase I unexpectedly declared unbounded.",
                    iterations=total_iterations,
                )

            # Check feasibility: w* = sum(a_i) must be approximately 0
            w_star = float(np.sum(phase1_res.x[model_sf.artificial_indices]))
            if w_star > tol.phase1_feasibility_tolerance:
                return Solution(
                    status=SolutionStatus.INFEASIBLE,
                    variable_values={},
                    objective_value=None,
                    message=(
                        f"Problem is infeasible; Phase-I objective is {w_star:.6e} "
                        f"(> tolerance {tol.phase1_feasibility_tolerance:.6e})."
                    ),
                    iterations=total_iterations,
                )

            # Reset monitor phase state for Phase II
            if monitor is not None:
                monitor.reset_phase_state()

            # ── Phase-I -> Phase-II Transition: Basis Cleanup ────────────────
            (
                A_phase2,
                b_phase2,
                c_phase2,
                basis_phase2,
                nonbasic_phase2,
                col_mapping,
            ) = self._cleanup_phase1_basis(
                model_sf=model_sf,
                final_basis=phase1_res.basis_indices,
                final_nonbasic=phase1_res.nonbasic_indices,
                tolerances=tol,
            )

            # Inherit Phase-I basis factorization if no artificial variables were basic
            initial_p2_fact = None
            if phase1_res.basis_factorization is not None:
                artificial_set = set(model_sf.artificial_indices)
                if not any(b in artificial_set for b in phase1_res.basis_indices):
                    initial_p2_fact = phase1_res.basis_factorization

            # ── PHASE II ─────────────────────────────────────────────────────
            phase2_res = self._simplex_loop(
                A=A_phase2,
                b=b_phase2,
                c=c_phase2,
                basis_indices=basis_phase2,
                nonbasic_indices=nonbasic_phase2,
                sense=model_sf.objective_sense,
                tolerances=tol,
                initial_iterations=total_iterations,
                numerical_monitor=monitor,
                stop_event=stop_ev,
                initial_basis_factorization=initial_p2_fact,
                pricing=phase2_pricing,
            )
            total_iterations = phase2_res.iterations
            self.degenerate_pivots_count += phase2_res.degenerate_pivots
            if phase2_res.cycle_fallback_triggered:
                self.fallback_triggered = True
                self.fallback_reason = phase2_res.cycle_fallback_reason
            self.last_iteration_result = phase2_res

            if phase2_res.status != SolutionStatus.OPTIMAL:
                return Solution(
                    status=phase2_res.status,
                    variable_values={},
                    objective_value=None,
                    message=phase2_res.message,
                    iterations=total_iterations,
                )

            # Extract variable values for the original model variables
            var_values = self._extract_original_variables(
                model_sf=model_sf,
                phase2_x=phase2_res.x,
                col_mapping=col_mapping,
                tolerances=tol,
            )

            original_obj = phase2_res.objective_value + model_sf.objective_constant
            if col_mapping is not None:
                rev_mapping = {new: old for old, new in col_mapping.items()}
                sf_basis_indices = [rev_mapping[j] for j in phase2_res.basis_indices]
            else:
                sf_basis_indices = list(phase2_res.basis_indices)
            self.last_standard_form_model = model_sf
            self.last_standard_form_basis_indices = sf_basis_indices
            return Solution(
                status=SolutionStatus.OPTIMAL,
                variable_values=var_values,
                objective_value=original_obj,
                message="Optimal solution found.",
                iterations=total_iterations,
            )

        else:
            # ── DIRECT PHASE II (no artificial variables) ────────────────────
            phase2_res = self._simplex_loop(
                A=model_sf.A,
                b=model_sf.b,
                c=model_sf.c,
                basis_indices=list(model_sf.phase1_basis_indices),
                nonbasic_indices=list(model_sf.phase1_nonbasic_indices),
                sense=model_sf.objective_sense,
                tolerances=tol,
                initial_iterations=total_iterations,
                numerical_monitor=monitor,
                stop_event=stop_ev,
                pricing=active_pricing,
            )
            total_iterations = phase2_res.iterations
            self.degenerate_pivots_count += phase2_res.degenerate_pivots
            if phase2_res.cycle_fallback_triggered:
                self.fallback_triggered = True
                self.fallback_reason = phase2_res.cycle_fallback_reason
            self.last_iteration_result = phase2_res

            if phase2_res.status != SolutionStatus.OPTIMAL:
                return Solution(
                    status=phase2_res.status,
                    variable_values={},
                    objective_value=None,
                    message=phase2_res.message,
                    iterations=total_iterations,
                )

            # Map directly
            var_values = self._extract_original_variables(
                model_sf=model_sf,
                phase2_x=phase2_res.x,
                col_mapping=None,
                tolerances=tol,
            )

            original_obj = phase2_res.objective_value + model_sf.objective_constant
            self.last_standard_form_model = model_sf
            self.last_standard_form_basis_indices = list(phase2_res.basis_indices)
            return Solution(
                status=SolutionStatus.OPTIMAL,
                variable_values=var_values,
                objective_value=original_obj,
                message="Optimal solution found.",
                iterations=total_iterations,
            )

    # -----------------------------------------------------------------------
    # Pricing Strategy (Entering Variable Selection)
    # -----------------------------------------------------------------------

    @staticmethod
    def _select_entering_variable(
        pricing: str,
        r_all: np.ndarray,
        nonbasic: list[int],
        sense: ObjectiveSense,
        tol: float,
    ) -> int | None:
        """Select entering variable according to the configured pricing strategy.

        Parameters
        ----------
        pricing : str
            Pricing rule ('bland' or 'dantzig').
        r_all : np.ndarray
            Reduced costs for all variables (length n).
        nonbasic : list[int]
            Nonbasic variable indices, maintained in sorted ascending order.
        sense : ObjectiveSense
            MINIMIZE or MAXIMIZE.
        tol : float
            Optimality tolerance (reduced_cost_tolerance).

        Returns
        -------
        int | None
            Selected entering variable column index, or None if optimal.
        """
        if pricing == "bland":
            if sense is ObjectiveSense.MINIMIZE:
                for j in nonbasic:
                    if r_all[j] < -tol:
                        return j
            else:  # MAXIMIZE
                for j in nonbasic:
                    if r_all[j] > tol:
                        return j
            return None

        elif pricing == "dantzig":
            best_j = None
            if sense is ObjectiveSense.MINIMIZE:
                best_r = -tol
                for j in nonbasic:
                    rj = r_all[j]
                    if rj < best_r:
                        best_r = rj
                        best_j = j
            else:  # MAXIMIZE
                best_r = tol
                for j in nonbasic:
                    rj = r_all[j]
                    if rj > best_r:
                        best_r = rj
                        best_j = j
            return best_j

        else:
            raise ValueError(
                f"Unsupported pricing strategy: '{pricing}'. Supported options are 'bland' and 'dantzig'."
            )

    # -----------------------------------------------------------------------
    # Core Revised Simplex Loop
    # -----------------------------------------------------------------------

    def _simplex_loop(
        self,
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        basis_indices: list[int],
        nonbasic_indices: list[int],
        sense: ObjectiveSense,
        tolerances: SimplexTolerances,
        initial_iterations: int = 0,
        numerical_monitor: NumericalStabilityMonitor | None = None,
        stop_event: Optional[threading.Event] = None,
        initial_basis_factorization: BasisFactorization | None = None,
        pricing: str | None = None,
    ) -> SimplexIterationResult:
        """Execute the Revised Simplex iteration loop.

        Parameters
        ----------
        A : np.ndarray
            Constraint matrix (m x n).
        b : np.ndarray
            Right-hand side vector (m,).
        c : np.ndarray
            Objective coefficient vector (n,).
        basis_indices : list[int]
            Initial basis column indices (length m).
        nonbasic_indices : list[int]
            Initial nonbasic column indices (length n - m).
        sense : ObjectiveSense
            MINIMIZE or MAXIMIZE.
        tolerances : SimplexTolerances
            Numerical tolerances.
        initial_iterations : int
            Prior iteration count to carry forward.
        numerical_monitor : NumericalStabilityMonitor | None
            Optional numerical stability monitor.
        stop_event : Optional[threading.Event]
            Cancellation signal.
        initial_basis_factorization : BasisFactorization | None
            Optional pre-computed basis factorization for the initial basis.

        Returns
        -------
        SimplexIterationResult
            Termination outcome with final basis, x vector, and status.
        """
        m, n = A.shape
        basis = list(basis_indices)
        nonbasic = list(nonbasic_indices)
        iterations = initial_iterations
        inherited_fact = initial_basis_factorization

        active_pricing = pricing if pricing is not None else tolerances.pricing
        if active_pricing not in ("bland", "dantzig"):
            raise ValueError(
                f"Unsupported pricing strategy: '{active_pricing}'. Supported options are 'bland' and 'dantzig'."
            )

        # Anti-cycling tracking state
        consecutive_degenerate_pivots = 0
        visited_bases: set[tuple[int, ...]] = set()
        cycle_fallback_triggered = False
        cycle_fallback_reason: str | None = None
        degenerate_pivots_count = 0

        # Incremental basis factorization prototype (Stage 8.7 & 8.14)
        inc_fact: PivotUpdateFactorization | IncrementalBasisFactorization | None = None
        if tolerances.use_pivot_factorization:
            try:
                inc_fact = PivotUpdateFactorization(
                    A[:, basis],
                    pivot_tolerance=tolerances.pivot_tolerance,
                    max_updates=tolerances.max_basis_updates,
                    residual_tolerance=tolerances.incremental_residual_tolerance,
                    base_factorization=initial_basis_factorization,
                    fast_pivot_update=tolerances.use_fast_pivot_update,
                    use_scipy_triangular_solve=tolerances.use_scipy_triangular_solve,
                    scipy_triangular_min_dimension=tolerances.scipy_triangular_min_dimension,
                    triangular_solve_provider=tolerances.triangular_solve_provider,
                    use_scipy_lu_factor=tolerances.use_scipy_lu_factor,
                    scipy_lu_min_dimension=tolerances.scipy_lu_min_dimension,
                    lu_factor_provider=tolerances.lu_factor_provider,
                )
            except SingularBasisError as err:
                if numerical_monitor is not None:
                    numerical_monitor.record_event(
                        event_type=NumericalEventType.LU_FACTORIZATION_FAILURE,
                        measured_value=0.0,
                        threshold=0.0,
                        iteration=iterations,
                        message=f"Basis matrix B is singular at iteration {iterations}: {err}",
                    )
                raise SingularBasisError(
                    f"Basis matrix B is singular at iteration {iterations}: {err}"
                ) from err
        elif tolerances.use_incremental_basis:
            try:
                inc_fact = IncrementalBasisFactorization(
                    A[:, basis],
                    pivot_tolerance=tolerances.pivot_tolerance,
                    max_updates=tolerances.max_basis_updates,
                    residual_tolerance=tolerances.incremental_residual_tolerance,
                    base_factorization=initial_basis_factorization,
                    use_scipy_triangular_solve=tolerances.use_scipy_triangular_solve,
                    scipy_triangular_min_dimension=tolerances.scipy_triangular_min_dimension,
                    triangular_solve_provider=tolerances.triangular_solve_provider,
                    use_scipy_lu_factor=tolerances.use_scipy_lu_factor,
                    scipy_lu_min_dimension=tolerances.scipy_lu_min_dimension,
                    lu_factor_provider=tolerances.lu_factor_provider,
                )
            except SingularBasisError as err:
                if numerical_monitor is not None:
                    numerical_monitor.record_event(
                        event_type=NumericalEventType.LU_FACTORIZATION_FAILURE,
                        measured_value=0.0,
                        threshold=0.0,
                        iteration=iterations,
                        message=f"Basis matrix B is singular at iteration {iterations}: {err}",
                    )
                raise SingularBasisError(
                    f"Basis matrix B is singular at iteration {iterations}: {err}"
                ) from err

        # Stage 8.13: Pre-allocate reusable buffers for matrix-vector products
        ATy_buf = np.empty(n, dtype=np.float64)
        r_all = np.empty(n, dtype=np.float64)
        x_B: np.ndarray | None = None
        need_solve_xB = True

        while iterations < tolerances.max_iterations:
            # Cooperative cancellation check — safe boundary between pivots
            if stop_event is not None and stop_event.is_set():
                x = np.zeros(n, dtype=np.float64)
                try:
                    B = A[:, basis]
                    bf = BasisFactorization(
                        B,
                        pivot_tolerance=tolerances.pivot_tolerance,
                        use_scipy_triangular_solve=tolerances.use_scipy_triangular_solve,
                        scipy_triangular_min_dimension=tolerances.scipy_triangular_min_dimension,
                        triangular_solve_provider=tolerances.triangular_solve_provider,
                        use_scipy_lu_factor=tolerances.use_scipy_lu_factor,
                        scipy_lu_min_dimension=tolerances.scipy_lu_min_dimension,
                        lu_factor_provider=tolerances.lu_factor_provider,
                    )
                    x_B_cancel = bf.solve(b)
                    for k, col in enumerate(basis):
                        x[col] = max(0.0, float(x_B_cancel[k]))
                except Exception:
                    pass
                return SimplexIterationResult(
                    status=SolutionStatus.ITERATION_LIMIT,
                    basis_indices=basis,
                    nonbasic_indices=nonbasic,
                    x=x,
                    objective_value=float(np.dot(c, x)),
                    iterations=iterations,
                    message="Solver cancelled (stop signal received).",
                )
            # 1. Validate basis invariant
            if len(basis) != m:
                raise ValueError(
                    f"Basis length {len(basis)} does not match row count {m}."
                )
            if len(set(basis)) != m:
                raise ValueError(f"Duplicate column indices in basis: {basis}.")

            # Anti-cycling safeguard B: Basis repetition detection
            # Dantzig's rule does not guarantee termination on degenerate LPs and can cycle
            # indefinitely between a subset of degenerate bases without objective progress
            # (as proven by Beale, 1955). To guarantee finite termination while retaining
            # Dantzig's fast convergence on non-cycling instances, we maintain a canonical
            # sorted basis signature history. If any basis repeats while Dantzig is active,
            # we immediately and permanently switch to Bland's rule for the remainder of
            # the solve, invoking Bland's finite termination theorem without terminating the solve.
            if tolerances.enable_basis_repetition_detection:
                basis_sig = tuple(sorted(basis))
                if basis_sig in visited_bases:
                    if active_pricing != "bland":
                        cycle_fallback_triggered = True
                        cycle_fallback_reason = (
                            f"Basis repetition detected at iteration {iterations}."
                        )
                        # Switch permanently to Bland for the remainder of this solve
                        active_pricing = "bland"
                else:
                    visited_bases.add(basis_sig)

            # Periodic re-solve of x_B every 25 pivots to prevent cumulative floating-point drift
            if (iterations - initial_iterations) % 25 == 0:
                need_solve_xB = True

            # 2. Extract basis matrix B and factorize using BasisFactorization or incremental prototype
            if inc_fact is not None:
                basis_fact = inc_fact
                try:
                    if need_solve_xB or x_B is None:
                        x_B = inc_fact.solve(b, check_residual=True)
                        need_solve_xB = False
                except SingularBasisError as err:
                    if numerical_monitor is not None:
                        numerical_monitor.record_event(
                            event_type=NumericalEventType.LU_FACTORIZATION_FAILURE,
                            measured_value=0.0,
                            threshold=0.0,
                            iteration=iterations,
                            message=f"Basis matrix B is singular at iteration {iterations}: {err}",
                        )
                    raise SingularBasisError(
                        f"Basis matrix B is singular at iteration {iterations}: {err}"
                    ) from err
            elif inherited_fact is not None:
                basis_fact = inherited_fact
                inherited_fact = None
                try:
                    x_B = basis_fact.solve(b)
                    need_solve_xB = False
                except SingularBasisError as err:
                    if numerical_monitor is not None:
                        numerical_monitor.record_event(
                            event_type=NumericalEventType.LU_FACTORIZATION_FAILURE,
                            measured_value=0.0,
                            threshold=0.0,
                            iteration=iterations,
                            message=f"Basis matrix B is singular at iteration {iterations}: {err}",
                        )
                    raise SingularBasisError(
                        f"Basis matrix B is singular at iteration {iterations}: {err}"
                    ) from err
            else:
                B = A[:, basis]
                try:
                    basis_fact = BasisFactorization(
                        B,
                        pivot_tolerance=tolerances.pivot_tolerance,
                        use_scipy_triangular_solve=tolerances.use_scipy_triangular_solve,
                        scipy_triangular_min_dimension=tolerances.scipy_triangular_min_dimension,
                        triangular_solve_provider=tolerances.triangular_solve_provider,
                        use_scipy_lu_factor=tolerances.use_scipy_lu_factor,
                        scipy_lu_min_dimension=tolerances.scipy_lu_min_dimension,
                        lu_factor_provider=tolerances.lu_factor_provider,
                    )
                    # Stage 8.13: Avoid redundant solve(b) when x_B was updated mathematically
                    if need_solve_xB or x_B is None:
                        x_B = basis_fact.solve(b)
                        need_solve_xB = False
                except SingularBasisError as err:
                    # Record LU factorization failure
                    if numerical_monitor is not None:
                        numerical_monitor.record_event(
                            event_type=NumericalEventType.LU_FACTORIZATION_FAILURE,
                            measured_value=0.0,
                            threshold=0.0,
                            iteration=iterations,
                            message=f"Basis matrix B is singular at iteration {iterations}: {err}",
                        )
                    raise SingularBasisError(
                        f"Basis matrix B is singular at iteration {iterations}: {err}"
                    ) from err

            # Record LU factorization diagnostics
            if numerical_monitor is not None:
                numerical_monitor.record_lu_factorization(
                    lu=basis_fact.lu,
                    iteration=iterations,
                    success=True,
                )

            # Numerical feasibility clamp
            recomputed_xB = False
            for k in range(m):
                if -tolerances.feasibility_tolerance <= x_B[k] < 0.0:
                    x_B[k] = 0.0
                elif x_B[k] < -tolerances.feasibility_tolerance:
                    if not recomputed_xB:
                        x_B = basis_fact.solve(b)
                        recomputed_xB = True
                        if -tolerances.feasibility_tolerance <= x_B[k] < 0.0:
                            x_B[k] = 0.0
                            continue
                    # Infeasible basic solution encountered
                    raise ValueError(
                        f"Infeasible basic solution at iteration {iterations}: "
                        f"x_B[{k}] = {x_B[k]:.6e} < -{tolerances.feasibility_tolerance:.6e}"
                    )

            # 3. Solve B^T y = c_B for simplex multipliers (dual vector y) using existing factorization
            c_B = c[basis]
            try:
                y = basis_fact.solve_transpose(c_B)
            except SingularBasisError as err:
                if numerical_monitor is not None:
                    numerical_monitor.record_event(
                        event_type=NumericalEventType.LU_FACTORIZATION_FAILURE,
                        measured_value=0.0,
                        threshold=0.0,
                        iteration=iterations,
                        message=f"Basis transpose B^T is singular at iteration {iterations}: {err}",
                    )
                raise SingularBasisError(
                    f"Basis transpose B^T is singular at iteration {iterations}: {err}"
                ) from err

            # 4. Reduced Costs: r_j = c_j - a_j^T y
            # Compute true dual residual: ||A^T y - c||_∞ over ALL variables
            # Stage 8.13: In-place GEMV to eliminate temporary array allocations
            np.dot(A.T, y, out=ATy_buf)
            np.subtract(c, ATy_buf, out=r_all)
            dual_residual = float(np.max(np.abs(r_all)))

            # Identify candidate entering variables based on optimization sense and pricing strategy
            tol = tolerances.reduced_cost_tolerance
            j_enter = self._select_entering_variable(
                pricing=active_pricing,
                r_all=r_all,
                nonbasic=nonbasic,
                sense=sense,
                tol=tol,
            )

            # 5. Optimality Check
            if j_enter is None:
                # No candidate can improve the objective -> optimal!
                # Refresh x_B to guarantee exact primal feasibility at termination
                x_B = basis_fact.solve(b)
                for k in range(m):
                    if -tolerances.feasibility_tolerance <= x_B[k] < 0.0:
                        x_B[k] = 0.0
                x = np.zeros(n, dtype=np.float64)
                for k, col in enumerate(basis):
                    x[col] = x_B[k]
                obj_val = float(np.dot(c, x))

                # Record final snapshot
                if numerical_monitor is not None:
                    # Compute primal residual ||Ax - b|| (respecting sampling)
                    compute_residual = numerical_monitor.should_compute_residuals(iterations)
                    primal_residual = float(np.linalg.norm(A @ x - b)) if compute_residual else None
                    # Constraint violation: max_i |A_i x - b_i| (same as primal residual but max-norm)
                    constraint_violation = float(np.max(np.abs(A @ x - b))) if compute_residual else None
                    # Bound violation: max_j max(0, -x_j) for x >= 0
                    bound_violation = float(np.max(np.maximum(0.0, -x))) if compute_residual else None
                    # Integrality violation: not applicable in LP (no integer variables)
                    integrality_violation = None
                    # Dual residual is ||A^T y - c||_∞ (true dual residual)
                    # Condition number (respecting sampling)
                    compute_condition = numerical_monitor.should_compute_condition(iterations)
                    condition_estimate = basis_fact.condition_number if compute_condition else None
                    numerical_monitor.record_snapshot(
                        iteration=iterations,
                        primal_residual=primal_residual,
                        dual_residual=dual_residual,
                        constraint_violation=constraint_violation,
                        bound_violation=bound_violation,
                        integrality_violation=integrality_violation,
                        pivot_magnitude=None,
                        condition_estimate=condition_estimate,
                        diagnostics=basis_fact.diagnostics(),
                    )

                return SimplexIterationResult(
                    status=SolutionStatus.OPTIMAL,
                    basis_indices=basis,
                    nonbasic_indices=nonbasic,
                    x=x,
                    objective_value=obj_val,
                    iterations=iterations,
                    message="Optimal solution found.",
                    basis_factorization=basis_fact if (hasattr(basis_fact, "solve") and hasattr(basis_fact, "solve_transpose")) else None,
                    degenerate_pivots=degenerate_pivots_count,
                    cycle_fallback_triggered=cycle_fallback_triggered,
                    cycle_fallback_reason=cycle_fallback_reason,
                    y=np.array(y, copy=True),
                    reduced_costs=np.array(r_all, copy=True),
                )

            # 6. Bland's Rule: j_enter already chosen as smallest-index candidate above

            # 7. Search Direction: Solve B d = a_{j_enter} using existing factorization
            a_enter = A[:, j_enter]
            try:
                d = basis_fact.solve(a_enter)
            except SingularBasisError as err:
                if numerical_monitor is not None:
                    numerical_monitor.record_event(
                        event_type=NumericalEventType.LU_FACTORIZATION_FAILURE,
                        measured_value=0.0,
                        threshold=0.0,
                        iteration=iterations,
                        message=f"Basis matrix B is singular when computing direction: {err}",
                    )
                raise SingularBasisError(
                    f"Basis matrix B is singular when computing direction: {err}"
                ) from err

            # 8. Ratio Test: Select leaving variable among rows where d_i > pivot_tolerance
            # Stage 8.13: Vectorized eligible row scanning and ratio computation
            eligible_mask = d > tolerances.pivot_tolerance
            eligible_indices = np.flatnonzero(eligible_mask)

            if eligible_indices.size == 0:
                # No positive direction components -> unbounded!
                x_B = basis_fact.solve(b)
                for k in range(m):
                    if -tolerances.feasibility_tolerance <= x_B[k] < 0.0:
                        x_B[k] = 0.0
                x = np.zeros(n, dtype=np.float64)
                for k, col in enumerate(basis):
                    x[col] = x_B[k]
                obj_val = float(np.dot(c, x))

                # Record snapshot for unbounded case
                if numerical_monitor is not None:
                    compute_residual = numerical_monitor.should_compute_residuals(iterations)
                    primal_residual = float(np.linalg.norm(A @ x - b)) if compute_residual else None
                    constraint_violation = float(np.max(np.abs(A @ x - b))) if compute_residual else None
                    bound_violation = float(np.max(np.maximum(0.0, -x))) if compute_residual else None
                    integrality_violation = None
                    # Compute true dual residual: ||A^T y - c||_∞
                    ATy = A.T @ y
                    dual_residual = float(np.max(np.abs(ATy - c)))
                    compute_condition = numerical_monitor.should_compute_condition(iterations)
                    condition_estimate = basis_fact.condition_number if compute_condition else None
                    numerical_monitor.record_snapshot(
                        iteration=iterations,
                        primal_residual=primal_residual,
                        dual_residual=dual_residual,
                        constraint_violation=constraint_violation,
                        bound_violation=bound_violation,
                        integrality_violation=integrality_violation,
                        pivot_magnitude=None,
                        condition_estimate=condition_estimate,
                        diagnostics=basis_fact.diagnostics(),
                    )

                return SimplexIterationResult(
                    status=SolutionStatus.UNBOUNDED,
                    basis_indices=basis,
                    nonbasic_indices=nonbasic,
                    x=x,
                    objective_value=obj_val,
                    iterations=iterations,
                    message=(
                        f"Problem is unbounded: entering variable column {j_enter} "
                        "has no positive direction entries in ratio test."
                    ),
                    degenerate_pivots=degenerate_pivots_count,
                    cycle_fallback_triggered=cycle_fallback_triggered,
                    cycle_fallback_reason=cycle_fallback_reason,
                )

            eligible_d = d[eligible_indices]
            eligible_xB = np.maximum(0.0, x_B[eligible_indices])
            ratios = eligible_xB / eligible_d
            min_ratio = np.min(ratios)

            # Bland's rule tie-breaking:
            # Among rows achieving the minimum ratio (within pivot tolerance 1e-11),
            # select the row whose basic variable index is smallest.
            tie_mask = np.abs(ratios - min_ratio) <= 1e-11
            tie_indices = eligible_indices[tie_mask]
            if tie_indices.size == 1:
                i_leave = int(tie_indices[0])
            else:
                i_leave = min(tie_indices, key=lambda i: basis[i])
            leaving_var = basis[i_leave]

            # Record pivot magnitude
            pivot_magnitude = abs(d[i_leave])
            if numerical_monitor is not None:
                numerical_monitor.record_event(
                    event_type=NumericalEventType.PIVOT_MAGNITUDE,
                    measured_value=pivot_magnitude,
                    threshold=numerical_monitor.thresholds.small_pivot_warning,
                    iteration=iterations,
                    message=f"Pivot magnitude {pivot_magnitude:.3e} at row {i_leave}, col {j_enter}",
                    context={"pivot_row": i_leave, "pivot_col": j_enter, "entering_var": j_enter, "leaving_var": leaving_var},
                )

            # Stage 8.13: Mathematical Primal Solution Update:
            # x_B_new = x_B - theta* * d with x_B_new[i_leave] = theta*
            # This preserves exact linear-algebraic primal balance: B_new x_B_new = b
            # while eliminating an expensive linear solve B \ b on every pivot.
            theta_star = max(0.0, float(x_B[i_leave])) / d[i_leave]

            # Anti-cycling safeguard A: Consecutive degenerate pivot counter
            # Degenerate pivots (step length theta == 0 or within numerical tolerance)
            # do not improve the objective value and are the necessary prerequisite for cycling.
            # If the solver undergoes a long unbroken sequence of degenerate pivots under Dantzig,
            # there is high risk of an impending cycle or stall. Switching permanently to Bland's
            # rule ensures finite termination by enforcing monotonic combinatorial basis progression.
            # A pivot is degenerate when step length theta <= degeneracy_tolerance
            if theta_star <= tolerances.degeneracy_tolerance:
                degenerate_pivots_count += 1
                consecutive_degenerate_pivots += 1
                if (
                    tolerances.enable_degenerate_pivot_fallback
                    and active_pricing != "bland"
                    and consecutive_degenerate_pivots >= tolerances.max_consecutive_degenerate_pivots
                ):
                    cycle_fallback_triggered = True
                    cycle_fallback_reason = (
                        f"Consecutive degenerate pivots reached threshold "
                        f"({consecutive_degenerate_pivots}) at iteration {iterations}."
                    )
                    # Switch permanently to Bland for the remainder of this solve
                    active_pricing = "bland"
            else:
                consecutive_degenerate_pivots = 0

            x_B -= theta_star * d
            x_B[i_leave] = theta_star

            # 9. Basis Update
            basis[i_leave] = j_enter
            # Stage 8.13: Maintain sorted nonbasic list using bisect in O(N) rather than O(N log N)
            idx = bisect.bisect_left(nonbasic, j_enter)
            if idx < len(nonbasic) and nonbasic[idx] == j_enter:
                nonbasic.pop(idx)
            bisect.insort(nonbasic, leaving_var)

            if inc_fact is not None:
                prev_refactors = inc_fact.total_refactorizations
                updated = inc_fact.replace_column(i_leave, a_enter, search_direction=d)
                if not updated:
                    need_solve_xB = True
                    if inc_fact.total_refactorizations > prev_refactors:
                        if numerical_monitor is not None:
                            numerical_monitor.record_event(
                                event_type=NumericalEventType.REFACTORIZATION,
                                measured_value=float(inc_fact.total_refactorizations),
                                threshold=float(tolerances.max_basis_updates),
                                iteration=iterations,
                                message=f"Basis refactorization triggered at iteration {iterations}.",
                            )

            # Record iteration snapshot
            if numerical_monitor is not None:
                compute_residual = numerical_monitor.should_compute_residuals(iterations)
                if compute_residual:
                    # Compute primal residual ||Ax - b||
                    x = np.zeros(n, dtype=np.float64)
                    for k, col in enumerate(basis):
                        x[col] = x_B[k]
                    primal_residual = float(np.linalg.norm(A @ x - b))
                    constraint_violation = float(np.max(np.abs(A @ x - b)))
                    bound_violation = float(np.max(np.maximum(0.0, -x)))
                else:
                    primal_residual = None
                    constraint_violation = None
                    bound_violation = None
                integrality_violation = None
                compute_condition = numerical_monitor.should_compute_condition(iterations)
                condition_estimate = basis_fact.condition_number if compute_condition else None

                numerical_monitor.record_snapshot(
                    iteration=iterations,
                    primal_residual=primal_residual,
                    dual_residual=dual_residual,
                    constraint_violation=constraint_violation,
                    bound_violation=bound_violation,
                    integrality_violation=integrality_violation,
                    pivot_magnitude=pivot_magnitude,
                    condition_estimate=condition_estimate,
                    diagnostics=basis_fact.diagnostics(),
                )

            iterations += 1

        # Iteration limit reached
        x = np.zeros(n, dtype=np.float64)
        B = A[:, basis]
        try:
            basis_fact = BasisFactorization(
                B,
                pivot_tolerance=tolerances.pivot_tolerance,
                use_scipy_triangular_solve=tolerances.use_scipy_triangular_solve,
                scipy_triangular_min_dimension=tolerances.scipy_triangular_min_dimension,
                triangular_solve_provider=tolerances.triangular_solve_provider,
                use_scipy_lu_factor=tolerances.use_scipy_lu_factor,
                scipy_lu_min_dimension=tolerances.scipy_lu_min_dimension,
                lu_factor_provider=tolerances.lu_factor_provider,
            )
            x_B = basis_fact.solve(b)
            for k, col in enumerate(basis):
                x[col] = max(0.0, float(x_B[k]))
        except SingularBasisError:
            pass

        obj_val = float(np.dot(c, x))

        # Record final snapshot for iteration limit
        if numerical_monitor is not None:
            compute_residual = numerical_monitor.should_compute_residuals(iterations)
            primal_residual = float(np.linalg.norm(A @ x - b)) if compute_residual else None
            constraint_violation = float(np.max(np.abs(A @ x - b))) if compute_residual else None
            bound_violation = float(np.max(np.maximum(0.0, -x))) if compute_residual else None
            integrality_violation = None
            # Compute true dual residual: ||A^T y - c||_∞ if we have a valid basis
            dual_residual = None
            try:
                B = A[:, basis]
                basis_fact = BasisFactorization(
                    B,
                    pivot_tolerance=tolerances.pivot_tolerance,
                    use_scipy_triangular_solve=tolerances.use_scipy_triangular_solve,
                    scipy_triangular_min_dimension=tolerances.scipy_triangular_min_dimension,
                    triangular_solve_provider=tolerances.triangular_solve_provider,
                    use_scipy_lu_factor=tolerances.use_scipy_lu_factor,
                    scipy_lu_min_dimension=tolerances.scipy_lu_min_dimension,
                    lu_factor_provider=tolerances.lu_factor_provider,
                )
                x_B = basis_fact.solve(b)
                c_B = c[basis]
                y = basis_fact.solve_transpose(c_B)
                ATy = A.T @ y
                dual_residual = float(np.max(np.abs(ATy - c)))
            except (SingularBasisError, np.linalg.LinAlgError):
                pass
            numerical_monitor.record_snapshot(
                iteration=iterations,
                primal_residual=primal_residual,
                dual_residual=dual_residual,
                constraint_violation=constraint_violation,
                bound_violation=bound_violation,
                integrality_violation=integrality_violation,
                pivot_magnitude=None,
                condition_estimate=None,
            )

        return SimplexIterationResult(
            status=SolutionStatus.ITERATION_LIMIT,
            basis_indices=basis,
            nonbasic_indices=nonbasic,
            x=x,
            objective_value=obj_val,
            iterations=iterations,
            message=f"Iteration limit ({tolerances.max_iterations}) reached.",
            degenerate_pivots=degenerate_pivots_count,
            cycle_fallback_triggered=cycle_fallback_triggered,
            cycle_fallback_reason=cycle_fallback_reason,
        )

    # -----------------------------------------------------------------------
    # Phase-I -> Phase-II Basis Cleanup
    # -----------------------------------------------------------------------

    def _cleanup_phase1_basis(
        self,
        model_sf: StandardFormModel,
        final_basis: list[int],
        final_nonbasic: list[int],
        tolerances: SimplexTolerances,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int], dict[int, int]]:
        """Drive out zero-valued basic artificial variables and drop artificial columns.

        Parameters
        ----------
        model_sf : StandardFormModel
            The standard-form model containing Phase-I metadata.
        final_basis : list[int]
            The basis returned by Phase I.
        final_nonbasic : list[int]
            The nonbasic variables returned by Phase I.
        tolerances : SimplexTolerances
            Numerical tolerances.

        Returns
        -------
        tuple
            (A_phase2, b_phase2, c_phase2, basis_phase2, nonbasic_phase2, col_mapping)
        """
        artificial_set = set(model_sf.artificial_indices)
        m = model_sf.num_constraints

        # Fast path: When no artificial variables remain in the final Phase-I basis,
        # skip redundant full-matrix copies and row drive-out loop entirely.
        has_basic_artificial = any(b in artificial_set for b in final_basis)
        if not has_basic_artificial:
            total_cols = model_sf.A.shape[1]
            keep_cols = [j for j in range(total_cols) if j not in artificial_set]
            col_mapping: dict[int, int] = {
                old_idx: new_idx for new_idx, old_idx in enumerate(keep_cols)
            }
            A_phase2 = model_sf.A[:, keep_cols]
            b_phase2 = np.array(model_sf.b, copy=True)
            c_phase2 = model_sf.c[keep_cols]
            basis_phase2 = [col_mapping[old_idx] for old_idx in final_basis]
            nonbasic_phase2 = [
                col_mapping[old_idx]
                for old_idx in final_nonbasic
                if old_idx not in artificial_set
            ]
            nonbasic_phase2.sort()
            return (
                A_phase2,
                b_phase2,
                c_phase2,
                basis_phase2,
                nonbasic_phase2,
                col_mapping,
            )

        curr_A = np.array(model_sf.A, copy=True)
        curr_b = np.array(model_sf.b, copy=True)
        curr_basis = list(final_basis)
        curr_nonbasic = list(final_nonbasic)

        # 1. Drive out basic artificial variables
        redundant_rows: list[int] = []
        for i in range(m):
            if curr_basis[i] in artificial_set:
                # Artificial variable is basic at value ~ 0.
                # Compute tableau row i: e_i^T B^{-1} A
                B = curr_A[:, curr_basis]
                e_i = np.zeros(m, dtype=np.float64)
                e_i[i] = 1.0
                try:
                    basis_fact = BasisFactorization(
                        B,
                        pivot_tolerance=tolerances.pivot_tolerance,
                        use_scipy_triangular_solve=tolerances.use_scipy_triangular_solve,
                        scipy_triangular_min_dimension=tolerances.scipy_triangular_min_dimension,
                        triangular_solve_provider=tolerances.triangular_solve_provider,
                        use_scipy_lu_factor=tolerances.use_scipy_lu_factor,
                        scipy_lu_min_dimension=tolerances.scipy_lu_min_dimension,
                        lu_factor_provider=tolerances.lu_factor_provider,
                    )
                    v = basis_fact.solve_transpose(e_i)
                except SingularBasisError as err:
                    raise SingularBasisError(
                        f"Basis matrix is singular during Phase-I cleanup: {err}"
                    ) from err
                a_bar_i = np.dot(v, curr_A)

                # Find a non-artificial nonbasic column with |a_bar_i[j]| > pivot_tolerance
                pivoted = False
                for j in curr_nonbasic:
                    if j not in artificial_set and abs(a_bar_i[j]) > tolerances.pivot_tolerance:
                        # Pivot non-artificial column j into basis row i
                        leaving = curr_basis[i]
                        curr_basis[i] = j
                        curr_nonbasic.remove(j)
                        curr_nonbasic.append(leaving)
                        curr_nonbasic.sort()
                        pivoted = True
                        break

                if not pivoted:
                    # Row i has no non-artificial columns with non-zero coefficients:
                    # It represents a redundant constraint.
                    redundant_rows.append(i)

        # If redundant rows exist, remove them from curr_A, curr_b, and curr_basis
        if redundant_rows:
            keep_rows = [r for r in range(m) if r not in redundant_rows]
            curr_A = curr_A[keep_rows, :]
            curr_b = curr_b[keep_rows]
            curr_basis = [curr_basis[r] for r in keep_rows]

        # 2. Remove all artificial variable columns
        total_cols = curr_A.shape[1]
        keep_cols = [j for j in range(total_cols) if j not in artificial_set]

        # Mapping: old column index -> new Phase-II column index
        col_mapping: dict[int, int] = {
            old_idx: new_idx for new_idx, old_idx in enumerate(keep_cols)
        }

        A_phase2 = curr_A[:, keep_cols]
        b_phase2 = curr_b
        c_phase2 = model_sf.c[keep_cols]

        basis_phase2 = [col_mapping[old_idx] for old_idx in curr_basis]
        nonbasic_phase2 = [
            col_mapping[old_idx]
            for old_idx in curr_nonbasic
            if old_idx not in artificial_set
        ]
        nonbasic_phase2.sort()

        return (
            A_phase2,
            b_phase2,
            c_phase2,
            basis_phase2,
            nonbasic_phase2,
            col_mapping,
        )

    # -----------------------------------------------------------------------
    # Solution Mapping Helper
    # -----------------------------------------------------------------------

    def _extract_original_variables(
        self,
        model_sf: StandardFormModel,
        phase2_x: np.ndarray,
        col_mapping: dict[int, int] | None,
        tolerances: SimplexTolerances,
    ) -> dict[str, float]:
        """Extract primal values for the original user-facing decision variables."""
        var_values: dict[str, float] = {}
        for var_name, old_col in model_sf.original_variable_mapping.items():
            if col_mapping is not None:
                col = col_mapping[old_col]
            else:
                col = old_col

            val = float(phase2_x[col])
            # Clamp near-zero numerical noise to 0.0
            if abs(val) < tolerances.feasibility_tolerance:
                val = 0.0
            # Apply variable shift: x = y + L
            if old_col in model_sf.variable_shifts:
                val += model_sf.variable_shifts[old_col]
            var_values[var_name] = val

        return var_values
