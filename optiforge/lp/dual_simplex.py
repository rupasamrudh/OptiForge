"""Dual Revised Simplex solver for Linear Programming (LP).

Implements the Dual Revised Simplex method from first mathematical principles,
designed to solve LPs that are dual feasible but primal infeasible (e.g.,
re-optimizing LP relaxations after bound-tightening branches in MILP).

Mathematical Framework:
-----------------------
Problem:
    min c^T x  (or max c^T x)
    subject to A x = b, x >= 0

Primal basic solution:
    x_B = B^{-1} b,  x_N = 0

Dual basic solution:
    B^T y = c_B  =>  y = B^{-T} c_B

Reduced costs:
    r = c - A^T y

Starting Condition:
    Dual feasible:
        For MIN: r_j >= -tol for all j in nonbasic
        For MAX: r_j <= tol for all j in nonbasic
    Primal infeasible:
        x_B[i] < -tol for at least one basic row i

Dual Simplex Iteration:
    1. Optimality check: If x_B >= -feasibility_tol for all rows, STOP (OPTIMAL).
    2. Leaving row: Choose p with x_B[p] < -feasibility_tol (most infeasible with Bland tie-breaking).
    3. Dual direction: Solve B^T v = e_p.
    4. Tableau row: alpha = A^T v.
    5. Infeasibility check: If alpha_j >= -pivot_tol for all j in nonbasic, STOP (INFEASIBLE).
    6. Ratio test:
           min_{j in nonbasic, alpha_j < -pivot_tol} r_bar_j / (-alpha_j)
       where r_bar_j = r_j for MIN and -r_j for MAX.
    7. Entering column: Choose q achieving min ratio (Bland tie-breaking).
    8. Basis update: basis[p] = q, update factorization, repeat.
"""

from __future__ import annotations

import bisect
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from optiforge.core.solution import Solution, SolutionStatus
from optiforge.lp.profiling import SimplexProfileStats, SimplexProfiler
from optiforge.lp.simplex import SimplexTolerances
from optiforge.lp.standard_form import (
    SingularBasisError,
    StandardFormModel,
    to_standard_form,
)
from optiforge.model.model import Model
from optiforge.model.objective import ObjectiveSense
from optiforge.numerics.basis import BasisFactorization
from optiforge.numerics.incremental_basis import IncrementalBasisFactorization
from optiforge.numerics.pivot_basis import PivotUpdateFactorization
from optiforge.numerics.numerical_monitor import (
    NumericalEventType,
    NumericalStabilityMonitor,
)
from optiforge.numerics.triangular_provider import get_provider_stats


@dataclass
class DualSimplexResult:
    """Internal result of a Dual Revised Simplex solve."""

    status: SolutionStatus
    basis_indices: list[int]
    nonbasic_indices: list[int]
    x: np.ndarray
    y: np.ndarray
    objective_value: float
    iterations: int
    message: str = ""
    basis_factorization: Any | None = None
    btran_y_count: int = 0
    btran_v_count: int = 0
    aty_count: int = 0
    atv_count: int = 0
    profile_stats: Optional[SimplexProfileStats] = None
    reduced_costs: Optional[np.ndarray] = None
    warm_started: bool = False
    initial_basis: tuple[int, ...] | None = None
    initial_primal_feasible: bool = False
    initial_dual_feasible: bool = False
    reoptimization_iterations: int = 0


class DualRevisedSimplexSolver:
    """A mathematically transparent Dual Revised Simplex solver.

    Parameters
    ----------
    tolerances : SimplexTolerances | None, optional
        Numerical tolerances. If None, default tolerances are used.
    numerical_monitor : NumericalStabilityMonitor | None, optional
        Optional numerical stability monitor for diagnostics.
    use_incremental_basis : bool, optional
        Whether to use Stage 8.7 IncrementalBasisFactorization instead of
        DenseLU refactorization on every pivot. Default False.
    use_pivot_factorization : bool, optional
        Whether to use Stage 8.14 PivotUpdateFactorization (PFI) instead of
        DenseLU refactorization on every pivot. Default False.
    use_incremental_reduced_costs : bool | None, optional
        Tri-state configuration for Stage 8.31B incremental reduced-cost and dual
        multiplier updates (Δy = -θ_D v and Δr = θ_D α) in Dual Revised Simplex.
        (Note: Primal Revised Simplex continues to evaluate B^T y = c_B directly for pricing).
        - None (default): AUTO. Automatically activates if and only if
          PivotUpdateFactorization (PFI) is actually active for problem dimension m.
          If DenseLU is active, resolves to OFF.
        - False: Explicit OFF. Strictly forces full BTRAN solves on every pivot.
        - True: Explicit ON. Forces incremental updates; safely falls back to full
          solves if DenseLU factorizes from scratch.

        Behavioral Compatibility Contract:
        - API Compatibility: 100% backward compatible (accepts bool, None, or default).
        - Default Solves: Unconfigured solves run DenseLU, keeping incremental RC OFF.
        - Explicit False / True: Exact behavior preserved.
        - PFI-enabled Solves with Unspecified RC: OLD behavior was OFF (full BTRAN solves);
          NEW behavior is AUTO -> ON when PFI is active. This is an intentional behavioral
          enhancement delivering 1.15x-1.35x speedup without changing final solutions.
    incremental_rc_refresh_interval : int | None, optional
        Periodic full re-evaluation interval for incremental reduced costs.
        Defaults to tolerances.incremental_rc_refresh_interval (25).
    stop_event : threading.Event | None, optional
        Cooperative cancellation event.
    """

    def __init__(
        self,
        tolerances: SimplexTolerances | None = None,
        numerical_monitor: NumericalStabilityMonitor | None = None,
        use_incremental_basis: bool = False,
        use_pivot_factorization: bool = False,
        use_adaptive_pivot_factorization: bool = False,
        adaptive_pfi_min_dimension: int | None = None,
        use_scipy_triangular_solve: bool = False,
        scipy_triangular_min_dimension: int | None = None,
        use_incremental_reduced_costs: bool | None = None,
        incremental_rc_refresh_interval: int | None = None,
        use_optimized_triangular_solves: bool | None = None,
        triangular_solve_provider: str | None = None,
        use_scipy_lu_factor: bool | None = None,
        scipy_lu_min_dimension: int | None = None,
        lu_factor_provider: str | None = None,
        enable_profiling: bool = False,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        self.tolerances: SimplexTolerances = tolerances or SimplexTolerances()
        self.numerical_monitor: NumericalStabilityMonitor | None = numerical_monitor
        self.use_incremental_basis: bool = use_incremental_basis
        self.use_pivot_factorization: bool = use_pivot_factorization
        self.use_adaptive_pivot_factorization: bool = use_adaptive_pivot_factorization
        self.adaptive_pfi_min_dimension: int | None = adaptive_pfi_min_dimension
        opt_tri = (
            use_optimized_triangular_solves
            if use_optimized_triangular_solves is not None
            else getattr(self.tolerances, "use_optimized_triangular_solves", False)
        )
        tri_prov = (
            triangular_solve_provider
            if triangular_solve_provider is not None
            else getattr(self.tolerances, "triangular_solve_provider", "python")
        )
        self.triangular_solve_provider: str = tri_prov
        self.use_scipy_triangular_solve: bool = (
            use_scipy_triangular_solve
            or opt_tri
            or (tri_prov == "scipy")
            or self.tolerances.use_scipy_triangular_solve
        )
        self.scipy_triangular_min_dimension: int = (
            scipy_triangular_min_dimension
            if scipy_triangular_min_dimension is not None
            else getattr(self.tolerances, "scipy_triangular_min_dimension", 10)
        )
        eff_lu_prov = (
            lu_factor_provider
            if lu_factor_provider is not None
            else getattr(self.tolerances, "lu_factor_provider", "python")
        )
        self.lu_factor_provider: str = eff_lu_prov
        self.use_scipy_lu_factor: bool = (
            (use_scipy_lu_factor is True)
            or (eff_lu_prov == "scipy")
            or getattr(self.tolerances, "use_scipy_lu_factor", False)
        )
        self.scipy_lu_min_dimension: int = (
            scipy_lu_min_dimension
            if scipy_lu_min_dimension is not None
            else getattr(self.tolerances, "scipy_lu_min_dimension", 10)
        )
        self.use_incremental_reduced_costs: bool | None = use_incremental_reduced_costs
        self.incremental_rc_refresh_interval: int | None = incremental_rc_refresh_interval
        self.enable_profiling: bool = enable_profiling or getattr(self.tolerances, "enable_profiling", False)
        self.last_profile_stats: SimplexProfileStats | None = None
        self.stop_event: Optional[threading.Event] = stop_event

    def should_use_incremental_reduced_costs(
        self,
        m_or_tolerances: int | SimplexTolerances | None = None,
        tolerances: SimplexTolerances | None = None,
    ) -> bool:
        """Determine whether to use incremental reduced-cost updating.

        Policy (Stage 8.34 Tri-State):
        - None (AUTO): Activates automatically only when PivotUpdateFactorization (PFI)
          is actually active for the problem dimension m.
        - False (Explicit OFF): Strictly disabled, performing full solves on every pivot.
        - True (Explicit ON): Strictly enabled.

        Precedence & Effective Resolution Table:
        Solver setting | Tolerance setting | PFI active | Effective RC mode
        ---------------------------------------------------------------------
        None           | None              | No         | OFF
        None           | None              | Yes        | AUTO -> ON
        None           | False             | Yes/No     | OFF
        None           | True              | Yes/No     | ON
        False          | any               | Yes/No     | OFF
        True           | any               | Yes        | ON
        True           | any               | No         | safe full-solve fallback

        Notes:
        - AUTO couples to ACTUAL runtime PFI activation (m >= threshold for adaptive PFI),
          not merely user intent.
        - Applies exclusively to Dual Revised Simplex; Primal Revised Simplex evaluates
          B^T y = c_B directly for pricing.
        """
        if isinstance(m_or_tolerances, (int, np.integer)):
            m = int(m_or_tolerances)
            tol = tolerances or self.tolerances
        else:
            m = None
            tol = m_or_tolerances if isinstance(m_or_tolerances, SimplexTolerances) else (tolerances or self.tolerances)

        # 1. Instance-level explicit configuration
        if self.use_incremental_reduced_costs is not None:
            return bool(self.use_incremental_reduced_costs)

        # 2. Tolerance-level explicit configuration
        tol_flag = getattr(tol, "use_incremental_reduced_costs", None)
        if tol_flag is not None:
            return bool(tol_flag)

        # 3. AUTO (None): Infer from actual runtime PFI activation
        if m is not None:
            return self.should_use_pivot_factorization(m, tol)
        # If m is omitted, check explicit PFI flags without dimension threshold
        if self.use_pivot_factorization or getattr(tol, "use_pivot_factorization", False):
            return True
        return False

    def get_incremental_rc_refresh_interval(
        self, tolerances: SimplexTolerances | None = None
    ) -> int:
        """Return the refresh interval for incremental reduced-cost updates."""
        if self.incremental_rc_refresh_interval is not None:
            return self.incremental_rc_refresh_interval
        tol = tolerances or self.tolerances
        return getattr(tol, "incremental_rc_refresh_interval", 25)

    def should_use_pivot_factorization(
        self, m: int, tolerances: SimplexTolerances | None = None
    ) -> bool:
        """Determine whether to use PivotUpdateFactorization based on problem dimension m.

        Policy:
        - If explicit `use_pivot_factorization` is True on instance or tolerances: True.
        - Else if `use_adaptive_pivot_factorization` is True on instance or tolerances:
            True if m >= adaptive_pfi_min_dimension, else False.
        - Otherwise: False.
        """
        tol = tolerances or self.tolerances
        if self.use_pivot_factorization or tol.use_pivot_factorization:
            return True
        if self.use_adaptive_pivot_factorization or tol.use_adaptive_pivot_factorization:
            threshold = (
                self.adaptive_pfi_min_dimension
                if self.adaptive_pfi_min_dimension is not None
                else tol.adaptive_pfi_min_dimension
            )
            return m >= threshold
        return False

    def solve(
        self,
        model: Model | StandardFormModel,
        initial_basis: list[int] | None = None,
        tolerances: SimplexTolerances | None = None,
        stop_event: Optional[threading.Event] = None,
    ) -> Solution:
        """Solve an LP using Dual Revised Simplex.

        Parameters
        ----------
        model : Model | StandardFormModel
            The model to solve.
        initial_basis : list[int] | None, optional
            Initial dual-feasible basis indices. If None and model is StandardFormModel,
            uses phase1_basis_indices.
        tolerances : SimplexTolerances | None, optional
            Optional tolerances overriding instance defaults.
        stop_event : threading.Event | None, optional
            Optional cancellation event.

        Returns
        -------
        Solution
            Standard OptiForge Solution container.
        """
        tol = tolerances or self.tolerances
        stop_ev = stop_event if stop_event is not None else self.stop_event

        if stop_ev is not None and stop_ev.is_set():
            return Solution(
                status=SolutionStatus.ITERATION_LIMIT,
                variable_values={},
                objective_value=None,
                message="Solver cancelled (stop signal received).",
                iterations=0,
            )

        # Convert high-level Model to StandardFormModel if needed
        col_mapping = None
        if isinstance(model, Model):
            model_sf = to_standard_form(model)
            col_mapping = model_sf.original_variable_mapping
        elif isinstance(model, StandardFormModel):
            model_sf = model
        else:
            raise TypeError(f"Expected Model or StandardFormModel, got {type(model).__name__}.")

        m, n = model_sf.A.shape

        is_pf = False
        if initial_basis is not None:
            basis = list(initial_basis)
            if len(basis) != m:
                raise ValueError(f"Initial basis length {len(basis)} does not match row count {m}.")
            if len(set(basis)) != m:
                raise ValueError("Initial basis contains duplicate column indices.")
            if any(idx < 0 or idx >= n for idx in basis):
                raise ValueError(f"Initial basis column indices must be within [0, {n - 1}].")

            try:
                bf_init = BasisFactorization(
                    model_sf.A[:, basis],
                    pivot_tolerance=tol.pivot_tolerance,
                    use_scipy_triangular_solve=self.use_scipy_triangular_solve,
                    scipy_triangular_min_dimension=self.scipy_triangular_min_dimension,
                    triangular_solve_provider=self.triangular_solve_provider,
                    use_scipy_lu_factor=self.use_scipy_lu_factor,
                    scipy_lu_min_dimension=self.scipy_lu_min_dimension,
                    lu_factor_provider=self.lu_factor_provider,
                )
            except SingularBasisError as err:
                raise ValueError(f"Incompatible initial basis: basis matrix is singular ({err}).") from err

            cB_init = model_sf.c[basis]
            y_init = bf_init.solve_transpose(cB_init)
            r_init = model_sf.c - np.dot(model_sf.A.T, y_init)
            art_set = set(getattr(model_sf, "artificial_indices", []))
            basis_set = set(basis)
            nonbasic_check = [j for j in range(n) if j not in basis_set and j not in art_set]
            is_min = (model_sf.objective_sense is ObjectiveSense.MINIMIZE)
            if is_min:
                is_df = all(r_init[j] >= -tol.pivot_tolerance for j in nonbasic_check) if nonbasic_check else True
            else:
                is_df = all(r_init[j] <= tol.pivot_tolerance for j in nonbasic_check) if nonbasic_check else True
            if not is_df:
                raise ValueError("Initial basis for Dual Simplex is not dual feasible.")

            xB_init = bf_init.solve(model_sf.b)
            is_pf = bool(np.all(xB_init >= -tol.feasibility_tolerance))
            all_cols = set(range(n))
            nonbasic = sorted(list(all_cols - set(basis) - art_set))
        else:
            basis = list(model_sf.phase1_basis_indices)
            if len(basis) != m:
                raise ValueError(f"Initial basis length {len(basis)} does not match row count {m}.")
            all_cols = set(range(n))
            nonbasic = sorted(list(all_cols - set(basis)))

        res = self.solve_standard_form(
            A=model_sf.A,
            b=model_sf.b,
            c=model_sf.c,
            basis=basis,
            nonbasic=nonbasic,
            sense=model_sf.objective_sense,
            tolerances=tol,
            stop_event=stop_ev,
        )

        if initial_basis is not None:
            res.warm_started = True
            res.initial_basis = tuple(initial_basis)
            res.initial_primal_feasible = is_pf
            res.initial_dual_feasible = True
            res.reoptimization_iterations = res.iterations

        meta = {
            "warm_started": getattr(res, "warm_started", False),
            "initial_basis": getattr(res, "initial_basis", None),
            "initial_primal_feasible": getattr(res, "initial_primal_feasible", False),
            "initial_dual_feasible": getattr(res, "initial_dual_feasible", False),
            "reoptimization_iterations": getattr(res, "reoptimization_iterations", 0),
        }

        if res.status != SolutionStatus.OPTIMAL:
            return Solution(
                status=res.status,
                variable_values={},
                objective_value=None,
                message=res.message,
                iterations=res.iterations,
                profile_stats=res.profile_stats,
                solver_metadata=meta,
            )

        # Extract variable values for original variables with shift adjustments
        if col_mapping is not None:
            var_values = {}
            for vname, col_idx in col_mapping.items():
                val = float(res.x[col_idx])
                if abs(val) < tol.feasibility_tolerance:
                    val = 0.0
                if col_idx in model_sf.variable_shifts:
                    val += model_sf.variable_shifts[col_idx]
                var_values[vname] = val
        else:
            var_values = {}
            for i in range(n):
                val = float(res.x[i])
                if abs(val) < tol.feasibility_tolerance:
                    val = 0.0
                if i in model_sf.variable_shifts:
                    val += model_sf.variable_shifts[i]
                var_values[model_sf.variable_names[i]] = val

        orig_obj = res.objective_value + model_sf.objective_constant
        self.last_standard_form_model = model_sf
        self.last_standard_form_basis_indices = list(res.basis_indices)
        self.last_iteration_result = res
        return Solution(
            status=SolutionStatus.OPTIMAL,
            variable_values=var_values,
            objective_value=orig_obj,
            message="Optimal solution found via Dual Simplex.",
            iterations=res.iterations,
            profile_stats=res.profile_stats,
            solver_metadata=meta,
        )

    def solve_standard_form(
        self,
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        basis: list[int],
        nonbasic: list[int],
        sense: ObjectiveSense,
        tolerances: SimplexTolerances,
        stop_event: Optional[threading.Event] = None,
        initial_basis_factorization: Any | None = None,
    ) -> DualSimplexResult:
        """Core Dual Revised Simplex iteration loop.

        Parameters
        ----------
        A : np.ndarray
            Constraint matrix (m x n).
        b : np.ndarray
            RHS vector (m,).
        c : np.ndarray
            Objective vector (n,).
        basis : list[int]
            Initial basis column indices (length m).
        nonbasic : list[int]
            Initial nonbasic column indices (length n - m).
        sense : ObjectiveSense
            MINIMIZE or MAXIMIZE.
        tolerances : SimplexTolerances
            Numerical tolerances.
        stop_event : threading.Event | None, optional
            Cancellation event.
        initial_basis_factorization : Any | None, optional
            Optional pre-computed or inherited basis factorization (e.g. BorderedBasisFactorization).

        Returns
        -------
        DualSimplexResult
        """
        m, n = A.shape
        current_basis = list(basis)
        current_nonbasic = list(nonbasic)
        iterations = 0

        prof_enabled = getattr(tolerances, "enable_profiling", False) or self.enable_profiling
        profiler = SimplexProfiler(enabled=prof_enabled)

        # Incremental / PFI basis factorization (if enabled)
        inc_fact: PivotUpdateFactorization | IncrementalBasisFactorization | None = None
        use_pfi = self.should_use_pivot_factorization(m, tolerances)
        use_inc = self.use_incremental_basis or tolerances.use_incremental_basis
        use_scipy = self.use_scipy_triangular_solve or getattr(tolerances, "use_scipy_triangular_solve", False)
        min_dim = self.scipy_triangular_min_dimension or getattr(tolerances, "scipy_triangular_min_dimension", 10)
        tri_prov = self.triangular_solve_provider or getattr(tolerances, "triangular_solve_provider", "python")
        use_scipy_lu = (
            (self.use_scipy_lu_factor is True)
            or (self.lu_factor_provider == "scipy")
            or getattr(tolerances, "use_scipy_lu_factor", False)
            or (getattr(tolerances, "lu_factor_provider", "python") == "scipy")
        )
        lu_prov = self.lu_factor_provider or getattr(tolerances, "lu_factor_provider", "python")
        min_dim_lu = self.scipy_lu_min_dimension or getattr(tolerances, "scipy_lu_min_dimension", 10)

        if prof_enabled:
            t_init = time.perf_counter()

        if use_pfi:
            fast_pfi = getattr(tolerances, "use_fast_pivot_update", False)
            try:
                inc_fact = PivotUpdateFactorization(
                    A[:, current_basis],
                    pivot_tolerance=tolerances.pivot_tolerance,
                    max_updates=tolerances.max_basis_updates,
                    residual_tolerance=tolerances.incremental_residual_tolerance,
                    base_factorization=initial_basis_factorization,
                    fast_pivot_update=fast_pfi,
                    use_scipy_triangular_solve=use_scipy,
                    scipy_triangular_min_dimension=min_dim,
                    triangular_solve_provider=tri_prov,
                    use_scipy_lu_factor=use_scipy_lu,
                    scipy_lu_min_dimension=min_dim_lu,
                    lu_factor_provider=lu_prov,
                )
            except SingularBasisError as err:
                # If initial_basis_factorization failed, attempt fallback to fresh DenseLU-backed PFI
                if initial_basis_factorization is not None:
                    try:
                        inc_fact = PivotUpdateFactorization(
                            A[:, current_basis],
                            pivot_tolerance=tolerances.pivot_tolerance,
                            max_updates=tolerances.max_basis_updates,
                            residual_tolerance=tolerances.incremental_residual_tolerance,
                            base_factorization=None,
                            fast_pivot_update=fast_pfi,
                            use_scipy_triangular_solve=use_scipy,
                            scipy_triangular_min_dimension=min_dim,
                            triangular_solve_provider=tri_prov,
                            use_scipy_lu_factor=use_scipy_lu,
                            scipy_lu_min_dimension=min_dim_lu,
                            lu_factor_provider=lu_prov,
                        )
                    except SingularBasisError as err2:
                        if self.numerical_monitor is not None:
                            self.numerical_monitor.record_event(
                                event_type=NumericalEventType.LU_FACTORIZATION_FAILURE,
                                measured_value=0.0,
                                threshold=0.0,
                                iteration=iterations,
                                message=f"Initial dual simplex basis B is singular: {err2}",
                            )
                        raise SingularBasisError(f"Initial dual simplex basis B is singular: {err2}") from err2
                else:
                    if self.numerical_monitor is not None:
                        self.numerical_monitor.record_event(
                            event_type=NumericalEventType.LU_FACTORIZATION_FAILURE,
                            measured_value=0.0,
                            threshold=0.0,
                            iteration=iterations,
                            message=f"Initial dual simplex basis B is singular: {err}",
                        )
                    raise SingularBasisError(f"Initial dual simplex basis B is singular: {err}") from err
        elif use_inc:
            try:
                inc_fact = IncrementalBasisFactorization(
                    A[:, current_basis],
                    pivot_tolerance=tolerances.pivot_tolerance,
                    max_updates=tolerances.max_basis_updates,
                    residual_tolerance=tolerances.incremental_residual_tolerance,
                    base_factorization=initial_basis_factorization,
                    use_scipy_triangular_solve=use_scipy,
                    scipy_triangular_min_dimension=min_dim,
                    triangular_solve_provider=tri_prov,
                    use_scipy_lu_factor=use_scipy_lu,
                    scipy_lu_min_dimension=min_dim_lu,
                    lu_factor_provider=lu_prov,
                )
            except SingularBasisError as err:
                if self.numerical_monitor is not None:
                    self.numerical_monitor.record_event(
                        event_type=NumericalEventType.LU_FACTORIZATION_FAILURE,
                        measured_value=0.0,
                        threshold=0.0,
                        iteration=iterations,
                        message=f"Initial dual simplex basis B is singular: {err}",
                    )
                raise SingularBasisError(f"Initial dual simplex basis B is singular: {err}") from err

        if prof_enabled:
            profiler.record("initial_factorization", time.perf_counter() - t_init)

        use_inc_rc = self.should_use_incremental_reduced_costs(m, tolerances)
        rc_refresh_interval = self.get_incremental_rc_refresh_interval(tolerances)

        # Stage 8.13: Pre-allocate reusable buffers
        ATy_buf = np.empty(n, dtype=np.float64)
        r = np.empty(n, dtype=np.float64)
        alpha = np.empty(n, dtype=np.float64)
        e_p = np.zeros(m, dtype=np.float64)
        y = np.zeros(m, dtype=np.float64)

        need_solve_y = True
        btran_y_count = 0
        btran_v_count = 0
        aty_count = 0
        atv_count = 0

        while iterations < tolerances.max_iterations:
            if stop_event is not None and stop_event.is_set():
                x = np.zeros(n, dtype=np.float64)
                stats = profiler.finish(iterations=iterations, provider_stats=get_provider_stats() if prof_enabled else None) if prof_enabled else None
                self.last_profile_stats = stats
                return DualSimplexResult(
                    status=SolutionStatus.ITERATION_LIMIT,
                    basis_indices=current_basis,
                    nonbasic_indices=current_nonbasic,
                    x=x,
                    y=np.zeros(m, dtype=np.float64),
                    objective_value=float(np.dot(c, x)),
                    iterations=iterations,
                    message="Solver cancelled (stop signal received).",
                    btran_y_count=btran_y_count,
                    btran_v_count=btran_v_count,
                    aty_count=aty_count,
                    atv_count=atv_count,
                    profile_stats=stats,
                )

            # 1. Factorize or reuse basis representation
            if prof_enabled:
                t_ps = time.perf_counter()
            if inc_fact is not None:
                basis_fact = inc_fact
                try:
                    x_B = inc_fact.solve(b, check_residual=True)
                except SingularBasisError as err:
                    # Defensive fallback: if inc_fact fails, fall back to fresh DenseLU
                    try:
                        inc_fact = None
                        B = A[:, current_basis]
                        basis_fact = BasisFactorization(
                            B,
                            pivot_tolerance=tolerances.pivot_tolerance,
                            use_scipy_triangular_solve=use_scipy,
                            scipy_triangular_min_dimension=min_dim,
                            triangular_solve_provider=tri_prov,
                            use_scipy_lu_factor=use_scipy_lu,
                            scipy_lu_min_dimension=min_dim_lu,
                            lu_factor_provider=lu_prov,
                        )
                        x_B = basis_fact.solve(b)
                        need_solve_y = True
                    except (SingularBasisError, ArithmeticError):
                        raise SingularBasisError(f"Basis B singular at iter {iterations}: {err}") from err
            else:
                B = A[:, current_basis]
                try:
                    basis_fact = BasisFactorization(
                        B,
                        pivot_tolerance=tolerances.pivot_tolerance,
                        use_scipy_triangular_solve=use_scipy,
                        scipy_triangular_min_dimension=min_dim,
                        triangular_solve_provider=tri_prov,
                        use_scipy_lu_factor=use_scipy_lu,
                        scipy_lu_min_dimension=min_dim_lu,
                        lu_factor_provider=lu_prov,
                    )
                    x_B = basis_fact.solve(b)
                    need_solve_y = True
                except SingularBasisError as err:
                    if self.numerical_monitor is not None:
                        self.numerical_monitor.record_event(
                            event_type=NumericalEventType.LU_FACTORIZATION_FAILURE,
                            measured_value=0.0,
                            threshold=0.0,
                            iteration=iterations,
                            message=f"Basis B singular at iter {iterations}: {err}",
                        )
                    raise SingularBasisError(f"Basis B singular at iter {iterations}: {err}") from err
            if prof_enabled:
                profiler.record("primal_solve", time.perf_counter() - t_ps)

            # 2. Dual solution: B^T y = c_B & 3. Reduced costs: r = c - A^T y
            if need_solve_y or not use_inc_rc:
                c_B = c[current_basis]
                if prof_enabled:
                    t_by = time.perf_counter()
                try:
                    if hasattr(basis_fact, "residual_tolerance"):
                        y = basis_fact.solve_transpose(c_B, check_residual=True)
                    else:
                        y = basis_fact.solve_transpose(c_B)
                except SingularBasisError as err:
                    if inc_fact is not None:
                        try:
                            inc_fact = None
                            B = A[:, current_basis]
                            basis_fact = BasisFactorization(
                                B,
                                pivot_tolerance=tolerances.pivot_tolerance,
                                use_scipy_triangular_solve=use_scipy,
                                scipy_triangular_min_dimension=min_dim,
                                triangular_solve_provider=tri_prov,
                                use_scipy_lu_factor=use_scipy_lu,
                                scipy_lu_min_dimension=min_dim_lu,
                                lu_factor_provider=lu_prov,
                            )
                            y = basis_fact.solve_transpose(c_B)
                        except (SingularBasisError, ArithmeticError):
                            raise SingularBasisError(f"Basis transpose B^T singular at iter {iterations}: {err}") from err
                    else:
                        raise SingularBasisError(f"Basis transpose B^T singular at iter {iterations}: {err}") from err
                if prof_enabled:
                    profiler.record("btran_y", time.perf_counter() - t_by)
                    t_aty = time.perf_counter()
                np.dot(A.T, y, out=ATy_buf)
                np.subtract(c, ATy_buf, out=r)
                if prof_enabled:
                    profiler.record("gemv_aty", time.perf_counter() - t_aty)
                btran_y_count += 1
                aty_count += 1
                need_solve_y = False

            # 4. Check primal feasibility (Optimality test)
            if prof_enabled:
                t_pricing = time.perf_counter()
            infeasible_mask = x_B < -tolerances.feasibility_tolerance
            infeasible_rows = np.flatnonzero(infeasible_mask)

            if infeasible_rows.size == 0:
                # All x_B >= -feasibility_tolerance: PRIMAL FEASIBLE & DUAL FEASIBLE => OPTIMAL
                if prof_enabled:
                    profiler.record("pricing", time.perf_counter() - t_pricing)
                c_B = c[current_basis]
                if use_inc_rc:
                    try:
                        y = basis_fact.solve_transpose(c_B)
                    except (SingularBasisError, ArithmeticError):
                        pass
                x = np.zeros(n, dtype=np.float64)
                for k, col in enumerate(current_basis):
                    x[col] = max(0.0, float(x_B[k]))
                obj_val = float(np.dot(c, x))

                stats = profiler.finish(iterations=iterations, provider_stats=get_provider_stats() if prof_enabled else None) if prof_enabled else None
                self.last_profile_stats = stats
                return DualSimplexResult(
                    status=SolutionStatus.OPTIMAL,
                    basis_indices=current_basis,
                    nonbasic_indices=current_nonbasic,
                    x=x,
                    y=y,
                    objective_value=obj_val,
                    iterations=iterations,
                    message="Optimal solution found via Dual Simplex.",
                    basis_factorization=basis_fact,
                    btran_y_count=btran_y_count,
                    btran_v_count=btran_v_count,
                    aty_count=aty_count,
                    atv_count=atv_count,
                    profile_stats=stats,
                    reduced_costs=np.array(r, copy=True),
                )

            # 5. Leaving row selection (Dual pricing)
            # Most infeasible row with Bland tie-breaking on basic variable index
            min_val = np.min(x_B[infeasible_rows])
            tie_mask = np.abs(x_B[infeasible_rows] - min_val) <= 1e-11
            tie_rows = infeasible_rows[tie_mask]
            if tie_rows.size == 1:
                p = int(tie_rows[0])
            else:
                p = min(tie_rows, key=lambda i: current_basis[i])
            leaving_var = current_basis[p]
            if prof_enabled:
                profiler.record("pricing", time.perf_counter() - t_pricing)

            # 6. Direction row in dual space (BTRAN): solve B^T v = e_p
            e_p[p] = 1.0
            if prof_enabled:
                t_bv = time.perf_counter()
            try:
                v = basis_fact.solve_transpose(e_p)
            except SingularBasisError as err:
                raise SingularBasisError(f"Solve for direction row failed: {err}") from err
            finally:
                e_p[p] = 0.0
            if prof_enabled:
                profiler.record("btran_v", time.perf_counter() - t_bv)
                t_atv = time.perf_counter()
            btran_v_count += 1

            # Compute tableau row across all nonbasic columns: alpha_j = A[:, j]^T v
            np.dot(A.T, v, out=alpha)
            if prof_enabled:
                profiler.record("gemv_atv", time.perf_counter() - t_atv)
            atv_count += 1

            # 7. Ratio Test: Select entering variable
            if prof_enabled:
                t_rt = time.perf_counter()
            # Eligible nonbasic columns must have alpha_j < -pivot_tolerance
            nonbasic_arr = np.asarray(current_nonbasic, dtype=np.intp)
            alpha_nb = alpha[nonbasic_arr]
            eligible_nb_mask = alpha_nb < -tolerances.pivot_tolerance

            if not np.any(eligible_nb_mask):
                if prof_enabled:
                    profiler.record("ratio_test", time.perf_counter() - t_rt)
                # No candidate can maintain dual feasibility while improving infeasible row
                # Dual is unbounded => Primal is INFEASIBLE
                if use_inc_rc:
                    c_B = c[current_basis]
                    try:
                        y = basis_fact.solve_transpose(c_B)
                    except (SingularBasisError, ArithmeticError):
                        pass
                x = np.zeros(n, dtype=np.float64)
                for k, col in enumerate(current_basis):
                    x[col] = float(x_B[k])
                obj_val = float(np.dot(c, x))
                stats = profiler.finish(iterations=iterations, provider_stats=get_provider_stats() if prof_enabled else None) if prof_enabled else None
                self.last_profile_stats = stats
                return DualSimplexResult(
                    status=SolutionStatus.INFEASIBLE,
                    basis_indices=current_basis,
                    nonbasic_indices=current_nonbasic,
                    x=x,
                    y=y,
                    objective_value=obj_val,
                    iterations=iterations,
                    message=f"Dual Simplex detected primal infeasibility at row {p} (all tableau entries non-negative).",
                    basis_factorization=basis_fact,
                    btran_y_count=btran_y_count,
                    btran_v_count=btran_v_count,
                    aty_count=aty_count,
                    atv_count=atv_count,
                    profile_stats=stats,
                )

            eligible_cols = nonbasic_arr[eligible_nb_mask]
            alpha_eligible = alpha[eligible_cols]
            r_eligible = r[eligible_cols]
            if sense is ObjectiveSense.MINIMIZE:
                r_bar = np.maximum(0.0, r_eligible)
            else:
                r_bar = np.maximum(0.0, -r_eligible)
            ratios = r_bar / (-alpha_eligible)
            min_ratio = np.min(ratios)

            # Bland's rule tie-breaking:
            min_tie_mask = np.abs(ratios - min_ratio) <= 1e-11
            tie_cols = eligible_cols[min_tie_mask]
            if tie_cols.size == 1:
                q = int(tie_cols[0])
            else:
                q = int(np.min(tie_cols))
            entering_var = q
            theta_D = min_ratio
            if prof_enabled:
                profiler.record("ratio_test", time.perf_counter() - t_rt)

            # 8. Pivot & Basis Update
            if prof_enabled:
                t_bu = time.perf_counter()
            current_basis[p] = entering_var
            # Maintain sorted nonbasic in O(N) using bisect
            idx = bisect.bisect_left(current_nonbasic, entering_var)
            if idx < len(current_nonbasic) and current_nonbasic[idx] == entering_var:
                current_nonbasic.pop(idx)
            bisect.insort(current_nonbasic, leaving_var)
            if prof_enabled:
                profiler.record("basis_update", time.perf_counter() - t_bu)

            # 9. Factorization Update
            if prof_enabled:
                t_pfi = time.perf_counter()
            if inc_fact is not None:
                # Search direction for column q: d = B^{-1} A[:, q]
                # Note: B currently represents basis before column p was replaced
                try:
                    prev_refactors = getattr(inc_fact, "total_refactorizations", 0)
                    col_q = A[:, entering_var]
                    d = inc_fact.solve(col_q)
                    updated = inc_fact.replace_column(p, col_q, search_direction=d)
                    curr_refactors = getattr(inc_fact, "total_refactorizations", 0)
                    if not updated or curr_refactors > prev_refactors:
                        need_solve_y = True
                    if not updated and self.numerical_monitor is not None and hasattr(inc_fact, "total_refactorizations"):
                        self.numerical_monitor.record_event(
                            event_type=NumericalEventType.REFACTORIZATION,
                            measured_value=float(inc_fact.total_refactorizations),
                            threshold=float(tolerances.max_basis_updates),
                            iteration=iterations,
                            message=f"Dual Simplex basis refactorization triggered at iteration {iterations}.",
                        )
                except (SingularBasisError, ArithmeticError):
                    # Clean fallback: discard incremental factorization, continue with fresh DenseLU
                    inc_fact = None
                    need_solve_y = True
            else:
                need_solve_y = True
            if prof_enabled:
                profiler.record("pfi_update", time.perf_counter() - t_pfi)

            # 10. Incremental reduced-cost & dual multiplier update (Stage 8.31B)
            if use_inc_rc:
                if (iterations + 1) % rc_refresh_interval == 0:
                    need_solve_y = True
                elif not need_solve_y:
                    if prof_enabled:
                        t_rc = time.perf_counter()
                    if sense is ObjectiveSense.MINIMIZE:
                        y -= theta_D * v
                        r += theta_D * alpha
                    else:
                        y += theta_D * v
                        r -= theta_D * alpha
                    # Basic variable entering basis has zero reduced cost
                    r[entering_var] = 0.0
                    if prof_enabled:
                        profiler.record("rc_update", time.perf_counter() - t_rc)

            iterations += 1

        # Iteration limit reached
        if use_inc_rc:
            c_B = c[current_basis]
            try:
                y = basis_fact.solve_transpose(c_B)
            except (SingularBasisError, ArithmeticError):
                pass
        x = np.zeros(n, dtype=np.float64)
        for k, col in enumerate(current_basis):
            x[col] = float(x_B[k])
        stats = profiler.finish(iterations=iterations, provider_stats=get_provider_stats() if prof_enabled else None) if prof_enabled else None
        self.last_profile_stats = stats
        return DualSimplexResult(
            status=SolutionStatus.ITERATION_LIMIT,
            basis_indices=current_basis,
            nonbasic_indices=current_nonbasic,
            x=x,
            y=y,
            objective_value=float(np.dot(c, x)),
            iterations=iterations,
            message=f"Dual Simplex reached maximum iteration limit ({tolerances.max_iterations}).",
            basis_factorization=basis_fact if inc_fact is not None else None,
            btran_y_count=btran_y_count,
            btran_v_count=btran_v_count,
            aty_count=aty_count,
            atv_count=atv_count,
            profile_stats=stats,
        )
