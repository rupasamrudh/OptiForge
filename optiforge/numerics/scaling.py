"""Iterative Ruiz dynamic scaling and equilibration for OptiForge.

Stage 8.49: Sparse Iterative Condition Estimation & Dynamic Scaling.

Implements iterative Ruiz equilibration (Ruiz, 2001) for LP constraint matrices:
    A_s = D_r A D_c

where D_r = diag(r) and D_c = diag(c) are positive diagonal matrices designed to
equilibrate row and column infinity-norms towards 1.0, reducing numerical
instability, ill-conditioning, and pivot roundoff error.

Transformation Contract:
------------------------
Primal problem:
    min c^T x  s.t.  A x = b, l <= x <= u
Scaled problem:
    x = D_c x_s  <=>  x_s = D_c^-1 x
    A_s = D_r A D_c
    b_s = D_r b
    c_s = D_c c
    l_s = D_c^-1 l,  u_s = D_c^-1 u

Dual problem and reduced costs:
    y = D_r y_s      (dual multipliers for constraints)
    r = D_c^-1 r_s   (reduced costs for variables)

Objective consistency:
    c^T x = c^T (D_c x_s) = (D_c c)^T x_s = c_s^T x_s
    Objective value is mathematically identical between unscaled and scaled forms.

Safety & Invariants:
--------------------
1. NEVER mutates the user's original Model.
2. Completely reversible transformations.
3. Safe handling of zero rows/columns (scaling factor remains 1.0).
4. Strictly positive, bounded scaling factors (clamped to [min_factor, max_factor]).
5. Composes seamlessly with presolve, postsolve, and variable shifts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from optiforge.core.basis import BasisInfo
from optiforge.core.solution import Solution, SolutionStatus
from optiforge.model.constraint import Constraint, ConstraintSense
from optiforge.model.model import Model
from optiforge.model.objective import Objective, ObjectiveSense
from optiforge.model.variable import Variable, VariableType


@dataclass(frozen=True)
class MatrixMagnitudeDiagnostics:
    """Matrix magnitude statistics and dynamic range diagnostics."""

    max_element: float
    min_nonzero_element: float
    dynamic_range: float
    row_max: np.ndarray
    col_max: np.ndarray


def compute_magnitude_diagnostics(A: np.ndarray) -> MatrixMagnitudeDiagnostics:
    """Compute magnitude diagnostics across matrix A."""
    A_arr = np.asarray(A, dtype=np.float64)
    if A_arr.size == 0:
        return MatrixMagnitudeDiagnostics(
            max_element=0.0,
            min_nonzero_element=0.0,
            dynamic_range=0.0,
            row_max=np.zeros(A_arr.shape[0]),
            col_max=np.zeros(A_arr.shape[1] if A_arr.ndim > 1 else 0),
        )

    abs_A = np.abs(A_arr)
    max_el = float(np.max(abs_A)) if abs_A.size > 0 else 0.0
    nonzeros = abs_A[abs_A > 1e-15]
    min_el = float(np.min(nonzeros)) if len(nonzeros) > 0 else 0.0
    dyn_range = float(max_el / min_el) if min_el > 0 else 0.0
    row_max = np.max(abs_A, axis=1) if abs_A.ndim > 1 else abs_A
    col_max = np.max(abs_A, axis=0) if abs_A.ndim > 1 else abs_A

    return MatrixMagnitudeDiagnostics(
        max_element=max_el,
        min_nonzero_element=min_el,
        dynamic_range=dyn_range,
        row_max=row_max,
        col_max=col_max,
    )


@dataclass
class EquilibrationScaling:
    """Simple 1-pass row/column equilibration scaler for backward compatibility."""

    row_scale: np.ndarray
    col_scale: np.ndarray

    @classmethod
    def from_max_norm(cls, A: np.ndarray) -> EquilibrationScaling:
        """Create equilibration scaling based on row maximum norms."""
        A_arr = np.asarray(A, dtype=np.float64)
        m, n = A_arr.shape
        row_max = np.max(np.abs(A_arr), axis=1)
        row_scale = np.where(row_max > 1e-15, 1.0 / row_max, 1.0)
        col_scale = np.ones(n, dtype=np.float64)
        return cls(row_scale=row_scale, col_scale=col_scale)

    def scale_system(
        self,
        A: np.ndarray,
        b: np.ndarray | None = None,
        c: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Scale system matrices and vectors."""
        A_arr = np.asarray(A, dtype=np.float64)
        A_scaled = self.row_scale[:, np.newaxis] * A_arr * self.col_scale[np.newaxis, :]
        b_scaled = (self.row_scale * b) if b is not None else None
        c_scaled = (self.col_scale * c) if c is not None else None
        return A_scaled, b_scaled, c_scaled

    def unscale_primal(self, x_scaled: np.ndarray) -> np.ndarray:
        """Unscale primal variables."""
        return x_scaled * self.col_scale

    def unscale_dual(self, y_scaled: np.ndarray) -> np.ndarray:
        """Unscale dual variables."""
        return y_scaled * self.row_scale


@dataclass(frozen=True)
class RuizScalingResult:
    """Immutable metadata from iterative Ruiz equilibration.

    Attributes
    ----------
    row_scale : np.ndarray
        Diagonal entries r_i of row scaling matrix D_r (length m).
    col_scale : np.ndarray
        Diagonal entries c_j of column scaling matrix D_c (length n).
    iterations : int
        Number of Ruiz equilibration iterations executed.
    converged : bool
        True if max row and column infinity-norm deviations fell within tolerance.
    max_row_imbalance : float
        Max deviation |1 - ||A_s[i, :]||_inf| across non-zero rows.
    max_col_imbalance : float
        Max deviation |1 - ||A_s[:, j]||_inf| across non-zero columns.
    scaling_range : tuple[float, float]
        Minimum and maximum scaling factors across D_r and D_c.
    details : str
        Human-readable diagnostic summary.
    """

    row_scale: np.ndarray
    col_scale: np.ndarray
    iterations: int
    converged: bool
    max_row_imbalance: float
    max_col_imbalance: float
    scaling_range: tuple[float, float]
    details: str

    def __repr__(self) -> str:
        conv_str = "CONVERGED" if self.converged else "UNCONVERGED"
        min_f, max_f = self.scaling_range
        return (
            f"RuizScalingResult({conv_str}, iters={self.iterations}, "
            f"range=[{min_f:.2e}, {max_f:.2e}], "
            f"row_imb={self.max_row_imbalance:.2e}, col_imb={self.max_col_imbalance:.2e})"
        )


@dataclass(frozen=True)
class ModelScalingResult:
    """Structured mapping and metadata for model-level scaling.

    Attributes
    ----------
    original_model : Model
        Reference to the unscaled model.
    scaled_model : Model
        Newly created scaled model instance.
    row_scales : dict[str, float]
        Mapping from constraint name to row scale factor r_i.
    col_scales : dict[str, float]
        Mapping from variable name to column scale factor c_j.
    ruiz_result : RuizScalingResult
        Numerical equilibration diagnostics.
    """

    original_model: Model
    scaled_model: Model
    row_scales: dict[str, float]
    col_scales: dict[str, float]
    ruiz_result: RuizScalingResult


def ruiz_equilibration(
    A: np.ndarray,
    max_iterations: int = 15,
    tolerance: float = 1e-3,
    min_factor: float = 1e-5,
    max_factor: float = 1e5,
    zero_tolerance: float = 1e-12,
) -> tuple[np.ndarray, RuizScalingResult]:
    """Perform iterative Ruiz equilibration on matrix A.

    Goal:
        A_s = D_r A D_c
    such that each non-zero row and column has infinity norm close to 1.0.

    Parameters
    ----------
    A : np.ndarray
        Input matrix of shape (m, n).
    max_iterations : int, optional
        Maximum equilibration rounds (default 15).
    tolerance : float, optional
        Termination tolerance for norm balance (default 1e-3).
    min_factor : float, optional
        Lower clamp threshold for scaling multipliers (default 1e-5).
    max_factor : float, optional
        Upper clamp threshold for scaling multipliers (default 1e5).
    zero_tolerance : float, optional
        Threshold below which a row or column is treated as empty/zero (default 1e-12).

    Returns
    -------
    tuple[np.ndarray, RuizScalingResult]
        (scaled_A, scaling_metadata)
    """
    A_arr = np.asarray(A, dtype=np.float64)
    if A_arr.ndim != 2:
        raise ValueError(f"Matrix A must be 2-dimensional, got shape {A_arr.shape}.")

    m, n = A_arr.shape
    r = np.ones(m, dtype=np.float64)
    c = np.ones(n, dtype=np.float64)

    if m == 0 or n == 0:
        res = RuizScalingResult(
            row_scale=r,
            col_scale=c,
            iterations=0,
            converged=True,
            max_row_imbalance=0.0,
            max_col_imbalance=0.0,
            scaling_range=(1.0, 1.0),
            details="Empty matrix requires no scaling.",
        )
        return A_arr.copy(), res

    A_cur = A_arr.copy()
    converged = False
    iters_run = 0
    max_row_imb = 0.0
    max_col_imb = 0.0

    for k in range(1, max_iterations + 1):
        iters_run = k

        # 1. Row scaling: d_r = 1 / sqrt(||A_{i, :}||_inf)
        row_norms = np.max(np.abs(A_cur), axis=1)
        d_r = np.ones(m, dtype=np.float64)
        for i in range(m):
            if row_norms[i] > zero_tolerance:
                val = 1.0 / math.sqrt(row_norms[i])
                target_r = max(min_factor, min(max_factor, r[i] * val))
                actual_d = target_r / r[i]
                d_r[i] = actual_d
                r[i] = target_r

        A_cur = d_r[:, np.newaxis] * A_cur

        # 2. Column scaling: d_c = 1 / sqrt(||A_{:, j}||_inf)
        col_norms = np.max(np.abs(A_cur), axis=0)
        d_c = np.ones(n, dtype=np.float64)
        for j in range(n):
            if col_norms[j] > zero_tolerance:
                val = 1.0 / math.sqrt(col_norms[j])
                target_c = max(min_factor, min(max_factor, c[j] * val))
                actual_d = target_c / c[j]
                d_c[j] = actual_d
                c[j] = target_c

        A_cur = A_cur * d_c[np.newaxis, :]

        # 3. Convergence check
        cur_row_norms = np.max(np.abs(A_cur), axis=1)
        cur_col_norms = np.max(np.abs(A_cur), axis=0)

        nz_r = cur_row_norms > zero_tolerance
        nz_c = cur_col_norms > zero_tolerance

        row_err = float(np.max(np.abs(cur_row_norms[nz_r] - 1.0))) if np.any(nz_r) else 0.0
        col_err = float(np.max(np.abs(cur_col_norms[nz_c] - 1.0))) if np.any(nz_c) else 0.0

        max_row_imb = row_err
        max_col_imb = col_err

        if row_err <= tolerance and col_err <= tolerance:
            converged = True
            break

    # Recompute final scaled A analytically from original A to prevent floating-point drift
    A_scaled = r[:, np.newaxis] * A_arr * c[np.newaxis, :]

    all_scales = np.concatenate([r, c])
    s_min = float(np.min(all_scales))
    s_max = float(np.max(all_scales))

    details = (
        f"Ruiz equilibration {'converged' if converged else 'terminated'} in {iters_run} iterations; "
        f"scaling factors in [{s_min:.2e}, {s_max:.2e}]."
    )

    res = RuizScalingResult(
        row_scale=r,
        col_scale=c,
        iterations=iters_run,
        converged=converged,
        max_row_imbalance=max_row_imb,
        max_col_imbalance=max_col_imb,
        scaling_range=(s_min, s_max),
        details=details,
    )
    return A_scaled, res


def scale_linear_system(
    A: np.ndarray,
    b: np.ndarray | None = None,
    c: np.ndarray | None = None,
    lower_bounds: np.ndarray | None = None,
    upper_bounds: np.ndarray | None = None,
    scaling_result: RuizScalingResult | None = None,
    policy: Any = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, RuizScalingResult]:
    """Scale an explicit linear programming system (A, b, c, lb, ub).

    Parameters
    ----------
    A : np.ndarray
        Constraint matrix (m, n).
    b : np.ndarray | None, optional
        RHS vector (m,).
    c : np.ndarray | None, optional
        Objective vector (n,).
    lower_bounds : np.ndarray | None, optional
        Variable lower bounds (n,).
    upper_bounds : np.ndarray | None, optional
        Variable upper bounds (n,).
    scaling_result : RuizScalingResult | None, optional
        Pre-computed scaling result. If None, computed from A.
    policy : Any, optional
        Numerical policy with scaling tolerances.

    Returns
    -------
    tuple
        (A_s, b_s, c_s, lb_s, ub_s, scaling_result)
    """
    from optiforge.numerics.policy import NumericalPolicy

    pol = NumericalPolicy.from_tolerances(policy)

    if scaling_result is None:
        A_s, res = ruiz_equilibration(
            A,
            max_iterations=pol.scaling_max_iterations,
            tolerance=pol.scaling_balance_tolerance,
            min_factor=pol.scaling_min_factor,
            max_factor=pol.scaling_max_factor,
            zero_tolerance=pol.zero_tolerance,
        )
    else:
        res = scaling_result
        A_s = res.row_scale[:, np.newaxis] * A * res.col_scale[np.newaxis, :]

    r = res.row_scale
    c_fac = res.col_scale

    b_s = (b * r) if b is not None else None
    c_s = (c * c_fac) if c is not None else None

    lb_s = None
    if lower_bounds is not None:
        lb_s = np.where(
            np.isneginf(lower_bounds),
            -math.inf,
            np.where(np.isposinf(lower_bounds), math.inf, lower_bounds / c_fac),
        )

    ub_s = None
    if upper_bounds is not None:
        ub_s = np.where(
            np.isposinf(upper_bounds),
            math.inf,
            np.where(np.isneginf(upper_bounds), -math.inf, upper_bounds / c_fac),
        )

    return A_s, b_s, c_s, lb_s, ub_s, res


def unscale_solution_vectors(
    x_s: np.ndarray | None = None,
    y_s: np.ndarray | None = None,
    r_s: np.ndarray | None = None,
    scaling_result: RuizScalingResult | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Unscale solution vectors back to original unscaled problem coordinates.

    Relations:
        x = D_c x_s
        y = D_r y_s
        r = D_c^-1 r_s

    Parameters
    ----------
    x_s : np.ndarray | None, optional
        Scaled primal solution.
    y_s : np.ndarray | None, optional
        Scaled dual multipliers.
    r_s : np.ndarray | None, optional
        Scaled reduced costs.
    scaling_result : RuizScalingResult | None, optional
        Scaling metadata containing row_scale and col_scale.

    Returns
    -------
    tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]
        (x, y, r)
    """
    if scaling_result is None:
        return x_s, y_s, r_s

    r = scaling_result.row_scale
    c = scaling_result.col_scale

    x = (x_s * c) if x_s is not None else None
    y = (y_s * r) if y_s is not None else None
    red_cost = (r_s / c) if r_s is not None else None

    return x, y, red_cost


def scale_model(
    model: Model,
    policy: Any = None,
) -> tuple[Model, ModelScalingResult]:
    """Create an equilibrated, mathematically equivalent copy of a user Model.

    Never modifies the original Model.

    Parameters
    ----------
    model : Model
        Original user Model.
    policy : Any, optional
        Numerical policy specifying scaling parameters.

    Returns
    -------
    tuple[Model, ModelScalingResult]
        (scaled_model, model_scaling_result)
    """
    from optiforge.numerics.policy import NumericalPolicy

    pol = NumericalPolicy.from_tolerances(policy)

    var_names = list(model.variables.keys())
    con_names = list(model.constraints.keys())
    n = len(var_names)
    m = len(con_names)

    var_to_idx = {name: j for j, name in enumerate(var_names)}
    A = np.zeros((m, n), dtype=np.float64)

    for i, c_name in enumerate(con_names):
        con = model.constraints[c_name]
        for v_name, coeff in con.coefficients.items():
            if v_name in var_to_idx:
                A[i, var_to_idx[v_name]] = float(coeff)

    # Perform Ruiz equilibration
    _, ruiz_res = ruiz_equilibration(
        A,
        max_iterations=pol.scaling_max_iterations,
        tolerance=pol.scaling_balance_tolerance,
        min_factor=pol.scaling_min_factor,
        max_factor=pol.scaling_max_factor,
        zero_tolerance=pol.zero_tolerance,
    )

    r_vec = ruiz_res.row_scale
    c_vec = ruiz_res.col_scale

    row_scales: dict[str, float] = {con_names[i]: float(r_vec[i]) for i in range(m)}
    col_scales: dict[str, float] = {var_names[j]: float(c_vec[j]) for j in range(n)}

    # Construct new Model with scaled parameters
    scaled_model = Model(name=f"{model.name}_scaled")

    # Add variables with scaled bounds: lb_s = lb / c_j, ub_s = ub / c_j
    for j, name in enumerate(var_names):
        orig_var = model.variables[name]
        c_j = c_vec[j]

        lb = orig_var.lower_bound
        if math.isfinite(lb):
            lb_s = float(lb / c_j)
        else:
            lb_s = float(lb)

        ub = orig_var.upper_bound
        if math.isfinite(ub):
            ub_s = float(ub / c_j)
        else:
            ub_s = float(ub)

        scaled_model.add_variable(
            Variable(
                name=name,
                lower_bound=lb_s,
                upper_bound=ub_s,
                variable_type=orig_var.variable_type,
            )
        )

    # Scaled objective: c_s = c_orig * c_j
    scaled_obj_coeffs: dict[str, float] = {}
    if model.objective is not None:
        for name, coeff in model.objective.coefficients.items():
            c_j = col_scales.get(name, 1.0)
            scaled_obj_coeffs[name] = float(coeff * c_j)

        scaled_model.set_objective(
            Objective(
                coefficients=scaled_obj_coeffs,
                sense=model.objective.sense,
            )
        )

    # Scaled constraints: A_s = r_i * A * c_j, b_s = r_i * b
    for i, name in enumerate(con_names):
        orig_con = model.constraints[name]
        r_i = r_vec[i]

        scaled_con_coeffs: dict[str, float] = {}
        for v_name, coeff in orig_con.coefficients.items():
            c_j = col_scales.get(v_name, 1.0)
            scaled_con_coeffs[v_name] = float(coeff * r_i * c_j)

        scaled_rhs = float(orig_con.rhs * r_i)
        scaled_model.add_constraint(
            Constraint(
                name=name,
                coefficients=scaled_con_coeffs,
                sense=orig_con.sense,
                rhs=scaled_rhs,
            )
        )

    result = ModelScalingResult(
        original_model=model,
        scaled_model=scaled_model,
        row_scales=row_scales,
        col_scales=col_scales,
        ruiz_result=ruiz_res,
    )
    return scaled_model, result


def unscale_model_solution(
    scaled_solution: Solution,
    model_scaling_result: ModelScalingResult,
) -> Solution:
    """Unscale a solved solution back to the original Model's variables and constraints.

    Preserves exact objective values, basis statuses, and problem fingerprints.

    Parameters
    ----------
    scaled_solution : Solution
        Solution produced on the scaled model.
    model_scaling_result : ModelScalingResult
        Scaling mappings and metadata.

    Returns
    -------
    Solution
        Reconstructed solution for the original model.
    """
    if scaled_solution.status is not SolutionStatus.OPTIMAL:
        return scaled_solution

    col_scales = model_scaling_result.col_scales
    row_scales = model_scaling_result.row_scales

    # 1. Unscale primal variables: x = c_j * x_s
    restored_vals: dict[str, float] = {}
    for var_name, val in scaled_solution.variable_values.items():
        c_j = col_scales.get(var_name, 1.0)
        restored_vals[var_name] = float(val * c_j)

    # 2. Objective value is invariant: c^T x = c_s^T x_s
    unscaled_obj = scaled_solution.objective_value

    # 3. Unscale BasisInfo if present
    restored_basis = None
    if scaled_solution.basis_info is not None:
        bi = scaled_solution.basis_info

        # Unscale primal basic values
        restored_pbv: dict[str, float] = {}
        for var_name, val in bi.primal_basic_values.items():
            c_j = col_scales.get(var_name, 1.0)
            restored_pbv[var_name] = float(val * c_j)

        # Unscale reduced costs: r = r_s / c_j
        restored_rc: dict[str, float] = {}
        for var_name, val in bi.reduced_costs.items():
            c_j = col_scales.get(var_name, 1.0)
            restored_rc[var_name] = float(val / c_j)

        # Unscale dual multipliers: y = r_i * y_s
        restored_duals: dict[str, float] = {}
        for con_name, val in bi.dual_multipliers.items():
            r_i = row_scales.get(con_name, 1.0)
            restored_duals[con_name] = float(val * r_i)

        restored_basis = BasisInfo(
            basic_variable_indices=bi.basic_variable_indices,
            nonbasic_variable_indices=bi.nonbasic_variable_indices,
            basic_variables=bi.basic_variables,
            nonbasic_variables=bi.nonbasic_variables,
            variable_status=bi.variable_status,
            primal_basic_values=restored_pbv,
            reduced_costs=restored_rc,
            dual_multipliers=restored_duals,
            objective_value=unscaled_obj if unscaled_obj is not None else bi.objective_value,
            standard_form_dimension=bi.standard_form_dimension,
            solver_selected=bi.solver_selected,
            is_lp=bi.is_lp,
            is_optimal=bi.is_optimal,
            metadata=dict(bi.metadata),
        )

    # 4. Attach scaling metadata
    ruiz = model_scaling_result.ruiz_result
    meta = dict(getattr(scaled_solution, "solver_metadata", {}) or {})
    meta["scaling"] = {
        "applied": True,
        "iterations": ruiz.iterations,
        "converged": ruiz.converged,
        "scaling_range": ruiz.scaling_range,
        "max_row_imbalance": ruiz.max_row_imbalance,
        "max_col_imbalance": ruiz.max_col_imbalance,
    }

    final_solution = Solution(
        status=scaled_solution.status,
        variable_values=restored_vals,
        objective_value=unscaled_obj,
        message=scaled_solution.message,
        iterations=scaled_solution.iterations,
        node_count=scaled_solution.node_count,
        numerical_snapshots=scaled_solution.numerical_snapshots,
        incumbent_objective=scaled_solution.incumbent_objective,
        best_bound=scaled_solution.best_bound,
        optimality_gap=scaled_solution.optimality_gap,
        absolute_gap=scaled_solution.absolute_gap,
        nodes_remaining=scaled_solution.nodes_remaining,
        max_depth=scaled_solution.max_depth,
        feasible_solutions_found=scaled_solution.feasible_solutions_found,
        profile_stats=scaled_solution.profile_stats,
        problem_fingerprint=scaled_solution.problem_fingerprint,
        solver_selected=scaled_solution.solver_selected,
        solver_reason=scaled_solution.solver_reason,
        solver_metadata=meta,
        basis_info=restored_basis,
        sensitivity=scaled_solution.sensitivity,
        diagnostics=scaled_solution.diagnostics,
    )
    final_solution._model = model_scaling_result.original_model
    return final_solution
