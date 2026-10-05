"""Basis quality assessment and numerical safety layer for OptiForge.

Stage 8.48: Numerical Robustness, Basis Quality & Solver Diagnostics.

Evaluates candidate simplex bases for:
1. Structural validity: correct dimension, unique indices, in-bound indices.
2. Nonsingularity & pivot magnitude: min/max diagonal pivots in LU decomposition.
3. Numerical rank: estimated via singular value decomposition.
4. Condition number estimate: 2-norm condition kappa_2(B) = sigma_max / sigma_min.
5. Primal residual: ||B x_B - b||_inf.
6. Dual residual: ||B^T y - c_B||_inf.
7. Complementarity residual: max_j |x_j * r_j|.
8. Numerical safety: overall safety determination preventing unstable warm-starts.

CRITICAL INVARIANTS:
- Does NOT expose raw LU/factorization buffers.
- Does NOT rely on determinant alone (scale-dependent).
- Immutable structured BasisQuality object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from optiforge.model.objective import ObjectiveSense
from optiforge.numerics.exceptions import SingularBasisError
from optiforge.numerics.lu import dense_lu_factor
from optiforge.numerics.policy import NumericalPolicy


@dataclass(frozen=True)
class BasisQuality:
    """Immutable structured evaluation of a candidate simplex basis.

    Attributes
    ----------
    is_valid : bool
        True if the basis satisfies structural requirements (length == m,
        all indices unique and within [0, n - 1]).
    is_numerically_safe : bool
        True if the basis is structurally valid, non-singular, well-conditioned,
        and produces acceptable primal/dual residuals.
    rank : int
        Numerical rank of the m x m basis matrix B.
    min_pivot : float
        Minimum absolute diagonal element min_k |U[k, k]| in LU decomposition.
    max_pivot : float
        Maximum absolute diagonal element max_k |U[k, k]| in LU decomposition.
    condition_estimate : float
        Estimated 2-norm condition number of B. inf if B is singular or computation fails.
    primal_residual : float
        Primal linear system residual ||B x_B - b||_inf.
    dual_residual : float
        Dual linear system residual ||B^T y - c_B||_inf.
    complementarity_residual : float
        Maximum complementarity violation max_j |x_j * r_j| across all standard-form columns.
    details : str
        Human-readable summary of the basis quality.
    warnings : tuple[str, ...]
        Specific numerical warnings flagged during evaluation.
    """

    is_valid: bool
    is_numerically_safe: bool
    rank: int
    min_pivot: float
    max_pivot: float
    condition_estimate: float
    primal_residual: float
    dual_residual: float
    complementarity_residual: float
    details: str
    warnings: tuple[str, ...] = field(default_factory=tuple)
    condition_method: str = "svd_2norm"
    condition_converged: bool = True
    condition_status: str = "numerically_acceptable"

    def __repr__(self) -> str:
        safe_str = "SAFE" if self.is_numerically_safe else "UNSAFE"
        return (
            f"BasisQuality({safe_str}, rank={self.rank}, cond={self.condition_estimate:.2e} "
            f"[{self.condition_method}], min_piv={self.min_pivot:.2e}, "
            f"res_p={self.primal_residual:.2e}, res_d={self.dual_residual:.2e})"
        )


def evaluate_basis_quality(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    basis_indices: Sequence[int],
    sense: ObjectiveSense = ObjectiveSense.MINIMIZE,
    policy: NumericalPolicy | None = None,
) -> BasisQuality:
    """Evaluate candidate basis quality and numerical safety against standard form.

    Parameters
    ----------
    A : np.ndarray
        Standard-form constraint matrix (m x n).
    b : np.ndarray
        Standard-form RHS vector (m,).
    c : np.ndarray
        Standard-form objective vector (n,).
    basis_indices : Sequence[int]
        Column indices forming the candidate basis B = A[:, basis_indices].
    sense : ObjectiveSense, optional
        Objective sense (MINIMIZE or MAXIMIZE). Default MINIMIZE.
    policy : NumericalPolicy | None, optional
        Numerical policy and tolerances. If None, default NumericalPolicy is used.

    Returns
    -------
    BasisQuality
        Immutable diagnostic assessment.
    """
    pol = policy or NumericalPolicy()
    A_arr = np.asarray(A, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    c_arr = np.asarray(c, dtype=np.float64)

    if A_arr.ndim != 2:
        return BasisQuality(
            is_valid=False,
            is_numerically_safe=False,
            rank=0,
            min_pivot=0.0,
            max_pivot=0.0,
            condition_estimate=float("inf"),
            primal_residual=float("inf"),
            dual_residual=float("inf"),
            complementarity_residual=float("inf"),
            details="Constraint matrix A is not 2-dimensional.",
            warnings=("invalid_matrix_dimension",),
        )

    m, n = A_arr.shape
    basis_list = list(basis_indices)

    # 1. Structural Validation
    structural_warnings: list[str] = []
    if len(basis_list) != m:
        structural_warnings.append(
            f"basis_length_mismatch: expected {m} basic indices, got {len(basis_list)}"
        )
    if len(set(basis_list)) != len(basis_list):
        structural_warnings.append("duplicate_basis_indices")
    if any(idx < 0 or idx >= n for idx in basis_list):
        structural_warnings.append("basis_indices_out_of_bounds")

    if structural_warnings:
        return BasisQuality(
            is_valid=False,
            is_numerically_safe=False,
            rank=0,
            min_pivot=0.0,
            max_pivot=0.0,
            condition_estimate=float("inf"),
            primal_residual=float("inf"),
            dual_residual=float("inf"),
            complementarity_residual=float("inf"),
            details="; ".join(structural_warnings),
            warnings=tuple(structural_warnings),
        )

    # Extract basis submatrix B (m x m)
    B = A_arr[:, basis_list]
    warnings_list: list[str] = []

    # 2. LU Factorization & Pivot Magnitudes
    try:
        lu = dense_lu_factor(
            B,
            pivot_tolerance=pol.basis_singularity_tolerance,
        )
        diag_u = np.abs(np.diag(lu.U))
        min_piv = float(np.min(diag_u)) if len(diag_u) > 0 else 0.0
        max_piv = float(np.max(diag_u)) if len(diag_u) > 0 else 0.0
    except (SingularBasisError, Exception) as err:
        min_piv = 0.0
        max_piv = 0.0
        warnings_list.append(f"lu_factorization_failed: {err}")
        return BasisQuality(
            is_valid=True,
            is_numerically_safe=False,
            rank=0,
            min_pivot=min_piv,
            max_pivot=max_piv,
            condition_estimate=float("inf"),
            primal_residual=float("inf"),
            dual_residual=float("inf"),
            complementarity_residual=float("inf"),
            details=f"Singular basis: LU factorization failed ({err}).",
            warnings=tuple(warnings_list),
            condition_method="none",
            condition_converged=False,
            condition_status="singular",
        )

    # 3. Condition Number & Numerical Rank Estimation
    method_pref = pol.condition_estimation_method.lower()
    use_iterative = (
        method_pref in ("iterative", "hager_higham")
        or (method_pref == "auto" and m > 200)
    )

    if use_iterative:
        from optiforge.numerics.condition import ConditionStatus, estimate_condition_1norm

        cond_diag = estimate_condition_1norm(
            B=B,
            solver=lu,
            max_iterations=pol.condition_estimation_max_iterations,
            tolerance=pol.condition_estimation_tolerance,
            condition_limit=pol.condition_limit,
            singularity_tolerance=pol.basis_singularity_tolerance,
        )
        cond_est = cond_diag.condition_estimate
        cond_method = cond_diag.method
        cond_converged = cond_diag.converged
        cond_status = cond_diag.status.value
        # Rank from LU pivots
        rank = int(np.count_nonzero(diag_u >= pol.basis_singularity_tolerance))
        if not cond_converged:
            warnings_list.append("condition_estimation_unconverged")
    else:
        # SVD gold standard for small/dense bases (m <= 200)
        try:
            s = np.linalg.svdvals(B)
            s_max = float(s[0]) if len(s) > 0 else 0.0
            s_min = float(s[-1]) if len(s) > 0 else 0.0

            eps = float(np.finfo(np.float64).eps)
            tol_rank = s_max * max(m, m) * eps
            rank = int(np.count_nonzero(s > tol_rank))

            if s_min > 1e-15:
                cond_est = float(s_max / s_min)
            else:
                cond_est = float("inf")
        except Exception:
            cond_est = float("inf")
            rank = 0

        cond_method = "svd_2norm"
        cond_converged = True
        if math.isinf(cond_est):
            cond_status = "singular"
        elif cond_est > pol.condition_limit:
            cond_status = "ill_conditioned"
        else:
            cond_status = "numerically_acceptable"

    # 4. Primal Linear System Solve: B x_B = b
    try:
        xB = lu.solve(b_arr)
        if not np.all(np.isfinite(xB)):
            warnings_list.append("non_finite_primal_solution")
            res_p = float("inf")
        else:
            res_p = float(np.max(np.abs(B @ xB - b_arr)))
    except Exception as err:
        warnings_list.append(f"primal_solve_failed: {err}")
        xB = np.zeros(m)
        res_p = float("inf")

    # 5. Dual Linear System Solve: B^T y = c_B
    cB = c_arr[basis_list]
    try:
        y = lu.solve_transpose(cB)
        if not np.all(np.isfinite(y)):
            warnings_list.append("non_finite_dual_solution")
            res_d = float("inf")
            r = np.zeros(n)
        else:
            res_d = float(np.max(np.abs(B.T @ y - cB)))
            r = c_arr - A_arr.T @ y
    except Exception as err:
        warnings_list.append(f"dual_solve_failed: {err}")
        y = np.zeros(m)
        res_d = float("inf")
        r = np.zeros(n)

    # 6. Complementary Slackness Residual
    # Full standard-form vector x: x_B in basis columns, 0.0 nonbasic
    x_full = np.zeros(n, dtype=np.float64)
    if np.all(np.isfinite(xB)):
        for i, col in enumerate(basis_list):
            x_full[col] = xB[i]
        comp_res = float(np.max(np.abs(x_full * r)))
    else:
        comp_res = float("inf")

    # 7. Numerical Safety Determination
    is_safe = True
    if rank < m:
        is_safe = False
        warnings_list.append(f"deficient_rank: rank {rank} < dimension {m}")

    if min_piv < pol.basis_singularity_tolerance:
        is_safe = False
        warnings_list.append(
            f"singular_pivot: min pivot {min_piv:.2e} < singularity tol {pol.basis_singularity_tolerance:.2e}"
        )
    elif min_piv < pol.min_safe_pivot:
        # Near singular pivot
        is_safe = False
        warnings_list.append(
            f"small_pivot: min pivot {min_piv:.2e} < safe threshold {pol.min_safe_pivot:.2e}"
        )

    if math.isinf(cond_est) or cond_est > pol.condition_limit:
        is_safe = False
        warnings_list.append(
            f"ill_conditioned: condition estimate {cond_est:.2e} > limit {pol.condition_limit:.2e}"
        )

    if res_p > pol.max_safe_residual:
        is_safe = False
        warnings_list.append(
            f"large_primal_residual: {res_p:.2e} > threshold {pol.max_safe_residual:.2e}"
        )

    if res_d > pol.max_safe_residual:
        is_safe = False
        warnings_list.append(
            f"large_dual_residual: {res_d:.2e} > threshold {pol.max_safe_residual:.2e}"
        )

    details = "Numerically safe basis" if is_safe else f"Numerically unsafe basis: {'; '.join(warnings_list)}"

    return BasisQuality(
        is_valid=True,
        is_numerically_safe=is_safe,
        rank=rank,
        min_pivot=min_piv,
        max_pivot=max_piv,
        condition_estimate=cond_est,
        primal_residual=res_p,
        dual_residual=res_d,
        complementarity_residual=comp_res,
        details=details,
        warnings=tuple(warnings_list),
        condition_method=cond_method,
        condition_converged=cond_converged,
        condition_status=cond_status,
    )
