"""Sparse iterative condition estimation for OptiForge.

Stage 8.49: Sparse Iterative Condition Estimation & Dynamic Scaling.

Implements the Hager (1984) and Higham (1988) iterative 1-norm matrix condition
estimator:
    kappa_1(B) = ||B||_1 * ||B^-1||_1

Key Invariants:
1. Matrix-free inverse estimation: NEVER explicitly forms B^-1.
2. Reuses existing LU factorization solves (forward/backward and transpose substitution).
3. Evaluates ||B^-1||_1 via a convex optimization formulation over the unit 1-ball.
4. Distinguishes:
   - NUMERICALLY_ACCEPTABLE
   - ILL_CONDITIONED
   - SINGULAR
   - UNAVAILABLE
5. Never returns NaN or Inf as a valid safety estimate.
6. Designed for dense matrices and linear operators (sparse-ready).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Sequence

import numpy as np

from optiforge.numerics.exceptions import SingularBasisError


class ConditionStatus(Enum):
    """Categorical assessment of matrix condition number."""

    NUMERICALLY_ACCEPTABLE = "numerically_acceptable"
    """Basis is non-singular and well-conditioned (kappa <= condition_limit)."""

    ILL_CONDITIONED = "ill_conditioned"
    """Basis is non-singular but ill-conditioned (kappa > condition_limit)."""

    SINGULAR = "singular"
    """Basis is mathematically or numerically singular (pivot = 0, rank deficient)."""

    UNAVAILABLE = "unavailable"
    """Condition estimation failed or was inconclusive due to numerical overflow/cancellation."""


@dataclass(frozen=True)
class ConditionDiagnostics:
    """Immutable structured diagnostics for matrix condition estimation.

    Attributes
    ----------
    condition_estimate : float
        Estimated condition number (1-norm or 2-norm). inf if singular.
    status : ConditionStatus
        Structured condition classification.
    converged : bool
        True if the iterative estimator reached stagnation or gradient alignment.
    iterations : int
        Number of matrix-vector solve pairs performed.
    matrix_norm_1 : float
        Exact 1-norm of the matrix ||B||_1 = max_j sum_i |B_{ij}|.
    inverse_norm_1 : float
        Estimated 1-norm of the inverse ||B^-1||_1.
    method : str
        Algorithm name ('hager_higham_1norm', 'svd_2norm').
    details : str
        Human-readable diagnostic summary.
    is_safe : bool
        True if condition <= condition_limit and status is NUMERICALLY_ACCEPTABLE.
    """

    condition_estimate: float
    status: ConditionStatus
    converged: bool
    iterations: int
    matrix_norm_1: float
    inverse_norm_1: float
    method: str
    details: str
    is_safe: bool

    def __repr__(self) -> str:
        safe_str = "SAFE" if self.is_safe else "UNSAFE"
        return (
            f"ConditionDiagnostics({safe_str}, cond={self.condition_estimate:.2e}, "
            f"status={self.status.value}, method='{self.method}', iters={self.iterations})"
        )


@dataclass(frozen=True)
class ConditionComparison:
    """Immutable comparison between Hager-Higham 1-norm and SVD 2-norm condition estimates.

    Attributes
    ----------
    hager_higham_cond : float
        1-norm condition estimate via Hager-Higham iteration.
    svd_cond : float
        2-norm condition estimate via exact Singular Value Decomposition.
    ratio : float
        Ratio kappa_1 / kappa_2.
    order_of_magnitude_match : bool
        True if the two condition estimates agree within bounded norms (ratio in [1/m, m]).
    both_safe : bool
        True if both estimates classify the matrix with identical safety outcomes.
    details : str
        Summary of the cross-check analysis.
    """

    hager_higham_cond: float
    svd_cond: float
    ratio: float
    order_of_magnitude_match: bool
    both_safe: bool
    details: str


def estimate_inverse_1norm(
    solver: Any = None,
    solve_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    solve_transpose_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    dimension: int | None = None,
    max_iterations: int = 10,
    tolerance: float = 1e-7,
) -> tuple[float, bool, int, bool]:
    """Estimate ||B^-1||_1 using the Hager-Higham iterative algorithm.

    Formulation:
        max ||B^-1 x||_1  subject to  ||x||_1 <= 1.

    At each iteration:
        1. y = B^-1 x
        2. xi = sign(y)
        3. z = (B^T)^-1 xi
        4. If ||z||_inf <= z^T x: converged (local maximum reached).
        5. Else x = e_j where j = argmax |z_i|.

    Parameters
    ----------
    solver : Any, optional
        Object providing .solve(rhs), .solve_transpose(rhs), and .m (or shape).
    solve_fn : Callable[[np.ndarray], np.ndarray] | None, optional
        Function to compute B^-1 rhs.
    solve_transpose_fn : Callable[[np.ndarray], np.ndarray] | None, optional
        Function to compute (B^T)^-1 rhs.
    dimension : int | None, optional
        System dimension m.
    max_iterations : int, optional
        Maximum number of iterative power iterations (default 10).
    tolerance : float, optional
        Convergence tolerance for stopping test (default 1e-7).

    Returns
    -------
    tuple[float, bool, int, bool]
        (estimated_norm, converged, iterations, is_singular)
    """
    # Resolve solve functions and dimension
    if solver is not None:
        if solve_fn is None and hasattr(solver, "solve"):
            solve_fn = solver.solve
        if solve_transpose_fn is None and hasattr(solver, "solve_transpose"):
            solve_transpose_fn = solver.solve_transpose
        if dimension is None:
            if hasattr(solver, "m"):
                dimension = int(solver.m)
            elif hasattr(solver, "shape"):
                dimension = int(solver.shape[0])

    if solve_fn is None or solve_transpose_fn is None:
        raise ValueError("solve_fn and solve_transpose_fn must be provided directly or via solver.")
    if dimension is None:
        raise ValueError("System dimension must be provided.")

    m = dimension
    if m <= 0:
        return 0.0, True, 0, False
    if m == 1:
        try:
            y0 = solve_fn(np.array([1.0], dtype=np.float64))
            val = float(abs(y0[0]))
            if not math.isfinite(val) or val > 1e16:
                return float("inf"), False, 1, True
            return val, True, 1, False
        except (SingularBasisError, ArithmeticError, Exception):
            return float("inf"), False, 1, True

    # Initial vector: x = [1/m, ..., 1/m]
    x = np.full(m, 1.0 / m, dtype=np.float64)
    visited_indices: set[int] = set()
    gamma_max = 0.0
    converged = False
    iterations_run = 0

    for k in range(1, max_iterations + 1):
        iterations_run = k
        # 1. Forward solve: y = B^-1 x
        try:
            y = solve_fn(x)
        except (SingularBasisError, ArithmeticError, Exception):
            return float("inf"), False, k, True

        if not np.all(np.isfinite(y)):
            return float("inf"), False, k, True

        # Current 1-norm estimate: gamma = ||y||_1
        gamma = float(np.sum(np.abs(y)))
        if gamma > gamma_max:
            gamma_max = gamma

        # If gamma is extremely large, declare numerical singularity early
        if gamma > 1e16:
            return float("inf"), False, k, True

        # 2. Dual sign vector: xi = sign(y)
        xi = np.where(y >= 0.0, 1.0, -1.0).astype(np.float64)

        # 3. Transpose solve: z = (B^T)^-1 xi
        try:
            z = solve_transpose_fn(xi)
        except (SingularBasisError, ArithmeticError, Exception):
            return float("inf"), False, k, True

        if not np.all(np.isfinite(z)):
            return float("inf"), False, k, True

        # 4. Convergence check: ||z||_inf <= z^T x (after at least one unit-vector step)
        z_abs = np.abs(z)
        z_max = float(np.max(z_abs))
        z_dot_x = float(np.dot(z, x))

        if k > 1 and z_max <= z_dot_x + tolerance:
            converged = True
            break

        # 5. Select next test direction: unit vector e_j corresponding to max |z_j|
        j = int(np.argmax(z_abs))
        if j in visited_indices:
            converged = True
            break

        visited_indices.add(j)
        x = np.zeros(m, dtype=np.float64)
        x[j] = 1.0

    # Higham refinement: check an alternating vector if not converged
    if not converged:
        try:
            alt_x = np.array([(-1.0) ** i for i in range(m)], dtype=np.float64) / m
            alt_y = solve_fn(alt_x)
            if np.all(np.isfinite(alt_y)):
                alt_gamma = float(np.sum(np.abs(alt_y)))
                if alt_gamma > gamma_max:
                    gamma_max = alt_gamma
        except Exception:
            pass

    return gamma_max, converged, iterations_run, False


def estimate_condition_1norm(
    B: np.ndarray,
    solver: Any = None,
    max_iterations: int = 10,
    tolerance: float = 1e-7,
    condition_limit: float = 1e12,
    singularity_tolerance: float = 1e-12,
) -> ConditionDiagnostics:
    """Estimate the 1-norm condition number kappa_1(B) = ||B||_1 * ||B^-1||_1.

    Parameters
    ----------
    B : np.ndarray
        Square matrix of shape (m, m).
    solver : Any, optional
        Pre-factorized solver (e.g. BasisFactorization or DenseLU). If None,
        B will be factorized using dense LU with partial pivoting.
    max_iterations : int, optional
        Maximum Hager-Higham iterations (default 10).
    tolerance : float, optional
        Stopping tolerance (default 1e-7).
    condition_limit : float, optional
        Condition threshold for numerical safety (default 1e12).
    singularity_tolerance : float, optional
        Pivot threshold for singularity detection (default 1e-12).

    Returns
    -------
    ConditionDiagnostics
        Structured immutable condition estimate.
    """
    B_arr = np.asarray(B, dtype=np.float64)
    if B_arr.ndim != 2 or B_arr.shape[0] != B_arr.shape[1]:
        return ConditionDiagnostics(
            condition_estimate=float("inf"),
            status=ConditionStatus.UNAVAILABLE,
            converged=False,
            iterations=0,
            matrix_norm_1=float("inf"),
            inverse_norm_1=float("inf"),
            method="hager_higham_1norm",
            details="Matrix must be square and 2-dimensional.",
            is_safe=False,
        )

    m = B_arr.shape[0]
    if m == 0:
        return ConditionDiagnostics(
            condition_estimate=1.0,
            status=ConditionStatus.NUMERICALLY_ACCEPTABLE,
            converged=True,
            iterations=0,
            matrix_norm_1=0.0,
            inverse_norm_1=0.0,
            method="hager_higham_1norm",
            details="Empty 0x0 matrix is trivially conditioned.",
            is_safe=True,
        )

    # 1. Exact 1-norm of B: maximum absolute column sum
    norm_B = float(np.max(np.sum(np.abs(B_arr), axis=0)))
    if norm_B < singularity_tolerance or not math.isfinite(norm_B):
        return ConditionDiagnostics(
            condition_estimate=float("inf"),
            status=ConditionStatus.SINGULAR,
            converged=True,
            iterations=0,
            matrix_norm_1=norm_B,
            inverse_norm_1=float("inf"),
            method="hager_higham_1norm",
            details=f"Matrix norm {norm_B:.2e} is below singularity threshold.",
            is_safe=False,
        )

    # 2. Factorize B if solver not supplied
    effective_solver = solver
    if effective_solver is None:
        from optiforge.numerics.lu import dense_lu_factor

        try:
            effective_solver = dense_lu_factor(B_arr, pivot_tolerance=singularity_tolerance)
            min_piv = float(np.min(np.abs(np.diag(effective_solver.U))))
            if min_piv < singularity_tolerance:
                return ConditionDiagnostics(
                    condition_estimate=float("inf"),
                    status=ConditionStatus.SINGULAR,
                    converged=True,
                    iterations=0,
                    matrix_norm_1=norm_B,
                    inverse_norm_1=float("inf"),
                    method="hager_higham_1norm",
                    details=f"Singular matrix: min LU pivot {min_piv:.2e} < {singularity_tolerance:.2e}.",
                    is_safe=False,
                )
        except (SingularBasisError, ArithmeticError, Exception) as err:
            return ConditionDiagnostics(
                condition_estimate=float("inf"),
                status=ConditionStatus.SINGULAR,
                converged=False,
                iterations=0,
                matrix_norm_1=norm_B,
                inverse_norm_1=float("inf"),
                method="hager_higham_1norm",
                details=f"LU factorization failed: {err}",
                is_safe=False,
            )

    # 3. Estimate ||B^-1||_1
    inv_norm, converged, iters, is_sing = estimate_inverse_1norm(
        solver=effective_solver,
        dimension=m,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )

    if is_sing or math.isinf(inv_norm) or not math.isfinite(inv_norm):
        return ConditionDiagnostics(
            condition_estimate=float("inf"),
            status=ConditionStatus.SINGULAR,
            converged=converged,
            iterations=iters,
            matrix_norm_1=norm_B,
            inverse_norm_1=float("inf"),
            method="hager_higham_1norm",
            details="Iterative solve encountered singular or non-finite direction.",
            is_safe=False,
        )

    # 4. Condition number kappa_1(B)
    cond = float(norm_B * inv_norm)
    if not math.isfinite(cond) or cond < 1.0:
        cond = max(1.0, cond) if math.isfinite(cond) else float("inf")

    if math.isinf(cond):
        status = ConditionStatus.SINGULAR
        is_safe = False
        details = "Infinite condition number (singular)."
    elif cond > condition_limit:
        status = ConditionStatus.ILL_CONDITIONED
        is_safe = False
        details = f"Ill-conditioned: condition {cond:.2e} > limit {condition_limit:.2e}."
    else:
        status = ConditionStatus.NUMERICALLY_ACCEPTABLE
        is_safe = True
        details = f"Numerically acceptable: condition {cond:.2e} <= limit {condition_limit:.2e}."

    return ConditionDiagnostics(
        condition_estimate=cond,
        status=status,
        converged=converged,
        iterations=iters,
        matrix_norm_1=norm_B,
        inverse_norm_1=inv_norm,
        method="hager_higham_1norm",
        details=details,
        is_safe=is_safe,
    )


def estimate_basis_condition(
    basis_factorization: Any,
    policy: Any = None,
) -> ConditionDiagnostics:
    """Estimate condition number of a BasisFactorization instance.

    Reuses existing LU factors without refactorization.

    Parameters
    ----------
    basis_factorization : BasisFactorization
        Factorized simplex basis.
    policy : NumericalPolicy | None, optional
        Numerical policy containing condition_limit and tolerances.

    Returns
    -------
    ConditionDiagnostics
    """
    from optiforge.numerics.policy import NumericalPolicy

    pol = NumericalPolicy.from_tolerances(policy)
    B = getattr(basis_factorization, "B", None)
    if B is None:
        raise ValueError("BasisFactorization has no underlying basis matrix B.")

    return estimate_condition_1norm(
        B=B,
        solver=basis_factorization,
        max_iterations=pol.condition_estimation_max_iterations,
        tolerance=pol.condition_estimation_tolerance,
        condition_limit=pol.condition_limit,
        singularity_tolerance=pol.basis_singularity_tolerance,
    )


def compare_condition_estimators(
    B: np.ndarray,
    condition_limit: float = 1e12,
) -> ConditionComparison:
    """Compare Hager-Higham 1-norm estimate with exact SVD 2-norm condition number.

    Useful for development diagnostics, validation, and benchmarking.

    Mathematical guarantee:
        (1 / m) * kappa_2(B) <= kappa_1(B) <= m * kappa_2(B)

    Parameters
    ----------
    B : np.ndarray
        Square matrix of shape (m, m).
    condition_limit : float, optional
        Condition threshold (default 1e12).

    Returns
    -------
    ConditionComparison
    """
    B_arr = np.asarray(B, dtype=np.float64)
    m = B_arr.shape[0]

    # 1. Hager-Higham 1-norm
    diag1 = estimate_condition_1norm(B_arr, condition_limit=condition_limit)
    cond1 = diag1.condition_estimate

    # 2. SVD 2-norm
    try:
        s = np.linalg.svdvals(B_arr)
        s_max = float(s[0]) if len(s) > 0 else 0.0
        s_min = float(s[-1]) if len(s) > 0 else 0.0
        cond2 = float(s_max / s_min) if s_min > 1e-15 else float("inf")
    except Exception:
        cond2 = float("inf")

    both_inf = math.isinf(cond1) and math.isinf(cond2)
    both_safe = (cond1 <= condition_limit) == (cond2 <= condition_limit)

    if both_inf:
        ratio = 1.0
        match = True
    elif math.isinf(cond1) or math.isinf(cond2):
        ratio = float("inf")
        match = False
    else:
        ratio = float(cond1 / cond2) if cond2 > 0 else float("inf")
        # Check order of magnitude match: ratio within factor of max(10, m * 5)
        bound = max(10.0, float(m) * 5.0)
        match = (1.0 / bound) <= ratio <= bound

    details = (
        f"1-norm: {cond1:.2e}, 2-norm (SVD): {cond2:.2e}, ratio: {ratio:.2f}, "
        f"match={match}, both_safe={both_safe}"
    )

    return ConditionComparison(
        hager_higham_cond=cond1,
        svd_cond=cond2,
        ratio=ratio,
        order_of_magnitude_match=match,
        both_safe=both_safe,
        details=details,
    )
