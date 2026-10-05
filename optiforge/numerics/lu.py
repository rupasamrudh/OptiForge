"""Dense LU Factorization with Partial Pivoting.

This module provides a from-scratch implementation of dense LU factorization
with row partial pivoting, forward substitution, backward substitution, and
transpose solve capabilities for square linear systems:

    P B = L U

where:
    P : Permutation matrix (stored as pivot permutation indices)
    L : Unit lower triangular matrix (L[i, i] == 1.0)
    U : Upper triangular matrix

Transpose solve:
    B^T x = rhs  <=>  (P^T L U)^T x = rhs  <=>  U^T L^T P x = rhs
Reuses the existing L and U factors directly without refactorizing B^T.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from optiforge.numerics.exceptions import NumericalSingularityError


@dataclass
class DenseLU:
    """Stores the factors of a permutation-aware LU decomposition.

    P B = L U

    Attributes
    ----------
    L : np.ndarray
        Unit lower triangular matrix of shape (m, m).
    U : np.ndarray
        Upper triangular matrix of shape (m, m).
    perm : np.ndarray
        Integer array of shape (m,) where row i of (P B) is row perm[i] of B.
    det_sign : int
        Sign of the permutation determinant (+1 or -1).
    pivot_tolerance : float
        Numerical pivot threshold used during factorization.
    """

    L: np.ndarray
    U: np.ndarray
    perm: np.ndarray
    det_sign: int
    pivot_tolerance: float
    use_scipy_triangular_solve: bool = False
    scipy_triangular_min_dimension: int = 10
    use_scipy_lu_factor: bool = False
    scipy_lu_min_dimension: int = 10
    triangular_solve_provider: str = "python"
    lu_factor_provider: str = "python"

    def __post_init__(self) -> None:
        if self.use_scipy_triangular_solve:
            self.triangular_solve_provider = "scipy"
        elif self.triangular_solve_provider == "scipy":
            self.use_scipy_triangular_solve = True

        if self.use_scipy_lu_factor:
            self.lu_factor_provider = "scipy"
        elif self.lu_factor_provider == "scipy":
            self.use_scipy_lu_factor = True

    @property
    def m(self) -> int:
        """Dimension of the square matrix."""
        return int(self.L.shape[0])

    @property
    def P(self) -> np.ndarray:
        """Construct the explicit (m, m) permutation matrix P.

        P[i, perm[i]] = 1.0
        """
        P_mat = np.zeros((self.m, self.m), dtype=np.float64)
        for i in range(self.m):
            P_mat[i, self.perm[i]] = 1.0
        return P_mat

    def solve(
        self,
        rhs: np.ndarray,
        use_scipy: bool | None = None,
        min_dimension: int | None = None,
        provider: str | None = None,
    ) -> np.ndarray:
        """Solve B x = rhs using the stored LU factors."""
        prov = provider if provider is not None else self.triangular_solve_provider
        scipy_flag = self.use_scipy_triangular_solve if use_scipy is None else use_scipy
        effective_prov = "scipy" if (scipy_flag or prov == "scipy") else "python"
        return lu_solve(
            self,
            rhs,
            use_scipy=scipy_flag,
            min_dimension=self.scipy_triangular_min_dimension if min_dimension is None else min_dimension,
            provider=effective_prov,
        )

    def solve_transpose(
        self,
        rhs: np.ndarray,
        use_scipy: bool | None = None,
        min_dimension: int | None = None,
        provider: str | None = None,
    ) -> np.ndarray:
        """Solve B^T x = rhs using the stored LU factors."""
        prov = provider if provider is not None else self.triangular_solve_provider
        scipy_flag = self.use_scipy_triangular_solve if use_scipy is None else use_scipy
        effective_prov = "scipy" if (scipy_flag or prov == "scipy") else "python"
        return lu_solve_transpose(
            self,
            rhs,
            use_scipy=scipy_flag,
            min_dimension=self.scipy_triangular_min_dimension if min_dimension is None else min_dimension,
            provider=effective_prov,
        )


def dense_lu_factor_reference(
    B: np.ndarray,
    pivot_tolerance: float = 1e-12,
    use_scipy_triangular_solve: bool = False,
    scipy_triangular_min_dimension: int = 10,
) -> DenseLU:
    """Historical baseline dense LU factorization reference implementation.

    Preserved unoptimized for Stage 8.23 verification and compatibility oracle.
    """
    B_arr = np.asarray(B, dtype=np.float64)
    if B_arr.ndim != 2:
        raise ValueError(f"Matrix B must be 2-dimensional, got ndim={B_arr.ndim}.")
    m, n = B_arr.shape
    if m != n:
        raise ValueError(f"Matrix B must be square, got shape ({m}, {n}).")

    if m == 0:
        return DenseLU(
            L=np.zeros((0, 0), dtype=np.float64),
            U=np.zeros((0, 0), dtype=np.float64),
            perm=np.zeros(0, dtype=np.int64),
            det_sign=1,
            pivot_tolerance=pivot_tolerance,
        )

    U = np.array(B_arr, copy=True, dtype=np.float64)
    L = np.eye(m, dtype=np.float64)
    perm = np.arange(m, dtype=np.int64)
    det_sign = 1

    for k in range(m):
        col_abs = np.abs(U[k:, k])
        p_rel = int(col_abs.argmax())
        p = k + p_rel
        pivot_val = U[p, k]
        pivot_mag = abs(pivot_val)

        if pivot_mag < pivot_tolerance:
            raise NumericalSingularityError(
                f"Matrix is singular or numerically singular at column {k}: "
                f"pivot magnitude {pivot_mag:.3e} below tolerance {pivot_tolerance:.3e}.",
                pivot=float(pivot_val),
                step=k,
            )

        if p != k:
            U[[k, p], :] = U[[p, k], :]
            if k > 0:
                L[[k, p], :k] = L[[p, k], :k]
            perm[k], perm[p] = perm[p], perm[k]
            det_sign = -det_sign

        if k < m - 1:
            pivot = U[k, k]
            factors = U[k + 1:, k] / pivot
            L[k + 1:, k] = factors
            U[k + 1:, k + 1:] -= factors[:, None] * U[k, k + 1:]
            U[k + 1:, k] = 0.0

    return DenseLU(
        L=L,
        U=U,
        perm=perm,
        det_sign=det_sign,
        pivot_tolerance=pivot_tolerance,
        use_scipy_triangular_solve=use_scipy_triangular_solve,
        scipy_triangular_min_dimension=scipy_triangular_min_dimension,
    )


_triangular_lu_solve = None
_triangular_lu_solve_transpose = None
_get_lu_provider_fn = None


def _get_triangular_solvers():
    global _triangular_lu_solve, _triangular_lu_solve_transpose
    if _triangular_lu_solve is None:
        from optiforge.numerics.triangular_provider import (
            triangular_lu_solve,
            triangular_lu_solve_transpose,
        )
        _triangular_lu_solve = triangular_lu_solve
        _triangular_lu_solve_transpose = triangular_lu_solve_transpose
    return _triangular_lu_solve, _triangular_lu_solve_transpose


def _get_lu_provider_cached():
    global _get_lu_provider_fn
    if _get_lu_provider_fn is None:
        from optiforge.numerics.lu_provider import get_lu_provider
        _get_lu_provider_fn = get_lu_provider
    return _get_lu_provider_fn


def dense_lu_factor(
    B: np.ndarray,
    pivot_tolerance: float = 1e-12,
    use_scipy_triangular_solve: bool = False,
    scipy_triangular_min_dimension: int = 10,
    use_scipy_lu_factor: bool = False,
    scipy_lu_min_dimension: int = 10,
    triangular_solve_provider: str = "python",
    lu_factor_provider: str | None = None,
) -> DenseLU:
    """Compute the dense LU factorization with partial pivoting: P B = L U.

    Parameters
    ----------
    B : np.ndarray
        Square matrix of shape (m, m).
    pivot_tolerance : float, optional
        Threshold below which a pivot element |B[p, k]| is deemed singular.
        Default is 1e-12.
    use_scipy_triangular_solve : bool, optional
        Whether to attempt SciPy solve_triangular acceleration for subsequent solves.
        Default is False.
    scipy_triangular_min_dimension : int, optional
        Dimension threshold for SciPy triangular solve dispatch. Default is 10.
    use_scipy_lu_factor : bool, optional
        Whether to attempt optional SciPy LAPACK dgetrf factorization.
        Default is False.
    scipy_lu_min_dimension : int, optional
        Dimension threshold for SciPy lu_factor dispatch. Default is 10.
    triangular_solve_provider : str, optional
        Triangular solve provider ('python' or 'scipy'). Default 'python'.
    lu_factor_provider : str | None, optional
        LU factorization provider ('python' or 'scipy'). Default is None
        (resolving to 'scipy' if use_scipy_lu_factor=True, else 'python').

    Returns
    -------
    DenseLU
        Factorization container holding L, U, permutation indices, and sign.

    Raises
    ------
    ValueError
        If B is not a 2D square matrix.
    NumericalSingularityError
        If B is singular or numerically singular (|pivot| < pivot_tolerance).
    """
    eff_lu_prov = (
        lu_factor_provider
        if lu_factor_provider is not None
        else ("scipy" if use_scipy_lu_factor else "python")
    )
    eff_tri_prov = "scipy" if (use_scipy_triangular_solve or triangular_solve_provider == "scipy") else "python"
    eff_use_scipy_tri = (eff_tri_prov == "scipy")

    provider = _get_lu_provider_cached()(eff_lu_prov, min_dimension=scipy_lu_min_dimension)
    return provider.factorize(
        B,
        pivot_tolerance=pivot_tolerance,
        use_scipy_triangular_solve=eff_use_scipy_tri,
        scipy_triangular_min_dimension=scipy_triangular_min_dimension,
        triangular_solve_provider=eff_tri_prov,
    )


def forward_substitution(
    L: np.ndarray,
    b: np.ndarray,
    unit_diag: bool = True,
) -> np.ndarray:
    """Solve lower triangular system L y = b via forward substitution.

    Parameters
    ----------
    L : np.ndarray
        Lower triangular matrix of shape (m, m).
    b : np.ndarray
        Right-hand side vector of shape (m,) or matrix of shape (m, k).
    unit_diag : bool, optional
        True if L has ones on the diagonal (unit lower triangular). Default True.

    Returns
    -------
    np.ndarray
        Solution y of same shape as b.
    """
    m = L.shape[0]
    b_arr = np.asarray(b, dtype=np.float64)
    is_1d = b_arr.ndim == 1

    if is_1d:
        if b_arr.shape[0] != m:
            raise ValueError(f"RHS length {b_arr.shape[0]} does not match matrix dimension {m}.")
        y = np.zeros(m, dtype=np.float64)
        for i in range(m):
            sum_val = np.dot(L[i, :i], y[:i]) if i > 0 else 0.0
            diff = b_arr[i] - sum_val
            y[i] = diff if unit_diag else diff / L[i, i]
        return y
    else:
        if b_arr.shape[0] != m:
            raise ValueError(f"RHS rows {b_arr.shape[0]} does not match matrix dimension {m}.")
        k = b_arr.shape[1]
        y_mat = np.zeros((m, k), dtype=np.float64)
        for i in range(m):
            sum_val = L[i, :i] @ y_mat[:i, :] if i > 0 else 0.0
            diff = b_arr[i, :] - sum_val
            y_mat[i, :] = diff if unit_diag else diff / L[i, i]
        return y_mat


def back_substitution(
    U: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Solve upper triangular system U x = y via back substitution.

    Parameters
    ----------
    U : np.ndarray
        Upper triangular matrix of shape (m, m).
    y : np.ndarray
        Right-hand side vector of shape (m,) or matrix of shape (m, k).

    Returns
    -------
    np.ndarray
        Solution x of same shape as y.
    """
    m = U.shape[0]
    y_arr = np.asarray(y, dtype=np.float64)
    is_1d = y_arr.ndim == 1

    if is_1d:
        if y_arr.shape[0] != m:
            raise ValueError(f"RHS length {y_arr.shape[0]} does not match matrix dimension {m}.")
        x = np.zeros(m, dtype=np.float64)
        for i in range(m - 1, -1, -1):
            sum_val = np.dot(U[i, i + 1:], x[i + 1:]) if i < m - 1 else 0.0
            x[i] = (y_arr[i] - sum_val) / U[i, i]
        return x
    else:
        if y_arr.shape[0] != m:
            raise ValueError(f"RHS rows {y_arr.shape[0]} does not match matrix dimension {m}.")
        k = y_arr.shape[1]
        x_mat = np.zeros((m, k), dtype=np.float64)
        for i in range(m - 1, -1, -1):
            sum_val = U[i, i + 1:] @ x_mat[i + 1:, :] if i < m - 1 else 0.0
            x_mat[i, :] = (y_arr[i, :] - sum_val) / U[i, i]
        return x_mat


def forward_substitution_transpose(
    U: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve U^T z = rhs via forward substitution (U is upper, U^T is lower).

    Parameters
    ----------
    U : np.ndarray
        Upper triangular matrix of shape (m, m).
    rhs : np.ndarray
        Right-hand side vector (m,) or matrix (m, k).

    Returns
    -------
    np.ndarray
        Solution array z of matching shape.
    """
    m = U.shape[0]
    rhs_arr = np.asarray(rhs, dtype=np.float64)
    if rhs_arr.shape[0] != m:
        raise ValueError(f"RHS length {rhs_arr.shape[0]} does not match matrix dimension {m}.")

    if rhs_arr.ndim == 1:
        z = np.zeros(m, dtype=np.float64)
        for i in range(m):
            sum_val = np.dot(U[:i, i], z[:i]) if i > 0 else 0.0
            z[i] = (rhs_arr[i] - sum_val) / U[i, i]
        return z
    elif rhs_arr.ndim == 2:
        k = rhs_arr.shape[1]
        z_mat = np.zeros((m, k), dtype=np.float64)
        for i in range(m):
            sum_val = U[:i, i] @ z_mat[:i, :] if i > 0 else 0.0
            z_mat[i, :] = (rhs_arr[i, :] - sum_val) / U[i, i]
        return z_mat
    else:
        raise ValueError(f"RHS must be 1D or 2D, got ndim={rhs_arr.ndim}.")


def back_substitution_transpose(
    L: np.ndarray,
    z: np.ndarray,
    unit_diag: bool = True,
) -> np.ndarray:
    """Solve L^T w = z via back substitution (L is lower, L^T is upper).

    Parameters
    ----------
    L : np.ndarray
        Lower triangular matrix of shape (m, m).
    z : np.ndarray
        Right-hand side vector (m,) or matrix (m, k).
    unit_diag : bool, optional
        True if L is unit lower triangular (L[i, i] == 1.0). Default True.

    Returns
    -------
    np.ndarray
        Solution array w of matching shape.
    """
    m = L.shape[0]
    z_arr = np.asarray(z, dtype=np.float64)
    if z_arr.shape[0] != m:
        raise ValueError(f"RHS length {z_arr.shape[0]} does not match matrix dimension {m}.")

    if z_arr.ndim == 1:
        w = np.zeros(m, dtype=np.float64)
        for i in range(m - 1, -1, -1):
            sum_val = np.dot(L[i + 1:, i], w[i + 1:]) if i < m - 1 else 0.0
            diff = z_arr[i] - sum_val
            w[i] = diff if unit_diag else diff / L[i, i]
        return w
    elif z_arr.ndim == 2:
        k = z_arr.shape[1]
        w_mat = np.zeros((m, k), dtype=np.float64)
        for i in range(m - 1, -1, -1):
            sum_val = L[i + 1:, i] @ w_mat[i + 1:, :] if i < m - 1 else 0.0
            diff = z_arr[i, :] - sum_val
            w_mat[i, :] = diff if unit_diag else diff / L[i, i]
        return w_mat
    else:
        raise ValueError(f"RHS must be 1D or 2D, got ndim={z_arr.ndim}.")


def lu_solve(
    lu: DenseLU,
    rhs: np.ndarray,
    use_scipy: bool | None = None,
    min_dimension: int | None = None,
    provider: str | None = None,
) -> np.ndarray:
    """Solve B x = rhs given P B = L U.

    Steps:
    1. Permute rhs: b' = P rhs (row i becomes rhs[perm[i]])
    2. Forward substitution: L y = b'
    3. Back substitution: U x = y

    Parameters
    ----------
    lu : DenseLU
        Computed LU factorization container.
    rhs : np.ndarray
        Right-hand side vector of length m or matrix of shape (m, k).
    use_scipy : bool | None, optional
        Whether to attempt SciPy solve_triangular acceleration. If None,
        defaults to `lu.use_scipy_triangular_solve`.
    min_dimension : int | None, optional
        Dimension threshold for SciPy dispatch. If None, defaults to
        `lu.scipy_triangular_min_dimension`.
    provider : str | None, optional
        Triangular solve provider override ('python' or 'scipy').

    Returns
    -------
    np.ndarray
        Solution vector x of length m or matrix of shape (m, k).
    """
    scipy_flag = lu.use_scipy_triangular_solve if use_scipy is None else use_scipy
    prov = provider if provider is not None else getattr(lu, "triangular_solve_provider", "python")
    min_dim = lu.scipy_triangular_min_dimension if min_dimension is None else min_dimension
    if scipy_flag or prov == "scipy":
        tri_solve, _ = _get_triangular_solvers()
        return tri_solve(lu, rhs, use_scipy=True, min_dimension=min_dim, provider="scipy")

    rhs_arr = np.asarray(rhs, dtype=np.float64)
    m = lu.m

    if rhs_arr.ndim == 1:
        if rhs_arr.shape[0] != m:
            raise ValueError(f"RHS dimension {rhs_arr.shape[0]} != matrix dimension {m}.")
        # 1. Permute rhs: b'[i] = rhs[perm[i]]
        b_prime = rhs_arr[lu.perm]
        # 2. Forward substitution: L y = b'
        y = forward_substitution(lu.L, b_prime, unit_diag=True)
        # 3. Back substitution: U x = y
        x = back_substitution(lu.U, y)
        return x
    elif rhs_arr.ndim == 2:
        if rhs_arr.shape[0] != m:
            raise ValueError(f"RHS rows {rhs_arr.shape[0]} != matrix dimension {m}.")
        b_prime = rhs_arr[lu.perm, :]
        y = forward_substitution(lu.L, b_prime, unit_diag=True)
        x = back_substitution(lu.U, y)
        return x
    else:
        raise ValueError(f"RHS must be 1D or 2D, got ndim={rhs_arr.ndim}.")


def lu_solve_transpose(
    lu: DenseLU,
    rhs: np.ndarray,
    use_scipy: bool | None = None,
    min_dimension: int | None = None,
    provider: str | None = None,
) -> np.ndarray:
    """Solve B^T x = rhs given P B = L U.

    Mathematical Derivation:
    ------------------------
    Since P B = L U and P is an orthogonal permutation matrix (P^T P = I):
        B = P^T L U
    Taking the transpose:
        B^T = (P^T L U)^T = U^T L^T P

    The system B^T x = rhs becomes:
        U^T L^T P x = rhs

    Define:
        w = P x  =>  x = P^T w
        z = L^T w

    Then:
        U^T z = rhs

    Solve steps:
    1. Forward solve U^T z = rhs (U is upper, U^T is lower)
    2. Backward solve L^T w = z (L is unit lower, L^T is unit upper)
    3. Invert permutation x = P^T w: x[perm[i]] = w[i]

    Parameters
    ----------
    lu : DenseLU
        Computed LU factorization container.
    rhs : np.ndarray
        Right-hand side vector of length m or matrix of shape (m, k).
    use_scipy : bool | None, optional
        Whether to attempt SciPy solve_triangular acceleration. If None,
        defaults to `lu.use_scipy_triangular_solve`.
    min_dimension : int | None, optional
        Dimension threshold for SciPy dispatch. If None, defaults to
        `lu.scipy_triangular_min_dimension`.
    provider : str | None, optional
        Triangular solve provider override ('python' or 'scipy').

    Returns
    -------
    np.ndarray
        Solution vector x of length m or matrix of shape (m, k).
    """
    scipy_flag = lu.use_scipy_triangular_solve if use_scipy is None else use_scipy
    prov = provider if provider is not None else getattr(lu, "triangular_solve_provider", "python")
    min_dim = lu.scipy_triangular_min_dimension if min_dimension is None else min_dimension
    if scipy_flag or prov == "scipy":
        _, tri_solve_t = _get_triangular_solvers()
        return tri_solve_t(lu, rhs, use_scipy=True, min_dimension=min_dim, provider="scipy")

    rhs_arr = np.asarray(rhs, dtype=np.float64)
    m = lu.m

    if rhs_arr.ndim not in (1, 2):
        raise ValueError(f"RHS must be 1D or 2D, got ndim={rhs_arr.ndim}.")
    if rhs_arr.shape[0] != m:
        raise ValueError(f"RHS dimension {rhs_arr.shape[0]} != matrix dimension {m}.")

    # Step 1: Solve U^T z = rhs (U^T is lower triangular)
    z = forward_substitution_transpose(lu.U, rhs_arr)

    # Step 2: Solve L^T w = z (L^T is unit upper triangular)
    w = back_substitution_transpose(lu.L, z, unit_diag=True)

    # Step 3: x = P^T w <=> x[perm[i]] = w[i]
    if rhs_arr.ndim == 1:
        x = np.empty_like(w)
        x[lu.perm] = w
        return x
    else:
        x = np.empty_like(w)
        x[lu.perm, :] = w
        return x
