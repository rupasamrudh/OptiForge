"""Bordered / Schur-Complement Basis Factorization Inheritance for OptiForge.

Provides an O(m^2) basis factorization update for child LP relaxations in Branch-and-Bound,
inheriting the parent basis factorization directly instead of recomputing an O(m^3)
Dense LU factorization from scratch.

Mathematical Foundation:
------------------------
When branching adds an explicit branch constraint row and a slack column, the child basis
matrix has the bordered block lower-triangular form:

    B_child = [ B_parent   0 ]
              [   a_B^T    d ]

where:
- B_parent is the m x m parent basis matrix.
- a_B^T is the 1 x m row vector: a_B^T = A_child[m, parent_basis].
- d is the pivot scalar (typically d = 1.0 for standard slack variables).
- 0 is the m x 1 zero column.

Linear Solves via Block Elimination:
-------------------------------------
1. Primal Solve: B_child x = rhs
   Let x = [x_1; x_2] and rhs = [b_1; b_2] with b_1 in R^m, b_2 in R:
       B_parent x_1 = b_1       =>  x_1 = B_parent^{-1} b_1
       a_B^T x_1 + d x_2 = b_2  =>  x_2 = (b_2 - a_B^T x_1) / d
   Cost: 1 parent solve (O(m^2) for LU) + 1 inner product (O(m)).

2. Transpose Solve: B_child^T y = rhs
   Let y = [y_1; y_2] and rhs = [r_1; r_2]:
       [ B_parent^T   a_B ] [ y_1 ] = [ r_1 ]
       [     0^T       d  ] [ y_2 ]   [ r_2 ]
   From row 2:
       d y_2 = r_2              =>  y_2 = r_2 / d
   From row 1:
       B_parent^T y_1 + y_2 a_B = r_1  =>  B_parent^T y_1 = r_1 - y_2 a_B
   Cost: 1 vector scale/subtraction (O(m)) + 1 parent transpose solve (O(m^2)).

Inverse Representation:
-----------------------
    B_child^{-1} = [   B_parent^{-1}       0  ]
                   [ -d^{-1} a_B^T B^{-1}  d^{-1} ]

Since det(B_child) = det(B_parent) * d, B_child is unconditionally nonsingular whenever
B_parent is nonsingular and d != 0.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np

from optiforge.numerics.diagnostics import NumericalDiagnostics, compute_diagnostics
from optiforge.numerics.exceptions import (
    NumericalSingularityError,
    SingularBasisError,
)
from optiforge.numerics.lu import DenseLU, dense_lu_factor, lu_solve, lu_solve_transpose


class BorderedBasisFactorization:
    """Bordered basis factorization inheriting from a parent basis factorization.

    Solves B_child x = rhs and B_child^T y = rhs in O(m^2) operations by reusing
    the parent factorization and block triangular back-substitution.

    Parameters
    ----------
    parent_factorization : Any
        Parent factorization object providing `solve(b)` and `solve_transpose(c)`.
        Can be a `DenseLU`, `BasisFactorization`, `IncrementalBasisFactorization`,
        or another `BorderedBasisFactorization`.
    a_B : np.ndarray
        Bordering row vector of length m_parent.
    B : np.ndarray
        Explicit basis matrix B_child of shape (m_parent + 1, m_parent + 1).
    d : float, optional
        Diagonal scalar of the bordering row (default 1.0).
    pivot_tolerance : float, optional
        Minimum acceptable absolute value for d (default 1e-12).
    residual_tolerance : float, optional
        Relative residual threshold triggering automatic DenseLU fallback (default 1e-7).
    """

    def __init__(
        self,
        parent_factorization: Any,
        a_B: np.ndarray,
        B: np.ndarray,
        d: float = 1.0,
        pivot_tolerance: float = 1e-12,
        residual_tolerance: float = 1e-7,
    ) -> None:
        self.parent_factorization = parent_factorization
        self.a_B = np.array(a_B, dtype=np.float64, copy=True)
        self._B = np.array(B, dtype=np.float64, copy=True)
        self.d = float(d)
        self.pivot_tolerance = pivot_tolerance
        self.residual_tolerance = residual_tolerance

        self.triangular_solve_provider = getattr(parent_factorization, "triangular_solve_provider", "python")
        self.use_scipy_triangular_solve = getattr(parent_factorization, "use_scipy_triangular_solve", False)
        self.scipy_triangular_min_dimension = getattr(parent_factorization, "scipy_triangular_min_dimension", 10)
        self.lu_factor_provider = getattr(parent_factorization, "lu_factor_provider", "python")
        self.use_scipy_lu_factor = getattr(parent_factorization, "use_scipy_lu_factor", False)
        self.scipy_lu_min_dimension = getattr(parent_factorization, "scipy_lu_min_dimension", 10)

        m_child, n_child = self._B.shape
        if m_child != n_child:
            raise ValueError(f"Basis matrix must be square, got shape ({m_child}, {n_child}).")
        self._m = m_child

        if abs(self.d) < self.pivot_tolerance:
            raise SingularBasisError(
                f"Bordered basis diagonal element |d| = {abs(self.d):.2e} < pivot_tolerance {self.pivot_tolerance:.2e}"
            )

        if len(self.a_B) != self._m - 1:
            raise ValueError(f"Length of a_B ({len(self.a_B)}) must equal m_child - 1 ({self._m - 1}).")

        self._fallback_lu: DenseLU | None = None
        self._fallback_count: int = 0
        self._diagnostics: NumericalDiagnostics | None = None

        if isinstance(parent_factorization, BorderedBasisFactorization) and not parent_factorization.is_fallback:
            self._root_factorization = parent_factorization._root_factorization
            self._levels = list(parent_factorization._levels)
            self._levels.append((self.a_B, self.d))
        else:
            self._root_factorization = parent_factorization
            self._levels = [(self.a_B, self.d)]

    @property
    def m(self) -> int:
        """Matrix dimension of B_child."""
        return self._m

    @property
    def B(self) -> np.ndarray:
        """Explicit basis matrix B_child."""
        return self._B

    @property
    def fallback_count(self) -> int:
        """Number of times fallback to DenseLU was triggered."""
        return self._fallback_count

    @property
    def is_fallback(self) -> bool:
        """Whether this factorization has fallen back to full DenseLU."""
        return self._fallback_lu is not None

    def _trigger_fallback(self) -> DenseLU:
        """Fall back to full DenseLU factorization of B_child."""
        if self._fallback_lu is None:
            self._fallback_count += 1
            try:
                self._fallback_lu = dense_lu_factor(
                    self._B,
                    pivot_tolerance=self.pivot_tolerance,
                    use_scipy_triangular_solve=self.use_scipy_triangular_solve,
                    scipy_triangular_min_dimension=self.scipy_triangular_min_dimension,
                    triangular_solve_provider=self.triangular_solve_provider,
                    use_scipy_lu_factor=self.use_scipy_lu_factor,
                    scipy_lu_min_dimension=self.scipy_lu_min_dimension,
                    lu_factor_provider=self.lu_factor_provider,
                )
            except (NumericalSingularityError, ValueError) as err:
                raise SingularBasisError(f"Fallback DenseLU factorization failed: {err}") from err
        return self._fallback_lu

    def solve(self, rhs: np.ndarray, check_residual: bool = False) -> np.ndarray:
        """Solve B_child x = rhs.

        Parameters
        ----------
        rhs : np.ndarray
            RHS vector of length m or 2D matrix of shape (m, k).
        check_residual : bool, optional
            Whether to check relative residual and fall back to DenseLU if violated.

        Returns
        -------
        np.ndarray
            Solution vector or matrix x.
        """
        if self._fallback_lu is not None:
            return lu_solve(self._fallback_lu, rhs)

        rhs_arr = rhs if (isinstance(rhs, np.ndarray) and rhs.dtype == np.float64) else np.asarray(rhs, dtype=np.float64)
        m_root = self._m - len(self._levels)

        try:
            if rhs_arr.ndim == 1:
                x = np.empty(self._m, dtype=np.float64)
                x[:m_root] = self._root_factorization.solve(rhs_arr[:m_root])
                curr_m = m_root
                for a_k, d_k in self._levels:
                    x[curr_m] = (rhs_arr[curr_m] - float(np.dot(a_k, x[:curr_m]))) / d_k
                    curr_m += 1
            else:
                k = rhs_arr.shape[1]
                x = np.empty((self._m, k), dtype=np.float64)
                x[:m_root, :] = self._root_factorization.solve(rhs_arr[:m_root, :])
                curr_m = m_root
                for a_k, d_k in self._levels:
                    x[curr_m, :] = (rhs_arr[curr_m, :] - a_k @ x[:curr_m, :]) / d_k
                    curr_m += 1

            if not np.isfinite(x).all():
                raise ValueError("Non-finite values encountered in bordered solve.")

        except Exception:
            fallback_lu = self._trigger_fallback()
            return lu_solve(fallback_lu, rhs_arr)

        if check_residual:
            rel_res = self.compute_residual(rhs_arr, x)
            if rel_res > self.residual_tolerance or not np.isfinite(rel_res):
                fallback_lu = self._trigger_fallback()
                return lu_solve(fallback_lu, rhs_arr)

        return x

    def solve_transpose(self, rhs: np.ndarray, check_residual: bool = False) -> np.ndarray:
        """Solve B_child^T y = rhs.

        Parameters
        ----------
        rhs : np.ndarray
            RHS vector of length m or 2D matrix of shape (m, k).
        check_residual : bool, optional
            Whether to check relative residual and fall back to DenseLU if violated.

        Returns
        -------
        np.ndarray
            Solution vector or matrix y.
        """
        if self._fallback_lu is not None:
            return lu_solve_transpose(self._fallback_lu, rhs)

        rhs_arr = rhs if (isinstance(rhs, np.ndarray) and rhs.dtype == np.float64) else np.asarray(rhs, dtype=np.float64)
        m_root = self._m - len(self._levels)

        try:
            if rhs_arr.ndim == 1:
                y = np.empty(self._m, dtype=np.float64)
                r = rhs_arr.copy()
                curr_m = self._m - 1
                for a_k, d_k in reversed(self._levels):
                    y_val = r[curr_m] / d_k
                    y[curr_m] = y_val
                    r[:curr_m] -= y_val * a_k
                    curr_m -= 1
                y[:m_root] = self._root_factorization.solve_transpose(r[:m_root])
            else:
                k = rhs_arr.shape[1]
                y = np.empty((self._m, k), dtype=np.float64)
                r = rhs_arr.copy()
                curr_m = self._m - 1
                for a_k, d_k in reversed(self._levels):
                    y_val = r[curr_m, :] / d_k
                    y[curr_m, :] = y_val
                    r[:curr_m, :] -= np.outer(a_k, y_val)
                    curr_m -= 1
                y[:m_root, :] = self._root_factorization.solve_transpose(r[:m_root, :])

            if not np.isfinite(y).all():
                raise ValueError("Non-finite values encountered in bordered transpose solve.")

        except Exception:
            fallback_lu = self._trigger_fallback()
            return lu_solve_transpose(fallback_lu, rhs_arr)

        if check_residual:
            rel_res = self.compute_transpose_residual(rhs_arr, y)
            if rel_res > self.residual_tolerance or not np.isfinite(rel_res):
                fallback_lu = self._trigger_fallback()
                return lu_solve_transpose(fallback_lu, rhs_arr)

        return y

    def compute_residual(self, rhs: np.ndarray, x: np.ndarray) -> float:
        """Compute relative forward residual ||B x - rhs||_inf / (||B||_inf * ||x||_inf + ||rhs||_inf)."""
        res = self._B @ x - rhs
        res_norm = float(np.max(np.abs(res)))
        b_norm = float(np.max(np.abs(rhs)))
        B_norm = float(np.max(np.sum(np.abs(self._B), axis=1)))
        x_norm = float(np.max(np.abs(x)))
        denom = B_norm * x_norm + b_norm + 1e-15
        return res_norm / denom

    def compute_transpose_residual(self, rhs: np.ndarray, y: np.ndarray) -> float:
        """Compute relative transpose residual ||B^T y - rhs||_inf / (||B^T||_inf * ||y||_inf + ||rhs||_inf)."""
        res = self._B.T @ y - rhs
        res_norm = float(np.max(np.abs(res)))
        r_norm = float(np.max(np.abs(rhs)))
        BT_norm = float(np.max(np.sum(np.abs(self._B), axis=0)))
        y_norm = float(np.max(np.abs(y)))
        denom = BT_norm * y_norm + r_norm + 1e-15
        return res_norm / denom

    def diagnostics(self) -> NumericalDiagnostics:
        """Compute numerical diagnostics for B_child."""
        if self._diagnostics is None:
            self._diagnostics = compute_diagnostics(self._B)
        return self._diagnostics
