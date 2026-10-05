r"""Incremental Basis Factorization for Revised Simplex.

This module provides an incremental basis factorization mechanism using the
Product Form of the Inverse (PFI) with elementary eta matrices, backed by
periodic full DenseLU refactorizations and automatic numerical fallback.

Mathematical Invariant:
-----------------------
Let B_0 be the basis matrix at the last full refactorization (P_0 B_0 = L_0 U_0).
After k column replacements:
    B_k = B_0 E_1 E_2 ... E_k
    B_k^{-1} = E_k^{-1} ... E_1^{-1} B_0^{-1}

For any right-hand side b:
    Primal solve:
        x = B_k^{-1} b = E_k^{-1} ( ... ( E_1^{-1} (B_0^{-1} b) ) ... )
    Dual / Transpose solve:
        y = B_k^{-T} c = B_0^{-T} ( E_1^{-T} ( ... ( E_k^{-T} c ) ... ) )

Each elementary eta inverse E_\ell^{-1} is applied in O(m) operations, avoiding
the O(m^3) full LU factorization on every simplex pivot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from optiforge.numerics.diagnostics import NumericalDiagnostics, compute_diagnostics
from optiforge.numerics.exceptions import (
    NumericalSingularityError,
    SingularBasisError,
)
from optiforge.numerics.lu import DenseLU, dense_lu_factor, lu_solve, lu_solve_transpose


@dataclass(frozen=True)
class EtaUpdate:
    """An elementary eta matrix update in product form.

    E^{-1} = I + (eta - e_p) e_p^T

    Attributes
    ----------
    p : int
        The replaced column index (pivot row).
    eta : np.ndarray
        The eta vector of length m:
            eta[i] = -d[i] / d[p] for i != p
            eta[p] = 1.0 / d[p]
    pivot_val : float
        The pivot element d[p].
    """

    p: int
    eta: np.ndarray
    pivot_val: float


class IncrementalBasisFactorization:
    """Incremental basis representation using Product Form of the Inverse (PFI).

    Maintains an explicit copy of the current basis matrix B alongside the base
    factorization B_0 and a chain of eta updates. Solves are performed via
    forward/backward eta updates. Automatically triggers full refactorizations
    upon update limits, small pivots, or residual degradation.

    Parameters
    ----------
    B : np.ndarray
        Initial square basis matrix of shape (m, m).
    pivot_tolerance : float, optional
        Pivot threshold below which singularity is declared. Default is 1e-12.
    max_updates : int, optional
        Maximum number of incremental column updates before forcing a full
        DenseLU refactorization. Default is 30.
    residual_tolerance : float, optional
        Relative residual threshold triggering automatic refactorization.
        Default is 1e-7.
    """

    def __init__(
        self,
        B: np.ndarray,
        pivot_tolerance: float = 1e-12,
        max_updates: int = 30,
        residual_tolerance: float = 1e-7,
        base_factorization: Any | None = None,
        use_scipy_triangular_solve: bool = False,
        scipy_triangular_min_dimension: int = 10,
        triangular_solve_provider: str = "python",
        use_scipy_lu_factor: bool = False,
        scipy_lu_min_dimension: int = 10,
        lu_factor_provider: str | None = None,
    ) -> None:
        self.pivot_tolerance = pivot_tolerance
        self.max_updates = max_updates
        self.residual_tolerance = residual_tolerance
        prov = "scipy" if (use_scipy_triangular_solve or triangular_solve_provider == "scipy") else "python"
        self.triangular_solve_provider = prov
        self.use_scipy_triangular_solve = (prov == "scipy")
        self.scipy_triangular_min_dimension = scipy_triangular_min_dimension
        eff_lu_prov = lu_factor_provider if lu_factor_provider is not None else ("scipy" if use_scipy_lu_factor else "python")
        self.lu_factor_provider = eff_lu_prov
        self.use_scipy_lu_factor = (eff_lu_prov == "scipy")
        self.scipy_lu_min_dimension = scipy_lu_min_dimension

        # Working explicit basis matrix B
        self._B = np.array(B, dtype=np.float64, copy=True)
        m, n = self._B.shape
        if m != n:
            raise ValueError(f"Basis matrix must be square, got shape ({m}, {n}).")
        self._m = m

        # Base factorization B_0 and eta sequence
        self._base_lu: DenseLU | None = None
        self._base_factorization: Any | None = base_factorization
        self._etas: list[EtaUpdate] = []
        self._update_count: int = 0
        self._total_updates: int = 0
        self._total_refactorizations: int = 0
        self._total_fallbacks: int = 0
        self._diagnostics: NumericalDiagnostics | None = None
        self._row_abs_sums: np.ndarray | None = None
        self._col_abs_sums: np.ndarray | None = None
        self._diff_buf: np.ndarray | None = None

        if self._base_factorization is None:
            # Factorize initial basis
            self._factorize_base()

    def _factorize_base(self) -> None:
        """Perform full DenseLU factorization of current explicit basis B."""
        try:
            self._base_lu = dense_lu_factor(
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
            raise SingularBasisError(f"Basis matrix B is singular: {err}") from err

        self._base_factorization = None
        self._etas.clear()
        self._update_count = 0
        self._total_refactorizations += 1
        self._diagnostics = None
        self._row_abs_sums = np.sum(np.abs(self._B), axis=1)
        self._col_abs_sums = np.sum(np.abs(self._B), axis=0)

    @classmethod
    def factorize(
        cls,
        B: np.ndarray,
        pivot_tolerance: float = 1e-12,
        max_updates: int = 30,
        residual_tolerance: float = 1e-7,
        base_factorization: Any | None = None,
        use_scipy_triangular_solve: bool = False,
        scipy_triangular_min_dimension: int = 10,
        triangular_solve_provider: str = "python",
        use_scipy_lu_factor: bool = False,
        scipy_lu_min_dimension: int = 10,
        lu_factor_provider: str | None = None,
    ) -> IncrementalBasisFactorization:
        """Factory constructor."""
        return cls(
            B,
            pivot_tolerance=pivot_tolerance,
            max_updates=max_updates,
            residual_tolerance=residual_tolerance,
            base_factorization=base_factorization,
            use_scipy_triangular_solve=use_scipy_triangular_solve,
            scipy_triangular_min_dimension=scipy_triangular_min_dimension,
            triangular_solve_provider=triangular_solve_provider,
            use_scipy_lu_factor=use_scipy_lu_factor,
            scipy_lu_min_dimension=scipy_lu_min_dimension,
            lu_factor_provider=lu_factor_provider,
        )

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def m(self) -> int:
        """Matrix dimension."""
        return self._m

    @property
    def B(self) -> np.ndarray:
        """Current explicit basis matrix (read-only copy or direct view)."""
        return self._B

    @property
    def lu(self) -> DenseLU:
        """Access underlying base LU (for monitor logging compatibility)."""
        if self._base_lu is None:
            self._base_lu = dense_lu_factor(
                self._B,
                pivot_tolerance=self.pivot_tolerance,
                use_scipy_triangular_solve=self.use_scipy_triangular_solve,
                scipy_triangular_min_dimension=self.scipy_triangular_min_dimension,
                triangular_solve_provider=self.triangular_solve_provider,
                use_scipy_lu_factor=self.use_scipy_lu_factor,
                scipy_lu_min_dimension=self.scipy_lu_min_dimension,
                lu_factor_provider=self.lu_factor_provider,
            )
        return self._base_lu

    @property
    def update_count(self) -> int:
        """Number of incremental updates since last full refactorization."""
        return self._update_count

    @property
    def total_updates(self) -> int:
        """Total incremental updates performed over solver lifetime."""
        return self._total_updates

    @property
    def total_refactorizations(self) -> int:
        """Total full DenseLU factorizations performed."""
        return self._total_refactorizations

    @property
    def total_fallbacks(self) -> int:
        """Total unexpected fallback refactorizations triggered by numerical checks."""
        return self._total_fallbacks

    @property
    def num_etas(self) -> int:
        """Number of currently active eta updates."""
        return len(self._etas)

    # -----------------------------------------------------------------------
    # Solves
    # -----------------------------------------------------------------------

    def solve(self, rhs: np.ndarray, check_residual: bool = False) -> np.ndarray:
        """Solve B x = rhs using PFI eta sequence.

        Parameters
        ----------
        rhs : np.ndarray
            Right-hand side vector of length m or matrix of shape (m, k).
        check_residual : bool, optional
            Whether to verify solution residual and fall back to full
            DenseLU refactorization if residual exceeds tolerance. Default False.

        Returns
        -------
        np.ndarray
            Solution x of matching shape.
        """
        assert self._base_lu is not None or self._base_factorization is not None
        rhs_arr = np.asarray(rhs, dtype=np.float64)

        try:
            x = self._raw_solve(rhs_arr)
        except Exception:
            # Safe fallback on unexpected failure
            self._total_fallbacks += 1
            self._factorize_base()
            return lu_solve(self._base_lu, rhs_arr)

        if check_residual and self._update_count > 0:
            rel_res = self.compute_residual(rhs_arr, x)
            if rel_res > self.residual_tolerance or not np.isfinite(rel_res):
                # Fallback to fresh refactorization
                self._total_fallbacks += 1
                self._factorize_base()
                return lu_solve(self._base_lu, rhs_arr)

        return x

    def _raw_solve(self, rhs_arr: np.ndarray) -> np.ndarray:
        """Raw PFI forward solve without fallback wrapper."""
        # 1. Base solve: v_0 = B_0^{-1} rhs
        if self._base_factorization is not None:
            v = self._base_factorization.solve(rhs_arr)
        else:
            assert self._base_lu is not None
            v = lu_solve(self._base_lu, rhs_arr)

        if not self._etas:
            return v

        is_1d = rhs_arr.ndim == 1
        if is_1d:
            # 2. Sequential forward application: v_ell = E_ell^{-1} v_{ell-1}
            for eta_up in self._etas:
                p = eta_up.p
                eta = eta_up.eta
                alpha = v[p]
                v += alpha * eta
                v[p] = alpha * eta[p]
            return v
        else:
            # 2D case: shape (m, k)
            for eta_up in self._etas:
                p = eta_up.p
                eta = eta_up.eta
                alpha = v[p, :].copy()  # shape (k,)
                v += eta[:, None] * alpha[None, :]
                v[p, :] = alpha * eta[p]
            return v

    def solve_transpose(self, rhs: np.ndarray, check_residual: bool = False) -> np.ndarray:
        """Solve B^T y = rhs using PFI eta sequence.

        Parameters
        ----------
        rhs : np.ndarray
            Right-hand side vector of length m or matrix of shape (m, k).
        check_residual : bool, optional
            Whether to verify transpose residual and trigger fallback if needed.

        Returns
        -------
        np.ndarray
            Solution y of matching shape.
        """
        assert self._base_lu is not None or self._base_factorization is not None
        rhs_arr = rhs if (isinstance(rhs, np.ndarray) and rhs.dtype == np.float64) else np.asarray(rhs, dtype=np.float64)

        try:
            y = self._raw_solve_transpose(rhs_arr)
        except Exception:
            self._total_fallbacks += 1
            self._factorize_base()
            return lu_solve_transpose(self._base_lu, rhs_arr)

        if check_residual and self._update_count > 0:
            rel_res = self.compute_transpose_residual(rhs_arr, y)
            if rel_res > self.residual_tolerance or not np.isfinite(rel_res):
                self._total_fallbacks += 1
                self._factorize_base()
                return lu_solve_transpose(self._base_lu, rhs_arr)

        return y

    def _raw_solve_transpose(self, rhs_arr: np.ndarray) -> np.ndarray:
        """Raw PFI transpose solve: y = B_0^{-T} E_1^{-T} ... E_k^{-T} rhs."""
        if not self._etas:
            if self._base_factorization is not None:
                return self._base_factorization.solve_transpose(rhs_arr)
            else:
                assert self._base_lu is not None
                return lu_solve_transpose(self._base_lu, rhs_arr)

        is_1d = rhs_arr.ndim == 1

        if is_1d:
            u = rhs_arr.copy()
            # Apply (E_ell^{-1})^T in reverse chronological order
            for eta_up in reversed(self._etas):
                p = eta_up.p
                eta = eta_up.eta
                u[p] = float(np.dot(eta, u))
            if self._base_factorization is not None:
                return self._base_factorization.solve_transpose(u)
            else:
                assert self._base_lu is not None
                return lu_solve_transpose(self._base_lu, u)
        else:
            u = rhs_arr.copy()
            for eta_up in reversed(self._etas):
                p = eta_up.p
                eta = eta_up.eta
                # u[p, :] = eta^T @ u
                u[p, :] = eta @ u
            if self._base_factorization is not None:
                return self._base_factorization.solve_transpose(u)
            else:
                assert self._base_lu is not None
                return lu_solve_transpose(self._base_lu, u)

    # -----------------------------------------------------------------------
    # Column Replacement & Update
    # -----------------------------------------------------------------------

    def replace_column(
        self,
        col_index: int,
        new_col: np.ndarray,
        search_direction: np.ndarray | None = None,
    ) -> bool:
        """Replace column col_index of basis B with new_col.

        Parameters
        ----------
        col_index : int
            Index of the basis column to replace (pivot row p, 0 <= p < m).
        new_col : np.ndarray
            New column vector a_q of shape (m,).
        search_direction : np.ndarray | None, optional
            Pre-computed direction vector d = B^{-1} a_q. If None, it is
            computed using self.solve(new_col). Default is None.

        Returns
        -------
        bool
            True if incremental eta update succeeded; False if full
            refactorization was performed instead.
        """
        if col_index < 0 or col_index >= self._m:
            raise ValueError(f"Column index {col_index} out of range [0, {self._m}).")

        new_col_arr = np.asarray(new_col, dtype=np.float64).ravel()
        if new_col_arr.shape[0] != self._m:
            raise ValueError(f"New column dimension {new_col_arr.shape[0]} != {self._m}.")

        # Update row and col abs sums before updating explicit basis matrix
        if self._row_abs_sums is not None:
            self._row_abs_sums += np.abs(new_col_arr) - np.abs(self._B[:, col_index])
        if self._col_abs_sums is not None:
            self._col_abs_sums[col_index] = float(np.sum(np.abs(new_col_arr)))

        # Update the explicit basis matrix B
        self._B[:, col_index] = new_col_arr
        self._diagnostics = None

        # Check if periodic refactorization limit reached
        if self._update_count >= self.max_updates:
            self._factorize_base()
            return False

        # Compute search direction if not provided
        if search_direction is None:
            d = self.solve(new_col_arr)
        else:
            d = np.asarray(search_direction, dtype=np.float64).ravel()

        pivot_val = float(d[col_index])
        abs_pivot = abs(pivot_val)

        # Check pivot quality: if too small, do not create ill-conditioned eta
        if abs_pivot < self.pivot_tolerance or not np.isfinite(pivot_val):
            self._factorize_base()
            return False

        # Construct eta vector:
        # eta[i] = -d[i] / d[p] for i != p
        # eta[p] = 1.0 / d[p]
        inv_pivot = 1.0 / pivot_val
        eta = -d * inv_pivot
        eta[col_index] = inv_pivot

        # Record eta update
        self._etas.append(EtaUpdate(p=col_index, eta=eta, pivot_val=pivot_val))
        self._update_count += 1
        self._total_updates += 1
        return True

    def refactorize(self) -> None:
        """Explicitly force a full DenseLU refactorization from current B."""
        self._factorize_base()

    def needs_refactorization(self) -> bool:
        """Check whether next update or solve would trigger a refactorization."""
        return self._update_count >= self.max_updates

    # -----------------------------------------------------------------------
    # Residuals & Diagnostics
    # -----------------------------------------------------------------------

    def compute_residual(self, rhs: np.ndarray, x: np.ndarray) -> float:
        """Compute relative primal residual: ||B x - rhs||_inf / (||B||_inf * ||x||_inf + ||rhs||_inf)."""
        rhs_arr = np.asarray(rhs, dtype=np.float64)
        x_arr = np.asarray(x, dtype=np.float64)

        if rhs_arr.ndim == 1 and x_arr.ndim == 1 and rhs_arr.shape[0] == self._m:
            if self._diff_buf is None or self._diff_buf.shape[0] != self._m:
                self._diff_buf = np.empty(self._m, dtype=np.float64)
            np.dot(self._B, x_arr, out=self._diff_buf)
            self._diff_buf -= rhs_arr
            norm_res = float(np.max(np.abs(self._diff_buf)))
            norm_b = float(np.max(np.abs(rhs_arr)))
            norm_x = float(np.max(np.abs(x_arr)))
            if self._row_abs_sums is not None:
                norm_B = float(np.max(self._row_abs_sums))
            else:
                norm_B = float(np.max(np.sum(np.abs(self._B), axis=1)))
            scale = norm_B * norm_x + norm_b + 1e-15
            return norm_res / scale
        else:
            diff = self._B @ x_arr - rhs_arr
            norm_res = float(np.max(np.abs(diff)))
            norm_b = float(np.max(np.abs(rhs_arr)))
            norm_x = float(np.max(np.abs(x_arr)))
            norm_B = float(np.max(np.sum(np.abs(self._B), axis=1)))
            scale = norm_B * norm_x + norm_b + 1e-15
            return norm_res / scale

    def compute_transpose_residual(self, rhs: np.ndarray, y: np.ndarray) -> float:
        """Compute relative transpose residual: ||B^T y - rhs||_inf / (||B^T||_inf * ||y||_inf + ||rhs||_inf)."""
        rhs_arr = np.asarray(rhs, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        if rhs_arr.ndim == 1 and y_arr.ndim == 1 and rhs_arr.shape[0] == self._m:
            if self._diff_buf is None or self._diff_buf.shape[0] != self._m:
                self._diff_buf = np.empty(self._m, dtype=np.float64)
            np.dot(self._B.T, y_arr, out=self._diff_buf)
            self._diff_buf -= rhs_arr
            norm_res = float(np.max(np.abs(self._diff_buf)))
            norm_b = float(np.max(np.abs(rhs_arr)))
            norm_y = float(np.max(np.abs(y_arr)))
            if self._col_abs_sums is not None:
                norm_BT = float(np.max(self._col_abs_sums))
            else:
                norm_BT = float(np.max(np.sum(np.abs(self._B), axis=0)))
            scale = norm_BT * norm_y + norm_b + 1e-15
            return norm_res / scale
        else:
            diff = self._B.T @ y_arr - rhs_arr
            norm_res = float(np.max(np.abs(diff)))
            norm_b = float(np.max(np.abs(rhs_arr)))
            norm_y = float(np.max(np.abs(y_arr)))
            norm_BT = float(np.max(np.sum(np.abs(self._B), axis=0)))
            scale = norm_BT * norm_y + norm_b + 1e-15
            return norm_res / scale

    def diagnostics(self, estimate_condition: bool = True) -> NumericalDiagnostics:
        """Compute or return cached diagnostics for the current basis."""
        if self._diagnostics is None:
            # Compute fresh DenseLU on current B for diagnostic consistency
            lu_diag = dense_lu_factor(self._B, pivot_tolerance=self.pivot_tolerance)
            self._diagnostics = compute_diagnostics(
                self._B,
                lu_diag,
                estimate_condition=estimate_condition,
            )
        return self._diagnostics

    @property
    def condition_number(self) -> float:
        """Condition number of current basis matrix."""
        return self.diagnostics().condition_number
