"""Incremental Basis Factorization for Revised Simplex Pivot Updates.

Stage 8.25: Replaces the sequential EtaMatrixUpdate chain with a Compact WY
Matrix representation for O(m * k) vectorized FTRAN/BTRAN instead of k serial
O(m) loops.  Public API, error-handling paths, and numerical contracts are
preserved exactly from Stage 8.14 / 8.18.

Mathematical Framework — Compact WY PFI:
-----------------------------------------
Let B_0 be the initial basis matrix at the most recent full DenseLU
refactorization.  After k successive column replacements the Product Form of
the Inverse is:

    B_k^{-1} = E_k^{-1} ... E_1^{-1} B_0^{-1}

Each elementary factor E_j^{-1} = I + w_j e_{p_j}^T where w_j is the shifted
eta vector (eta_j - e_{p_j}).

The Compact WY representation collects all updates into three arrays:

    W  in R^{m x k}   — columns are w_1, ..., w_k (F-order for BLAS locality)
    S  in R^{k x k}   — strictly lower triangular induction; S[j, :j] = W[p_j, :j] @ S[:j, :j]; S[j,j]=1
    P  in Z^k         — pivot row indices p_1, ..., p_k

FTRAN (primal solve):  x = B_k^{-1} b
    v   = B_0^{-1} b
    z   = S[:k,:k] @ v[P[:k]]
    x   = v + W[:,:k] @ z

BTRAN (transpose solve):  y = B_k^{-T} c
    q   = W[:,:k].T @ c
    r   = S[:k,:k].T @ q
    u   = c.copy(); u[P[j]] += r[j]  for j in 0..k-1
    y   = B_0^{-T} u

Stage 8.14 sequential operations (reference):
    B_k = B_0 E_1 E_2 ... E_k
    B_k^{-1} = E_k^{-1} ... E_1^{-1} B_0^{-1}
    B_k^{-T} = B_0^{-T} E_1^{-T} ... E_k^{-T}
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np

from optiforge.numerics.diagnostics import NumericalDiagnostics, compute_diagnostics
from optiforge.numerics.exceptions import (
    NumericalSingularityError,
    SingularBasisError,
)
from optiforge.numerics.lu import DenseLU, dense_lu_factor, lu_solve, lu_solve_transpose


class RefactorizationReason(Enum):
    """Structured enumeration of triggers forcing full basis refactorization."""
    NONE = "none"
    UPDATE_LIMIT = "update_limit"
    RESIDUAL_LIMIT = "residual_limit"
    PIVOT_INSTABILITY = "pivot_instability"
    NUMERICAL_DRIFT = "numerical_drift"
    EXPLICIT_REQUEST = "explicit_request"
    SINGULAR_BASE = "singular_base"
    UNEXPECTED_EXCEPTION = "unexpected_exception"


@dataclass(frozen=True)
class EtaMatrixUpdate:
    """An elementary eta matrix update representing column replacement at row p.

    Kept for API compatibility; no longer used as internal storage (Stage 8.25).

    Attributes
    ----------
    p : int
        Leaving row / replaced column index in the basis (0 <= p < m).
    eta : np.ndarray
        Eta vector of length m: eta[p] = 1/d[p], eta[i] = -d[i]/d[p] for i != p.
    pivot_val : float
        The pivot element d[p].
    """
    p: int
    eta: np.ndarray
    pivot_val: float


class PivotUpdateFactorization:
    """Incremental basis factorization using Product Form of the Inverse (PFI).

    Maintains the explicit basis matrix B, base factorization B_0, and a
    Compact WY representation of eta updates.  Provides primal solve,
    transpose solve, and automatic refactorization on update count, pivot
    magnitude, or residual limit.

    Stage 8.25: Internal eta-chain storage replaced by Compact WY Matrix
    (W, S, pivots) for vectorized FTRAN/BTRAN.  All public API, error
    handling, and numerical contracts are identical to Stage 8.14/8.18.

    Parameters
    ----------
    B : np.ndarray
        Initial square basis matrix (m x m).
    pivot_tolerance : float, optional
        Pivot threshold below which singularity/instability is declared. Default 1e-12.
    max_updates : int, optional
        Maximum number of incremental updates before forcing a full DenseLU
        refactorization. Default 30.
    residual_tolerance : float, optional
        Relative residual threshold triggering automatic refactorization. Default 1e-7.
    base_factorization : Any | None, optional
        Optional pre-computed base factorization (e.g. inherited BorderedBasisFactorization).
    """

    def __init__(
        self,
        B: np.ndarray,
        pivot_tolerance: float = 1e-12,
        max_updates: int = 30,
        residual_tolerance: float = 1e-7,
        base_factorization: Any | None = None,
        fast_pivot_update: bool = False,
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
        self.fast_pivot_update = fast_pivot_update
        prov = "scipy" if (use_scipy_triangular_solve or triangular_solve_provider == "scipy") else "python"
        self.triangular_solve_provider = prov
        self.use_scipy_triangular_solve = (prov == "scipy")
        self.scipy_triangular_min_dimension = scipy_triangular_min_dimension
        eff_lu_prov = lu_factor_provider if lu_factor_provider is not None else ("scipy" if use_scipy_lu_factor else "python")
        self.lu_factor_provider = eff_lu_prov
        self.use_scipy_lu_factor = (eff_lu_prov == "scipy")
        self.scipy_lu_min_dimension = scipy_lu_min_dimension

        self._B = np.array(B, dtype=np.float64, copy=True)
        m, n = self._B.shape
        if m != n:
            raise ValueError(f"Basis matrix must be square, got shape ({m}, {n}).")
        self._m = m

        self._base_lu: DenseLU | None = None
        self._base_factorization: Any | None = base_factorization

        # --- Compact WY Matrix state (Stage 8.25) ---
        # W: (m, max_updates), F-order for column-access locality
        # S: (max_updates, max_updates), lower triangular induction coefficients
        # _pivots: (max_updates,) pivot row indices
        # _k: number of accumulated updates since last refactorization
        cap = max(max_updates, 1)
        self._W: np.ndarray = np.zeros((m, cap), dtype=np.float64, order="F")
        self._S: np.ndarray = np.zeros((cap, cap), dtype=np.float64)
        self._pivots: np.ndarray = np.zeros(cap, dtype=np.intp)
        self._k: int = 0

        self._update_count: int = 0
        self._total_updates: int = 0
        self._total_refactorizations: int = 0
        self._total_fallbacks: int = 0
        self._last_refactor_reason: RefactorizationReason = RefactorizationReason.NONE
        self._diagnostics: NumericalDiagnostics | None = None

        # Stage 8.40: Reusable buffers and cached matrix 1-norm / inf-norm state
        self._row_abs_sums: np.ndarray | None = None
        self._col_abs_sums: np.ndarray | None = None
        self._diff_buf: np.ndarray | None = None

        if self._base_factorization is None:
            self._factorize_base(reason=RefactorizationReason.EXPLICIT_REQUEST)

    @classmethod
    def factorize(
        cls,
        A: np.ndarray,
        basis: list[int] | np.ndarray,
        pivot_tolerance: float = 1e-12,
        max_updates: int = 30,
        residual_tolerance: float = 1e-7,
        base_factorization: Any | None = None,
        fast_pivot_update: bool = False,
        use_scipy_triangular_solve: bool = False,
        scipy_triangular_min_dimension: int = 10,
        triangular_solve_provider: str = "python",
        use_scipy_lu_factor: bool = False,
        scipy_lu_min_dimension: int = 10,
        lu_factor_provider: str | None = None,
    ) -> PivotUpdateFactorization:
        """Factory constructor extracting basis matrix from constraint matrix A."""
        B = A[:, basis]
        return cls(
            B,
            pivot_tolerance=pivot_tolerance,
            max_updates=max_updates,
            residual_tolerance=residual_tolerance,
            base_factorization=base_factorization,
            fast_pivot_update=fast_pivot_update,
            use_scipy_triangular_solve=use_scipy_triangular_solve,
            scipy_triangular_min_dimension=scipy_triangular_min_dimension,
            triangular_solve_provider=triangular_solve_provider,
            use_scipy_lu_factor=use_scipy_lu_factor,
            scipy_lu_min_dimension=scipy_lu_min_dimension,
            lu_factor_provider=lu_factor_provider,
        )

    def _factorize_base(self, reason: RefactorizationReason) -> None:
        """Perform full DenseLU factorization of current explicit basis matrix B."""
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
            self._last_refactor_reason = RefactorizationReason.SINGULAR_BASE
            raise SingularBasisError(f"Basis matrix B is singular: {err}") from err

        self._base_factorization = None
        # Reset Compact WY state — no etas active after a full refactorization
        self._k = 0
        self._update_count = 0
        self._total_refactorizations += 1
        self._last_refactor_reason = reason
        self._diagnostics = None
        # Initialize row and col abs sums for O(m) residual scaling
        self._row_abs_sums = np.sum(np.abs(self._B), axis=1)
        self._col_abs_sums = np.sum(np.abs(self._B), axis=0)

    # -----------------------------------------------------------------------
    # Properties & Status
    # -----------------------------------------------------------------------

    @property
    def m(self) -> int:
        """Matrix row/column dimension."""
        return self._m

    @property
    def B(self) -> np.ndarray:
        """Current explicit basis matrix (read-only view or copy)."""
        return self._B

    @property
    def lu(self) -> DenseLU:
        """Access underlying base LU (for numerical monitor compatibility)."""
        if self._base_lu is None:
            if self._base_factorization is not None and hasattr(self._base_factorization, "lu"):
                return self._base_factorization.lu
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
        """Number of currently active eta updates (Compact WY columns)."""
        return self._k

    def get_update_count(self) -> int:
        """Return number of incremental updates since last full refactorization."""
        return self._update_count

    def get_total_updates(self) -> int:
        """Return total incremental updates performed over factorization lifetime."""
        return self._total_updates

    def get_total_refactorizations(self) -> int:
        """Return total number of full DenseLU factorizations performed."""
        return self._total_refactorizations

    def get_total_fallbacks(self) -> int:
        """Return count of emergency fallbacks to DenseLU."""
        return self._total_fallbacks

    def get_refactorization_reason(self) -> RefactorizationReason:
        """Return the reason code for the most recent refactorization."""
        return self._last_refactor_reason

    def needs_refactorization(self) -> bool:
        """Check whether update count limit has been reached."""
        return self._update_count >= self.max_updates

    # -----------------------------------------------------------------------
    # Primal Solve: B x = rhs
    # -----------------------------------------------------------------------

    def solve(self, rhs: np.ndarray, check_residual: bool = False) -> np.ndarray:
        """Solve B x = rhs using PFI Compact WY representation.

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
        if self.fast_pivot_update:
            rhs_arr = rhs if (isinstance(rhs, np.ndarray) and rhs.dtype == np.float64 and rhs.ndim in (1, 2)) else np.asarray(rhs, dtype=np.float64)
        else:
            rhs_arr = np.asarray(rhs, dtype=np.float64)

        try:
            x = self._raw_solve(rhs_arr)
        except Exception:
            # Deterministic fallback on unexpected numerical exception
            self._total_fallbacks += 1
            self._factorize_base(reason=RefactorizationReason.UNEXPECTED_EXCEPTION)
            return lu_solve(self._base_lu, rhs_arr)

        if check_residual and self._update_count > 0:
            rel_res = self.residual(rhs_arr, x)
            if rel_res > self.residual_tolerance or not np.isfinite(rel_res):
                self._total_fallbacks += 1
                self._factorize_base(reason=RefactorizationReason.RESIDUAL_LIMIT)
                return lu_solve(self._base_lu, rhs_arr)

        return x

    def _raw_solve(self, rhs_arr: np.ndarray) -> np.ndarray:
        """Internal Compact WY forward solve without fallback wrapper.

        FTRAN:
            v  = B_0^{-1} rhs
            z  = S[:k,:k] @ v[pivots[:k]]    (k-vector)
            x  = v + W[:,:k] @ z             (rank-k correction)

        Handles both 1-D (m,) and 2-D (m, ncols) RHS.
        """
        # 1. Base solve: v = B_0^{-1} rhs
        if self._base_factorization is not None:
            v = self._base_factorization.solve(rhs_arr)
        else:
            assert self._base_lu is not None
            v = lu_solve(self._base_lu, rhs_arr)

        k = self._k
        if k == 0:
            return v

        pivs = self._pivots[:k]          # shape (k,)
        W_k  = self._W[:, :k]            # shape (m, k) — F-order view
        S_k  = self._S[:k, :k]           # shape (k, k)

        if rhs_arr.ndim == 1:
            # z = S_k @ v[pivs]  →  shape (k,)
            z = S_k @ v[pivs]
            # x = v + W_k @ z (in-place accumulation into v)
            v += W_k @ z
        else:
            # 2-D case: rhs shape (m, ncols)
            # v[pivs, :]  →  shape (k, ncols)
            # S_k @ ...   →  shape (k, ncols)
            z = S_k @ v[pivs, :]         # (k, ncols)
            v += W_k @ z                 # (m, ncols)

        return v

    # -----------------------------------------------------------------------
    # Transpose Solve: B^T y = rhs
    # -----------------------------------------------------------------------

    def solve_transpose(self, rhs: np.ndarray, check_residual: bool = False) -> np.ndarray:
        """Solve B^T y = rhs using PFI Compact WY representation.

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
            self._factorize_base(reason=RefactorizationReason.UNEXPECTED_EXCEPTION)
            return lu_solve_transpose(self._base_lu, rhs_arr)

        if check_residual and self._update_count > 0:
            rel_res = self.transpose_residual(rhs_arr, y)
            if rel_res > self.residual_tolerance or not np.isfinite(rel_res):
                self._total_fallbacks += 1
                self._factorize_base(reason=RefactorizationReason.RESIDUAL_LIMIT)
                return lu_solve_transpose(self._base_lu, rhs_arr)

        return y

    def _raw_solve_transpose(self, rhs_arr: np.ndarray) -> np.ndarray:
        """Internal Compact WY transpose solve.

        BTRAN:
            q   = W[:,:k].T @ rhs           (k-vector or k x ncols)
            r   = S[:k,:k].T @ q            (k-vector or k x ncols)
            u   = rhs.copy()
            u[pivots[j]] += r[j]  for j in 0..k-1   (scatter add)
            y   = B_0^{-T} u

        Handles both 1-D (m,) and 2-D (m, ncols) RHS.
        Note: scatter-add uses np.add.at to correctly handle repeated pivot indices.
        """
        k = self._k
        if k == 0:
            if self._base_factorization is not None:
                return self._base_factorization.solve_transpose(rhs_arr)
            else:
                assert self._base_lu is not None
                return lu_solve_transpose(self._base_lu, rhs_arr)

        pivs = self._pivots[:k]   # shape (k,)
        W_k  = self._W[:, :k]     # (m, k)
        S_k  = self._S[:k, :k]    # (k, k)

        if rhs_arr.ndim == 1:
            q = W_k.T @ rhs_arr          # (k,)
            r = S_k.T @ q                # (k,)
            u = rhs_arr.copy()
            # Scatter-add: handles repeated pivot indices correctly
            np.add.at(u, pivs, r)
        else:
            # 2-D: rhs_arr shape (m, ncols)
            q = W_k.T @ rhs_arr          # (k, ncols)
            r = S_k.T @ q                # (k, ncols)
            u = rhs_arr.copy()
            # np.add.at along axis 0 for 2-D
            np.add.at(u, pivs, r)

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
        position: int,
        new_column: np.ndarray,
        search_direction: np.ndarray | None = None,
    ) -> bool:
        """Replace column at position with new_column.

        Parameters
        ----------
        position : int
            Index of the basis column to replace (0 <= position < m).
        new_column : np.ndarray
            New column vector a_q of shape (m,).
        search_direction : np.ndarray | None, optional
            Pre-computed direction vector d = B^{-1} a_q. If None, it is
            computed using self.solve(new_column).

        Returns
        -------
        bool
            True if incremental eta update succeeded; False if full
            DenseLU refactorization was performed instead.
        """
        if position < 0 or position >= self._m:
            raise ValueError(f"Column index {position} out of range [0, {self._m}).")

        if self.fast_pivot_update:
            if isinstance(new_column, np.ndarray) and new_column.dtype == np.float64 and new_column.ndim == 1:
                new_col_arr = new_column
            else:
                new_col_arr = np.asarray(new_column, dtype=np.float64).ravel()

            if new_col_arr.shape[0] != self._m:
                raise ValueError(f"New column dimension {new_col_arr.shape[0]} != {self._m}.")

            if not np.all(np.isfinite(new_col_arr)):
                self._last_refactor_reason = RefactorizationReason.PIVOT_INSTABILITY
                raise SingularBasisError("New column contains non-finite values (NaN or Inf).")

            self._diagnostics = None

            # Check update count trigger
            if self._update_count >= self.max_updates:
                old_col = self._B[:, position].copy()
                self._B[:, position] = new_col_arr
                try:
                    self._factorize_base(reason=RefactorizationReason.UPDATE_LIMIT)
                except SingularBasisError:
                    self._B[:, position] = old_col
                    raise
                return False

            # Compute search direction if not provided
            if search_direction is None:
                d = self.solve(new_col_arr)
            elif isinstance(search_direction, np.ndarray) and search_direction.dtype == np.float64 and search_direction.ndim == 1:
                d = search_direction
            else:
                d = np.asarray(search_direction, dtype=np.float64).ravel()

            pivot_val = float(d[position])
            abs_pivot = abs(pivot_val)

            # Check pivot quality: if too small or non-finite, refactorize
            if abs_pivot < self.pivot_tolerance or not np.isfinite(pivot_val):
                old_col = self._B[:, position].copy()
                self._B[:, position] = new_col_arr
                try:
                    self._factorize_base(reason=RefactorizationReason.PIVOT_INSTABILITY)
                except SingularBasisError:
                    self._B[:, position] = old_col
                    raise
                return False

            # Update row and col abs sums before updating explicit basis matrix
            if self._row_abs_sums is not None:
                self._row_abs_sums += np.abs(new_col_arr) - np.abs(self._B[:, position])
            if self._col_abs_sums is not None:
                self._col_abs_sums[position] = float(np.sum(np.abs(new_col_arr)))

            # Update explicit basis matrix
            self._B[:, position] = new_col_arr

            # Build eta and accumulate into Compact WY state
            self._update_compact_wy(d, position, pivot_val)

            self._update_count += 1
            self._total_updates += 1
            return True
        else:
            # Baseline path — avoid copying old_col unless refactorization actually occurs
            if isinstance(new_column, np.ndarray) and new_column.dtype == np.float64 and new_column.ndim == 1:
                new_col_arr = new_column
            else:
                new_col_arr = np.asarray(new_column, dtype=np.float64).ravel()

            if new_col_arr.shape[0] != self._m:
                raise ValueError(f"New column dimension {new_col_arr.shape[0]} != {self._m}.")

            if not np.all(np.isfinite(new_col_arr)):
                self._last_refactor_reason = RefactorizationReason.PIVOT_INSTABILITY
                raise SingularBasisError("New column contains non-finite values (NaN or Inf).")

            self._diagnostics = None

            # Check update count trigger
            if self._update_count >= self.max_updates:
                old_col = self._B[:, position].copy()
                self._B[:, position] = new_col_arr
                try:
                    self._factorize_base(reason=RefactorizationReason.UPDATE_LIMIT)
                except SingularBasisError:
                    self._B[:, position] = old_col
                    raise
                return False

            # Compute search direction if not provided
            if search_direction is None:
                d = self.solve(new_col_arr)
            elif isinstance(search_direction, np.ndarray) and search_direction.dtype == np.float64 and search_direction.ndim == 1:
                d = search_direction
            else:
                d = np.asarray(search_direction, dtype=np.float64).ravel()

            pivot_val = float(d[position])
            abs_pivot = abs(pivot_val)

            # Check pivot quality: if too small or non-finite, refactorize
            if abs_pivot < self.pivot_tolerance or not np.isfinite(pivot_val):
                old_col = self._B[:, position].copy()
                self._B[:, position] = new_col_arr
                try:
                    self._factorize_base(reason=RefactorizationReason.PIVOT_INSTABILITY)
                except SingularBasisError:
                    self._B[:, position] = old_col
                    raise
                return False

            # Update row and col abs sums before updating explicit basis matrix
            if self._row_abs_sums is not None:
                self._row_abs_sums += np.abs(new_col_arr) - np.abs(self._B[:, position])
            if self._col_abs_sums is not None:
                self._col_abs_sums[position] = float(np.sum(np.abs(new_col_arr)))

            self._B[:, position] = new_col_arr

            # Build eta and accumulate into Compact WY state
            self._update_compact_wy(d, position, pivot_val)

            self._update_count += 1
            self._total_updates += 1
            return True

    def _update_compact_wy(self, d: np.ndarray, position: int, pivot_val: float) -> None:
        """Accumulate one new eta update into the Compact WY buffers (W, S, pivots).

        Given search direction d = B^{-1} a_q and pivot row p = position:

            eta[p] = 1 / pivot_val
            eta[i] = -d[i] / pivot_val  for i != p

            w_k = eta - e_p  (the "offset" eta column)

        Induction for S:
            S[k, :k] = W[p, :k] @ S[:k, :k]
            S[k,  k] = 1

        Parameters
        ----------
        d : np.ndarray
            Search direction vector of shape (m,).
        position : int
            Pivot row index p.
        pivot_val : float
            d[position] (non-zero, validated by caller).
        """
        k = self._k
        inv_pivot = 1.0 / pivot_val

        # Vectorized in-place write directly into column k of W:
        col = self._W[:, k]
        np.multiply(d, -inv_pivot, out=col)
        col[position] = inv_pivot - 1.0

        # Store pivot row index
        self._pivots[k] = position

        # Update S induction without intermediate allocations
        if k > 0:
            np.dot(self._W[position, :k], self._S[:k, :k], out=self._S[k, :k])
        self._S[k, k] = 1.0

        self._k += 1

    # Stage 8.40: _accumulate_eta aliases _update_compact_wy for contract compatibility
    _accumulate_eta = _update_compact_wy

    def refactorize(self, reason: RefactorizationReason = RefactorizationReason.EXPLICIT_REQUEST) -> None:
        """Explicitly force a full DenseLU refactorization from current B."""
        self._factorize_base(reason=reason)

    # -----------------------------------------------------------------------
    # Residuals & Diagnostics
    # -----------------------------------------------------------------------

    def residual(self, rhs: np.ndarray, x: np.ndarray) -> float:
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
        else:
            diff = self._B @ x_arr - rhs_arr
            norm_res = float(np.max(np.abs(diff)))
            norm_b = float(np.max(np.abs(rhs_arr)))
            norm_x = float(np.max(np.abs(x_arr)))
            norm_B = float(np.max(np.sum(np.abs(self._B), axis=1 if rhs_arr.ndim == 1 else 0)))
        scale = norm_B * norm_x + norm_b + 1e-15
        return norm_res / scale

    def transpose_residual(self, rhs: np.ndarray, y: np.ndarray) -> float:
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
