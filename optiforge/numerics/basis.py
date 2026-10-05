"""Basis factorization interface for Simplex solvers.

This module provides the BasisFactorization abstraction that encapsulates dense
LU decomposition, forward/backward triangular solves, transpose solves, and
numerical diagnostics for simplex basis matrices.
"""

from __future__ import annotations

import numpy as np

from optiforge.numerics.diagnostics import NumericalDiagnostics, compute_diagnostics
from optiforge.numerics.lu import DenseLU, dense_lu_factor


class BasisFactorization:
    """Encapsulates the numerical factorization and linear solves for a basis matrix B.

    Mathematically:
        P B = L U

    Provides:
        solve(rhs):            B x = rhs
        solve_transpose(rhs):  B^T x = rhs
        diagnostics():         NumericalDiagnostics metrics
    """

    def __init__(
        self,
        B: np.ndarray,
        pivot_tolerance: float = 1e-12,
        use_scipy_triangular_solve: bool = False,
        scipy_triangular_min_dimension: int = 10,
        triangular_solve_provider: str = "python",
        use_scipy_lu_factor: bool = False,
        scipy_lu_min_dimension: int = 10,
        lu_factor_provider: str | None = None,
    ) -> None:
        """Factorize basis matrix B using dense LU with partial pivoting.

        Parameters
        ----------
        B : np.ndarray
            Square basis matrix of shape (m, m).
        pivot_tolerance : float, optional
            Pivot threshold below which singularity is declared. Default is 1e-12.
        use_scipy_triangular_solve : bool, optional
            Whether to attempt SciPy solve_triangular acceleration. Default False.
        scipy_triangular_min_dimension : int, optional
            Dimension threshold for SciPy dispatch. Default 10.
        triangular_solve_provider : str, optional
            Triangular solve provider selection ('python' or 'scipy'). Default 'python'.
        use_scipy_lu_factor : bool, optional
            Whether to attempt SciPy LAPACK dgetrf factorization. Default False.
        scipy_lu_min_dimension : int, optional
            Dimension threshold for SciPy lu_factor dispatch. Default 10.
        lu_factor_provider : str | None, optional
            LU factorization provider selection ('python' or 'scipy'). Default None.
        """
        self.B = np.asarray(B, dtype=np.float64)
        self.pivot_tolerance = pivot_tolerance
        prov = "scipy" if (use_scipy_triangular_solve or triangular_solve_provider == "scipy") else "python"
        self.triangular_solve_provider = prov
        self.use_scipy_triangular_solve = (prov == "scipy")
        self.scipy_triangular_min_dimension = scipy_triangular_min_dimension
        eff_lu_prov = lu_factor_provider if lu_factor_provider is not None else ("scipy" if use_scipy_lu_factor else "python")
        self.lu_factor_provider = eff_lu_prov
        self.use_scipy_lu_factor = (eff_lu_prov == "scipy")
        self.scipy_lu_min_dimension = scipy_lu_min_dimension

        self.lu: DenseLU = dense_lu_factor(
            self.B,
            pivot_tolerance=pivot_tolerance,
            use_scipy_triangular_solve=self.use_scipy_triangular_solve,
            scipy_triangular_min_dimension=scipy_triangular_min_dimension,
            triangular_solve_provider=self.triangular_solve_provider,
            use_scipy_lu_factor=self.use_scipy_lu_factor,
            scipy_lu_min_dimension=self.scipy_lu_min_dimension,
            lu_factor_provider=self.lu_factor_provider,
        )
        self._diagnostics: NumericalDiagnostics | None = None

    @classmethod
    def factorize(
        cls,
        B: np.ndarray,
        pivot_tolerance: float = 1e-12,
        use_scipy_triangular_solve: bool = False,
        scipy_triangular_min_dimension: int = 10,
        triangular_solve_provider: str = "python",
        use_scipy_lu_factor: bool = False,
        scipy_lu_min_dimension: int = 10,
        lu_factor_provider: str | None = None,
    ) -> BasisFactorization:
        """Factory method to construct a BasisFactorization."""
        return cls(
            B,
            pivot_tolerance=pivot_tolerance,
            use_scipy_triangular_solve=use_scipy_triangular_solve,
            scipy_triangular_min_dimension=scipy_triangular_min_dimension,
            triangular_solve_provider=triangular_solve_provider,
            use_scipy_lu_factor=use_scipy_lu_factor,
            scipy_lu_min_dimension=scipy_lu_min_dimension,
            lu_factor_provider=lu_factor_provider,
        )

    @property
    def m(self) -> int:
        """Matrix dimension."""
        return self.lu.m

    @property
    def L(self) -> np.ndarray:
        """Unit lower triangular matrix L."""
        return self.lu.L

    @property
    def U(self) -> np.ndarray:
        """Upper triangular matrix U."""
        return self.lu.U

    @property
    def P(self) -> np.ndarray:
        """Explicit permutation matrix P."""
        return self.lu.P

    @property
    def perm(self) -> np.ndarray:
        """Permutation vector."""
        return self.lu.perm

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve B x = rhs using the existing LU factorization.

        Parameters
        ----------
        rhs : np.ndarray
            Right-hand side vector of length m or matrix of shape (m, k).

        Returns
        -------
        np.ndarray
            Solution x of matching shape.
        """
        return self.lu.solve(rhs)

    def solve_transpose(self, rhs: np.ndarray) -> np.ndarray:
        """Solve B^T x = rhs using the existing LU factorization.

        Parameters
        ----------
        rhs : np.ndarray
            Right-hand side vector of length m or matrix of shape (m, k).

        Returns
        -------
        np.ndarray
            Solution x of matching shape.
        """
        return self.lu.solve_transpose(rhs)

    def diagnostics(self, estimate_condition: bool = True) -> NumericalDiagnostics:
        """Compute or return cached numerical diagnostics for this basis.

        Parameters
        ----------
        estimate_condition : bool, optional
            Whether to compute condition number via np.linalg.cond. Default True.

        Returns
        -------
        NumericalDiagnostics
        """
        if self._diagnostics is None:
            self._diagnostics = compute_diagnostics(
                self.B,
                self.lu,
                estimate_condition=estimate_condition,
            )
        return self._diagnostics

    @property
    def condition_number(self) -> float:
        """Condition number of the basis matrix."""
        return self.diagnostics().condition_number
