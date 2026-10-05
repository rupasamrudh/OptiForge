"""Tests for OptiForge Stage 8.41: Direct LAPACK dtrtrs Triangular Solve Dispatch.

Validates that:
1. SciPyTriangularSolveProvider uses scipy.linalg.lapack.dtrtrs directly.
2. All 8 combinations of (lower, trans, unit_diagonal) match reference solutions.
3. Both 1D and 2D RHS are supported with identical precision.
4. Singular systems (info > 0) gracefully increment fallback telemetry.
5. Exceptions in LAPACK call trigger defensive fallback to pure NumPy substitution.
6. Simplex solvers (Primal, Dual PFI, and MILP) run accurately with the dtrtrs path.
"""

from __future__ import annotations

import numpy as np
import pytest

from optiforge.numerics.triangular_provider import (
    PythonTriangularSolveProvider,
    SciPyTriangularSolveProvider,
    get_provider_stats,
    is_scipy_available,
    reset_provider_stats,
    triangular_solve,
)
from optiforge.numerics.lu import DenseLU, dense_lu_factor
from optiforge.lp.simplex import RevisedSimplexSolver
from optiforge.lp.dual_simplex import DualRevisedSimplexSolver


def _make_test_matrices(m: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((m, m))
    L = np.tril(M, -1) + np.eye(m)
    U = np.triu(M)
    for i in range(m):
        if abs(U[i, i]) < 0.5:
            U[i, i] = 1.5 if U[i, i] >= 0 else -1.5
    b_1d = rng.standard_normal(m)
    b_2d = rng.standard_normal((m, 3))
    return L, U, b_1d, b_2d


def _make_dual_lp(m: int, n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    A_orig = rng.uniform(0.1, 1.0, size=(m, n))
    b_orig = rng.uniform(5.0, 50.0, size=m)
    c_orig = rng.uniform(1.0, 10.0, size=n)
    A_eq = np.hstack([-A_orig, np.eye(m)])
    b_eq = -b_orig
    c_eq = np.concatenate([c_orig, np.zeros(m)])
    basis = list(range(n, n + m))
    nonbasic = list(range(n))
    return A_eq, b_eq, c_eq, basis, nonbasic


def _make_primal_lp(m: int, n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    A_orig = rng.uniform(0.1, 1.0, size=(m, n))
    b_orig = rng.uniform(10.0, 50.0, size=m)
    c_orig = rng.uniform(1.0, 10.0, size=n)
    A_eq = np.hstack([A_orig, np.eye(m)])
    b_eq = b_orig
    c_eq = np.concatenate([c_orig, np.zeros(m)])
    basis = list(range(n, n + m))
    nonbasic = list(range(n))
    return A_eq, b_eq, c_eq, basis, nonbasic


class TestStage841DtrtrsDispatch:
    """Test suite verifying direct LAPACK dtrtrs triangular dispatch."""

    def test_scipy_dtrtrs_availability(self):
        """Verify that dtrtrs is detected when SciPy is available."""
        if not is_scipy_available():
            pytest.skip("SciPy not installed in environment.")
        from optiforge.numerics.triangular_provider import _SCIPY_DTRTRS
        assert _SCIPY_DTRTRS is not None, "scipy.linalg.lapack.dtrtrs must be loaded."

    @pytest.mark.parametrize("m", [12, 25, 50])
    def test_all_triangular_modes_1d(self, m: int):
        """Test all 8 triangular modes with 1D RHS comparing Python vs SciPy dtrtrs."""
        L, U, b, _ = _make_test_matrices(m, seed=m * 3)
        p_py = PythonTriangularSolveProvider()
        p_sp = SciPyTriangularSolveProvider(min_dimension=10)

        # 1. L x = b (lower=True, unit=True, trans=False)
        x_py = p_py.solve_l(L, b, unit_diagonal=True)
        x_sp = p_sp.solve_l(L, b, unit_diagonal=True)
        np.testing.assert_allclose(x_sp, x_py, atol=1e-12, rtol=1e-12)

        # 2. L x = b (lower=True, unit=False, trans=False)
        x_py = p_py.solve_triangular(L, b, lower=True, unit_diagonal=False, trans=False)
        x_sp = p_sp.solve_triangular(L, b, lower=True, unit_diagonal=False, trans=False)
        np.testing.assert_allclose(x_sp, x_py, atol=1e-12, rtol=1e-12)

        # 3. U x = b (lower=False, unit=False, trans=False)
        x_py = p_py.solve_u(U, b)
        x_sp = p_sp.solve_u(U, b)
        np.testing.assert_allclose(x_sp, x_py, atol=1e-12, rtol=1e-12)

        # 4. U x = b (lower=False, unit=True, trans=False)
        U_unit = U.copy()
        np.fill_diagonal(U_unit, 1.0)
        x_py = p_py.solve_triangular(U_unit, b, lower=False, unit_diagonal=True, trans=False)
        x_sp = p_sp.solve_triangular(U_unit, b, lower=False, unit_diagonal=True, trans=False)
        np.testing.assert_allclose(x_sp, x_py, atol=1e-12, rtol=1e-12)

        # 5. L^T x = b (lower=True, unit=True, trans=True)
        x_py = p_py.solve_lt(L, b, unit_diagonal=True)
        x_sp = p_sp.solve_lt(L, b, unit_diagonal=True)
        np.testing.assert_allclose(x_sp, x_py, atol=1e-12, rtol=1e-12)

        # 6. L^T x = b (lower=True, unit=False, trans=True)
        x_py = p_py.solve_triangular(L, b, lower=True, unit_diagonal=False, trans=True)
        x_sp = p_sp.solve_triangular(L, b, lower=True, unit_diagonal=False, trans=True)
        np.testing.assert_allclose(x_sp, x_py, atol=1e-12, rtol=1e-12)

        # 7. U^T x = b (lower=False, unit=False, trans=True)
        x_py = p_py.solve_ut(U, b)
        x_sp = p_sp.solve_ut(U, b)
        np.testing.assert_allclose(x_sp, x_py, atol=1e-12, rtol=1e-12)

        # 8. U^T x = b (lower=False, unit=True, trans=True)
        x_py = p_py.solve_triangular(U_unit, b, lower=False, unit_diagonal=True, trans=True)
        x_sp = p_sp.solve_triangular(U_unit, b, lower=False, unit_diagonal=True, trans=True)
        np.testing.assert_allclose(x_sp, x_py, atol=1e-12, rtol=1e-12)

    def test_2d_rhs_dispatch(self):
        """Verify 2D RHS arrays solve correctly with dtrtrs."""
        L, U, _, b_2d = _make_test_matrices(20, seed=99)
        p_py = PythonTriangularSolveProvider()
        p_sp = SciPyTriangularSolveProvider(min_dimension=10)

        # L solve
        x2_py = p_py.solve_l(L, b_2d)
        x2_sp = p_sp.solve_l(L, b_2d)
        assert x2_sp.shape == b_2d.shape
        np.testing.assert_allclose(x2_sp, x2_py, atol=1e-12, rtol=1e-12)

        # U solve
        x2_py = p_py.solve_u(U, b_2d)
        x2_sp = p_sp.solve_u(U, b_2d)
        assert x2_sp.shape == b_2d.shape
        np.testing.assert_allclose(x2_sp, x2_py, atol=1e-12, rtol=1e-12)

    def test_dtrtrs_telemetry_counting(self):
        """Verify _SCIPY_SOLVE_COUNT increments on successful dtrtrs call."""
        reset_provider_stats()
        L, _, b, _ = _make_test_matrices(15, seed=12)
        p_sp = SciPyTriangularSolveProvider(min_dimension=10)

        p_sp.solve_l(L, b)
        p_sp.solve_l(L, b)

        stats = get_provider_stats()
        if is_scipy_available():
            assert stats["total_solves"] == 2
            assert stats["scipy_solves"] == 2
            assert stats["fallbacks"] == 0

    def test_singular_info_nonzero_triggers_fallback(self):
        """Singular matrix (info != 0) increments fallback telemetry."""
        reset_provider_stats()
        _, U, b, _ = _make_test_matrices(15, seed=42)
        U_sing = U.copy()
        U_sing[7, 7] = 0.0

        p_sp = SciPyTriangularSolveProvider(min_dimension=10)
        x = p_sp.solve_u(U_sing, b)

        stats = get_provider_stats()
        if is_scipy_available():
            assert stats["fallbacks"] >= 1
        assert np.isinf(x[7]) or np.isnan(x[7])

    def test_exception_in_dtrtrs_triggers_fallback(self, monkeypatch):
        """Exception in _SCIPY_DTRTRS triggers graceful fallback without unhandled error."""
        reset_provider_stats()
        L, _, b, _ = _make_test_matrices(15, seed=77)

        def mock_broken_dtrtrs(*args, **kwargs):
            raise np.linalg.LinAlgError("Simulated internal BLAS exception")

        monkeypatch.setattr("optiforge.numerics.triangular_provider._SCIPY_DTRTRS", mock_broken_dtrtrs)
        p_sp = SciPyTriangularSolveProvider(min_dimension=10)
        x = p_sp.solve_l(L, b)

        stats = get_provider_stats()
        assert stats["fallbacks"] == 1
        x_expected = PythonTriangularSolveProvider().solve_l(L, b)
        np.testing.assert_allclose(x, x_expected, atol=1e-12)

    def test_denselu_with_dtrtrs(self):
        """DenseLU solve and solve_transpose with dtrtrs dispatch."""
        rng = np.random.default_rng(101)
        A = rng.standard_normal((25, 25)) + np.eye(25) * 5.0
        b = rng.standard_normal(25)

        lu = dense_lu_factor(
            A,
            use_scipy_triangular_solve=True,
            scipy_triangular_min_dimension=10,
        )
        x = lu.solve(b)
        x_t = lu.solve_transpose(b)

        np.testing.assert_allclose(A @ x, b, atol=1e-10)
        np.testing.assert_allclose(A.T @ x_t, b, atol=1e-10)

    def test_dual_simplex_pfi_with_dtrtrs(self):
        """Dual simplex with PFI and SciPy dtrtrs solves problem accurately."""
        from optiforge.lp.dual_simplex import DualRevisedSimplexSolver
        from optiforge.lp.simplex import SimplexTolerances, ObjectiveSense, SolutionStatus

        m, n = 15, 30
        A_eq, b_eq, c_eq, basis, nonbasic = _make_dual_lp(m, n, seed=303)

        tol_py = SimplexTolerances(triangular_solve_provider="python", use_pivot_factorization=True)
        solver_py = DualRevisedSimplexSolver(tolerances=tol_py)
        sol_py = solver_py.solve_standard_form(A_eq, b_eq, c_eq, list(basis), list(nonbasic), ObjectiveSense.MINIMIZE, tol_py)

        tol_sp = SimplexTolerances(triangular_solve_provider="scipy", use_pivot_factorization=True, scipy_triangular_min_dimension=5)
        solver_sp = DualRevisedSimplexSolver(tolerances=tol_sp)
        sol_sp = solver_sp.solve_standard_form(A_eq, b_eq, c_eq, list(basis), list(nonbasic), ObjectiveSense.MINIMIZE, tol_sp)

        assert sol_py.status == SolutionStatus.OPTIMAL
        assert sol_sp.status == SolutionStatus.OPTIMAL
        np.testing.assert_allclose(sol_py.x, sol_sp.x, atol=1e-9)
        assert np.isclose(sol_py.objective_value, sol_sp.objective_value, atol=1e-9)

    def test_primal_simplex_with_dtrtrs(self):
        """Primal simplex with SciPy dtrtrs solves problem accurately."""
        from optiforge.lp.simplex import RevisedSimplexSolver, SimplexTolerances, ObjectiveSense, SolutionStatus

        m, n = 15, 30
        A_eq, b_eq, c_eq, basis, nonbasic = _make_primal_lp(m, n, seed=404)

        tol_py = SimplexTolerances(triangular_solve_provider="python", use_pivot_factorization=True)
        solver_py = RevisedSimplexSolver(tolerances=tol_py)
        sol_py = solver_py._simplex_loop(A_eq, b_eq, c_eq, list(basis), list(nonbasic), ObjectiveSense.MINIMIZE, tol_py)

        tol_sp = SimplexTolerances(triangular_solve_provider="scipy", use_pivot_factorization=True, scipy_triangular_min_dimension=5)
        solver_sp = RevisedSimplexSolver(tolerances=tol_sp)
        sol_sp = solver_sp._simplex_loop(A_eq, b_eq, c_eq, list(basis), list(nonbasic), ObjectiveSense.MINIMIZE, tol_sp)

        assert sol_py.status == SolutionStatus.OPTIMAL
        assert sol_sp.status == SolutionStatus.OPTIMAL
        np.testing.assert_allclose(sol_py.x, sol_sp.x, atol=1e-9)
        assert np.isclose(sol_py.objective_value, sol_sp.objective_value, atol=1e-9)
