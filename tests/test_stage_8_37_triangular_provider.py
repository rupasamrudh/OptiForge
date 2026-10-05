"""Stage 8.37 Dedicated Test Suite: Triangular Solve Provider Integration.

Tests cover:
Phase 5: Numerical Equivalence Tests
- Small triangular systems (m < 10)
- Random well-conditioned systems
- Ill-conditioned triangular systems
- Diagonal-dominant systems
- Unit-diagonal systems
- 1-D RHS
- 2-D RHS
- Forward substitution (L x = b)
- Back substitution (U x = b)
- Transposed solves (L^T x = b, U^T x = b)
- Comparison against numpy.linalg.solve via residual checks

Phase 6: Failure / Edge Cases
- Singular triangular matrices (zero diagonal)
- Extremely small diagonal (near machine eps)
- Non-finite inputs (NaN, Inf)
- Empty dimension (m = 0)
- Incompatible RHS dimensions (dimension mismatch, 3-D array)
- Integer dtypes normalized to float64
- Float32 dtypes normalized to float64

Phase 7: Integration Tests
- DenseLU solve and transpose solve
- PFI / Compact WY FTRAN and BTRAN
- Primal Revised Simplex end-to-end equivalence
- Dual Revised Simplex end-to-end equivalence
- MILP Branch-and-Bound warm-start equivalence
"""

from __future__ import annotations

import numpy as np
import pytest

from optiforge.core.solution import SolutionStatus
from optiforge.lp.dual_simplex import DualRevisedSimplexSolver
from optiforge.lp.simplex import RevisedSimplexSolver, SimplexTolerances
from optiforge.milp.solver import MILPSolver, MILPTolerances
from optiforge.model.constraint import Constraint, ConstraintSense
from optiforge.model.model import Model
from optiforge.model.objective import Objective, ObjectiveSense
from optiforge.model.variable import Variable, VariableType
from optiforge.numerics.basis import BasisFactorization
from optiforge.numerics.exceptions import NumericalSingularityError, SingularBasisError
from optiforge.numerics.lu import (
    DenseLU,
    back_substitution,
    back_substitution_transpose,
    dense_lu_factor,
    forward_substitution,
    forward_substitution_transpose,
    lu_solve,
    lu_solve_transpose,
)
from optiforge.numerics.pivot_basis import PivotUpdateFactorization
from optiforge.numerics.triangular_provider import (
    DEFAULT_SCIPY_MIN_DIMENSION,
    PythonTriangularSolveProvider,
    SciPyTriangularSolveProvider,
    TriangularProviderConfig,
    TriangularProviderType,
    TriangularSolveProvider,
    backward_solve,
    forward_solve,
    get_global_provider_config,
    get_provider_stats,
    get_triangular_provider,
    is_scipy_available,
    reset_provider_stats,
    set_global_provider_config,
    triangular_lu_solve,
    triangular_lu_solve_transpose,
    triangular_solve,
)


# ===========================================================================
# Helper Generators
# ===========================================================================

def _make_triangular_pair(m: int, seed: int = 42, diag_scale: float = 2.0):
    rng = np.random.default_rng(seed)
    L = np.tril(rng.standard_normal((m, m)) * 0.1, -1) + np.eye(m)
    U = np.triu(rng.standard_normal((m, m)) * 0.1, 1) + np.eye(m) * diag_scale
    return L, U


def _relative_residual(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """Compute relative residual ||A x - b|| / (||A|| * ||x|| + ||b||)."""
    res = np.linalg.norm(A @ x - b)
    denom = np.linalg.norm(A) * np.linalg.norm(x) + np.linalg.norm(b) + 1e-16
    return float(res / denom)


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


# ===========================================================================
# Phase 5: Numerical Equivalence Tests
# ===========================================================================

class TestPhase5NumericalEquivalence:
    """Rigorous equivalence tests between Python and SciPy providers."""

    @pytest.mark.parametrize("m", [2, 4, 7, 9])
    def test_small_triangular_systems(self, m):
        """1. Small triangular systems below default crossover (m < 10)."""
        L, U = _make_triangular_pair(m, seed=100 + m)
        b = np.arange(1.0, m + 1.0)

        # Force SciPy provider with min_dimension=0 to test actual SciPy kernel on small m
        p_py = PythonTriangularSolveProvider()
        p_sp = SciPyTriangularSolveProvider(min_dimension=0)

        # Forward solve L x = b
        x_py_f = p_py.solve_l(L, b)
        x_sp_f = p_sp.solve_l(L, b)
        assert np.allclose(x_py_f, x_sp_f, atol=1e-12, rtol=1e-12)
        assert _relative_residual(L, x_sp_f, b) < 1e-13

        # Backward solve U x = b
        x_py_b = p_py.solve_u(U, b)
        x_sp_b = p_sp.solve_u(U, b)
        assert np.allclose(x_py_b, x_sp_b, atol=1e-12, rtol=1e-12)
        assert _relative_residual(U, x_sp_b, b) < 1e-13

    @pytest.mark.parametrize("m", [15, 30, 50])
    def test_random_well_conditioned_systems(self, m):
        """2. Random well-conditioned systems comparing both providers and numpy.linalg.solve."""
        L, U = _make_triangular_pair(m, seed=200 + m, diag_scale=5.0)
        rng = np.random.default_rng(300 + m)
        b = rng.standard_normal(m)

        p_py = PythonTriangularSolveProvider()
        p_sp = SciPyTriangularSolveProvider(min_dimension=10)

        # Compare L x = b against np.linalg.solve
        x_np_L = np.linalg.solve(L, b)
        x_py_L = p_py.solve_l(L, b)
        x_sp_L = p_sp.solve_l(L, b)
        assert np.allclose(x_py_L, x_np_L, atol=1e-12, rtol=1e-12)
        assert np.allclose(x_sp_L, x_np_L, atol=1e-12, rtol=1e-12)

        # Compare U x = b against np.linalg.solve
        x_np_U = np.linalg.solve(U, b)
        x_py_U = p_py.solve_u(U, b)
        x_sp_U = p_sp.solve_u(U, b)
        assert np.allclose(x_py_U, x_np_U, atol=1e-12, rtol=1e-12)
        assert np.allclose(x_sp_U, x_np_U, atol=1e-12, rtol=1e-12)

    def test_ill_conditioned_triangular_systems(self):
        """3. Ill-conditioned systems with decaying diagonal (cond ~ 1e9)."""
        m = 25
        rng = np.random.default_rng(42)
        # Create upper triangular matrix with geometrically decaying diagonal
        diag = np.logspace(0, -9, m)
        U = np.triu(rng.standard_normal((m, m)) * 0.05, 1) + np.diag(diag)
        b = rng.standard_normal(m)

        p_py = PythonTriangularSolveProvider()
        p_sp = SciPyTriangularSolveProvider(min_dimension=10)

        x_py = p_py.solve_u(U, b)
        x_sp = p_sp.solve_u(U, b)

        # For ill-conditioned systems, forward error between BLAS and loops can be larger,
        # but relative residuals should both remain within machine precision tolerance.
        res_py = _relative_residual(U, x_py, b)
        res_sp = _relative_residual(U, x_sp, b)
        assert res_py < 1e-11
        assert res_sp < 1e-11
        assert np.allclose(res_py, res_sp, atol=1e-11)

    def test_diagonal_dominant_systems(self):
        """4. Strongly diagonal-dominant systems."""
        m = 20
        rng = np.random.default_rng(99)
        L = np.tril(rng.standard_normal((m, m)) * 0.01, -1) + np.eye(m) * 10.0
        b = rng.standard_normal(m)

        p_py = PythonTriangularSolveProvider()
        p_sp = SciPyTriangularSolveProvider(min_dimension=10)

        x_py = p_py.solve_triangular(L, b, lower=True, unit_diagonal=False)
        x_sp = p_sp.solve_triangular(L, b, lower=True, unit_diagonal=False)
        assert np.allclose(x_py, x_sp, atol=1e-13, rtol=1e-13)

    def test_unit_diagonal_systems(self):
        """5. Unit-diagonal lower triangular systems."""
        m = 25
        L, _ = _make_triangular_pair(m, seed=55)
        # Verify L has ones on diagonal
        assert np.allclose(np.diag(L), 1.0)
        b = np.ones(m)

        p_py = PythonTriangularSolveProvider()
        p_sp = SciPyTriangularSolveProvider(min_dimension=10)

        x_py = p_py.solve_l(L, b, unit_diagonal=True)
        x_sp = p_sp.solve_l(L, b, unit_diagonal=True)
        assert np.allclose(x_py, x_sp, atol=1e-13, rtol=1e-13)

    def test_1d_and_2d_rhs_consistency(self):
        """6 & 7. Equivalence of 1-D RHS vector and 2-D multi-column matrix RHS."""
        m = 20
        k = 5
        L, U = _make_triangular_pair(m, seed=77)
        rng = np.random.default_rng(88)
        B_mat = rng.standard_normal((m, k))

        p_py = PythonTriangularSolveProvider()
        p_sp = SciPyTriangularSolveProvider(min_dimension=10)

        # 2-D matrix solve
        X_py_2d = p_py.solve_u(U, B_mat)
        X_sp_2d = p_sp.solve_u(U, B_mat)
        assert np.allclose(X_py_2d, X_sp_2d, atol=1e-12, rtol=1e-12)

        # Column-by-column 1-D solves must match 2-D columns
        for col in range(k):
            b_col = B_mat[:, col]
            x_py_1d = p_py.solve_u(U, b_col)
            x_sp_1d = p_sp.solve_u(U, b_col)
            assert np.allclose(X_py_2d[:, col], x_py_1d, atol=1e-14)
            assert np.allclose(X_sp_2d[:, col], x_sp_1d, atol=1e-14)

    def test_transposed_solves(self):
        """8, 9, 10. Forward/Back substitution under transpose: L^T x = b and U^T x = b."""
        m = 22
        L, U = _make_triangular_pair(m, seed=12)
        rng = np.random.default_rng(34)
        b = rng.standard_normal(m)

        p_py = PythonTriangularSolveProvider()
        p_sp = SciPyTriangularSolveProvider(min_dimension=10)

        # L^T x = b
        x_py_lt = p_py.solve_lt(L, b, unit_diagonal=True)
        x_sp_lt = p_sp.solve_lt(L, b, unit_diagonal=True)
        x_np_lt = np.linalg.solve(L.T, b)
        assert np.allclose(x_py_lt, x_np_lt, atol=1e-12, rtol=1e-12)
        assert np.allclose(x_sp_lt, x_np_lt, atol=1e-12, rtol=1e-12)

        # U^T x = b
        x_py_ut = p_py.solve_ut(U, b)
        x_sp_ut = p_sp.solve_ut(U, b)
        x_np_ut = np.linalg.solve(U.T, b)
        assert np.allclose(x_py_ut, x_np_ut, atol=1e-12, rtol=1e-12)
        assert np.allclose(x_sp_ut, x_np_ut, atol=1e-12, rtol=1e-12)

        # Transpose with 2-D RHS
        B_mat = rng.standard_normal((m, 3))
        X_py_lt2 = p_py.solve_lt(L, B_mat, unit_diagonal=True)
        X_sp_lt2 = p_sp.solve_lt(L, B_mat, unit_diagonal=True)
        assert np.allclose(X_py_lt2, X_sp_lt2, atol=1e-12, rtol=1e-12)

        X_py_ut2 = p_py.solve_ut(U, B_mat)
        X_sp_ut2 = p_sp.solve_ut(U, B_mat)
        assert np.allclose(X_py_ut2, X_sp_ut2, atol=1e-12, rtol=1e-12)


# ===========================================================================
# Phase 6: Failure / Edge Cases
# ===========================================================================

class TestPhase6FailureAndEdgeCases:
    """Failure, edge case, and parameter validation tests."""

    def test_singular_triangular_matrix_zero_diagonal(self):
        """Exact zero on diagonal triggers graceful fallback without unhandled exception."""
        reset_provider_stats()
        m = 12
        _, U = _make_triangular_pair(m, seed=50)
        U_zero = U.copy()
        U_zero[5, 5] = 0.0
        b = np.ones(m)

        p_sp = SciPyTriangularSolveProvider(min_dimension=10)
        # Under SciPy, LinAlgError is caught and deterministically falls back to pure NumPy
        x = p_sp.solve_u(U_zero, b)
        stats = get_provider_stats()
        if is_scipy_available():
            assert stats["fallbacks"] >= 1
        assert np.isinf(x[5]) or np.isnan(x[5])

    def test_extremely_small_diagonal(self):
        """Near-zero diagonal (1e-16) solves backward stably."""
        m = 15
        _, U = _make_triangular_pair(m, seed=60)
        U_near = U.copy()
        U_near[m - 1, m - 1] = 1e-16
        b = np.ones(m)

        x_py = PythonTriangularSolveProvider().solve_u(U_near, b)
        x_sp = SciPyTriangularSolveProvider(min_dimension=10).solve_u(U_near, b)
        assert np.isclose(x_py[-1], x_sp[-1], rtol=1e-5)

    def test_nan_and_inf_inputs(self):
        """Non-finite values propagate IEEE 754 float semantics gracefully."""
        m = 12
        L, _ = _make_triangular_pair(m, seed=70)
        b_nan = np.ones(m)
        b_nan[3] = np.nan

        p_py = PythonTriangularSolveProvider()
        p_sp = SciPyTriangularSolveProvider(min_dimension=10)

        x_py = p_py.solve_l(L, b_nan)
        x_sp = p_sp.solve_l(L, b_nan)
        assert np.isnan(x_py[3])
        assert np.isnan(x_sp[3])

        b_inf = np.ones(m)
        b_inf[2] = np.inf
        x_py_inf = p_py.solve_l(L, b_inf)
        x_sp_inf = p_sp.solve_l(L, b_inf)
        assert np.isinf(x_py_inf[2])
        assert np.isinf(x_sp_inf[2])

    def test_empty_dimensions(self):
        """Empty systems (m = 0) return empty arrays of matching shape."""
        L0 = np.zeros((0, 0), dtype=np.float64)
        b0_1d = np.zeros(0, dtype=np.float64)
        b0_2d = np.zeros((0, 3), dtype=np.float64)

        p_py = PythonTriangularSolveProvider()
        p_sp = SciPyTriangularSolveProvider(min_dimension=0)

        assert p_py.solve_l(L0, b0_1d).shape == (0,)
        assert p_sp.solve_l(L0, b0_1d).shape == (0,)
        assert p_py.solve_l(L0, b0_2d).shape == (0, 3)
        assert p_sp.solve_l(L0, b0_2d).shape == (0, 3)

    def test_incompatible_dimensions(self):
        """Dimension mismatches and non-2D matrices raise ValueError."""
        L, _ = _make_triangular_pair(10)
        p_py = PythonTriangularSolveProvider()
        p_sp = SciPyTriangularSolveProvider()

        # Wrong RHS length
        with pytest.raises(ValueError, match="RHS dimension"):
            p_py.solve_l(L, np.ones(8))
        with pytest.raises(ValueError, match="RHS dimension"):
            p_sp.solve_l(L, np.ones(8))

        # 3-D RHS
        with pytest.raises(ValueError, match="RHS must be 1D or 2D"):
            p_py.solve_l(L, np.ones((10, 2, 2)))
        with pytest.raises(ValueError, match="RHS must be 1D or 2D"):
            p_sp.solve_l(L, np.ones((10, 2, 2)))

        # Non-square matrix
        with pytest.raises(ValueError, match="Matrix A must be square"):
            p_py.solve_l(np.ones((10, 8)), np.ones(10))
        with pytest.raises(ValueError, match="Matrix A must be square"):
            p_sp.solve_l(np.ones((10, 8)), np.ones(10))

    def test_integer_and_float32_normalization(self):
        """Integer and float32 inputs are safely normalized to float64."""
        L = np.array([[1, 0], [2, 1]], dtype=np.int32)
        b = np.array([3, 4], dtype=np.int32)

        p_py = PythonTriangularSolveProvider()
        p_sp = SciPyTriangularSolveProvider(min_dimension=0)

        x_py = p_py.solve_l(L, b)
        x_sp = p_sp.solve_l(L, b)
        assert x_py.dtype == np.float64
        assert x_sp.dtype == np.float64
        assert np.allclose(x_py, x_sp)

        # Float32 inputs
        L_f32 = L.astype(np.float32)
        b_f32 = b.astype(np.float32)
        x_f32_py = p_py.solve_l(L_f32, b_f32)
        x_f32_sp = p_sp.solve_l(L_f32, b_f32)
        assert x_f32_py.dtype == np.float64
        assert x_f32_sp.dtype == np.float64
        assert np.allclose(x_f32_py, x_f32_sp)

    def test_provider_selection_mechanism(self):
        """Configuration and factory function return correct providers."""
        p_default = get_triangular_provider()
        assert isinstance(p_default, PythonTriangularSolveProvider)

        p_py = get_triangular_provider("python")
        assert isinstance(p_py, PythonTriangularSolveProvider)

        p_sp = get_triangular_provider("scipy")
        assert isinstance(p_sp, SciPyTriangularSolveProvider)

        p_bool_t = get_triangular_provider(True)
        assert isinstance(p_bool_t, SciPyTriangularSolveProvider)

        p_bool_f = get_triangular_provider(False)
        assert isinstance(p_bool_f, PythonTriangularSolveProvider)

        with pytest.raises(ValueError, match="Unknown triangular solve provider"):
            get_triangular_provider("invalid_provider")


# ===========================================================================
# Phase 7: Integration Tests
# ===========================================================================

class TestPhase7Integration:
    """Full solver integration tests comparing Python and SciPy provider paths."""

    def test_denselu_solve_and_transpose_solve(self):
        """DenseLU solve() and solve_transpose() across both providers."""
        m = 25
        rng = np.random.default_rng(101)
        B = rng.standard_normal((m, m)) + np.eye(m) * 6.0
        b = rng.standard_normal(m)

        lu_py = dense_lu_factor(B, triangular_solve_provider="python")
        lu_sp = dense_lu_factor(B, triangular_solve_provider="scipy", scipy_triangular_min_dimension=10)

        # Primal solve
        x_py = lu_py.solve(b)
        x_sp = lu_sp.solve(b)
        assert np.allclose(x_py, x_sp, atol=1e-12)
        assert _relative_residual(B, x_sp, b) < 1e-12

        # Transpose solve
        xt_py = lu_py.solve_transpose(b)
        xt_sp = lu_sp.solve_transpose(b)
        assert np.allclose(xt_py, xt_sp, atol=1e-12)
        assert _relative_residual(B.T, xt_sp, b) < 1e-12

    def test_pfi_compact_wy_ftran_and_btran(self):
        """PivotUpdateFactorization FTRAN and BTRAN with multiple pivots."""
        m = 20
        rng = np.random.default_rng(202)
        B = rng.standard_normal((m, m)) + np.eye(m) * 5.0

        pfi_py = PivotUpdateFactorization(B, triangular_solve_provider="python")
        pfi_sp = PivotUpdateFactorization(B, triangular_solve_provider="scipy", scipy_triangular_min_dimension=10)

        # Perform 5 pivot updates using replace_column
        for p in range(5):
            col_new = rng.standard_normal(m) + np.eye(m)[:, p] * 3.0
            pfi_py.replace_column(p, col_new)
            pfi_sp.replace_column(p, col_new)

        rhs = rng.standard_normal(m)

        # FTRAN
        x_py = pfi_py.solve(rhs)
        x_sp = pfi_sp.solve(rhs)
        assert np.allclose(x_py, x_sp, atol=1e-12)
        assert np.allclose(pfi_sp.B @ x_sp, rhs, atol=1e-9)

        # BTRAN
        y_py = pfi_py.solve_transpose(rhs)
        y_sp = pfi_sp.solve_transpose(rhs)
        assert np.allclose(y_py, y_sp, atol=1e-12)
        assert np.allclose(pfi_sp.B.T @ y_sp, rhs, atol=1e-9)

    def test_dual_revised_simplex_equivalence(self):
        """Dual Revised Simplex end-to-end equivalence under Python vs SciPy."""
        m, n = 20, 40
        A_eq, b_eq, c_eq, basis, nonbasic = _make_dual_lp(m, n, seed=303)

        tol_py = SimplexTolerances(
            triangular_solve_provider="python",
            use_pivot_factorization=True,
        )
        solver_py = DualRevisedSimplexSolver(tolerances=tol_py)
        sol_py = solver_py.solve_standard_form(
            A_eq, b_eq, c_eq, list(basis), list(nonbasic), ObjectiveSense.MINIMIZE, tol_py
        )

        tol_sp = SimplexTolerances(
            triangular_solve_provider="scipy",
            use_pivot_factorization=True,
            scipy_triangular_min_dimension=5,
        )
        solver_sp = DualRevisedSimplexSolver(tolerances=tol_sp)
        sol_sp = solver_sp.solve_standard_form(
            A_eq, b_eq, c_eq, list(basis), list(nonbasic), ObjectiveSense.MINIMIZE, tol_sp
        )

        assert sol_py.status == SolutionStatus.OPTIMAL
        assert sol_sp.status == SolutionStatus.OPTIMAL
        assert abs(sol_py.objective_value - sol_sp.objective_value) < 1e-9
        assert np.allclose(sol_py.x, sol_sp.x, atol=1e-9)
        assert sol_py.iterations == sol_sp.iterations

    def test_primal_revised_simplex_equivalence(self):
        """Primal Revised Simplex end-to-end equivalence under Python vs SciPy."""
        m, n = 15, 30
        A_eq, b_eq, c_eq, basis, nonbasic = _make_primal_lp(m, n, seed=404)

        tol_py = SimplexTolerances(
            triangular_solve_provider="python",
            use_pivot_factorization=True,
        )
        solver_py = RevisedSimplexSolver(tolerances=tol_py)
        sol_py = solver_py._simplex_loop(
            A_eq, b_eq, c_eq, list(basis), list(nonbasic), ObjectiveSense.MINIMIZE, tol_py
        )

        tol_sp = SimplexTolerances(
            triangular_solve_provider="scipy",
            use_pivot_factorization=True,
            scipy_triangular_min_dimension=5,
        )
        solver_sp = RevisedSimplexSolver(tolerances=tol_sp)
        sol_sp = solver_sp._simplex_loop(
            A_eq, b_eq, c_eq, list(basis), list(nonbasic), ObjectiveSense.MINIMIZE, tol_sp
        )

        assert sol_py.status == SolutionStatus.OPTIMAL
        assert sol_sp.status == SolutionStatus.OPTIMAL
        assert abs(sol_py.objective_value - sol_sp.objective_value) < 1e-9
        assert np.allclose(sol_py.x, sol_sp.x, atol=1e-9)
        assert sol_py.iterations == sol_sp.iterations

    def test_milp_branch_and_bound_equivalence(self):
        """MILP Branch-and-Bound integer solution equivalence."""
        model = Model(name="stage_8_37_milp_test")
        x = model.add_variable(Variable("x", variable_type=VariableType.INTEGER, lower_bound=0.0, upper_bound=10.0))
        y = model.add_variable(Variable("y", variable_type=VariableType.INTEGER, lower_bound=0.0, upper_bound=10.0))
        z = model.add_variable(Variable("z", variable_type=VariableType.CONTINUOUS, lower_bound=0.0))

        # 2x + 3y + z <= 14.5
        model.add_constraint(
            Constraint(
                name="c1",
                coefficients={"x": 2.0, "y": 3.0, "z": 1.0},
                sense=ConstraintSense.LESS_EQUAL,
                rhs=14.5,
            )
        )
        # x + 2y <= 9.2
        model.add_constraint(
            Constraint(
                name="c2",
                coefficients={"x": 1.0, "y": 2.0},
                sense=ConstraintSense.LESS_EQUAL,
                rhs=9.2,
            )
        )
        # Maximize 3x + 4y + 0.5z
        model.set_objective(
            Objective(
                coefficients={"x": 3.0, "y": 4.0, "z": 0.5},
                sense=ObjectiveSense.MAXIMIZE,
            )
        )

        tol_py = MILPTolerances(triangular_solve_provider="python")
        solver_py = MILPSolver(tolerances=tol_py)
        res_py = solver_py.solve(model)

        tol_sp = MILPTolerances(triangular_solve_provider="scipy", scipy_triangular_min_dimension=2)
        solver_sp = MILPSolver(tolerances=tol_sp)
        res_sp = solver_sp.solve(model)

        assert res_py.status == SolutionStatus.OPTIMAL
        assert res_sp.status == SolutionStatus.OPTIMAL
        assert abs(res_py.objective_value - res_sp.objective_value) < 1e-10
        assert res_py.node_count == res_sp.node_count
        assert np.isclose(res_py.variable_values["x"], round(res_py.variable_values["x"]))
        assert np.isclose(res_sp.variable_values["x"], round(res_sp.variable_values["x"]))
        assert np.isclose(res_py.variable_values["y"], round(res_py.variable_values["y"]))
        assert np.isclose(res_sp.variable_values["y"], round(res_sp.variable_values["y"]))
