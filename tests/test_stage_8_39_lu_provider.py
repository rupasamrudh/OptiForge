"""Tests for Stage 8.39: Controlled LU Factorization Provider.

Verifies:
1. Python reference LU provider preserves exact historical behavior and mathematical contract.
2. SciPy LAPACK dgetrf LU provider satisfies exact PB = LU contract and matches reference.
3. Deterministic matrices:
   - Well-conditioned dense matrices
   - Random matrices
   - Small matrices (< crossover threshold)
   - Medium matrices (>= crossover threshold)
   - Matrices requiring row pivoting
   - Ill-conditioned / nearly singular matrices
   - Singular matrices (raises NumericalSingularityError)
   - Identity matrices
   - Upper and lower triangular matrices
4. Downstream solves:
   - solve() and solve_transpose() across all 4 decoupled provider combinations:
     (Python LU, Python Tri), (Python LU, SciPy Tri), (SciPy LU, Python Tri), (SciPy LU, SciPy Tri)
5. Simplex & PFI integration with both providers.
6. Telemetry and fallback mechanisms.
"""

from __future__ import annotations

import numpy as np
import pytest

from optiforge.numerics.exceptions import NumericalSingularityError
from optiforge.numerics.lu import (
    DenseLU,
    dense_lu_factor,
    dense_lu_factor_reference,
    lu_solve,
    lu_solve_transpose,
)
from optiforge.numerics.lu_provider import (
    LUFactorizationProvider,
    LUProviderConfig,
    LUProviderType,
    PythonLUFactorizationProvider,
    SciPyLUFactorizationProvider,
    get_global_lu_provider_config,
    get_lu_provider,
    get_lu_provider_stats,
    reset_lu_provider_stats,
    set_global_lu_provider_config,
)
from optiforge.numerics.triangular_provider import (
    get_provider_stats,
    is_scipy_available,
    reset_provider_stats,
)
from optiforge.numerics.pivot_basis import PivotUpdateFactorization
from optiforge.lp.dual_simplex import DualRevisedSimplexSolver
from optiforge.lp.simplex import RevisedSimplexSolver, SimplexTolerances
from optiforge.model import Constraint, ConstraintSense, Model, Objective, ObjectiveSense, Variable
from optiforge.core.solution import SolutionStatus


def _verify_lu_contract(lu: DenseLU, B: np.ndarray, tol: float = 1e-11) -> None:
    """Validate all structural and mathematical invariants of DenseLU."""
    m = B.shape[0]
    assert lu.L.shape == (m, m)
    assert lu.U.shape == (m, m)
    assert lu.perm.shape == (m,)
    assert set(lu.perm) == set(range(m))
    assert lu.det_sign in (1, -1)

    # Unit lower triangular
    assert np.allclose(np.diag(lu.L), 1.0)
    assert np.allclose(np.triu(lu.L, 1), 0.0)

    # Upper triangular
    assert np.allclose(np.tril(lu.U, -1), 0.0)

    # Reconstruction: P B = L U
    P = lu.P
    PB = P @ B
    LU = lu.L @ lu.U
    res = np.linalg.norm(PB - LU, ord=np.inf)
    assert res < tol, f"Reconstruction error {res:.3e} exceeds tolerance {tol:.3e}"


# ===========================================================================
# 1. Deterministic Matrix Verification
# ===========================================================================

class TestLUProviderDeterministicMatrices:
    """Verify LU factorization across varied deterministic matrix topologies."""

    def test_identity_matrix(self):
        """Identity matrix has L=I, U=I, perm=0..m-1, det_sign=1."""
        for m in (1, 4, 12):
            I = np.eye(m)
            # Python provider
            lu_py = dense_lu_factor(I, lu_factor_provider="python")
            _verify_lu_contract(lu_py, I)
            assert np.allclose(lu_py.L, I)
            assert np.allclose(lu_py.U, I)
            assert np.array_equal(lu_py.perm, np.arange(m))
            assert lu_py.det_sign == 1

            if is_scipy_available():
                lu_sp = dense_lu_factor(I, lu_factor_provider="scipy", scipy_lu_min_dimension=m)
                _verify_lu_contract(lu_sp, I)
                assert np.allclose(lu_sp.L, I)
                assert np.allclose(lu_sp.U, I)
                assert np.array_equal(lu_sp.perm, np.arange(m))
                assert lu_sp.det_sign == 1

    def test_upper_triangular_matrix(self):
        """Upper triangular matrix should produce L=I, U=B if no pivoting occurs."""
        rng = np.random.RandomState(101)
        m = 12
        U_mat = np.triu(rng.randn(m, m)) + 2.0 * np.eye(m)

        lu_py = dense_lu_factor(U_mat, lu_factor_provider="python")
        _verify_lu_contract(lu_py, U_mat)

        if is_scipy_available():
            lu_sp = dense_lu_factor(U_mat, lu_factor_provider="scipy", scipy_lu_min_dimension=10)
            _verify_lu_contract(lu_sp, U_mat)

    def test_lower_triangular_matrix(self):
        """Lower triangular matrix should be factorized accurately."""
        rng = np.random.RandomState(102)
        m = 12
        L_mat = np.tril(rng.randn(m, m)) + 3.0 * np.eye(m)

        lu_py = dense_lu_factor(L_mat, lu_factor_provider="python")
        _verify_lu_contract(lu_py, L_mat)

        if is_scipy_available():
            lu_sp = dense_lu_factor(L_mat, lu_factor_provider="scipy", scipy_lu_min_dimension=10)
            _verify_lu_contract(lu_sp, L_mat)

    def test_matrices_requiring_row_pivoting(self):
        """Matrix where pivots must be swapped to avoid zero / small diagonal elements."""
        # Matrix with 0 on the diagonal
        m = 6
        B = np.array([
            [0.0, 1.0, 2.0, 0.0, 1.0, 0.0],
            [2.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            [1.0, 3.0, 0.0, 2.0, 1.0, 0.0],
            [0.0, 2.0, 1.0, 0.0, 3.0, 2.0],
            [3.0, 1.0, 0.0, 1.0, 0.0, 4.0],
            [1.0, 0.0, 2.0, 3.0, 1.0, 0.0],
        ], dtype=np.float64)

        lu_py = dense_lu_factor(B, lu_factor_provider="python")
        _verify_lu_contract(lu_py, B)
        # Verify row 0 was actually swapped (perm[0] != 0)
        assert lu_py.perm[0] != 0

        if is_scipy_available():
            lu_sp = dense_lu_factor(B, lu_factor_provider="scipy", scipy_lu_min_dimension=4)
            _verify_lu_contract(lu_sp, B)
            assert lu_sp.perm[0] != 0

    def test_ill_conditioned_matrix(self):
        """Nearly singular (Hilbert-like) matrix remains solvable and matches contract."""
        m = 8
        H = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            for j in range(m):
                H[i, j] = 1.0 / (i + j + 1.0)

        lu_py = dense_lu_factor(H, lu_factor_provider="python", pivot_tolerance=1e-15)
        _verify_lu_contract(lu_py, H, tol=1e-10)

        if is_scipy_available():
            lu_sp = dense_lu_factor(H, lu_factor_provider="scipy", scipy_lu_min_dimension=5, pivot_tolerance=1e-15)
            _verify_lu_contract(lu_sp, H, tol=1e-10)

    def test_singular_matrix_detection(self):
        """Exact zero row or column triggers NumericalSingularityError in both providers."""
        m = 12
        rng = np.random.RandomState(103)
        B = rng.randn(m, m)
        B[3, :] = 0.0  # Singular row

        with pytest.raises(NumericalSingularityError):
            dense_lu_factor(B, lu_factor_provider="python")

        if is_scipy_available():
            with pytest.raises(NumericalSingularityError):
                dense_lu_factor(B, lu_factor_provider="scipy", scipy_lu_min_dimension=10)


# ===========================================================================
# 2. Crossover Threshold & Fallback Behavior
# ===========================================================================

class TestLUProviderCrossoverAndFallback:
    """Verify crossover dimension threshold and driver fallback logic."""

    @pytest.mark.skipif(not is_scipy_available(), reason="SciPy not installed")
    def test_crossover_threshold(self):
        """Matrices below scipy_lu_min_dimension use Python fallback transparently."""
        reset_lu_provider_stats()
        rng = np.random.RandomState(104)
        m = 5
        B = rng.randn(m, m) + 3.0 * np.eye(m)

        # Min dimension is 10, matrix is 5 -> should fall back to Python
        lu = dense_lu_factor(B, lu_factor_provider="scipy", scipy_lu_min_dimension=10)
        _verify_lu_contract(lu, B)

        stats = get_lu_provider_stats()
        # SciPy factors should not increment because m < min_dimension
        assert stats["scipy_factors"] == 0
        assert stats["total_factors"] >= 1

    @pytest.mark.skipif(not is_scipy_available(), reason="SciPy not installed")
    def test_medium_dimension_activates_scipy(self):
        """Matrices at or above scipy_lu_min_dimension invoke compiled SciPy."""
        reset_lu_provider_stats()
        rng = np.random.RandomState(105)
        m = 15
        B = rng.randn(m, m) + 4.0 * np.eye(m)

        lu = dense_lu_factor(B, lu_factor_provider="scipy", scipy_lu_min_dimension=10)
        _verify_lu_contract(lu, B)

        assert lu.use_scipy_lu_factor is True
        assert lu.lu_factor_provider == "scipy"
        stats = get_lu_provider_stats()
        assert stats["scipy_factors"] >= 1


# ===========================================================================
# 3. Downstream Solves across All 4 Provider Permutations
# ===========================================================================

class TestDownstreamSolveDecoupledPermutations:
    """Verify solve() and solve_transpose() across all 4 decoupled provider combinations:
    1. Python LU + Python Triangular Solve
    2. Python LU + SciPy Triangular Solve
    3. SciPy LU  + Python Triangular Solve
    4. SciPy LU  + SciPy Triangular Solve
    """

    @pytest.mark.parametrize("m", [15, 35])
    def test_all_four_quadrants(self, m: int):
        rng = np.random.RandomState(106)
        B = rng.randn(m, m) + 6.0 * np.eye(m)
        b = rng.randn(m)
        c = rng.randn(m)

        # Reference solve via pure Python reference
        lu_ref = dense_lu_factor_reference(B)
        x_ref = lu_solve(lu_ref, b)
        y_ref = lu_solve_transpose(lu_ref, c)

        quadrants = [
            ("python", "python"),
            ("python", "scipy"),
        ]
        if is_scipy_available():
            quadrants.extend([
                ("scipy", "python"),
                ("scipy", "scipy"),
            ])

        for lu_prov, tri_prov in quadrants:
            lu = dense_lu_factor(
                B,
                lu_factor_provider=lu_prov,
                triangular_solve_provider=tri_prov,
                scipy_lu_min_dimension=10,
                scipy_triangular_min_dimension=10,
            )
            _verify_lu_contract(lu, B)

            x = lu.solve(b, provider=tri_prov)
            y = lu.solve_transpose(c, provider=tri_prov)

            diff_x = np.linalg.norm(x - x_ref, ord=np.inf)
            diff_y = np.linalg.norm(y - y_ref, ord=np.inf)

            assert diff_x < 1e-10, f"Quadrant ({lu_prov}, {tri_prov}) primal solve diff {diff_x:.3e}"
            assert diff_y < 1e-10, f"Quadrant ({lu_prov}, {tri_prov}) dual solve diff {diff_y:.3e}"


# ===========================================================================
# 4. Solvers & Simplex Integration
# ===========================================================================

class TestSimplexIntegrationWithLUProvider:
    """Verify RevisedSimplex and DualRevisedSimplex solutions are numerically identical
    under Python vs SciPy LU factorization providers.
    """

    def test_dual_revised_simplex_equality(self):
        """Dual simplex LP solved with Python LU vs SciPy LU yields identical result."""
        # Dual-feasible, primal-infeasible problem:
        # Min x1 + 2*x2 s.t. -x1 - x2 + s1 = -2, -2*x1 - x2 + s2 = -3
        c = np.array([1.0, 2.0, 0.0, 0.0])
        A = np.array([
            [-1.0, -1.0, 1.0, 0.0],
            [-2.0, -1.0, 0.0, 1.0],
        ])
        b = np.array([-2.0, -3.0])
        basis = [2, 3]
        nonbasic = [0, 1]

        # Solve with Python LU
        tol_py = SimplexTolerances(
            use_pivot_factorization=True,
            lu_factor_provider="python",
        )
        solver_py = DualRevisedSimplexSolver(tolerances=tol_py)
        res_py = solver_py.solve_standard_form(
            A=A, b=b, c=c, basis=basis, nonbasic=nonbasic, sense=ObjectiveSense.MINIMIZE, tolerances=tol_py
        )

        assert res_py.status == SolutionStatus.OPTIMAL
        assert np.isclose(res_py.objective_value, 2.0)

        if is_scipy_available():
            tol_sp = SimplexTolerances(
                use_pivot_factorization=True,
                lu_factor_provider="scipy",
                scipy_lu_min_dimension=2,
            )
            solver_sp = DualRevisedSimplexSolver(tolerances=tol_sp)
            res_sp = solver_sp.solve_standard_form(
                A=A, b=b, c=c, basis=basis, nonbasic=nonbasic, sense=ObjectiveSense.MINIMIZE, tolerances=tol_sp
            )

            assert res_sp.status == SolutionStatus.OPTIMAL
            assert np.isclose(res_sp.objective_value, res_py.objective_value, atol=1e-10)
            assert res_sp.iterations == res_py.iterations

    def test_primal_revised_simplex_equality(self):
        """Primal simplex LP solved with Python LU vs SciPy LU yields identical result."""
        m = Model("primal_test")
        x1 = m.add_variable(Variable("x1", lower_bound=0.0))
        x2 = m.add_variable(Variable("x2", lower_bound=0.0))
        m.set_objective(Objective(coefficients={"x1": 3.0, "x2": 2.0}, sense=ObjectiveSense.MAXIMIZE))
        m.add_constraint(Constraint("c1", {"x1": 2.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 10.0))
        m.add_constraint(Constraint("c2", {"x1": 1.0, "x2": 3.0}, ConstraintSense.LESS_EQUAL, 15.0))

        tol_py = SimplexTolerances(lu_factor_provider="python")
        solver_py = RevisedSimplexSolver(tolerances=tol_py)
        res_py = solver_py.solve(m)

        assert res_py.status == SolutionStatus.OPTIMAL

        if is_scipy_available():
            tol_sp = SimplexTolerances(
                lu_factor_provider="scipy",
                scipy_lu_min_dimension=2,
            )
            solver_sp = RevisedSimplexSolver(tolerances=tol_sp)
            res_sp = solver_sp.solve(m)

            assert res_sp.status == SolutionStatus.OPTIMAL
            assert np.isclose(res_sp.objective_value, res_py.objective_value, atol=1e-10)
            assert res_sp.iterations == res_py.iterations


# ===========================================================================
# 5. Global Provider Config & Factory
# ===========================================================================

class TestGlobalLUProviderConfig:
    """Verify global provider configuration and factory functions."""

    def test_global_config_get_and_set(self):
        initial_cfg = get_global_lu_provider_config()
        assert initial_cfg.provider == "python"

        try:
            set_global_lu_provider_config(use_scipy=True, min_dimension=25)
            cfg = get_global_lu_provider_config()
            assert cfg.use_scipy is True
            assert cfg.provider == "scipy"
            assert cfg.min_dimension == 25

            prov = get_lu_provider()
            assert isinstance(prov, SciPyLUFactorizationProvider)
            assert prov.min_dimension == 25
        finally:
            set_global_lu_provider_config(use_scipy=False, min_dimension=10)
            cfg = get_global_lu_provider_config()
            assert cfg.provider == "python"
            assert isinstance(get_lu_provider(), PythonLUFactorizationProvider)
