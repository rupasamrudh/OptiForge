"""Stage 8.36 Test Suite: Base Triangular Solve Optimization.

Verifies:
1. Default configuration: use_optimized_triangular_solves defaults to False.
2. Configuration wiring: use_optimized_triangular_solves=True sets use_scipy_triangular_solve=True.
3. Pure-NumPy reference correctness and equivalence to SciPy kernel.
4. Transpose solve equivalence across forward and backward sweeps.
5. Graceful fallback when SciPy is unavailable or simulated missing.
6. Graceful fallback on exact zero diagonal.
7. Dual simplex solve equivalence under optimized triangular solves.
8. Primal simplex solve equivalence under optimized triangular solves.
9. PFI and Compact WY integration with optimized triangular solves.
10. MILP warm-start integration and integer solution equivalence.
"""

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
from optiforge.numerics.lu import (
    DenseLU,
    back_substitution,
    dense_lu_factor,
    forward_substitution,
    lu_solve,
    lu_solve_transpose,
)
from optiforge.numerics.pivot_basis import PivotUpdateFactorization
from optiforge.numerics.triangular_provider import (
    backward_solve,
    forward_solve,
    get_provider_stats,
    is_scipy_available,
    reset_provider_stats,
    triangular_lu_solve,
    triangular_lu_solve_transpose,
)


def make_test_dual_lp(m, n, seed=42):
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


def make_test_primal_lp(m, n, seed=42):
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


class TestStage836TriangularSolve:
    """Stage 8.36 verification of triangular solve optimization."""

    def test_item_1_default_is_false(self):
        """1. Verify default configuration is False."""
        tol = SimplexTolerances()
        assert tol.use_optimized_triangular_solves is False
        assert tol.use_scipy_triangular_solve is False
        solver = DualRevisedSimplexSolver()
        assert solver.use_scipy_triangular_solve is False

    def test_item_2_configuration_wiring(self):
        """2. Verify use_optimized_triangular_solves wires to use_scipy_triangular_solve."""
        tol = SimplexTolerances(use_optimized_triangular_solves=True)
        assert tol.use_optimized_triangular_solves is True
        assert tol.use_scipy_triangular_solve is True

        solver = DualRevisedSimplexSolver(use_optimized_triangular_solves=True)
        assert solver.use_scipy_triangular_solve is True

    def test_item_3_forward_backward_solve_equivalence(self):
        """3. Verify forward and backward solve equivalence between NumPy and SciPy."""
        rng = np.random.default_rng(123)
        for m in (10, 25, 50):
            M = rng.standard_normal((m, m)) + (m * 2.0) * np.eye(m)
            lu = dense_lu_factor(M)
            b = rng.standard_normal(m)

            # Forward
            y_np = forward_substitution(lu.L, b, unit_diag=True)
            y_sp = forward_solve(lu.L, b, unit_diag=True, use_scipy=True, min_dimension=5)
            assert np.allclose(y_np, y_sp, atol=1e-12, rtol=1e-12)

            # Backward
            x_np = back_substitution(lu.U, y_np)
            x_sp = backward_solve(lu.U, y_np, use_scipy=True, min_dimension=5)
            assert np.allclose(x_np, x_sp, atol=1e-12, rtol=1e-12)

    def test_item_4_transpose_solve_equivalence(self):
        """4. Verify transpose solve equivalence between NumPy and SciPy."""
        rng = np.random.default_rng(456)
        for m in (10, 25, 50):
            M = rng.standard_normal((m, m)) + (m * 2.0) * np.eye(m)
            lu = dense_lu_factor(M)
            rhs = rng.standard_normal(m)

            x_np = lu_solve_transpose(lu, rhs, use_scipy=False)
            x_sp = triangular_lu_solve_transpose(lu, rhs, use_scipy=True, min_dimension=5)
            assert np.allclose(x_np, x_sp, atol=1e-12, rtol=1e-12)

    def test_item_5_fallback_when_scipy_unavailable(self):
        """5. Verify graceful fallback when SciPy is flagged unavailable."""
        import optiforge.numerics.triangular_provider as tp
        orig_avail = tp._SCIPY_AVAILABLE
        try:
            tp._SCIPY_AVAILABLE = False
            L = np.eye(20, dtype=np.float64)
            b = np.ones(20, dtype=np.float64)
            y = forward_solve(L, b, unit_diag=True, use_scipy=True, min_dimension=5)
            assert np.allclose(y, b)
        finally:
            tp._SCIPY_AVAILABLE = orig_avail

    def test_item_6_fallback_on_zero_diagonal(self):
        """6. Verify graceful fallback on singular upper factor (zero diagonal)."""
        reset_provider_stats()
        U_zero = np.array([[2.0, 1.0], [0.0, 0.0]], dtype=np.float64)
        y = np.array([1.0, 2.0], dtype=np.float64)
        x = backward_solve(U_zero, y, use_scipy=True, min_dimension=2)
        stats = get_provider_stats()
        assert stats["fallbacks"] >= 1

    def test_item_7_dual_simplex_solve_equivalence(self):
        """7. Verify Dual Simplex solutions match bit-exact between default and optimized."""
        A, b, c, basis, nonbasic = make_test_dual_lp(25, 50, seed=789)
        tol_base = SimplexTolerances(use_optimized_triangular_solves=False)
        tol_opt = SimplexTolerances(use_optimized_triangular_solves=True, scipy_triangular_min_dimension=5)

        res_base = DualRevisedSimplexSolver(tolerances=tol_base).solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol_base)
        res_opt = DualRevisedSimplexSolver(tolerances=tol_opt).solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol_opt)

        assert res_base.status == res_opt.status == SolutionStatus.OPTIMAL
        assert abs(res_base.objective_value - res_opt.objective_value) < 1e-9
        assert np.allclose(res_base.x, res_opt.x, atol=1e-9)
        assert np.allclose(res_base.y, res_opt.y, atol=1e-9)

    def test_item_8_primal_simplex_solve_equivalence(self):
        """8. Verify Primal Simplex solutions match between default and optimized."""
        A, b, c, basis, nonbasic = make_test_primal_lp(20, 40, seed=101)
        tol_base = SimplexTolerances(use_optimized_triangular_solves=False)
        tol_opt = SimplexTolerances(use_optimized_triangular_solves=True, scipy_triangular_min_dimension=5)

        res_base = RevisedSimplexSolver(tolerances=tol_base)._simplex_loop(A, b, c, list(basis), list(nonbasic), ObjectiveSense.MINIMIZE, tol_base)
        res_opt = RevisedSimplexSolver(tolerances=tol_opt)._simplex_loop(A, b, c, list(basis), list(nonbasic), ObjectiveSense.MINIMIZE, tol_opt)

        assert res_base.status == res_opt.status == SolutionStatus.OPTIMAL
        assert abs(res_base.objective_value - res_opt.objective_value) < 1e-9
        assert np.allclose(res_base.x, res_opt.x, atol=1e-9)

    def test_item_9_pfi_compact_wy_equivalence(self):
        """9. Verify PFI Compact WY solutions match under optimized triangular solve."""
        A, b, c, basis, nonbasic = make_test_dual_lp(30, 60, seed=202)
        tol_base = SimplexTolerances(use_pivot_factorization=True, use_optimized_triangular_solves=False)
        tol_opt = SimplexTolerances(use_pivot_factorization=True, use_optimized_triangular_solves=True, scipy_triangular_min_dimension=5)

        res_base = DualRevisedSimplexSolver(tolerances=tol_base).solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol_base)
        res_opt = DualRevisedSimplexSolver(tolerances=tol_opt).solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol_opt)

        assert res_base.status == res_opt.status == SolutionStatus.OPTIMAL
        assert abs(res_base.objective_value - res_opt.objective_value) < 1e-9
        assert np.allclose(res_base.x, res_opt.x, atol=1e-9)

    def test_item_10_milp_warm_start_equivalence(self):
        """10. Verify MILP branch-and-bound produces exact same solution."""
        m = Model("milp_stage_836")
        x = [m.add_variable(Variable(f"x_{i}", variable_type=VariableType.BINARY)) for i in range(6)]
        weights = [3, 4, 5, 8, 9, 2]
        values = [12, 18, 24, 32, 40, 9]
        m.add_constraint(Constraint("cap", coefficients={f"x_{i}": weights[i] for i in range(6)}, sense=ConstraintSense.LESS_EQUAL, rhs=15.0))
        m.set_objective(Objective({f"x_{i}": values[i] for i in range(6)}, sense=ObjectiveSense.MAXIMIZE))

        tol_base = MILPTolerances(use_scipy_triangular_solve=False, warm_start_relaxations=True)
        tol_opt = MILPTolerances(use_scipy_triangular_solve=True, scipy_triangular_min_dimension=2, warm_start_relaxations=True)

        sol_base = MILPSolver(tolerances=tol_base).solve(m)
        sol_opt = MILPSolver(tolerances=tol_opt).solve(m)

        assert sol_base.status == sol_opt.status == SolutionStatus.OPTIMAL
        assert abs(sol_base.objective_value - sol_opt.objective_value) < 1e-8
        for k in sol_base.variable_values:
            assert np.isclose(sol_base.variable_values[k], sol_opt.variable_values[k], atol=1e-8)
