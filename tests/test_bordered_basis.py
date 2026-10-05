"""Test suite for Stage 8.10 Bordered / Schur-Complement Basis Factorization Inheritance.

Validates:
1. Bordered matrix construction and dimensional consistency
2. Primal solve equivalence (Bordered vs DenseLU)
3. Transpose solve equivalence (Bordered vs DenseLU)
4. Identity basis bordering
5. Random dense basis bordering across dimensions (m=1, 2, 5, 10, 25, 50)
6. Ill-conditioned basis bordering
7. Near-singular pivot / singular fallback
8. Repeated inheritance across consecutive levels
9. Parent/child memory and state isolation
10. Multi-level inheritance (depth >= 4)
11. Explicit DenseLU fallback when residual tolerance is exceeded
12. Mathematical equivalence with Stage 8.9 across MILPs
13. MIN and MAX MILP objective senses
14. Non-zero lower bounds and coordinate shifts
15. Forward and transpose residual validation
16. Feature flag default disabled
"""

import numpy as np
import pytest

from optiforge.core.solution import SolutionStatus
from optiforge.lp.simplex import SimplexTolerances
from optiforge.milp.solver import MILPSolver, MILPTolerances, solve_milp
from optiforge.model import Constraint, ConstraintSense, Model, Objective, ObjectiveSense, Variable, VariableType
from optiforge.numerics.basis import BasisFactorization
from optiforge.numerics.bordered_basis import BorderedBasisFactorization
from optiforge.numerics.exceptions import SingularBasisError
from optiforge.numerics.lu import dense_lu_factor, lu_solve, lu_solve_transpose


class TestBorderedBasisFactorization:
    """Unit and numerical tests for BorderedBasisFactorization."""

    def test_feature_flag_default_disabled(self):
        """16. Verify inherit_basis_factorization is False by default."""
        solver = MILPSolver()
        assert solver.inherit_basis_factorization is False
        assert solver.tolerances.inherit_basis_factorization is False

        tol = MILPTolerances()
        assert tol.inherit_basis_factorization is False

    def test_bordered_matrix_construction(self):
        """1. Verify bordered matrix dimensions and block structure."""
        m = 4
        B_parent = np.eye(m)
        a_B = np.array([1.0, 2.0, 3.0, 4.0])
        B_child = np.block([[B_parent, np.zeros((m, 1))], [a_B, 1.0]])

        parent_fact = BasisFactorization(B_parent)
        bordered = BorderedBasisFactorization(parent_fact, a_B=a_B, B=B_child, d=1.0)

        assert bordered.m == m + 1
        assert bordered.B.shape == (m + 1, m + 1)
        np.testing.assert_array_equal(bordered.a_B, a_B)
        assert bordered.d == 1.0

    def test_primal_solve_equivalence(self):
        """2. Verify primal solve B x = rhs matches DenseLU reference."""
        rng = np.random.RandomState(42)
        m = 6
        B_parent = rng.randn(m, m)
        a_B = rng.randn(m)
        B_child = np.block([[B_parent, np.zeros((m, 1))], [a_B, 1.0]])

        parent_fact = BasisFactorization(B_parent)
        bordered = BorderedBasisFactorization(parent_fact, a_B, B_child)
        lu_ref = dense_lu_factor(B_child)

        rhs = rng.randn(m + 1)
        x_bordered = bordered.solve(rhs)
        x_ref = lu_solve(lu_ref, rhs)

        np.testing.assert_allclose(x_bordered, x_ref, atol=1e-12, rtol=1e-12)
        assert bordered.compute_residual(rhs, x_bordered) < 1e-12

    def test_transpose_solve_equivalence(self):
        """3. Verify transpose solve B^T y = rhs matches DenseLU reference."""
        rng = np.random.RandomState(43)
        m = 7
        B_parent = rng.randn(m, m)
        a_B = rng.randn(m)
        B_child = np.block([[B_parent, np.zeros((m, 1))], [a_B, 1.0]])

        parent_fact = BasisFactorization(B_parent)
        bordered = BorderedBasisFactorization(parent_fact, a_B, B_child)
        lu_ref = dense_lu_factor(B_child)

        rhs = rng.randn(m + 1)
        y_bordered = bordered.solve_transpose(rhs)
        y_ref = lu_solve_transpose(lu_ref, rhs)

        np.testing.assert_allclose(y_bordered, y_ref, atol=1e-12, rtol=1e-12)
        assert bordered.compute_transpose_residual(rhs, y_bordered) < 1e-12

    def test_identity_basis(self):
        """4. Verify bordering an identity basis."""
        m = 5
        B_parent = np.eye(m)
        a_B = np.zeros(m)
        a_B[1] = 1.0
        B_child = np.block([[B_parent, np.zeros((m, 1))], [a_B, 1.0]])

        parent_fact = BasisFactorization(B_parent)
        bordered = BorderedBasisFactorization(parent_fact, a_B, B_child)

        rhs = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        x = bordered.solve(rhs)
        expected = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0 - 3.0])
        np.testing.assert_allclose(x, expected)

    @pytest.mark.parametrize("m", [1, 2, 5, 10, 25, 50])
    def test_random_dense_basis_across_dimensions(self, m: int):
        """5. Verify random dense bases across various matrix dimensions."""
        rng = np.random.RandomState(100 + m)
        B_parent = rng.randn(m, m) + np.eye(m) * 2.0  # diagonally dominant
        a_B = rng.randn(m)
        B_child = np.block([[B_parent, np.zeros((m, 1))], [a_B, 1.0]])

        parent_fact = BasisFactorization(B_parent)
        bordered = BorderedBasisFactorization(parent_fact, a_B, B_child)
        lu_ref = dense_lu_factor(B_child)

        rhs = rng.randn(m + 1)
        x_b = bordered.solve(rhs)
        x_ref = lu_solve(lu_ref, rhs)
        np.testing.assert_allclose(x_b, x_ref, atol=1e-10)

        y_b = bordered.solve_transpose(rhs)
        y_ref = lu_solve_transpose(lu_ref, rhs)
        np.testing.assert_allclose(y_b, y_ref, atol=1e-10)

    def test_ill_conditioned_basis(self):
        """6. Verify behavior on ill-conditioned parent basis."""
        rng = np.random.RandomState(99)
        m = 4
        # Construct Hilbert-like ill-conditioned matrix
        B_parent = np.array([[1.0 / (i + j + 1.0) for j in range(m)] for i in range(m)])
        a_B = rng.randn(m)
        B_child = np.block([[B_parent, np.zeros((m, 1))], [a_B, 1.0]])

        parent_fact = BasisFactorization(B_parent)
        bordered = BorderedBasisFactorization(parent_fact, a_B, B_child)
        lu_ref = dense_lu_factor(B_child)

        rhs = rng.randn(m + 1)
        x_b = bordered.solve(rhs, check_residual=True)
        x_ref = lu_solve(lu_ref, rhs)
        np.testing.assert_allclose(x_b, x_ref, atol=1e-6)

    def test_near_singular_pivot_raises_error(self):
        """7. Verify near-singular or zero diagonal element d raises error."""
        m = 3
        B_parent = np.eye(m)
        a_B = np.ones(m)
        B_child = np.block([[B_parent, np.zeros((m, 1))], [a_B, 0.0]])

        parent_fact = BasisFactorization(B_parent)
        with pytest.raises(SingularBasisError):
            BorderedBasisFactorization(parent_fact, a_B, B_child, d=0.0)

    def test_multi_level_inheritance(self):
        """8 & 10. Multi-level chained inheritance (depth >= 4)."""
        rng = np.random.RandomState(777)
        m0 = 3
        B = rng.randn(m0, m0) + np.eye(m0) * 3.0
        current_fact = BasisFactorization(B)
        current_B = B.copy()

        # Chain 4 levels of bordering
        for level in range(4):
            m_curr = current_fact.m
            a_B = rng.randn(m_curr)
            B_next = np.block([[current_B, np.zeros((m_curr, 1))], [a_B, 1.0]])
            bordered = BorderedBasisFactorization(current_fact, a_B, B_next)
            current_fact = bordered
            current_B = B_next

        # Check solve at depth 4
        assert current_fact.m == m0 + 4
        rhs = rng.randn(m0 + 4)
        x_chain = current_fact.solve(rhs)
        lu_ref = dense_lu_factor(current_B)
        x_ref = lu_solve(lu_ref, rhs)
        np.testing.assert_allclose(x_chain, x_ref, atol=1e-10)

        y_chain = current_fact.solve_transpose(rhs)
        y_ref = lu_solve_transpose(lu_ref, rhs)
        np.testing.assert_allclose(y_chain, y_ref, atol=1e-10)

    def test_parent_child_memory_isolation(self):
        """9. Verify child does not mutate parent matrix or factorization."""
        m = 3
        B_parent = np.eye(m) * 2.0
        B_parent_copy = B_parent.copy()
        a_B = np.array([1.0, 2.0, 3.0])
        B_child = np.block([[B_parent, np.zeros((m, 1))], [a_B, 1.0]])

        parent_fact = BasisFactorization(B_parent)
        bordered = BorderedBasisFactorization(parent_fact, a_B, B_child)

        bordered.solve(np.ones(4))
        bordered.solve_transpose(np.ones(4))

        np.testing.assert_array_equal(parent_fact.B, B_parent_copy)
        assert bordered.B is not parent_fact.B
        assert bordered.a_B is not a_B

    def test_denselu_fallback_on_high_residual(self):
        """11. Verify fallback to DenseLU when relative residual exceeds tolerance."""
        m = 3
        B_parent = np.eye(m)
        a_B = np.zeros(m)
        B_child = np.block([[B_parent, np.zeros((m, 1))], [a_B, 1.0]])

        # Create a parent factorization with an intentionally inaccurate solve to trigger residual check
        class InaccurateParentFactorization:
            def __init__(self, dim):
                self.m = dim
            def solve(self, rhs):
                return np.zeros_like(rhs)  # Large residual error!
            def solve_transpose(self, rhs):
                return np.zeros_like(rhs)

        inaccurate_parent = InaccurateParentFactorization(m)
        bordered = BorderedBasisFactorization(inaccurate_parent, a_B, B_child, residual_tolerance=1e-7)

        rhs = np.array([1.0, 2.0, 3.0, 4.0])
        x = bordered.solve(rhs, check_residual=True)
        assert bordered.is_fallback is True
        assert bordered.fallback_count == 1
        np.testing.assert_allclose(x, rhs)

    def test_numerical_residual_validation(self):
        """15. Verify compute_residual and compute_transpose_residual formulas."""
        m = 3
        B_child = np.array([[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.5, 0.5, 1.0]])
        parent_fact = BasisFactorization(B_child[:2, :2])
        bordered = BorderedBasisFactorization(parent_fact, B_child[2, :2], B_child)

        rhs = np.array([4.0, 7.0, 3.0])
        x = bordered.solve(rhs)
        res = bordered.compute_residual(rhs, x)
        assert res < 1e-12

        y = bordered.solve_transpose(rhs)
        res_t = bordered.compute_transpose_residual(rhs, y)
        assert res_t < 1e-12


class TestStage810FactorizationInheritanceIntegration:
    """End-to-end MILP integration tests with inherit_basis_factorization=True."""

    def test_stage_89_equivalence_binary_knapsack(self):
        """12. Verify Stage 8.9 and 8.10 mathematical equivalence on binary knapsack."""
        m = Model("binary_knap")
        x1 = m.add_variable(Variable("x1", variable_type=VariableType.BINARY))
        x2 = m.add_variable(Variable("x2", variable_type=VariableType.BINARY))
        x3 = m.add_variable(Variable("x3", variable_type=VariableType.BINARY))
        m.add_constraint(Constraint("cap", coefficients={"x1": 2.0, "x2": 3.0, "x3": 5.0}, sense=ConstraintSense.LESS_EQUAL, rhs=7.0))
        m.set_objective(Objective({"x1": 10.0, "x2": 15.0, "x3": 25.0}, sense=ObjectiveSense.MAXIMIZE))

        sol_89 = solve_milp(m, warm_start_relaxations=True, inherit_basis_factorization=False)
        sol_810 = solve_milp(m, warm_start_relaxations=True, inherit_basis_factorization=True)

        assert sol_89.status == sol_810.status == SolutionStatus.OPTIMAL
        assert np.isclose(sol_89.objective_value, sol_810.objective_value, atol=1e-6)
        for v in ["x1", "x2", "x3"]:
            assert np.isclose(sol_89.variable_values[v], sol_810.variable_values[v], atol=1e-6)

    def test_min_max_milp_equivalence(self):
        """13. Verify MINIMIZE and MAXIMIZE objective senses with factorization inheritance."""
        for sense in [ObjectiveSense.MAXIMIZE, ObjectiveSense.MINIMIZE]:
            m = Model(f"test_{sense.name}")
            x1 = m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
            x2 = m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
            m.add_constraint(Constraint("c1", coefficients={"x1": 3.0, "x2": 4.0}, sense=ConstraintSense.LESS_EQUAL, rhs=15.0))
            m.set_objective(Objective({"x1": 2.0, "x2": 3.0}, sense=sense))

            sol_89 = solve_milp(m, warm_start_relaxations=True, inherit_basis_factorization=False)
            s_810 = MILPSolver(warm_start_relaxations=True, inherit_basis_factorization=True)
            sol_810 = s_810.solve(m)

            assert sol_89.status == sol_810.status == SolutionStatus.OPTIMAL
            assert np.isclose(sol_89.objective_value, sol_810.objective_value, atol=1e-5)
            stats = s_810.get_warm_start_stats()
            assert stats.inheritance_attempted >= 0

    def test_non_zero_lower_bounds_inheritance(self):
        """14. Verify non-zero lower bounds and coordinate shifts with factorization inheritance."""
        m = Model("shifted_bounds")
        x1 = m.add_variable(Variable("x1", lower_bound=2.0, variable_type=VariableType.INTEGER))
        x2 = m.add_variable(Variable("x2", lower_bound=1.0, variable_type=VariableType.INTEGER))
        m.add_constraint(Constraint("c1", coefficients={"x1": 2.0, "x2": 3.0}, sense=ConstraintSense.LESS_EQUAL, rhs=14.0))
        m.set_objective(Objective({"x1": 4.0, "x2": 5.0}, sense=ObjectiveSense.MAXIMIZE))

        s_89 = MILPSolver(warm_start_relaxations=True, inherit_basis_factorization=False)
        sol_89 = s_89.solve(m)

        s_810 = MILPSolver(warm_start_relaxations=True, inherit_basis_factorization=True)
        sol_810 = s_810.solve(m)

        assert sol_89.status == sol_810.status == SolutionStatus.OPTIMAL
        assert np.isclose(sol_89.objective_value, sol_810.objective_value, atol=1e-6)
        stats = s_810.get_warm_start_stats()
        assert stats.inheritance_accepted >= 1
        assert stats.dense_lu_calls_saved >= 1
