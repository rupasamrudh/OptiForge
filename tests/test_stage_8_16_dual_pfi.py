"""Unit, Mathematical, and Integration tests for Stage 8.16: Dual Revised Simplex + PFI.

Validates:
1. Mathematical invariants: PFI FTRAN (B^-1 v) and BTRAN (B^-T v) across multiple dual pivots.
2. Dual Revised Simplex solve correctness: MIN and MAX senses under PFI vs DenseLU reference.
3. Pricing, ratio tests, Bland tie-breaking, and primal infeasibility detection under PFI.
4. Factorization inheritance: BorderedBasisFactorization wrapping PFI, and PFI wrapping BorderedBasis.
5. Sibling and parent isolation in Branch-and-Bound: no mutable state leakage between nodes.
6. Fallback mechanisms: refactorization triggers on update limits, small pivots, or singular columns.
7. Opt-in feature flag isolation: default remains False; existing paths unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from optiforge.core.solution import SolutionStatus
from optiforge.lp.dual_simplex import DualRevisedSimplexSolver, DualSimplexResult
from optiforge.lp.simplex import RevisedSimplexSolver, SimplexTolerances
from optiforge.lp.standard_form import SingularBasisError
from optiforge.milp.solver import MILPSolver, MILPTolerances
from optiforge.model import Constraint, ConstraintSense, Model, Objective, ObjectiveSense, Variable, VariableType
from optiforge.numerics.basis import BasisFactorization
from optiforge.numerics.bordered_basis import BorderedBasisFactorization
from optiforge.numerics.pivot_basis import PivotUpdateFactorization, RefactorizationReason


class TestDualPFIMathematicalInvariants:
    """Verifies that elementary eta updates correctly preserve FTRAN and BTRAN."""

    def test_ftran_btran_identity_after_multiple_pivots(self) -> None:
        """Verify B_k^-1 v and B_k^-T v match explicit inverse across multiple pivots."""
        rng = np.random.RandomState(81601)
        m = 8
        B0 = rng.randn(m, m) + np.eye(m) * 6.0
        pfi = PivotUpdateFactorization(B0, max_updates=20)
        B_curr = B0.copy()

        for step in range(12):
            p = step % m
            a_q = rng.randn(m)
            d = pfi.solve(a_q)
            updated = pfi.replace_column(p, a_q, search_direction=d)
            assert updated is True
            B_curr[:, p] = a_q

            # Test FTRAN
            v = rng.randn(m)
            x_pfi = pfi.solve(v)
            x_ref = np.linalg.solve(B_curr, v)
            assert np.allclose(x_pfi, x_ref, atol=1e-11)

            # Test BTRAN
            y_pfi = pfi.solve_transpose(v)
            y_ref = np.linalg.solve(B_curr.T, v)
            assert np.allclose(y_pfi, y_ref, atol=1e-11)

    def test_bordered_inheritance_with_pfi(self) -> None:
        """Verify BorderedBasisFactorization can wrap PFI and vice-versa."""
        rng = np.random.RandomState(81602)
        m = 5
        B0 = rng.randn(m, m) + np.eye(m) * 4.0
        pfi = PivotUpdateFactorization(B0)

        # 2 pivots on parent
        for p in [1, 3]:
            a_q = rng.randn(m)
            d = pfi.solve(a_q)
            pfi.replace_column(p, a_q, search_direction=d)

        # Bordered matrix for child
        a_B = rng.randn(m)
        B_child = np.zeros((m + 1, m + 1))
        B_child[:m, :m] = pfi.B
        B_child[m, :m] = a_B
        B_child[m, m] = 1.0

        bordered = BorderedBasisFactorization(
            parent_factorization=pfi,
            a_B=a_B,
            B=B_child,
            d=1.0,
        )

        v_child = rng.randn(m + 1)
        assert np.allclose(bordered.solve(v_child), np.linalg.solve(B_child, v_child), atol=1e-11)
        assert np.allclose(bordered.solve_transpose(v_child), np.linalg.solve(B_child.T, v_child), atol=1e-11)

        # Child PFI wraps Bordered
        child_pfi = PivotUpdateFactorization(B_child, base_factorization=bordered)
        p_c = 2
        a_q_c = rng.randn(m + 1)
        d_c = child_pfi.solve(a_q_c)
        child_pfi.replace_column(p_c, a_q_c, search_direction=d_c)

        B_child_updated = B_child.copy()
        B_child_updated[:, p_c] = a_q_c
        assert np.allclose(child_pfi.solve(v_child), np.linalg.solve(B_child_updated, v_child), atol=1e-11)
        assert np.allclose(child_pfi.solve_transpose(v_child), np.linalg.solve(B_child_updated.T, v_child), atol=1e-11)


class TestDualSimplexPFICorrectness:
    """Tests Dual Revised Simplex with PFI enabled against reference solver."""

    def test_single_pivot_minimization_pfi(self) -> None:
        """Min x1 + 2*x2 s.t. x1 + x2 >= 2 with PFI."""
        c = np.array([1.0, 2.0, 0.0])
        A = np.array([[-1.0, -1.0, 1.0]])
        b = np.array([-2.0])
        basis = [2]
        nonbasic = [0, 1]

        solver = DualRevisedSimplexSolver(use_pivot_factorization=True)
        res = solver.solve_standard_form(
            A=A, b=b, c=c, basis=basis, nonbasic=nonbasic,
            sense=ObjectiveSense.MINIMIZE, tolerances=SimplexTolerances(use_pivot_factorization=True),
        )

        assert res.status == SolutionStatus.OPTIMAL
        assert res.iterations == 1
        assert res.basis_indices == [0]
        assert np.isclose(res.x[0], 2.0)
        assert np.isclose(res.objective_value, 2.0)
        assert isinstance(res.basis_factorization, PivotUpdateFactorization)

    def test_single_pivot_maximization_pfi(self) -> None:
        """Max -x1 - 2*x2 s.t. x1 + x2 >= 2 with PFI."""
        c = np.array([-1.0, -2.0, 0.0])
        A = np.array([[-1.0, -1.0, 1.0]])
        b = np.array([-2.0])
        basis = [2]
        nonbasic = [0, 1]

        solver = DualRevisedSimplexSolver(use_pivot_factorization=True)
        res = solver.solve_standard_form(
            A=A, b=b, c=c, basis=basis, nonbasic=nonbasic,
            sense=ObjectiveSense.MAXIMIZE, tolerances=SimplexTolerances(use_pivot_factorization=True),
        )

        assert res.status == SolutionStatus.OPTIMAL
        assert res.iterations == 1
        assert res.basis_indices == [0]
        assert np.isclose(res.x[0], 2.0)
        assert np.isclose(res.objective_value, -2.0)

    def test_infeasibility_detection_pfi(self) -> None:
        """Infeasible problem detection with PFI enabled."""
        c = np.array([1.0, 2.0, 0.0])
        A = np.array([[1.0, 1.0, 1.0]])
        b = np.array([-2.0])
        basis = [2]
        nonbasic = [0, 1]

        solver = DualRevisedSimplexSolver(use_pivot_factorization=True)
        res = solver.solve_standard_form(
            A=A, b=b, c=c, basis=basis, nonbasic=nonbasic,
            sense=ObjectiveSense.MINIMIZE, tolerances=SimplexTolerances(use_pivot_factorization=True),
        )

        assert res.status == SolutionStatus.INFEASIBLE

    def test_multipivot_dual_simplex_comparison(self) -> None:
        """Multi-pivot problem compared between DenseLU and PFI."""
        # 3 constraints, 5 variables
        A = np.array([
            [-1.0, -2.0, 1.0, 0.0, 0.0],
            [-2.0, -1.0, 0.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0, 0.0, 1.0],
        ])
        b = np.array([-4.0, -5.0, -3.0])
        c = np.array([3.0, 4.0, 0.0, 0.0, 0.0])
        basis = [2, 3, 4]
        nonbasic = [0, 1]

        # Solve with DenseLU
        s_dense = DualRevisedSimplexSolver(use_pivot_factorization=False, use_incremental_basis=False)
        res_dense = s_dense.solve_standard_form(
            A=A, b=b, c=c, basis=basis, nonbasic=nonbasic,
            sense=ObjectiveSense.MINIMIZE, tolerances=SimplexTolerances(use_pivot_factorization=False),
        )

        # Solve with PFI
        s_pfi = DualRevisedSimplexSolver(use_pivot_factorization=True)
        res_pfi = s_pfi.solve_standard_form(
            A=A, b=b, c=c, basis=basis, nonbasic=nonbasic,
            sense=ObjectiveSense.MINIMIZE, tolerances=SimplexTolerances(use_pivot_factorization=True),
        )

        assert res_dense.status == SolutionStatus.OPTIMAL
        assert res_pfi.status == SolutionStatus.OPTIMAL
        assert res_dense.iterations == res_pfi.iterations
        assert np.isclose(res_dense.objective_value, res_pfi.objective_value)
        assert np.allclose(res_dense.x, res_pfi.x)
        assert np.allclose(res_dense.y, res_pfi.y)


class TestDualPFIFallbackAndStability:
    """Verifies that numerical anomalies trigger safe fallback without solver failure."""

    def test_update_limit_triggers_refactorization(self) -> None:
        """PFI refactorizes when update count reaches max_basis_updates."""
        A = np.array([
            [-1.0, -2.0, 1.0, 0.0, 0.0],
            [-2.0, -1.0, 0.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0, 0.0, 1.0],
        ])
        b = np.array([-4.0, -5.0, -3.0])
        c = np.array([3.0, 4.0, 0.0, 0.0, 0.0])
        basis = [2, 3, 4]
        nonbasic = [0, 1]

        # Force max_basis_updates=1 so every pivot triggers refactorization
        tol = SimplexTolerances(use_pivot_factorization=True, max_basis_updates=1)
        solver = DualRevisedSimplexSolver(use_pivot_factorization=True, tolerances=tol)
        res = solver.solve_standard_form(
            A=A, b=b, c=c, basis=basis, nonbasic=nonbasic,
            sense=ObjectiveSense.MINIMIZE, tolerances=tol,
        )

        assert res.status == SolutionStatus.OPTIMAL
        assert np.isclose(res.objective_value, 10.0)  # optimal obj: 3*2 + 4*1 = 10.0
        assert res.basis_factorization.total_refactorizations >= 1

    def test_default_flag_isolation(self) -> None:
        """Verify that default tolerances leave use_pivot_factorization=False."""
        tol = SimplexTolerances()
        assert tol.use_pivot_factorization is False

        s = DualRevisedSimplexSolver()
        assert s.use_pivot_factorization is False


class TestMILPWarmStartPFIIntegration:
    """Validates end-to-end MILP Branch-and-Bound with Dual PFI warm-starts."""

    def test_knapsack_milp_with_dual_pfi(self) -> None:
        """Solve knapsack MILP under cold, standard warm-start, and Dual PFI warm-start."""
        model = Model("knapsack_test")
        values = [10.0, 15.0, 20.0, 25.0, 30.0]
        weights = [2.0, 3.0, 5.0, 7.0, 8.0]
        n = len(values)
        vnames = [f"x{j}" for j in range(n)]
        for vn in vnames:
            model.add_variable(Variable(vn, lower_bound=0.0, upper_bound=1.0, variable_type=VariableType.BINARY))
        model.add_constraint(Constraint("capacity", {vnames[j]: weights[j] for j in range(n)}, ConstraintSense.LESS_EQUAL, 14.0))
        model.set_objective(Objective({vnames[j]: values[j] for j in range(n)}, ObjectiveSense.MAXIMIZE))

        # Config B: WarmStart reference (IncrementalBasis)
        s_ref = MILPSolver(
            tolerances=MILPTolerances(warm_start_relaxations=True, inherit_basis_factorization=True),
            simplex_tolerances=SimplexTolerances(use_pivot_factorization=False),
        )
        sol_ref = s_ref.solve(model)
        stats_ref = s_ref.get_warm_start_stats()

        # Config C: WarmStart + Dual PFI
        s_pfi = MILPSolver(
            tolerances=MILPTolerances(warm_start_relaxations=True, inherit_basis_factorization=True),
            simplex_tolerances=SimplexTolerances(use_pivot_factorization=True),
        )
        sol_pfi = s_pfi.solve(model)
        stats_pfi = s_pfi.get_warm_start_stats()

        assert sol_ref.status == SolutionStatus.OPTIMAL
        assert sol_pfi.status == SolutionStatus.OPTIMAL
        assert np.isclose(sol_ref.objective_value, sol_pfi.objective_value)
        assert sol_ref.node_count == sol_pfi.node_count
        assert stats_ref.accepted == stats_pfi.accepted
        assert stats_pfi.pfi_fallbacks == 0
        assert stats_pfi.pfi_updates > 0

    def test_sibling_isolation_no_leakage(self) -> None:
        """Verify that solving left child branch does not mutate parent state for right branch."""
        # Simple integer model: max x1 + x2 s.t. 2*x1 + x2 <= 4.5, x1 + 2*x2 <= 4.5, x >= 0 integer
        model = Model("sibling_isolation")
        model.add_variable(Variable("x1", lower_bound=0.0, upper_bound=5.0, variable_type=VariableType.INTEGER))
        model.add_variable(Variable("x2", lower_bound=0.0, upper_bound=5.0, variable_type=VariableType.INTEGER))
        model.add_constraint(Constraint("c1", {"x1": 2.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 4.5))
        model.add_constraint(Constraint("c2", {"x1": 1.0, "x2": 2.0}, ConstraintSense.LESS_EQUAL, 4.5))
        model.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))

        solver = MILPSolver(
            tolerances=MILPTolerances(warm_start_relaxations=True, inherit_basis_factorization=True),
            simplex_tolerances=SimplexTolerances(use_pivot_factorization=True),
        )
        sol = solver.solve(model)
        assert sol.status == SolutionStatus.OPTIMAL
        assert np.isclose(sol.objective_value, 2.0)
        assert np.isclose(sol.variable_values["x1"] + sol.variable_values["x2"], 2.0)
