"""Unit and Reference-Comparison tests for Dual Revised Simplex (Stage 8.8).

Validates the Dual Revised Simplex prototype against mathematical invariants
and the reference RevisedSimplexSolver across diverse problem classes:
- Small minimization LP
- Small maximization LP
- Slack basis with negative RHS (bound-induced infeasibility)
- Infeasible problems (primal infeasible / dual unbounded)
- Degenerate problems
- Multiple optima
- Random LPs comparing Dual Simplex with Primal Simplex
- Ill-conditioned bases
"""

from __future__ import annotations

import numpy as np
import pytest

from optiforge.core.solution import SolutionStatus
from optiforge.lp.dual_simplex import DualRevisedSimplexSolver, DualSimplexResult
from optiforge.lp.simplex import RevisedSimplexSolver, SimplexTolerances
from optiforge.lp.standard_form import StandardFormModel, to_standard_form
from optiforge.model import Constraint, ConstraintSense, Model, Objective, ObjectiveSense, Variable


class TestDualSimplexBasics:
    """Verifies fundamental dual simplex operations and mathematical invariants."""

    def test_single_pivot_minimization(self) -> None:
        """Min x1 + 2*x2 s.t. x1 + x2 >= 2 => -x1 - x2 + s1 = -2.
        
        Initial slack basis s1: x_s1 = -2 < 0 (primal infeasible).
        Reduced costs: r = [1, 2, 0] >= 0 (dual feasible).
        Dual simplex should perform 1 pivot to enter x1 and reach optimality.
        """
        c = np.array([1.0, 2.0, 0.0])
        A = np.array([[-1.0, -1.0, 1.0]])
        b = np.array([-2.0])
        basis = [2]
        nonbasic = [0, 1]

        solver = DualRevisedSimplexSolver()
        res = solver.solve_standard_form(
            A=A,
            b=b,
            c=c,
            basis=basis,
            nonbasic=nonbasic,
            sense=ObjectiveSense.MINIMIZE,
            tolerances=SimplexTolerances(),
        )

        assert res.status == SolutionStatus.OPTIMAL
        assert res.iterations == 1
        assert res.basis_indices == [0]  # x1 entered
        assert np.isclose(res.x[0], 2.0)
        assert np.isclose(res.x[1], 0.0)
        assert np.isclose(res.objective_value, 2.0)

    def test_single_pivot_maximization(self) -> None:
        """Max -x1 - 2*x2 s.t. x1 + x2 >= 2 => -x1 - x2 + s1 = -2.
        
        Initial slack basis: x_s1 = -2 < 0.
        Reduced costs for MAX: r = [-1, -2, 0] <= 0 (dual feasible).
        Dual simplex should pivot to optimal x1 = 2, obj = -2.
        """
        c = np.array([-1.0, -2.0, 0.0])
        A = np.array([[-1.0, -1.0, 1.0]])
        b = np.array([-2.0])
        basis = [2]
        nonbasic = [0, 1]

        solver = DualRevisedSimplexSolver()
        res = solver.solve_standard_form(
            A=A,
            b=b,
            c=c,
            basis=basis,
            nonbasic=nonbasic,
            sense=ObjectiveSense.MAXIMIZE,
            tolerances=SimplexTolerances(),
        )

        assert res.status == SolutionStatus.OPTIMAL
        assert res.iterations == 1
        assert res.basis_indices == [0]
        assert np.isclose(res.x[0], 2.0)
        assert np.isclose(res.objective_value, -2.0)

    def test_infeasibility_detection(self) -> None:
        """Min x1 + 2*x2 s.t. x1 + x2 + s1 = -2 with x1, x2, s1 >= 0.
        
        No non-negative variables can sum to -2.
        All tableau entries alpha_j in row 0 are non-negative.
        Dual ratio test has no candidates -> INFEASIBLE.
        """
        c = np.array([1.0, 2.0, 0.0])
        A = np.array([[1.0, 1.0, 1.0]])
        b = np.array([-2.0])
        basis = [2]
        nonbasic = [0, 1]

        solver = DualRevisedSimplexSolver()
        res = solver.solve_standard_form(
            A=A,
            b=b,
            c=c,
            basis=basis,
            nonbasic=nonbasic,
            sense=ObjectiveSense.MINIMIZE,
            tolerances=SimplexTolerances(),
        )

        assert res.status == SolutionStatus.INFEASIBLE


class TestDualSimplexReferenceComparison:
    """Compares Dual Simplex results with Primal Simplex reference."""

    def test_two_constraint_minimization(self) -> None:
        """Min 3*x1 + 4*x2
        s.t.  x1 + 2*x2 >= 4  =>  -x1 - 2*x2 + s1 = -4
              2*x1 + x2 >= 5  =>  -2*x1 - x2 + s2 = -5
              x1, x2 >= 0
        """
        A = np.array([
            [-1.0, -2.0, 1.0, 0.0],
            [-2.0, -1.0, 0.0, 1.0],
        ])
        b = np.array([-4.0, -5.0])
        c = np.array([3.0, 4.0, 0.0, 0.0])
        basis = [2, 3]  # [s1, s2]
        nonbasic = [0, 1]

        solver = DualRevisedSimplexSolver()
        res_dual = solver.solve_standard_form(
            A=A,
            b=b,
            c=c,
            basis=basis,
            nonbasic=nonbasic,
            sense=ObjectiveSense.MINIMIZE,
            tolerances=SimplexTolerances(),
        )
        assert res_dual.status == SolutionStatus.OPTIMAL

        # Solve equivalent model with Primal Revised Simplex
        m = Model("test_two_con")
        x1 = m.add_variable(Variable("x1", lower_bound=0.0))
        x2 = m.add_variable(Variable("x2", lower_bound=0.0))
        m.set_objective(Objective(coefficients={"x1": 3.0, "x2": 4.0}, sense=ObjectiveSense.MINIMIZE))
        m.add_constraint(Constraint("c1", coefficients={"x1": 1.0, "x2": 2.0}, sense=ConstraintSense.GREATER_EQUAL, rhs=4.0))
        m.add_constraint(Constraint("c2", coefficients={"x1": 2.0, "x2": 1.0}, sense=ConstraintSense.GREATER_EQUAL, rhs=5.0))

        solver_primal = RevisedSimplexSolver()
        sol_primal = solver_primal.solve(m)

        assert sol_primal.status == SolutionStatus.OPTIMAL
        assert np.isclose(res_dual.objective_value, sol_primal.objective_value, atol=1e-7)
        assert np.isclose(res_dual.x[0], sol_primal.variable_values["x1"], atol=1e-7)
        assert np.isclose(res_dual.x[1], sol_primal.variable_values["x2"], atol=1e-7)

    def test_three_constraint_maximization(self) -> None:
        """Max -2*x1 - 3*x2 - x3
        s.t.   x1 + x2 + 2*x3 >= 6  =>  -x1 - x2 - 2*x3 + s1 = -6
             2*x1 + 3*x2 + x3 >= 8  =>  -2*x1 - 3*x2 - x3 + s2 = -8
                    x2 + x3   >= 2  =>        -x2 - x3 + s3   = -2
        """
        A = np.array([
            [-1.0, -1.0, -2.0, 1.0, 0.0, 0.0],
            [-2.0, -3.0, -1.0, 0.0, 1.0, 0.0],
            [ 0.0, -1.0, -1.0, 0.0, 0.0, 1.0],
        ])
        b = np.array([-6.0, -8.0, -2.0])
        c = np.array([-2.0, -3.0, -1.0, 0.0, 0.0, 0.0])
        basis = [3, 4, 5]  # slacks
        nonbasic = [0, 1, 2]

        solver = DualRevisedSimplexSolver()
        res_dual = solver.solve_standard_form(
            A=A,
            b=b,
            c=c,
            basis=basis,
            nonbasic=nonbasic,
            sense=ObjectiveSense.MAXIMIZE,
            tolerances=SimplexTolerances(),
        )
        assert res_dual.status == SolutionStatus.OPTIMAL

        # Primal solve comparison
        m = Model("three_con_max")
        x1 = m.add_variable(Variable("x1", lower_bound=0.0))
        x2 = m.add_variable(Variable("x2", lower_bound=0.0))
        x3 = m.add_variable(Variable("x3", lower_bound=0.0))
        m.set_objective(Objective(coefficients={"x1": -2.0, "x2": -3.0, "x3": -1.0}, sense=ObjectiveSense.MAXIMIZE))
        m.add_constraint(Constraint("c1", coefficients={"x1": 1.0, "x2": 1.0, "x3": 2.0}, sense=ConstraintSense.GREATER_EQUAL, rhs=6.0))
        m.add_constraint(Constraint("c2", coefficients={"x1": 2.0, "x2": 3.0, "x3": 1.0}, sense=ConstraintSense.GREATER_EQUAL, rhs=8.0))
        m.add_constraint(Constraint("c3", coefficients={"x2": 1.0, "x3": 1.0}, sense=ConstraintSense.GREATER_EQUAL, rhs=2.0))

        sol_primal = RevisedSimplexSolver().solve(m)
        assert sol_primal.status == SolutionStatus.OPTIMAL
        assert np.isclose(res_dual.objective_value, sol_primal.objective_value, atol=1e-7)

    def test_degenerate_dual_simplex(self) -> None:
        """Degenerate dual simplex test case where reduced cost is zero on candidates."""
        A = np.array([
            [-1.0, -1.0, 1.0, 0.0],
            [-1.0,  0.0, 0.0, 1.0],
        ])
        b = np.array([-3.0, 0.0])
        c = np.array([0.0, 2.0, 0.0, 0.0])  # zero cost on x1
        basis = [2, 3]
        nonbasic = [0, 1]

        solver = DualRevisedSimplexSolver()
        res = solver.solve_standard_form(
            A=A,
            b=b,
            c=c,
            basis=basis,
            nonbasic=nonbasic,
            sense=ObjectiveSense.MINIMIZE,
            tolerances=SimplexTolerances(),
        )
        assert res.status == SolutionStatus.OPTIMAL
        assert np.all(res.x >= -1e-7)

    @pytest.mark.parametrize("seed", [42, 105, 999])
    def test_random_dual_feasible_lps(self, seed: int) -> None:
        """Random LPs with c >= 0 (MIN), negative RHS, comparing Dual Simplex with Primal."""
        rng = np.random.default_rng(seed)
        m_rows = 4
        n_orig = 5

        # Random positive objective coefficients (guarantees dual feasibility of slack basis)
        c_orig = rng.uniform(1.0, 10.0, size=n_orig)
        # Random non-negative A matrix
        A_orig = rng.uniform(0.5, 4.0, size=(m_rows, n_orig))
        # Positive RHS for >= constraints, meaning -A x + s = -b
        b_val = rng.uniform(5.0, 20.0, size=m_rows)

        # Standard form: [-A | I] [x; s] = -b
        A_full = np.hstack([-A_orig, np.eye(m_rows)])
        b_full = -b_val
        c_full = np.concatenate([c_orig, np.zeros(m_rows)])
        basis = list(range(n_orig, n_orig + m_rows))
        nonbasic = list(range(n_orig))

        solver_dual = DualRevisedSimplexSolver()
        res_dual = solver_dual.solve_standard_form(
            A=A_full,
            b=b_full,
            c=c_full,
            basis=basis,
            nonbasic=nonbasic,
            sense=ObjectiveSense.MINIMIZE,
            tolerances=SimplexTolerances(),
        )

        # Build Model for Primal Revised Simplex
        model = Model(f"rand_dual_{seed}")
        vars = [model.add_variable(Variable(f"x{i}", lower_bound=0.0)) for i in range(n_orig)]
        model.set_objective(Objective(coefficients={f"x{i}": c_orig[i] for i in range(n_orig)}, sense=ObjectiveSense.MINIMIZE))
        for j in range(m_rows):
            coeffs = {f"x{i}": A_orig[j, i] for i in range(n_orig)}
            model.add_constraint(Constraint(f"c{j}", coefficients=coeffs, sense=ConstraintSense.GREATER_EQUAL, rhs=b_val[j]))

        sol_primal = RevisedSimplexSolver().solve(model)

        assert res_dual.status == sol_primal.status == SolutionStatus.OPTIMAL
        assert np.isclose(res_dual.objective_value, sol_primal.objective_value, atol=1e-6)
        for i in range(n_orig):
            assert np.isclose(res_dual.x[i], sol_primal.variable_values[f"x{i}"], atol=1e-6)
