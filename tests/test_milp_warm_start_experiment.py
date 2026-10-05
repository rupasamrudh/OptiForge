"""Phase 5 & Phase 7: MILP Bound-Tightening Warm-Start Experiment.

Evaluates transferring parent LP relaxation bases to child nodes created by
Branch-and-Bound branching (x_j <= floor(x_j*) or x_j >= ceil(x_j*)), re-optimizing
with Dual Revised Simplex, and validating against cold solves with Primal Simplex.
"""

from __future__ import annotations

import numpy as np
import pytest

from optiforge.core.solution import SolutionStatus
from optiforge.lp.dual_simplex import DualRevisedSimplexSolver
from optiforge.lp.simplex import RevisedSimplexSolver, SimplexTolerances
from optiforge.lp.standard_form import to_standard_form
from optiforge.model import Constraint, ConstraintSense, Model, Objective, ObjectiveSense, Variable, VariableType


def warm_start_branch(
    parent_sf,
    parent_basis: list[int],
    branch_var_idx: int,
    bound_type: str,  # 'upper' (<=) or 'lower' (>=)
    bound_val: float,
    sense: ObjectiveSense,
    use_incremental_basis: bool = False,
) -> tuple[any, float]:
    """Helper to construct child standard form with bound row and solve via Dual Simplex."""
    m_rows, n_cols = parent_sf.A.shape
    A_child = np.zeros((m_rows + 1, n_cols + 1))
    A_child[:m_rows, :n_cols] = parent_sf.A

    if bound_type == "upper":
        # x_k <= bound_val  =>  x_k + s_new = bound_val
        A_child[m_rows, branch_var_idx] = 1.0
        A_child[m_rows, n_cols] = 1.0
        b_child = np.append(parent_sf.b, bound_val)
    elif bound_type == "lower":
        # x_k >= bound_val  =>  -x_k + s_new = -bound_val
        A_child[m_rows, branch_var_idx] = -1.0
        A_child[m_rows, n_cols] = 1.0
        b_child = np.append(parent_sf.b, -bound_val)
    else:
        raise ValueError(f"Unknown bound type: {bound_type}")

    c_child = np.append(parent_sf.c, 0.0)
    warm_basis = list(parent_basis) + [n_cols]
    all_cols = set(range(n_cols + 1))
    warm_nonbasic = sorted(list(all_cols - set(warm_basis)))

    ds = DualRevisedSimplexSolver(use_incremental_basis=use_incremental_basis)
    res = ds.solve_standard_form(
        A=A_child,
        b=b_child,
        c=c_child,
        basis=warm_basis,
        nonbasic=warm_nonbasic,
        sense=sense,
        tolerances=SimplexTolerances(),
    )
    return res, b_child[-1]


class TestMILPBoundTighteningWarmStart:
    """Rigorous validation of MILP parent-basis transfer and dual simplex re-optimization."""

    def test_maximization_upper_branch(self) -> None:
        """Max 5*x1 + 4*x2 s.t. 6*x1 + 4*x2 <= 24, x1 + 2*x2 <= 6.
        Parent optimal: x1=3.0, x2=1.5.
        Branch: x2 <= 1.0.
        """
        m = Model("max_branch_upper")
        x1 = m.add_variable(Variable("x1", lower_bound=0.0))
        x2 = m.add_variable(Variable("x2", lower_bound=0.0))
        m.set_objective(Objective(coefficients={"x1": 5.0, "x2": 4.0}, sense=ObjectiveSense.MAXIMIZE))
        m.add_constraint(Constraint("c1", coefficients={"x1": 6.0, "x2": 4.0}, sense=ConstraintSense.LESS_EQUAL, rhs=24.0))
        m.add_constraint(Constraint("c2", coefficients={"x1": 1.0, "x2": 2.0}, sense=ConstraintSense.LESS_EQUAL, rhs=6.0))

        sf_parent = to_standard_form(m)
        s_primal = RevisedSimplexSolver()
        res_parent = s_primal._simplex_loop(
            A=sf_parent.A, b=sf_parent.b, c=sf_parent.c,
            basis_indices=list(sf_parent.phase1_basis_indices),
            nonbasic_indices=list(sf_parent.phase1_nonbasic_indices),
            sense=sf_parent.objective_sense,
            tolerances=SimplexTolerances(),
        )
        assert res_parent.status == SolutionStatus.OPTIMAL
        assert np.isclose(res_parent.x[1], 1.5)

        # Branch x2 <= 1.0
        res_warm, _ = warm_start_branch(
            parent_sf=sf_parent,
            parent_basis=res_parent.basis_indices,
            branch_var_idx=1,
            bound_type="upper",
            bound_val=1.0,
            sense=ObjectiveSense.MAXIMIZE,
        )
        assert res_warm.status == SolutionStatus.OPTIMAL
        assert res_warm.iterations <= 2

        # Cold solve
        m_cold = Model("max_cold_upper")
        m_cold.add_variable(Variable("x1", lower_bound=0.0))
        m_cold.add_variable(Variable("x2", lower_bound=0.0, upper_bound=1.0))
        m_cold.set_objective(Objective(coefficients={"x1": 5.0, "x2": 4.0}, sense=ObjectiveSense.MAXIMIZE))
        m_cold.add_constraint(Constraint("c1", coefficients={"x1": 6.0, "x2": 4.0}, sense=ConstraintSense.LESS_EQUAL, rhs=24.0))
        m_cold.add_constraint(Constraint("c2", coefficients={"x1": 1.0, "x2": 2.0}, sense=ConstraintSense.LESS_EQUAL, rhs=6.0))
        sol_cold = s_primal.solve(m_cold)

        assert np.isclose(res_warm.objective_value, sol_cold.objective_value, atol=1e-7)
        assert np.isclose(res_warm.x[0], sol_cold.variable_values["x1"], atol=1e-7)
        assert np.isclose(res_warm.x[1], sol_cold.variable_values["x2"], atol=1e-7)

    def test_maximization_lower_branch(self) -> None:
        """Branch x2 >= 2.0 on same parent problem."""
        m = Model("max_branch_lower")
        x1 = m.add_variable(Variable("x1", lower_bound=0.0))
        x2 = m.add_variable(Variable("x2", lower_bound=0.0))
        m.set_objective(Objective(coefficients={"x1": 5.0, "x2": 4.0}, sense=ObjectiveSense.MAXIMIZE))
        m.add_constraint(Constraint("c1", coefficients={"x1": 6.0, "x2": 4.0}, sense=ConstraintSense.LESS_EQUAL, rhs=24.0))
        m.add_constraint(Constraint("c2", coefficients={"x1": 1.0, "x2": 2.0}, sense=ConstraintSense.LESS_EQUAL, rhs=6.0))

        sf_parent = to_standard_form(m)
        s_primal = RevisedSimplexSolver()
        res_parent = s_primal._simplex_loop(
            A=sf_parent.A, b=sf_parent.b, c=sf_parent.c,
            basis_indices=list(sf_parent.phase1_basis_indices),
            nonbasic_indices=list(sf_parent.phase1_nonbasic_indices),
            sense=sf_parent.objective_sense,
            tolerances=SimplexTolerances(),
        )

        res_warm, _ = warm_start_branch(
            parent_sf=sf_parent,
            parent_basis=res_parent.basis_indices,
            branch_var_idx=1,
            bound_type="lower",
            bound_val=2.0,
            sense=ObjectiveSense.MAXIMIZE,
        )
        assert res_warm.status == SolutionStatus.OPTIMAL

        m_cold = Model("max_cold_lower")
        m_cold.add_variable(Variable("x1", lower_bound=0.0))
        m_cold.add_variable(Variable("x2", lower_bound=2.0))
        m_cold.set_objective(Objective(coefficients={"x1": 5.0, "x2": 4.0}, sense=ObjectiveSense.MAXIMIZE))
        m_cold.add_constraint(Constraint("c1", coefficients={"x1": 6.0, "x2": 4.0}, sense=ConstraintSense.LESS_EQUAL, rhs=24.0))
        m_cold.add_constraint(Constraint("c2", coefficients={"x1": 1.0, "x2": 2.0}, sense=ConstraintSense.LESS_EQUAL, rhs=6.0))
        sol_cold = s_primal.solve(m_cold)

        assert np.isclose(res_warm.objective_value, sol_cold.objective_value, atol=1e-7)
        assert np.isclose(res_warm.x[0], sol_cold.variable_values["x1"], atol=1e-7)
        assert np.isclose(res_warm.x[1], sol_cold.variable_values["x2"], atol=1e-7)

    def test_minimization_branching(self) -> None:
        """Minimization problem with fractional variable branching."""
        m = Model("min_branch_parent")
        x1 = m.add_variable(Variable("x1", lower_bound=0.0))
        x2 = m.add_variable(Variable("x2", lower_bound=0.0))
        m.set_objective(Objective(coefficients={"x1": -3.0, "x2": -2.0}, sense=ObjectiveSense.MINIMIZE))
        m.add_constraint(Constraint("c1", coefficients={"x1": 2.0, "x2": 1.0}, sense=ConstraintSense.LESS_EQUAL, rhs=5.0))
        m.add_constraint(Constraint("c2", coefficients={"x1": 1.0, "x2": 3.0}, sense=ConstraintSense.LESS_EQUAL, rhs=6.0))

        sf_parent = to_standard_form(m)
        s_primal = RevisedSimplexSolver()
        res_parent = s_primal._simplex_loop(
            A=sf_parent.A, b=sf_parent.b, c=sf_parent.c,
            basis_indices=list(sf_parent.phase1_basis_indices),
            nonbasic_indices=list(sf_parent.phase1_nonbasic_indices),
            sense=sf_parent.objective_sense,
            tolerances=SimplexTolerances(),
        )
        assert res_parent.status == SolutionStatus.OPTIMAL
        # x1 = 1.8, branch x1 <= 1.0
        assert np.isclose(res_parent.x[0], 1.8)

        res_warm, _ = warm_start_branch(
            parent_sf=sf_parent,
            parent_basis=res_parent.basis_indices,
            branch_var_idx=0,
            bound_type="upper",
            bound_val=1.0,
            sense=ObjectiveSense.MINIMIZE,
        )
        assert res_warm.status == SolutionStatus.OPTIMAL

        m_cold = Model("min_cold_upper")
        m_cold.add_variable(Variable("x1", lower_bound=0.0, upper_bound=1.0))
        m_cold.add_variable(Variable("x2", lower_bound=0.0))
        m_cold.set_objective(Objective(coefficients={"x1": -3.0, "x2": -2.0}, sense=ObjectiveSense.MINIMIZE))
        m_cold.add_constraint(Constraint("c1", coefficients={"x1": 2.0, "x2": 1.0}, sense=ConstraintSense.LESS_EQUAL, rhs=5.0))
        m_cold.add_constraint(Constraint("c2", coefficients={"x1": 1.0, "x2": 3.0}, sense=ConstraintSense.LESS_EQUAL, rhs=6.0))
        sol_cold = s_primal.solve(m_cold)

        assert np.isclose(res_warm.objective_value, sol_cold.objective_value, atol=1e-7)
        assert np.isclose(res_warm.x[0], sol_cold.variable_values["x1"], atol=1e-7)
        assert np.isclose(res_warm.x[1], sol_cold.variable_values["x2"], atol=1e-7)

    def test_infeasible_branch_detection(self) -> None:
        """Branch into an empty region: Dual simplex must detect INFEASIBLE."""
        m = Model("infeasible_branch_parent")
        x = m.add_variable(Variable("x", lower_bound=0.0, upper_bound=3.0))
        m.set_objective(Objective(coefficients={"x": -1.0}, sense=ObjectiveSense.MINIMIZE))
        m.add_constraint(Constraint("c1", coefficients={"x": 1.0}, sense=ConstraintSense.LESS_EQUAL, rhs=3.0))

        sf_parent = to_standard_form(m)
        s_primal = RevisedSimplexSolver()
        res_parent = s_primal._simplex_loop(
            A=sf_parent.A, b=sf_parent.b, c=sf_parent.c,
            basis_indices=list(sf_parent.phase1_basis_indices),
            nonbasic_indices=list(sf_parent.phase1_nonbasic_indices),
            sense=sf_parent.objective_sense,
            tolerances=SimplexTolerances(),
        )

        # Branch x >= 5.0 (impossible since x <= 3.0 in problem)
        res_warm, _ = warm_start_branch(
            parent_sf=sf_parent,
            parent_basis=res_parent.basis_indices,
            branch_var_idx=0,
            bound_type="lower",
            bound_val=5.0,
            sense=ObjectiveSense.MINIMIZE,
        )
        assert res_warm.status == SolutionStatus.INFEASIBLE

    def test_warm_start_with_incremental_basis(self) -> None:
        """Verify warm-start works when use_incremental_basis=True."""
        m = Model("max_branch_inc")
        x1 = m.add_variable(Variable("x1", lower_bound=0.0))
        x2 = m.add_variable(Variable("x2", lower_bound=0.0))
        m.set_objective(Objective(coefficients={"x1": 5.0, "x2": 4.0}, sense=ObjectiveSense.MAXIMIZE))
        m.add_constraint(Constraint("c1", coefficients={"x1": 6.0, "x2": 4.0}, sense=ConstraintSense.LESS_EQUAL, rhs=24.0))
        m.add_constraint(Constraint("c2", coefficients={"x1": 1.0, "x2": 2.0}, sense=ConstraintSense.LESS_EQUAL, rhs=6.0))

        sf_parent = to_standard_form(m)
        s_primal = RevisedSimplexSolver()
        res_parent = s_primal._simplex_loop(
            A=sf_parent.A, b=sf_parent.b, c=sf_parent.c,
            basis_indices=list(sf_parent.phase1_basis_indices),
            nonbasic_indices=list(sf_parent.phase1_nonbasic_indices),
            sense=sf_parent.objective_sense,
            tolerances=SimplexTolerances(),
        )

        res_warm_inc, _ = warm_start_branch(
            parent_sf=sf_parent,
            parent_basis=res_parent.basis_indices,
            branch_var_idx=1,
            bound_type="upper",
            bound_val=1.0,
            sense=ObjectiveSense.MAXIMIZE,
            use_incremental_basis=True,
        )
        assert res_warm_inc.status == SolutionStatus.OPTIMAL
        assert np.isclose(res_warm_inc.objective_value, 20.666666666666668, atol=1e-7)
