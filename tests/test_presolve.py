import math

import pytest

from optiforge.core.solution import SolutionStatus
from optiforge.model import Constraint, ConstraintSense, Model, Objective, ObjectiveSense, Variable
from optiforge.presolve import presolve_model, postsolve_solution


def test_fixed_variable_elimination():
    m = Model("fixed_var")
    m.add_variable(Variable("x1", lower_bound=3.0, upper_bound=3.0))
    m.add_variable(Variable("x2", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": 2.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 10.0))
    m.set_objective(Objective({"x1": 4.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))

    result = presolve_model(m)

    assert result.status is SolutionStatus.FEASIBLE
    # x1 is fixed at 3.0; c1 then implies x2 <= 4, and x2 becomes a
    # zero-column with a positive MAX coefficient, so it is fixed at its
    # upper bound 4.0. Objective constant: 4*3 + 1*4 = 16.
    assert result.fixed_variables == {"x1": 3.0, "x2": 4.0}
    assert result.objective_constant == pytest.approx(16.0)
    assert result.reduced_model.num_variables == 0
    assert "x1" not in result.reduced_model.variables


def test_fixed_variable_multiple_constraints_and_objective_constant():
    m = Model("fixed_multi")
    m.add_variable(Variable("x", lower_bound=5.0, upper_bound=5.0))
    m.add_variable(Variable("y", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x": 1.0, "y": 1.0}, ConstraintSense.LESS_EQUAL, 10.0))
    m.add_constraint(Constraint("c2", {"x": 2.0, "y": 3.0}, ConstraintSense.GREATER_EQUAL, 13.0))
    m.set_objective(Objective({"x": 2.0, "y": 7.0}, ObjectiveSense.MAXIMIZE))

    result = presolve_model(m)

    # x fixed at 5 -> constant 10. c1 -> y <= 5, c2 -> y >= 1. y is then a
    # zero-column with positive MAX coefficient, fixed at its upper bound 5
    # -> constant += 35. Total objective constant = 45.
    assert result.objective_constant == pytest.approx(45.0)
    assert result.fixed_variables == {"x": 5.0, "y": 5.0}
    assert result.reduced_model.num_variables == 0


def test_zero_row_redundant_and_infeasible():
    m = Model("zero_row")
    m.add_variable(Variable("x", lower_bound=0.0))
    m.add_constraint(Constraint("r1", {"x": 0.0}, ConstraintSense.LESS_EQUAL, 2.0))
    m.add_constraint(Constraint("r2", {"x": 0.0}, ConstraintSense.GREATER_EQUAL, 5.0))
    m.set_objective(Objective({"x": 1.0}, ObjectiveSense.MINIMIZE))

    result = presolve_model(m)
    assert result.status is SolutionStatus.INFEASIBLE

    m2 = Model("zero_row_ok")
    m2.add_variable(Variable("x", lower_bound=0.0))
    m2.add_constraint(Constraint("r1", {"x": 0.0}, ConstraintSense.LESS_EQUAL, 2.0))
    m2.set_objective(Objective({"x": 1.0}, ObjectiveSense.MINIMIZE))
    result2 = presolve_model(m2)
    assert result2.status is SolutionStatus.FEASIBLE
    # The zero row is redundant; x is a zero-column with a positive MIN
    # coefficient, so it is fixed at its lower bound 0.0.
    assert result2.fixed_variables == {"x": 0.0}
    assert result2.reduced_model.num_variables == 0
    assert result2.reduced_model.num_constraints == 0


def test_zero_equality_redundant_and_infeasible():
    m = Model("zero_eq")
    m.add_variable(Variable("x", lower_bound=0.0))
    m.add_constraint(Constraint("eq1", {"x": 0.0}, ConstraintSense.EQUAL, 0.0))
    m.set_objective(Objective({"x": 1.0}, ObjectiveSense.MINIMIZE))
    assert presolve_model(m).status is SolutionStatus.FEASIBLE

    m2 = Model("zero_eq_bad")
    m2.add_variable(Variable("x", lower_bound=0.0))
    m2.add_constraint(Constraint("eq1", {"x": 0.0}, ConstraintSense.EQUAL, 4.0))
    m2.set_objective(Objective({"x": 1.0}, ObjectiveSense.MINIMIZE))
    assert presolve_model(m2).status is SolutionStatus.INFEASIBLE


def test_singleton_bound_reduction():
    m = Model("singleton")
    m.add_variable(Variable("x", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x": 3.0}, ConstraintSense.LESS_EQUAL, 12.0))
    m.add_constraint(Constraint("c2", {"x": 2.0}, ConstraintSense.GREATER_EQUAL, 10.0))
    m.add_constraint(Constraint("c3", {"x": -2.0}, ConstraintSense.LESS_EQUAL, 6.0))
    m.set_objective(Objective({"x": 1.0}, ObjectiveSense.MINIMIZE))

    result = presolve_model(m)
    # c1 -> x <= 4, c2 -> x >= 5: contradictory singleton bounds.
    assert result.status is SolutionStatus.INFEASIBLE


def test_contradictory_bounds_and_activity_infeasibility():
    # Variable construction rejects contradictory bounds (lb > ub), so
    # presolve detects infeasibility through constraint-derived bounds.
    m = Model("contradict")
    m.add_variable(Variable("x", lower_bound=0.0, upper_bound=10.0))
    m.add_constraint(Constraint("c1", {"x": 1.0}, ConstraintSense.GREATER_EQUAL, 12.0))
    m.set_objective(Objective({"x": 1.0}, ObjectiveSense.MINIMIZE))
    assert presolve_model(m).status is SolutionStatus.INFEASIBLE

    m2 = Model("activity_bad")
    m2.add_variable(Variable("x", lower_bound=0.0, upper_bound=10.0))
    m2.add_constraint(Constraint("c1", {"x": 1.0}, ConstraintSense.LESS_EQUAL, 1.0))
    m2.set_objective(Objective({"x": 1.0}, ObjectiveSense.MINIMIZE))
    assert presolve_model(m2).status is SolutionStatus.FEASIBLE


def test_bound_propagation_and_iterative_fixed_point():
    m = Model("propagate")
    m.add_variable(Variable("x1", lower_bound=0.0))
    m.add_variable(Variable("x2", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.add_constraint(Constraint("c2", {"x1": 2.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 8.0))
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MINIMIZE))

    result = presolve_model(m)
    assert result.status is SolutionStatus.FEASIBLE
    assert result.statistics["tightened_bounds"] >= 1
    assert result.statistics["passes"] >= 1


def test_zero_column_bounded_and_unbounded():
    m = Model("zero_col")
    m.add_variable(Variable("x", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x": 0.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.set_objective(Objective({"x": 3.0}, ObjectiveSense.MINIMIZE))
    result = presolve_model(m)
    assert result.status is SolutionStatus.FEASIBLE
    assert result.fixed_variables == {"x": 0.0}

    m2 = Model("zero_col_unbounded")
    m2.add_variable(Variable("x", lower_bound=0.0))
    m2.add_constraint(Constraint("c1", {"x": 0.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m2.set_objective(Objective({"x": 1.0}, ObjectiveSense.MAXIMIZE))
    result2 = presolve_model(m2)
    assert result2.status is SolutionStatus.UNBOUNDED


def test_postsolve_reconstruction_and_objective_recovery():
    m = Model("postsolve")
    m.add_variable(Variable("x1", lower_bound=0.0))
    m.add_variable(Variable("x2", lower_bound=0.0))
    m.add_variable(Variable("x3", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 2.0, "x3": 1.0}, ConstraintSense.LESS_EQUAL, 10.0))
    m.add_constraint(Constraint("c2", {"x2": 1.0}, ConstraintSense.EQUAL, 5.0))
    m.set_objective(Objective({"x1": 2.0, "x2": 3.0, "x3": 1.0}, ObjectiveSense.MAXIMIZE))

    presolve = presolve_model(m)
    reduced = presolve.reduced_model

    # x2 is fixed at 5 (singleton equality). c1 becomes x1 + x3 <= 0, which
    # propagates x1 <= 0 and x3 <= 0; with non-negative lower bounds both are
    # fixed at 0. The reduced model is empty and the objective constant is
    # 3*5 = 15 (x2's contribution).
    assert presolve.fixed_variables == {"x2": 5.0, "x1": 0.0, "x3": 0.0}
    assert presolve.objective_constant == pytest.approx(15.0)
    assert reduced.num_variables == 0

    restored = postsolve_solution({}, presolve)
    assert restored["x1"] == pytest.approx(0.0)
    assert restored["x2"] == pytest.approx(5.0)
    assert restored["x3"] == pytest.approx(0.0)


def test_noop_presolve_and_determinism():
    m = Model("noop")
    m.add_variable(Variable("x", lower_bound=0.0))
    m.add_variable(Variable("y", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x": 1.0, "y": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.set_objective(Objective({"x": 2.0, "y": 1.0}, ObjectiveSense.MAXIMIZE))

    r1 = presolve_model(m)
    r2 = presolve_model(m)
    assert r1.status is SolutionStatus.FEASIBLE
    assert r1.statistics == r2.statistics
    assert r1.reduced_model is not None


def test_model_solve_uses_presolve_for_fixed_variables():
    m = Model("solve_fixed")
    m.add_variable(Variable("x1", lower_bound=0.0))
    m.add_variable(Variable("x2", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.add_constraint(Constraint("c2", {"x1": 0.0, "x2": 1.0}, ConstraintSense.EQUAL, 1.0))
    m.set_objective(Objective({"x1": 3.0, "x2": 2.0}, ObjectiveSense.MAXIMIZE))

    sol = m.solve()

    # c2 fixes x2 = 1; c1 then gives x1 <= 3, and x1 is fixed at its upper
    # bound 3 (zero-column, positive MAX coefficient). Objective = 3*3 + 2*1 = 11.
    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.variable_values["x1"] == pytest.approx(3.0, abs=1e-6)
    assert sol.variable_values["x2"] == pytest.approx(1.0, abs=1e-6)
    assert sol.objective_value == pytest.approx(11.0, abs=1e-6)
