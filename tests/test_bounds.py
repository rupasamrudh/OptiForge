"""Focused tests for Stage 6 Phase 1: finite upper bounds and non-zero lower bounds.

Covers:
- finite upper bound (MAXIMIZE / MINIMIZE)
- lower bound > 0 (variable shift)
- lower + upper bound together
- equality constraint with non-zero lower bound
- infeasible bounds
- postsolve variable reconstruction
- objective constant recovery
"""

import math

import pytest

from optiforge.core.solution import SolutionStatus
from optiforge.lp.simplex import RevisedSimplexSolver
from optiforge.lp.standard_form import to_standard_form
from optiforge.model import (
    Constraint,
    ConstraintSense,
    Model,
    Objective,
    ObjectiveSense,
    Variable,
)


# ── finite upper bound ──────────────────────────────────────────────────────


def test_finite_upper_bound_maximize():
    """max x1 + x2 s.t. x1 <= 3, x2 <= 2, x1, x2 >= 0 -> x1=3, x2=2, obj=5."""
    m = Model("ub_max")
    m.add_variable(Variable("x1", lower_bound=0.0, upper_bound=3.0))
    m.add_variable(Variable("x2", lower_bound=0.0, upper_bound=2.0))
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))

    sol = m.solve()
    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.variable_values["x1"] == pytest.approx(3.0, abs=1e-6)
    assert sol.variable_values["x2"] == pytest.approx(2.0, abs=1e-6)
    assert sol.objective_value == pytest.approx(5.0, abs=1e-6)


def test_finite_upper_bound_minimize():
    """min x1 + x2 s.t. x1 <= 3, x2 <= 2, x1, x2 >= 0 -> x1=0, x2=0, obj=0."""
    m = Model("ub_min")
    m.add_variable(Variable("x1", lower_bound=0.0, upper_bound=3.0))
    m.add_variable(Variable("x2", lower_bound=0.0, upper_bound=2.0))
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MINIMIZE))

    sol = m.solve()
    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.variable_values["x1"] == pytest.approx(0.0, abs=1e-6)
    assert sol.variable_values["x2"] == pytest.approx(0.0, abs=1e-6)
    assert sol.objective_value == pytest.approx(0.0, abs=1e-6)


def test_upper_bound_with_constraint():
    """max 3x1 + 2x2 s.t. x1 + x2 <= 4, x1 <= 1, x1, x2 >= 0 -> x1=1, x2=3, obj=9."""
    m = Model("ub_constr")
    m.add_variable(Variable("x1", lower_bound=0.0, upper_bound=1.0))
    m.add_variable(Variable("x2", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.set_objective(Objective({"x1": 3.0, "x2": 2.0}, ObjectiveSense.MAXIMIZE))

    sol = m.solve()
    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.variable_values["x1"] == pytest.approx(1.0, abs=1e-6)
    assert sol.variable_values["x2"] == pytest.approx(3.0, abs=1e-6)
    assert sol.objective_value == pytest.approx(9.0, abs=1e-6)


# ── non-zero lower bound (variable shift) ───────────────────────────────────


def test_lower_bound_positive_maximize():
    """max x s.t. x >= 2, x <= 5 -> x=5, obj=5."""
    m = Model("lb_max")
    m.add_variable(Variable("x", lower_bound=2.0, upper_bound=5.0))
    m.set_objective(Objective({"x": 1.0}, ObjectiveSense.MAXIMIZE))

    sol = m.solve()
    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.variable_values["x"] == pytest.approx(5.0, abs=1e-6)
    assert sol.objective_value == pytest.approx(5.0, abs=1e-6)


def test_lower_bound_positive_minimize():
    """min x s.t. x >= 2, x <= 5 -> x=2, obj=2."""
    m = Model("lb_min")
    m.add_variable(Variable("x", lower_bound=2.0, upper_bound=5.0))
    m.set_objective(Objective({"x": 1.0}, ObjectiveSense.MINIMIZE))

    sol = m.solve()
    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.variable_values["x"] == pytest.approx(2.0, abs=1e-6)
    assert sol.objective_value == pytest.approx(2.0, abs=1e-6)


def test_lower_bound_with_constraint():
    """max 3x1 + 2x2 s.t. x1 + x2 <= 4, x1 >= 1, x1, x2 >= 0 -> x1=4, x2=0, obj=12."""
    m = Model("lb_constr")
    m.add_variable(Variable("x1", lower_bound=1.0))
    m.add_variable(Variable("x2", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.set_objective(Objective({"x1": 3.0, "x2": 2.0}, ObjectiveSense.MAXIMIZE))

    sol = m.solve()
    assert sol.status is SolutionStatus.OPTIMAL
    # x1 has the higher objective coefficient (3 > 2), so push x1 to its max.
    assert sol.variable_values["x1"] == pytest.approx(4.0, abs=1e-6)
    assert sol.variable_values["x2"] == pytest.approx(0.0, abs=1e-6)
    assert sol.objective_value == pytest.approx(12.0, abs=1e-6)


# ── lower + upper bound together ────────────────────────────────────────────


def test_lower_and_upper_bound_together():
    """max 2x s.t. 1 <= x <= 4 -> x=4, obj=8."""
    m = Model("lb_ub")
    m.add_variable(Variable("x", lower_bound=1.0, upper_bound=4.0))
    m.set_objective(Objective({"x": 2.0}, ObjectiveSense.MAXIMIZE))

    sol = m.solve()
    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.variable_values["x"] == pytest.approx(4.0, abs=1e-6)
    assert sol.objective_value == pytest.approx(8.0, abs=1e-6)


def test_lower_and_upper_bound_with_constraint():
    """max x1 + x2 s.t. x1 + x2 <= 5, 1 <= x1 <= 3, 2 <= x2 <= 4 -> x1=3, x2=2, obj=5."""
    m = Model("lb_ub_constr")
    m.add_variable(Variable("x1", lower_bound=1.0, upper_bound=3.0))
    m.add_variable(Variable("x2", lower_bound=2.0, upper_bound=4.0))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))

    sol = m.solve()
    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.variable_values["x1"] == pytest.approx(3.0, abs=1e-6)
    assert sol.variable_values["x2"] == pytest.approx(2.0, abs=1e-6)
    assert sol.objective_value == pytest.approx(5.0, abs=1e-6)


# ── equality with non-zero lower bound ──────────────────────────────────────


def test_equality_with_nonzero_lower_bound():
    """x == 3 with x >= 2 -> x=3. max x -> obj=3."""
    m = Model("eq_lb")
    m.add_variable(Variable("x", lower_bound=2.0))
    m.add_constraint(Constraint("c1", {"x": 1.0}, ConstraintSense.EQUAL, 3.0))
    m.set_objective(Objective({"x": 1.0}, ObjectiveSense.MAXIMIZE))

    sol = m.solve()
    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.variable_values["x"] == pytest.approx(3.0, abs=1e-6)
    assert sol.objective_value == pytest.approx(3.0, abs=1e-6)


def test_equality_with_lower_and_upper_bound():
    """x == 3 with 1 <= x <= 5 -> x=3. min x -> obj=3."""
    m = Model("eq_lb_ub")
    m.add_variable(Variable("x", lower_bound=1.0, upper_bound=5.0))
    m.add_constraint(Constraint("c1", {"x": 1.0}, ConstraintSense.EQUAL, 3.0))
    m.set_objective(Objective({"x": 1.0}, ObjectiveSense.MINIMIZE))

    sol = m.solve()
    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.variable_values["x"] == pytest.approx(3.0, abs=1e-6)
    assert sol.objective_value == pytest.approx(3.0, abs=1e-6)


# ── infeasible bounds ───────────────────────────────────────────────────────


def test_infeasible_upper_bound():
    """max x s.t. x <= 2 (constraint), x >= 5 (constraint) -> INFEASIBLE."""
    m = Model("inf_ub")
    m.add_variable(Variable("x", lower_bound=0.0, upper_bound=10.0))
    m.add_constraint(Constraint("c1", {"x": 1.0}, ConstraintSense.LESS_EQUAL, 2.0))
    m.add_constraint(Constraint("c2", {"x": 1.0}, ConstraintSense.GREATER_EQUAL, 5.0))
    m.set_objective(Objective({"x": 1.0}, ObjectiveSense.MAXIMIZE))

    sol = m.solve()
    assert sol.status is SolutionStatus.INFEASIBLE


def test_infeasible_bounds_via_constraint():
    """max x s.t. x <= 2 (constraint), x >= 5 (constraint) -> INFEASIBLE."""
    m = Model("inf_constr")
    m.add_variable(Variable("x", lower_bound=0.0, upper_bound=10.0))
    m.add_constraint(Constraint("c1", {"x": 1.0}, ConstraintSense.LESS_EQUAL, 2.0))
    m.add_constraint(Constraint("c2", {"x": 1.0}, ConstraintSense.GREATER_EQUAL, 5.0))
    m.set_objective(Objective({"x": 1.0}, ObjectiveSense.MAXIMIZE))

    sol = m.solve()
    assert sol.status is SolutionStatus.INFEASIBLE


def test_infeasible_equality_outside_bounds():
    """x == 7 with 1 <= x <= 5 -> INFEASIBLE."""
    m = Model("inf_eq")
    m.add_variable(Variable("x", lower_bound=1.0, upper_bound=5.0))
    m.add_constraint(Constraint("c1", {"x": 1.0}, ConstraintSense.EQUAL, 7.0))
    m.set_objective(Objective({"x": 1.0}, ObjectiveSense.MINIMIZE))

    sol = m.solve()
    assert sol.status is SolutionStatus.INFEASIBLE


# ── postsolve reconstruction & objective constant ───────────────────────────


def test_postsolve_reconstruction_with_shift():
    """max 3x1 + 2x2 s.t. x1 + x2 <= 4, x1 >= 1, x2 >= 2 -> x1=2, x2=2, obj=10."""
    m = Model("postsolve_shift")
    m.add_variable(Variable("x1", lower_bound=1.0))
    m.add_variable(Variable("x2", lower_bound=2.0))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.set_objective(Objective({"x1": 3.0, "x2": 2.0}, ObjectiveSense.MAXIMIZE))

    sol = m.solve()
    assert sol.status is SolutionStatus.OPTIMAL
    # x1 + x2 <= 4 with x1 >= 1, x2 >= 2: x2 is pinned at 2 (its lower bound),
    # leaving x1 <= 2. max 3x1+2x2 -> x1=2, x2=2 (obj=6+4=10).
    assert sol.variable_values["x1"] == pytest.approx(2.0, abs=1e-6)
    assert sol.variable_values["x2"] == pytest.approx(2.0, abs=1e-6)
    assert sol.objective_value == pytest.approx(10.0, abs=1e-6)


def test_objective_constant_recovery():
    """min 2x s.t. x >= 3, x <= 8 -> x=3, obj=6 (constant 6 from shift)."""
    m = Model("obj_const")
    m.add_variable(Variable("x", lower_bound=3.0, upper_bound=8.0))
    m.set_objective(Objective({"x": 2.0}, ObjectiveSense.MINIMIZE))

    sol = m.solve()
    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.variable_values["x"] == pytest.approx(3.0, abs=1e-6)
    assert sol.objective_value == pytest.approx(6.0, abs=1e-6)


def test_standard_form_shift_metadata():
    """to_standard_form records variable_shifts and objective_constant."""
    m = Model("sf_meta")
    m.add_variable(Variable("x", lower_bound=2.0, upper_bound=10.0))
    m.add_constraint(Constraint("c1", {"x": 1.0}, ConstraintSense.LESS_EQUAL, 10.0))
    m.set_objective(Objective({"x": 3.0}, ObjectiveSense.MAXIMIZE))

    sf = to_standard_form(m)
    assert sf.variable_shifts == {0: 2.0}
    assert sf.objective_constant == pytest.approx(6.0)  # 3 * 2
    # Upper bound row: y <= 8 (10 - 2)
    assert sf.has_upper_bounds is True
    # Constraint row: y <= 8 (10 - 2)
    # Both rows are y <= 8; the model is feasible with y in [0, 8].