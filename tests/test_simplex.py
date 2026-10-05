"""Unit tests for Stage 3: Baseline Revised Simplex LP solver."""

import math
import numpy as np
import pytest

from optiforge.core.solution import Solution, SolutionStatus
from optiforge.lp.simplex import RevisedSimplexSolver, SimplexTolerances
from optiforge.lp.standard_form import (
    SingularBasisError,
    StandardFormModel,
    to_standard_form,
)
from optiforge.model import (
    Constraint,
    ConstraintSense,
    Model,
    Objective,
    ObjectiveSense,
    Variable,
    VariableType,
)


# ── TEST 1: Canonical Maximization ──────────────────────────────────────────


def test_canonical_maximization():
    """Solve canonical 2-variable LP:
    maximize 3·x1 + 2·x2
    s.t.
        x1 +  x2 <= 4
       2·x1 + x2 <= 5
        x1, x2 >= 0
    Expected optimum: x1 = 1, x2 = 3, z = 9.
    """
    m = Model(name="canonical_max")
    m.add_variable(Variable("x1", lower_bound=0.0))
    m.add_variable(Variable("x2", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.add_constraint(Constraint("c2", {"x1": 2.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.set_objective(Objective({"x1": 3.0, "x2": 2.0}, ObjectiveSense.MAXIMIZE))

    sol = m.solve()

    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.variable_values["x1"] == pytest.approx(1.0, abs=1e-6)
    assert sol.variable_values["x2"] == pytest.approx(3.0, abs=1e-6)
    assert sol.objective_value == pytest.approx(9.0, abs=1e-6)
    assert sol.iterations == 2


# ── TEST 2: Simple Minimization ─────────────────────────────────────────────


def test_simple_minimization():
    """Solve:
    minimize x1 + 2·x2
    s.t.
        x1 +  x2 >= 3
        x1 + 3·x2 >= 6
        x1, x2 >= 0
    Intersection: x1 = 1.5, x2 = 1.5 -> z = 4.5.
    At (6, 0): z = 6. At (0, 3): z = 6.
    Expected optimum: x1 = 1.5, x2 = 1.5, z = 4.5.
    """
    m = Model(name="simple_min")
    m.add_variable(Variable("x1", lower_bound=0.0))
    m.add_variable(Variable("x2", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, 3.0))
    m.add_constraint(Constraint("c2", {"x1": 1.0, "x2": 3.0}, ConstraintSense.GREATER_EQUAL, 6.0))
    m.set_objective(Objective({"x1": 1.0, "x2": 2.0}, ObjectiveSense.MINIMIZE))

    solver = RevisedSimplexSolver()
    sol = solver.solve(m)

    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.variable_values["x1"] == pytest.approx(1.5, abs=1e-6)
    assert sol.variable_values["x2"] == pytest.approx(1.5, abs=1e-6)
    assert sol.objective_value == pytest.approx(4.5, abs=1e-6)
    assert sol.iterations > 0


# ── TEST 3: Already Optimal Initial BFS ──────────────────────────────────────


def test_already_optimal_bfs():
    """LP where the origin (initial BFS) is already optimal:
    minimize 2·x1 + 3·x2
    s.t.
        x1 + x2 <= 4
        x1, x2 >= 0
    Optimal at x1=0, x2=0, z=0. Zero pivots required.
    """
    m = Model(name="already_optimal")
    m.add_variable(Variable("x1", lower_bound=0.0))
    m.add_variable(Variable("x2", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.set_objective(Objective({"x1": 2.0, "x2": 3.0}, ObjectiveSense.MINIMIZE))

    sol = m.solve()

    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.variable_values["x1"] == pytest.approx(0.0)
    assert sol.variable_values["x2"] == pytest.approx(0.0)
    assert sol.objective_value == pytest.approx(0.0)
    assert sol.iterations == 0


# ── TEST 4: Phase-I Feasible LP ──────────────────────────────────────────────


def test_phase1_feasible_lp():
    """Mixed equality and inequality constraints requiring Phase I:
    maximize 2·x1 + x2
    s.t.
        x1 + x2 = 4
        x1 <= 3
        x1, x2 >= 0
    Vertices of feasible line segment:
        (3, 1) -> obj = 2(3) + 1 = 7
        (0, 4) -> obj = 2(0) + 4 = 4
    Optimal: x1 = 3, x2 = 1, z = 7.
    """
    m = Model(name="phase1_feasible")
    m.add_variable(Variable("x1", lower_bound=0.0))
    m.add_variable(Variable("x2", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.EQUAL, 4.0))
    m.add_constraint(Constraint("c2", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 3.0))
    m.set_objective(Objective({"x1": 2.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))

    sol = m.solve()

    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.variable_values["x1"] == pytest.approx(3.0, abs=1e-6)
    assert sol.variable_values["x2"] == pytest.approx(1.0, abs=1e-6)
    assert sol.objective_value == pytest.approx(7.0, abs=1e-6)
    # Artificial variables should not appear in original solution values
    assert "a1" not in sol.variable_values
    assert "s2" not in sol.variable_values


# ── TEST 5: Phase-I Infeasible LP ────────────────────────────────────────────


def test_phase1_infeasible_lp():
    """Infeasible LP:
    maximize x1
    s.t.
        x1 >= 2
        x1 <= 1
        x1 >= 0
    Expected: INFEASIBLE.
    """
    m = Model(name="infeasible_lp")
    m.add_variable(Variable("x1", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": 1.0}, ConstraintSense.GREATER_EQUAL, 2.0))
    m.add_constraint(Constraint("c2", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 1.0))
    m.set_objective(Objective({"x1": 1.0}, ObjectiveSense.MAXIMIZE))

    sol = m.solve()

    assert sol.status is SolutionStatus.INFEASIBLE
    assert sol.objective_value is None
    assert "infeasible" in sol.message.lower()


# ── TEST 6: Zero-Valued Basic Artificial Variable Cleanup ───────────────────


def test_zero_valued_basic_artificial():
    """Feasible LP with redundant equality constraint:
    Phase I reaches w* = 0, but an artificial variable remains basic at value 0.
    maximize x1 + x2
    s.t.
        x1 + x2 = 4
        2·x1 + 2·x2 = 8   (redundant row)
        x1 <= 3
        x1, x2 >= 0
    Expected: Optimum x1=3, x2=1, obj=4 (or any point on line x1+x2=4 with obj=4).
    """
    m = Model(name="redundant_equality")
    m.add_variable(Variable("x1", lower_bound=0.0))
    m.add_variable(Variable("x2", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.EQUAL, 4.0))
    m.add_constraint(Constraint("c2", {"x1": 2.0, "x2": 2.0}, ConstraintSense.EQUAL, 8.0))
    m.add_constraint(Constraint("c3", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 3.0))
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))

    sol = m.solve()

    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.objective_value == pytest.approx(4.0, abs=1e-6)
    assert sol.variable_values["x1"] + sol.variable_values["x2"] == pytest.approx(4.0, abs=1e-6)


# ── TEST 7: Unbounded LP ─────────────────────────────────────────────────────


def test_unbounded_lp():
    """Unbounded LP:
    maximize x1
    s.t.
        x1 - x2 <= 2
        x1, x2 >= 0
    As x2 -> inf, x1 -> inf without bound.
    Expected: UNBOUNDED.
    """
    m = Model(name="unbounded_lp")
    m.add_variable(Variable("x1", lower_bound=0.0))
    m.add_variable(Variable("x2", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": -1.0}, ConstraintSense.LESS_EQUAL, 2.0))
    m.set_objective(Objective({"x1": 1.0}, ObjectiveSense.MAXIMIZE))

    sol = m.solve()

    assert sol.status is SolutionStatus.UNBOUNDED
    assert "unbounded" in sol.message.lower()


# ── TEST 8: Degenerate BFS ───────────────────────────────────────────────────


def test_degenerate_bfs():
    """LP with degenerate BFS (RHS = 0 on constraint):
    maximize 2·x1 + x2
    s.t.
        x1 +  x2 <= 4
        x1 - 2·x2 <= 0   (passes through origin; s2 = 0 initially)
        x1, x2 >= 0
    Optimal solution:
        x1 - 2*x2 <= 0 -> x1 <= 2*x2
        x1 + x2 = 4 -> 3*x2 = 4 -> x2 = 8/3, x1 = 4/3 -> obj = 8/3 + 8/3 = 16/3 ≈ 5.333333
        or x1=0, x2=4 -> obj = 4.
        At (8/3, 4/3): 2(8/3) + 4/3 = 20/3 ? Wait:
        x1 = 2*x2 and x1 + x2 = 4 => 3*x2 = 4 => x2 = 4/3, x1 = 8/3.
        obj = 2*(8/3) + 4/3 = 20/3 ≈ 6.666667.
    """
    m = Model(name="degenerate_lp")
    m.add_variable(Variable("x1", lower_bound=0.0))
    m.add_variable(Variable("x2", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.add_constraint(Constraint("c2", {"x1": 1.0, "x2": -2.0}, ConstraintSense.LESS_EQUAL, 0.0))
    m.set_objective(Objective({"x1": 2.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))

    sol = m.solve()

    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.variable_values["x1"] == pytest.approx(8.0 / 3.0, abs=1e-6)
    assert sol.variable_values["x2"] == pytest.approx(4.0 / 3.0, abs=1e-6)
    assert sol.objective_value == pytest.approx(20.0 / 3.0, abs=1e-6)


# ── TEST 9: Multiple Ratio Ties ──────────────────────────────────────────────


def test_multiple_ratio_ties():
    """LP where multiple basic rows achieve the identical minimum ratio:
    maximize 3·x1
    s.t.
        x1 + x2 <= 4    (row 0: slack s1)
        x1 + 2·x2 <= 4  (row 1: slack s2)
        x1, x2 >= 0
    When x1 enters:
        row 0 ratio: 4/1 = 4
        row 1 ratio: 4/1 = 4
    Bland's rule breaks the tie deterministically (s1 has smaller index than s2).
    Optimal: x1 = 4, x2 = 0, z = 12.
    """
    m = Model(name="ratio_ties")
    m.add_variable(Variable("x1", lower_bound=0.0))
    m.add_variable(Variable("x2", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.add_constraint(Constraint("c2", {"x1": 1.0, "x2": 2.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.set_objective(Objective({"x1": 3.0}, ObjectiveSense.MAXIMIZE))

    sol = m.solve()

    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.variable_values["x1"] == pytest.approx(4.0, abs=1e-6)
    assert sol.variable_values["x2"] == pytest.approx(0.0, abs=1e-6)
    assert sol.objective_value == pytest.approx(12.0, abs=1e-6)


# ── TEST 10: Singular Basis ──────────────────────────────────────────────────


def test_singular_basis():
    """A singular basis must raise SingularBasisError."""
    m = Model("singular")
    m.add_variable(Variable("x1"))
    m.add_variable(Variable("x2"))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 2.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.add_constraint(Constraint("c2", {"x1": 2.0, "x2": 4.0}, ConstraintSense.LESS_EQUAL, 8.0))
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))

    sf = to_standard_form(m)
    solver = RevisedSimplexSolver()

    # Pass linearly dependent columns [0, 1] as initial basis
    with pytest.raises(SingularBasisError, match="singular"):
        solver._simplex_loop(
            A=sf.A,
            b=sf.b,
            c=sf.c,
            basis_indices=[0, 1],
            nonbasic_indices=[2, 3],
            sense=sf.objective_sense,
            tolerances=SimplexTolerances(),
        )


# ── TEST 11: Negative RHS Normalization ──────────────────────────────────────


def test_negative_rhs():
    """Negative RHS constraints normalized via Stage 2.5 and solved:
    maximize 3·x1 + 2·x2
    s.t.
        -x1 - x2 >= -4   (equivalent to x1 + x2 <= 4)
        2·x1 + x2 <= 5
        x1, x2 >= 0
    Expected optimum: x1 = 1, x2 = 3, z = 9.
    """
    m = Model(name="neg_rhs_lp")
    m.add_variable(Variable("x1", lower_bound=0.0))
    m.add_variable(Variable("x2", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": -1.0, "x2": -1.0}, ConstraintSense.GREATER_EQUAL, -4.0))
    m.add_constraint(Constraint("c2", {"x1": 2.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.set_objective(Objective({"x1": 3.0, "x2": 2.0}, ObjectiveSense.MAXIMIZE))

    sol = m.solve()

    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.variable_values["x1"] == pytest.approx(1.0, abs=1e-6)
    assert sol.variable_values["x2"] == pytest.approx(3.0, abs=1e-6)
    assert sol.objective_value == pytest.approx(9.0, abs=1e-6)


# ── TEST 12: MAX and MIN Consistency ─────────────────────────────────────────


def test_max_and_min_consistency():
    """max 3x1 + 2x2 and min -3x1 - 2x2 on the same region must yield the same x*."""
    m_max = Model(name="max_model")
    m_max.add_variable(Variable("x1", lower_bound=0.0))
    m_max.add_variable(Variable("x2", lower_bound=0.0))
    m_max.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m_max.add_constraint(Constraint("c2", {"x1": 2.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m_max.set_objective(Objective({"x1": 3.0, "x2": 2.0}, ObjectiveSense.MAXIMIZE))

    m_min = Model(name="min_model")
    m_min.add_variable(Variable("x1", lower_bound=0.0))
    m_min.add_variable(Variable("x2", lower_bound=0.0))
    m_min.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m_min.add_constraint(Constraint("c2", {"x1": 2.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m_min.set_objective(Objective({"x1": -3.0, "x2": -2.0}, ObjectiveSense.MINIMIZE))

    sol_max = m_max.solve()
    sol_min = m_min.solve()

    assert sol_max.status is SolutionStatus.OPTIMAL
    assert sol_min.status is SolutionStatus.OPTIMAL
    assert sol_max.variable_values["x1"] == pytest.approx(sol_min.variable_values["x1"])
    assert sol_max.variable_values["x2"] == pytest.approx(sol_min.variable_values["x2"])
    assert sol_max.objective_value == pytest.approx(-sol_min.objective_value)


# ── TEST 13: Numerical Tolerances ────────────────────────────────────────────


def test_numerical_tolerance():
    """Values near tolerance boundaries behave deterministically."""
    m = Model(name="tol_model")
    m.add_variable(Variable("x1", lower_bound=0.0))
    # RHS is slightly positive (1e-6)
    m.add_constraint(Constraint("c1", {"x1": 1.0}, ConstraintSense.GREATER_EQUAL, 1e-6))
    m.set_objective(Objective({"x1": 1.0}, ObjectiveSense.MINIMIZE))

    tol = SimplexTolerances(feasibility_tolerance=1e-7, reduced_cost_tolerance=1e-7)
    sol = m.solve(tolerances=tol)

    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.variable_values["x1"] == pytest.approx(1e-6, abs=1e-8)


# ── TEST 14: Iteration Limit ─────────────────────────────────────────────────


def test_iteration_limit():
    """Setting max_iterations < needed pivots returns ITERATION_LIMIT."""
    m = Model(name="iter_limit_model")
    m.add_variable(Variable("x1", lower_bound=0.0))
    m.add_variable(Variable("x2", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.add_constraint(Constraint("c2", {"x1": 2.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.set_objective(Objective({"x1": 3.0, "x2": 2.0}, ObjectiveSense.MAXIMIZE))

    # This problem takes 2 pivots to solve; limit to 1
    tol = SimplexTolerances(max_iterations=1)
    sol = m.solve(tolerances=tol)

    assert sol.status is SolutionStatus.ITERATION_LIMIT
    assert sol.iterations == 1
    assert "iteration limit" in sol.message.lower()


# ── TEST 15: Beale (1955) Degenerate Cycling & Anti-Cycling Safeguard ────────


def test_beale_cycling_safeguard():
    """Beale's (1955) degenerate problem cycles under pure Dantzig but terminates with safeguards.

    Verifies:
    1. Reference Bland rule terminates in 6 pivots with z* = -1.25.
    2. Dantzig + safeguards terminates cleanly with z* = -1.25.
    3. Anti-cycling safeguard is demonstrably triggered.
    """
    m = Model(name="beale_cycling")
    x1 = Variable("x1", lower_bound=0.0)
    x2 = Variable("x2", lower_bound=0.0)
    x3 = Variable("x3", lower_bound=0.0)
    x4 = Variable("x4", lower_bound=0.0)
    for v in [x1, x2, x3, x4]:
        m.add_variable(v)
    m.set_objective(Objective({"x1": -0.75, "x2": 20.0, "x3": -0.5, "x4": 6.0}, ObjectiveSense.MINIMIZE))
    m.add_constraint(Constraint("c1", {"x1": 0.25, "x2": -8.0, "x3": -1.0, "x4": 9.0}, ConstraintSense.LESS_EQUAL, 0.0))
    m.add_constraint(Constraint("c2", {"x1": 0.5, "x2": -12.0, "x3": -0.5, "x4": 3.0}, ConstraintSense.LESS_EQUAL, 0.0))
    m.add_constraint(Constraint("c3", {"x3": 1.0}, ConstraintSense.LESS_EQUAL, 1.0))

    # 1. Reference Bland
    s_bland = RevisedSimplexSolver(pricing="bland")
    sol_b = s_bland.solve(m)
    assert sol_b.status == SolutionStatus.OPTIMAL
    assert sol_b.objective_value == pytest.approx(-1.25, abs=1e-7)
    assert sol_b.iterations == 6

    # 2. Dantzig with anti-cycling safeguards
    s_dantzig = RevisedSimplexSolver(pricing="dantzig")
    sol_d = s_dantzig.solve(m)
    assert sol_d.status == SolutionStatus.OPTIMAL
    assert sol_d.objective_value == pytest.approx(-1.25, abs=1e-7)
    assert sol_d.iterations <= 20
    assert s_dantzig.fallback_triggered is True

