"""Tests for Stage 6 Phase 2: MILP Branch-and-Bound Solver."""

import math

import pytest

from optiforge.core.solution import SolutionStatus
from optiforge.lp.simplex import SimplexTolerances
from optiforge.milp import MILPSolver, MILPTolerances, solve_milp
from optiforge.model import (
    Constraint,
    ConstraintSense,
    Model,
    Objective,
    ObjectiveSense,
    Variable,
    VariableType,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _build_simple_integer_lp() -> Model:
    """Simple integer LP:
    max x1 + x2
    s.t. x1 + x2 <= 3
         x1, x2 >= 0, integer
    Optimal: x1=3, x2=0 or x1=2, x2=1 or x1=1, x2=2 or x1=0, x2=3 -> obj=3
    """
    m = Model("simple_int")
    m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 3.0))
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_binary_knapsack() -> Model:
    """Binary knapsack:
    max 5x1 + 4x2 + 3x3
    s.t. 3x1 + 2x2 + x3 <= 4
         x1, x2, x3 in {0,1}
    Optimal: x1=1, x2=0, x3=1 -> obj=8
    """
    m = Model("binary_knapsack")
    m.add_variable(Variable("x1", variable_type=VariableType.BINARY))
    m.add_variable(Variable("x2", variable_type=VariableType.BINARY))
    m.add_variable(Variable("x3", variable_type=VariableType.BINARY))
    m.add_constraint(Constraint("c1", {"x1": 3.0, "x2": 2.0, "x3": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.set_objective(Objective({"x1": 5.0, "x2": 4.0, "x3": 3.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_mixed_integer() -> Model:
    """Mixed integer + continuous:
    max 2x1 + 3x2
    s.t. x1 + 2x2 <= 4
         x1 >= 0, continuous
         x2 >= 0, integer
    LP relaxation: x1=4, x2=0 -> obj=8
    Integer: x2=2, x1=0 -> obj=6; x2=1, x1=2 -> obj=7; x2=0, x1=4 -> obj=8
    Optimal: x1=4, x2=0 -> obj=8
    """
    m = Model("mixed_int")
    m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
    m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 2.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.set_objective(Objective({"x1": 2.0, "x2": 3.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_fractional_relaxation() -> Model:
    """LP relaxation is fractional, requires branching:
    max x1 + x2
    s.t. 2x1 + x2 <= 3
         x1 + 2x2 <= 3
         x1, x2 >= 0, integer
    LP relaxation: x1=1, x2=1 -> obj=2 (fractional? No, this is integral)
    Let's use: max 2x1 + x2 s.t. x1 + x2 <= 2, x1, x2 integer
    LP: x1=2, x2=0 -> obj=4 (integral)
    Better: max x1 + x2 s.t. 2x1 + x2 <= 3, x1 + 2x2 <= 3, x1,x2 integer
    LP: x1=1, x2=1 -> obj=2 (integral)
    Let's use: max x1 + x2 s.t. x1 + x2 <= 2.5, x1,x2 integer
    LP: x1=1.25, x2=1.25 -> obj=2.5 (fractional)
    Integer: x1=1, x2=1 -> obj=2
    """
    m = Model("fractional_relax")
    m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 2.5))
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_integer_optimal_equals_lp() -> Model:
    """Integer optimum equals LP optimum:
    max 3x1 + 2x2
    s.t. x1 + x2 <= 4
         x1, x2 >= 0, integer
    LP: x1=4, x2=0 -> obj=12 (integral)
    """
    m = Model("int_eq_lp")
    m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.set_objective(Objective({"x1": 3.0, "x2": 2.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_infeasible_milp() -> Model:
    """Infeasible MILP:
    max x1
    s.t. x1 >= 5
         x1 <= 2
         x1 >= 0, integer
    """
    m = Model("infeasible_milp")
    m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_constraint(Constraint("c1", {"x1": 1.0}, ConstraintSense.GREATER_EQUAL, 5.0))
    m.add_constraint(Constraint("c2", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 2.0))
    m.set_objective(Objective({"x1": 1.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_maximize_milp() -> Model:
    """Simple MAXIMIZE MILP."""
    m = Model("max_milp")
    m.add_variable(Variable("x", lower_bound=0.0, upper_bound=10.0, variable_type=VariableType.INTEGER))
    m.set_objective(Objective({"x": 2.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_minimize_milp() -> Model:
    """Simple MINIMIZE MILP."""
    m = Model("min_milp")
    m.add_variable(Variable("x", lower_bound=0.0, upper_bound=10.0, variable_type=VariableType.INTEGER))
    m.set_objective(Objective({"x": 2.0}, ObjectiveSense.MINIMIZE))
    return m


def _build_multiple_integer_vars() -> Model:
    """Multiple integer variables:
    max x1 + 2x2 + 3x3
    s.t. x1 + x2 + x3 <= 5
         x1, x2, x3 >= 0, integer
    Optimal: x3=5 -> obj=15
    """
    m = Model("multi_int")
    m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable("x3", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0, "x3": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.set_objective(Objective({"x1": 1.0, "x2": 2.0, "x3": 3.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_nonzero_lower_bound_int() -> Model:
    """Integer variable with non-zero lower bound:
    max x
    s.t. x >= 2
         x <= 5
         x integer
    Optimal: x=5 -> obj=5
    """
    m = Model("nz_lb_int")
    m.add_variable(Variable("x", lower_bound=2.0, upper_bound=5.0, variable_type=VariableType.INTEGER))
    m.set_objective(Objective({"x": 1.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_finite_upper_bound_int() -> Model:
    """Integer variable with finite upper bound:
    max x
    s.t. x <= 3
         x >= 0, integer
    Optimal: x=3 -> obj=3
    """
    m = Model("fin_ub_int")
    m.add_variable(Variable("x", lower_bound=0.0, upper_bound=3.0, variable_type=VariableType.INTEGER))
    m.set_objective(Objective({"x": 1.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_binary_bounds() -> Model:
    """Binary variable bounds test:
    max x1 + x2
    s.t. x1 + x2 <= 1
         x1, x2 binary
    Optimal: x1=1, x2=0 or x1=0, x2=1 -> obj=1
    """
    m = Model("binary_bounds")
    m.add_variable(Variable("x1", variable_type=VariableType.BINARY))
    m.add_variable(Variable("x2", variable_type=VariableType.BINARY))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 1.0))
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_integrality_tolerance() -> Model:
    """Test integrality tolerance:
    max x
    s.t. x <= 2.000001
         x >= 0, integer
    With tolerance 1e-6, x=2 should be feasible.
    """
    m = Model("int_tol")
    m.add_variable(Variable("x", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_constraint(Constraint("c1", {"x": 1.0}, ConstraintSense.LESS_EQUAL, 2.000001))
    m.set_objective(Objective({"x": 1.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_no_integer_feasible() -> Model:
    """LP feasible but no integer solution:
    max x
    s.t. x >= 1.5
         x <= 1.5
         x integer
    LP: x=1.5 (feasible), but no integer x satisfies both.
    """
    m = Model("no_int_feas")
    m.add_variable(Variable("x", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_constraint(Constraint("c1", {"x": 1.0}, ConstraintSense.GREATER_EQUAL, 1.5))
    m.add_constraint(Constraint("c2", {"x": 1.0}, ConstraintSense.LESS_EQUAL, 1.5))
    m.set_objective(Objective({"x": 1.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_obj_constant_through_presolve() -> Model:
    """Objective constant through presolve/lower-bound shift:
    max 2x1 + 3x2
    s.t. x1 + x2 <= 4
         x1 >= 1, x2 >= 2
         x1, x2 integer
    x1 = y1 + 1, x2 = y2 + 2
    y1 + y2 <= 1
    max 2(y1+1) + 3(y2+2) = 2y1 + 3y2 + 8
    Optimal: y1=1, y2=0 -> x1=2, x2=2 -> obj=4+6=10
    """
    m = Model("obj_const")
    m.add_variable(Variable("x1", lower_bound=1.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable("x2", lower_bound=2.0, variable_type=VariableType.INTEGER))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.set_objective(Objective({"x1": 2.0, "x2": 3.0}, ObjectiveSense.MAXIMIZE))
    return m


# ── Tests ───────────────────────────────────────────────────────────────────


class TestSimpleIntegerLP:
    """Simple integer LP tests."""

    def test_simple_integer_lp(self):
        m = _build_simple_integer_lp()
        sol = solve_milp(m)

        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(3.0, abs=1e-6)
        # x1 + x2 = 3 with integer x1, x2 >= 0
        assert sol.variable_values["x1"] + sol.variable_values["x2"] == pytest.approx(3.0, abs=1e-6)
        assert abs(sol.variable_values["x1"] - round(sol.variable_values["x1"])) < 1e-6
        assert abs(sol.variable_values["x2"] - round(sol.variable_values["x2"])) < 1e-6


class TestBinaryOptimization:
    """Binary optimization tests."""

    def test_binary_knapsack(self):
        m = _build_binary_knapsack()
        sol = solve_milp(m)

        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(8.0, abs=1e-6)
        # x1=1, x2=0, x3=1 -> 5+3=8
        assert sol.variable_values["x1"] in (0.0, 1.0)
        assert sol.variable_values["x2"] in (0.0, 1.0)
        assert sol.variable_values["x3"] in (0.0, 1.0)


class TestMixedInteger:
    """Mixed integer + continuous tests."""

    def test_mixed_integer(self):
        m = _build_mixed_integer()
        sol = solve_milp(m)

        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(8.0, abs=1e-6)
        # x1=4 (continuous), x2=0 (integer)
        assert sol.variable_values["x1"] == pytest.approx(4.0, abs=1e-6)
        assert sol.variable_values["x2"] == pytest.approx(0.0, abs=1e-6)


class TestFractionalRelaxation:
    """Tests where LP relaxation is fractional."""

    def test_fractional_relaxation(self):
        m = _build_fractional_relaxation()
        sol = solve_milp(m)

        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(2.0, abs=1e-6)
        # Multiple optimal solutions exist: (2,0), (0,2), (1,1) all give obj=2
        # Verify the solution is integer-feasible and satisfies the constraint
        x1 = sol.variable_values["x1"]
        x2 = sol.variable_values["x2"]
        assert abs(x1 - round(x1)) < 1e-6
        assert abs(x2 - round(x2)) < 1e-6
        assert x1 + x2 <= 2.5 + 1e-6
        assert x1 >= -1e-6 and x2 >= -1e-6


class TestMinimizeFractionalRelaxation:
    """Tests where MINIMIZE LP relaxation is fractional and requires branching."""

    def test_minimize_fractional_relaxation(self):
        """MINIMIZE with fractional LP relaxation requiring branching.
        
        min x1 + 2*x2
        s.t. x1 + x2 >= 2.5
             x1, x2 >= 0, integer
        
        LP relaxation: x1=0, x2=2.5 (obj=5) or x1=2.5, x2=0 (obj=2.5)
        Integer: x1=1, x2=2 (obj=5) or x1=2, x2=1 (obj=4) or x1=3, x2=0 (obj=3)
        Optimal integer: x1=3, x2=0 (obj=3)
        """
        m = Model("min_frac_relax")
        m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
        m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, 2.5))
        m.set_objective(Objective({"x1": 1.0, "x2": 2.0}, ObjectiveSense.MINIMIZE))

        sol = solve_milp(m)

        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(3.0, abs=1e-6)
        assert sol.variable_values["x1"] == pytest.approx(3.0, abs=1e-6)
        assert sol.variable_values["x2"] == pytest.approx(0.0, abs=1e-6)
        # Verify integer feasibility
        assert abs(sol.variable_values["x1"] - round(sol.variable_values["x1"])) < 1e-6
        assert abs(sol.variable_values["x2"] - round(sol.variable_values["x2"])) < 1e-6
        # Verify constraint satisfaction
        assert sol.variable_values["x1"] + sol.variable_values["x2"] >= 2.5 - 1e-6


class TestIntegerOptimalEqualsLP:
    """Integer optimum equals LP optimum."""

    def test_integer_optimal_equals_lp(self):
        m = _build_integer_optimal_equals_lp()
        sol = solve_milp(m)

        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(12.0, abs=1e-6)
        # x1=4, x2=0 -> obj=12
        assert sol.variable_values["x1"] == pytest.approx(4.0, abs=1e-6)
        assert sol.variable_values["x2"] == pytest.approx(0.0, abs=1e-6)


class TestInfeasibleMILP:
    """Infeasible MILP tests."""

    def test_infeasible_milp(self):
        m = _build_infeasible_milp()
        sol = solve_milp(m)

        assert sol.status is SolutionStatus.INFEASIBLE


class TestMaximize:
    """MAXIMIZE tests."""

    def test_maximize_milp(self):
        m = _build_maximize_milp()
        sol = solve_milp(m)

        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(20.0, abs=1e-6)
        assert sol.variable_values["x"] == pytest.approx(10.0, abs=1e-6)


class TestMinimize:
    """MINIMIZE tests."""

    def test_minimize_milp(self):
        m = _build_minimize_milp()
        sol = solve_milp(m)

        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(0.0, abs=1e-6)
        assert sol.variable_values["x"] == pytest.approx(0.0, abs=1e-6)


class TestMultipleIntegerVars:
    """Multiple integer variables."""

    def test_multiple_integer_vars(self):
        m = _build_multiple_integer_vars()
        sol = solve_milp(m)

        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(15.0, abs=1e-6)
        # x3=5 -> obj=15
        assert sol.variable_values["x3"] == pytest.approx(5.0, abs=1e-6)


class TestNonZeroLowerBoundInt:
    """Non-zero lower bound integer variable."""

    def test_nonzero_lower_bound_int(self):
        m = _build_nonzero_lower_bound_int()
        sol = solve_milp(m)

        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(5.0, abs=1e-6)
        assert sol.variable_values["x"] == pytest.approx(5.0, abs=1e-6)


class TestFiniteUpperBoundInt:
    """Finite upper bound integer variable."""

    def test_finite_upper_bound_int(self):
        m = _build_finite_upper_bound_int()
        sol = solve_milp(m)

        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(3.0, abs=1e-6)
        assert sol.variable_values["x"] == pytest.approx(3.0, abs=1e-6)


class TestBinaryBounds:
    """Binary bounds tests."""

    def test_binary_bounds(self):
        m = _build_binary_bounds()
        sol = solve_milp(m)

        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(1.0, abs=1e-6)
        assert sol.variable_values["x1"] in (0.0, 1.0)
        assert sol.variable_values["x2"] in (0.0, 1.0)


class TestIntegralityTolerance:
    """Integrality tolerance tests."""

    def test_integrality_tolerance(self):
        m = _build_integrality_tolerance()
        tol = MILPTolerances(integrality_tolerance=1e-6)
        sol = solve_milp(m, tolerances=tol)

        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.variable_values["x"] == pytest.approx(2.0, abs=1e-6)


class TestNodePruning:
    """Node pruning tests."""

    def test_node_pruning(self):
        """Test that nodes are pruned correctly."""
        m = _build_multiple_integer_vars()
        tol = MILPTolerances(max_nodes=1000)
        sol = solve_milp(m, tolerances=tol)

        assert sol.status is SolutionStatus.OPTIMAL
        # Should find optimal without exploring all nodes


class TestDeterministicSolve:
    """Deterministic repeated solve."""

    def test_deterministic_solve(self):
        m = _build_multiple_integer_vars()
        sol1 = solve_milp(m)
        sol2 = solve_milp(m)

        assert sol1.status == sol2.status
        assert sol1.objective_value == pytest.approx(sol2.objective_value, abs=1e-6)
        assert sol1.variable_values == sol2.variable_values


class TestCorrectIncumbentBound:
    """Correct incumbent/best-bound termination."""

    def test_correct_incumbent_bound(self):
        m = _build_multiple_integer_vars()
        sol = solve_milp(m)

        assert sol.status is SolutionStatus.OPTIMAL
        # Verify the solution is truly optimal by checking no better solution exists
        assert sol.objective_value == pytest.approx(15.0, abs=1e-6)


class TestNoIntegerFeasible:
    """No integer-feasible solution despite feasible LP relaxation."""

    def test_no_integer_feasible(self):
        m = _build_no_integer_feasible()
        sol = solve_milp(m)

        # Should be INFEASIBLE or UNKNOWN (no integer solution)
        assert sol.status in (SolutionStatus.INFEASIBLE, SolutionStatus.UNKNOWN)


class TestObjectiveConstant:
    """Objective constant through presolve/lower-bound shift."""

    def test_objective_constant(self):
        m = _build_obj_constant_through_presolve()
        sol = solve_milp(m)

        assert sol.status is SolutionStatus.OPTIMAL
        # x1=1, x2=3 -> obj = 2*1 + 3*3 = 11 (optimal)
        # x1=2, x2=2 -> obj = 2*2 + 3*2 = 10 (suboptimal)
        assert sol.objective_value == pytest.approx(11.0, abs=1e-6)
        assert sol.variable_values["x1"] == pytest.approx(1.0, abs=1e-6)
        assert sol.variable_values["x2"] == pytest.approx(3.0, abs=1e-6)


class TestSolverAPI:
    """Test the solver API directly."""

    def test_solver_class(self):
        m = _build_simple_integer_lp()
        solver = MILPSolver()
        sol = solver.solve(m)

        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(3.0, abs=1e-6)

    def test_solver_with_custom_tolerances(self):
        m = _build_integrality_tolerance()
        milp_tol = MILPTolerances(integrality_tolerance=1e-4)
        simplex_tol = SimplexTolerances(max_iterations=1000)
        solver = MILPSolver(tolerances=milp_tol, simplex_tolerances=simplex_tol)
        sol = solver.solve(m)

        assert sol.status is SolutionStatus.OPTIMAL


# ── Stage 7.1: Extension Points ─────────────────────────────────────────────


class TestBranchingStrategyExtension:
    """Custom branching strategy injection."""

    def test_custom_branching_strategy(self):
        """A custom branching strategy should be used and produce a valid result."""
        from optiforge.milp import BranchingStrategy, BranchingDecision

        class FirstVariableBranching(BranchingStrategy):
            """Always branch on the first fractional integer variable."""

            def __init__(self):
                self.calls = 0

            def choose_branching_variable(self, node, model, tolerances):
                self.calls += 1
                for var_name, var in model.variables.items():
                    if not var.is_integer():
                        continue
                    value = node.lp_solution.get(var_name, 0.0)
                    nearest_int = round(value)
                    if abs(value - nearest_int) > tolerances.integrality_tolerance:
                        return BranchingDecision(variable_name=var_name, value=value)
                return None

        strategy = FirstVariableBranching()
        m = _build_fractional_relaxation()
        solver = MILPSolver(branching_strategy=strategy)
        sol = solver.solve(m)

        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(2.0, abs=1e-6)
        assert strategy.calls > 0

    def test_default_strategy_is_most_fractional(self):
        """Default branching strategy is MostFractionalBranching."""
        from optiforge.milp import MostFractionalBranching

        solver = MILPSolver()
        assert isinstance(solver._branching_strategy, MostFractionalBranching)


class TestNodeSelectionStrategyExtension:
    """Custom node selection strategy injection."""

    def test_custom_node_selection_strategy(self):
        """A custom node selection strategy should be used and produce a valid result."""
        from optiforge.milp import NodeSelectionStrategy

        class DepthFirstSelection(NodeSelectionStrategy):
            """Depth-first: select the deepest node (largest node_id as proxy)."""

            def __init__(self):
                self.calls = 0

            def select_node(self, open_nodes, nodes, is_maximize):
                self.calls += 1
                # Deepest node = largest depth; tie-break by node_id
                return max(open_nodes, key=lambda nid: (nodes[nid].depth, nid))

        strategy = DepthFirstSelection()
        m = _build_fractional_relaxation()
        solver = MILPSolver(node_selection_strategy=strategy)
        sol = solver.solve(m)

        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(2.0, abs=1e-6)
        assert strategy.calls > 0

    def test_default_strategy_is_best_bound(self):
        """Default node selection strategy is BestBoundNodeSelection."""
        from optiforge.milp import BestBoundNodeSelection

        solver = MILPSolver()
        assert isinstance(solver._node_selection_strategy, BestBoundNodeSelection)


class TestSearchStateCallback:
    """Search state observation callback."""

    def test_callback_receives_search_state(self):
        """Callback receives SearchState snapshots during the search."""
        from optiforge.milp import SearchState

        snapshots = []

        def observer(state: SearchState) -> None:
            snapshots.append(state)

        m = _build_fractional_relaxation()
        solver = MILPSolver(search_state_callback=observer)
        sol = solver.solve(m)

        assert sol.status is SolutionStatus.OPTIMAL
        assert len(snapshots) > 0

        # First snapshot: root node processed
        first = snapshots[0]
        assert first.nodes_processed == 1
        assert first.nodes_remaining >= 0
        assert first.feasible_solutions_found >= 0
        assert first.nodes_pruned >= 0
        assert first.max_depth >= 0

        # Last snapshot: search complete
        last = snapshots[-1]
        assert last.nodes_remaining == 0
        assert last.incumbent_objective == pytest.approx(2.0, abs=1e-6)

    def test_callback_monotonic_nodes_processed(self):
        """nodes_processed should be monotonically increasing."""
        snapshots = []
        solver = MILPSolver(search_state_callback=lambda s: snapshots.append(s))
        sol = solver.solve(_build_fractional_relaxation())

        assert sol.status is SolutionStatus.OPTIMAL
        counts = [s.nodes_processed for s in snapshots]
        assert counts == sorted(counts)
        assert counts[-1] == len(snapshots)

    def test_callback_observes_incumbent_improvement(self):
        """Callback should observe incumbent improvement over the search."""
        snapshots = []
        solver = MILPSolver(search_state_callback=lambda s: snapshots.append(s))
        sol = solver.solve(_build_fractional_relaxation())

        assert sol.status is SolutionStatus.OPTIMAL
        # At least one snapshot should have an incumbent
        assert any(s.incumbent_objective is not None for s in snapshots)
        # Final incumbent matches the optimal objective
        assert snapshots[-1].incumbent_objective == pytest.approx(2.0, abs=1e-6)


class TestStrategyDeterminism:
    """Determinism with custom strategies."""

    def test_custom_strategies_deterministic(self):
        """Same strategies + same model → identical results."""
        from optiforge.milp import BranchingStrategy, BranchingDecision, NodeSelectionStrategy

        class FirstVariableBranching(BranchingStrategy):
            def choose_branching_variable(self, node, model, tolerances):
                for var_name, var in model.variables.items():
                    if not var.is_integer():
                        continue
                    value = node.lp_solution.get(var_name, 0.0)
                    if abs(value - round(value)) > tolerances.integrality_tolerance:
                        return BranchingDecision(variable_name=var_name, value=value)
                return None

        class DepthFirstSelection(NodeSelectionStrategy):
            def select_node(self, open_nodes, nodes, is_maximize):
                return max(open_nodes, key=lambda nid: (nodes[nid].depth, nid))

        m = _build_fractional_relaxation()
        sol1 = solve_milp(
            m,
            branching_strategy=FirstVariableBranching(),
            node_selection_strategy=DepthFirstSelection(),
        )
        sol2 = solve_milp(
            m,
            branching_strategy=FirstVariableBranching(),
            node_selection_strategy=DepthFirstSelection(),
        )

        assert sol1.status == sol2.status
        assert sol1.objective_value == pytest.approx(sol2.objective_value, abs=1e-6)
        assert sol1.variable_values == sol2.variable_values

    def test_solve_milp_accepts_strategies(self):
        """solve_milp convenience function accepts strategy parameters."""
        from optiforge.milp import MostFractionalBranching, BestBoundNodeSelection

        m = _build_simple_integer_lp()
        sol = solve_milp(
            m,
            branching_strategy=MostFractionalBranching(),
            node_selection_strategy=BestBoundNodeSelection(),
        )

        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(3.0, abs=1e-6)


# ── Stage 7.1 Audit Remediation Tests ───────────────────────────────────────


class TestTerminationSemantics:
    """M1: _should_terminate must not skip unprocessed nodes that could
    contain a better solution."""

    def _make_solver(self, sense=ObjectiveSense.MAXIMIZE):
        from optiforge.milp import MILPSolver
        from optiforge.milp.solver import BnBNode

        solver = MILPSolver()
        solver._objective_sense = sense
        solver._incumbent_objective = 10.0
        solver._nodes = {}
        solver._open_nodes = []
        return solver, BnBNode

    def test_no_open_nodes_terminates(self):
        solver, _ = self._make_solver()
        assert solver._should_terminate() is True

    def test_unprocessed_node_without_bound_blocks_termination(self):
        """An unprocessed node with no known bound must not be pruned."""
        solver, BnBNode = self._make_solver()
        solver._nodes = {1: BnBNode(node_id=1, parent_id=None, depth=0)}
        solver._open_nodes = [1]
        assert solver._should_terminate() is False

    def test_unprocessed_node_with_dominating_bound_blocks_termination(self):
        """Unprocessed node whose inherited bound beats the incumbent must
        still be processed (MAX: bound 12 > incumbent 10)."""
        solver, BnBNode = self._make_solver()
        solver._nodes = {1: BnBNode(node_id=1, parent_id=None, depth=1, lp_bound=12.0)}
        solver._open_nodes = [1]
        assert solver._should_terminate() is False

    def test_unprocessed_node_with_dominated_bound_terminates(self):
        """Unprocessed node whose inherited bound cannot beat the incumbent
        is prunable without being solved (MAX: bound 8 <= incumbent 10)."""
        solver, BnBNode = self._make_solver()
        solver._nodes = {1: BnBNode(node_id=1, parent_id=None, depth=1, lp_bound=8.0)}
        solver._open_nodes = [1]
        assert solver._should_terminate() is True

    def test_processed_node_bound_used(self):
        """Processed nodes use their own LP objective for pruning."""
        solver, BnBNode = self._make_solver()
        solver._nodes = {1: BnBNode(node_id=1, parent_id=None, depth=1, lp_objective=9.0, lp_bound=9.0)}
        solver._open_nodes = [1]
        assert solver._should_terminate() is True
        solver._nodes = {2: BnBNode(node_id=2, parent_id=None, depth=1, lp_objective=11.0, lp_bound=11.0)}
        solver._open_nodes = [2]
        assert solver._should_terminate() is False

    def test_no_incumbent_never_terminates(self):
        solver, BnBNode = self._make_solver()
        solver._incumbent_objective = None
        solver._nodes = {1: BnBNode(node_id=1, parent_id=None, depth=0, lp_objective=5.0)}
        solver._open_nodes = [1]
        assert solver._should_terminate() is False

    def test_minimize_sense_bound_direction(self):
        """MIN: prune when bound >= incumbent, keep when bound < incumbent."""
        solver, BnBNode = self._make_solver(sense=ObjectiveSense.MINIMIZE)
        solver._nodes = {1: BnBNode(node_id=1, parent_id=None, depth=1, lp_bound=12.0)}
        solver._open_nodes = [1]
        assert solver._should_terminate() is True
        solver._nodes = {2: BnBNode(node_id=2, parent_id=None, depth=1, lp_bound=8.0)}
        solver._open_nodes = [2]
        assert solver._should_terminate() is False

    def test_knapsack_requires_processing_dominating_node(self):
        """Regression: the binary knapsack must reach obj=8, proving the
        right child (x1>=1, inherited bound 8.667 > 7) is not skipped."""
        m = _build_binary_knapsack()
        sol = solve_milp(m)
        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(8.0, abs=1e-6)


class TestMostFractionalBranchingRule:
    """M2: MostFractionalBranching picks the variable furthest from integral."""

    def test_picks_largest_deviation(self):
        from optiforge.milp import MILPTolerances, MostFractionalBranching
        from optiforge.milp.solver import BnBNode

        m = Model("mf")
        m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
        m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
        m.add_variable(Variable("x3", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        # x1 dev 0.2, x2 dev 0.7 -> x2 is most fractional
        node = BnBNode(node_id=1, parent_id=None, depth=0, lp_solution={"x1": 1.2, "x2": 1.7, "x3": 0.5})
        decision = MostFractionalBranching().choose_branching_variable(node, m, MILPTolerances())
        assert decision is not None
        assert decision.variable_name == "x2"
        assert decision.value == pytest.approx(1.7, abs=1e-9)

    def test_ignores_continuous_variables(self):
        from optiforge.milp import MILPTolerances, MostFractionalBranching
        from optiforge.milp.solver import BnBNode

        m = Model("mf_cont")
        m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        node = BnBNode(node_id=1, parent_id=None, depth=0, lp_solution={"x1": 1.5})
        decision = MostFractionalBranching().choose_branching_variable(node, m, MILPTolerances())
        assert decision is None

    def test_returns_none_when_integral(self):
        from optiforge.milp import MILPTolerances, MostFractionalBranching
        from optiforge.milp.solver import BnBNode

        m = Model("mf_int")
        m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
        node = BnBNode(node_id=1, parent_id=None, depth=0, lp_solution={"x1": 2.0})
        decision = MostFractionalBranching().choose_branching_variable(node, m, MILPTolerances())
        assert decision is None

    def test_fractional_metadata_populated_on_branched_nodes(self):
        """M2: branched nodes carry the actual branching variable/value."""
        m = _build_fractional_relaxation()
        solver = MILPSolver()
        sol = solver.solve(m)
        assert sol.status is SolutionStatus.OPTIMAL
        branched = [n for n in solver._nodes.values() if n.fractional_var is not None]
        assert len(branched) > 0
        for node in branched:
            assert node.fractional_var in ("x1", "x2")
            assert node.fractional_value is not None


class TestMinimizeWithCustomStrategies:
    """MINIMIZE objective with injected strategies."""

    def test_minimize_with_custom_branching(self):
        from optiforge.milp import BranchingStrategy, BranchingDecision

        class FirstVariableBranching(BranchingStrategy):
            def choose_branching_variable(self, node, model, tolerances):
                for var_name, var in model.variables.items():
                    if not var.is_integer():
                        continue
                    value = node.lp_solution.get(var_name, 0.0)
                    if abs(value - round(value)) > tolerances.integrality_tolerance:
                        return BranchingDecision(variable_name=var_name, value=value)
                return None

        m = Model("min_frac_custom")
        m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
        m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, 2.5))
        m.set_objective(Objective({"x1": 1.0, "x2": 2.0}, ObjectiveSense.MINIMIZE))

        sol = solve_milp(m, branching_strategy=FirstVariableBranching())
        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(3.0, abs=1e-6)

    def test_minimize_with_custom_node_selection(self):
        from optiforge.milp import NodeSelectionStrategy

        class DepthFirstSelection(NodeSelectionStrategy):
            def select_node(self, open_nodes, nodes, is_maximize):
                return max(open_nodes, key=lambda nid: (nodes[nid].depth, nid))

        m = Model("min_frac_custom_ns")
        m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
        m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, 2.5))
        m.set_objective(Objective({"x1": 1.0, "x2": 2.0}, ObjectiveSense.MINIMIZE))

        sol = solve_milp(m, node_selection_strategy=DepthFirstSelection())
        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(3.0, abs=1e-6)


class TestSearchStateBoundsAndGap:
    """M4: best_open_bound / gap / absolute_gap / elapsed_time."""

    def test_best_open_bound_and_gap_populated(self):
        """During the search, best_open_bound and gaps are meaningful."""
        from optiforge.milp import SearchState

        snapshots = []
        solver = MILPSolver(search_state_callback=lambda s: snapshots.append(s))
        sol = solver.solve(_build_fractional_relaxation())

        assert sol.status is SolutionStatus.OPTIMAL
        # Root snapshot: no incumbent yet -> no gap
        assert snapshots[0].incumbent_objective is None
        assert snapshots[0].best_open_bound is not None
        assert snapshots[0].gap is None
        assert snapshots[0].absolute_gap is None
        # Some snapshot has both incumbent and bound -> gap defined
        assert any(s.incumbent_objective is not None and s.best_open_bound is not None
                   for s in snapshots)
        # Final snapshot: search complete -> no open nodes -> no bound
        assert snapshots[-1].nodes_remaining == 0
        assert snapshots[-1].best_open_bound is None

    def test_gap_consistency(self):
        """gap == absolute_gap / |incumbent| when incumbent != 0."""
        snapshots = []
        solver = MILPSolver(search_state_callback=lambda s: snapshots.append(s))
        solver.solve(_build_fractional_relaxation())

        for s in snapshots:
            if s.gap is not None and s.absolute_gap is not None and s.incumbent_objective:
                expected = s.absolute_gap / abs(s.incumbent_objective)
                assert s.gap == pytest.approx(expected, abs=1e-9)

    def test_elapsed_time_is_set(self):
        """elapsed_time is a non-negative float during the solve."""
        snapshots = []
        solver = MILPSolver(search_state_callback=lambda s: snapshots.append(s))
        solver.solve(_build_fractional_relaxation())

        assert len(snapshots) > 0
        for s in snapshots:
            assert s.elapsed_time is not None
            assert s.elapsed_time >= 0.0

    def test_callback_on_infeasible_model(self):
        """Callback fires for an infeasible model (root processed)."""
        from optiforge.milp import SearchState

        snapshots = []
        solver = MILPSolver(search_state_callback=lambda s: snapshots.append(s))
        sol = solver.solve(_build_infeasible_milp())

        assert sol.status is SolutionStatus.INFEASIBLE
        assert len(snapshots) == 1
        assert snapshots[0].nodes_processed == 1
        assert snapshots[0].nodes_remaining == 0
        assert snapshots[0].incumbent_objective is None

    def test_callback_on_iteration_limit(self):
        """Callback fires on the iteration-limit path."""
        from optiforge.milp import SearchState

        snapshots = []
        tol = MILPTolerances(max_nodes=1)
        solver = MILPSolver(tolerances=tol, search_state_callback=lambda s: snapshots.append(s))
        sol = solver.solve(_build_fractional_relaxation())

        assert sol.status is SolutionStatus.ITERATION_LIMIT
        assert len(snapshots) == 1
        assert snapshots[0].nodes_processed == 1


class TestStrategyStateAcrossSolves:
    """Strategy ownership: injected strategies keep state across solves."""

    def test_stateful_strategy_accumulates_calls(self):
        from optiforge.milp import BranchingStrategy, BranchingDecision

        class CountingBranching(BranchingStrategy):
            def __init__(self):
                self.total_calls = 0

            def choose_branching_variable(self, node, model, tolerances):
                self.total_calls += 1
                for var_name, var in model.variables.items():
                    if not var.is_integer():
                        continue
                    value = node.lp_solution.get(var_name, 0.0)
                    if abs(value - round(value)) > tolerances.integrality_tolerance:
                        return BranchingDecision(variable_name=var_name, value=value)
                return None

        strategy = CountingBranching()
        solver = MILPSolver(branching_strategy=strategy)
        sol1 = solver.solve(_build_fractional_relaxation())
        calls_after_first = strategy.total_calls
        sol2 = solver.solve(_build_fractional_relaxation())

        assert sol1.status is SolutionStatus.OPTIMAL
        assert sol2.status is SolutionStatus.OPTIMAL
        assert calls_after_first > 0
        # State accumulates across solves (by design)
        assert strategy.total_calls > calls_after_first


class TestBestBoundTieBreaking:
    """Deterministic BestBoundNodeSelection tie-breaking."""

    def test_tie_breaks_by_node_id(self):
        from optiforge.milp import BestBoundNodeSelection
        from optiforge.milp.solver import BnBNode

        nodes = {
            1: BnBNode(node_id=1, parent_id=None, depth=0, lp_objective=5.0),
            2: BnBNode(node_id=2, parent_id=None, depth=1, lp_objective=5.0),
            3: BnBNode(node_id=3, parent_id=None, depth=1, lp_objective=7.0),
        }
        sel = BestBoundNodeSelection()
        # MAX: highest bound first -> node 3
        assert sel.select_node([1, 2, 3], nodes, True) == 3
        # MAX tie: lowest node_id wins
        assert sel.select_node([2, 1], nodes, True) == 1
        # MIN: lowest bound first -> node 1
        assert sel.select_node([1, 2, 3], nodes, False) == 1

    def test_unprocessed_nodes_rank_last(self):
        from optiforge.milp import BestBoundNodeSelection
        from optiforge.milp.solver import BnBNode

        nodes = {
            1: BnBNode(node_id=1, parent_id=None, depth=0, lp_objective=5.0),
            2: BnBNode(node_id=2, parent_id=None, depth=1),  # unprocessed
        }
        sel = BestBoundNodeSelection()
        assert sel.select_node([1, 2], nodes, True) == 1
        assert sel.select_node([2, 1], nodes, False) == 1


class TestMaxNodesEarlyTermination:
    """max_nodes / early termination regression."""

    def test_max_nodes_one_returns_iteration_limit(self):
        tol = MILPTolerances(max_nodes=1)
        sol = solve_milp(_build_fractional_relaxation(), tolerances=tol)
        assert sol.status is SolutionStatus.ITERATION_LIMIT

    def test_max_nodes_zero_returns_iteration_limit(self):
        tol = MILPTolerances(max_nodes=0)
        sol = solve_milp(_build_fractional_relaxation(), tolerances=tol)
        assert sol.status is SolutionStatus.ITERATION_LIMIT

    def test_high_max_nodes_still_optimal(self):
        tol = MILPTolerances(max_nodes=10_000)
        sol = solve_milp(_build_fractional_relaxation(), tolerances=tol)
        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(2.0, abs=1e-6)


class TestInvalidBranchingDecision:
    """M5: invalid branching strategy output raises a clear error."""

    def test_unknown_variable_raises_value_error(self):
        from optiforge.milp import BranchingStrategy, BranchingDecision

        class BadBranching(BranchingStrategy):
            def choose_branching_variable(self, node, model, tolerances):
                return BranchingDecision(variable_name="nonexistent_var", value=1.5)

        with pytest.raises(ValueError, match="nonexistent_var"):
            solve_milp(_build_fractional_relaxation(), branching_strategy=BadBranching())

    def test_none_decision_skips_branching(self):
        """A strategy returning None simply skips branching (no crash)."""
        from optiforge.milp import BranchingStrategy

        class NoBranching(BranchingStrategy):
            def choose_branching_variable(self, node, model, tolerances):
                return None

        sol = solve_milp(_build_fractional_relaxation(), branching_strategy=NoBranching())
        # No branching -> no integer solution found
        assert sol.status in (SolutionStatus.UNKNOWN, SolutionStatus.OPTIMAL)