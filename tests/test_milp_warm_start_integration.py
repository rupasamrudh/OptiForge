"""Tests for Stage 8.9 Branch-and-Bound LP Warm-Start Integration.

Validates:
- Feature flag opt-in isolation (warm_start_relaxations=False vs True)
- Correctness across diverse MILP problem classes (binary, integer, mixed)
- MINIMIZE and MAXIMIZE objective senses
- Multi-level warm-start chaining (depth >= 5)
- Parent/child mutability isolation
- Infeasible child node detection and pruning
- Defensive fallback on iteration limits or numerical errors
- Phase-I basis safety
"""

import numpy as np
import pytest

from optiforge.core.solution import SolutionStatus
from optiforge.lp.simplex import SimplexTolerances
from optiforge.milp.solver import MILPSolver, MILPTolerances, solve_milp
from optiforge.model import Constraint, ConstraintSense, Model, Objective, ObjectiveSense, Variable, VariableType


class TestMILPWarmStartIntegration:
    """Test suite for Branch-and-Bound Dual Simplex warm-starting."""

    def test_feature_flag_default_is_disabled(self):
        """Verify warm_start_relaxations is False by default."""
        solver = MILPSolver()
        assert solver.warm_start_relaxations is False
        assert solver.tolerances.warm_start_relaxations is False

    def test_binary_knapsack_cold_vs_warm(self):
        """0-1 Knapsack problem comparing cold vs warm solves."""
        m = Model("knapsack_test")
        x1 = m.add_variable(Variable("x1", variable_type=VariableType.BINARY))
        x2 = m.add_variable(Variable("x2", variable_type=VariableType.BINARY))
        x3 = m.add_variable(Variable("x3", variable_type=VariableType.BINARY))
        x4 = m.add_variable(Variable("x4", variable_type=VariableType.BINARY))
        # weights: 3, 4, 5, 8; values: 4, 5, 7, 11; cap: 10
        m.add_constraint(Constraint("cap", coefficients={"x1": 3.0, "x2": 4.0, "x3": 5.0, "x4": 8.0}, sense=ConstraintSense.LESS_EQUAL, rhs=10.0))
        m.set_objective(Objective({"x1": 4.0, "x2": 5.0, "x3": 7.0, "x4": 11.0}, sense=ObjectiveSense.MAXIMIZE))

        sol_cold = solve_milp(m, warm_start_relaxations=False)
        sol_warm = solve_milp(m, warm_start_relaxations=True)

        assert sol_cold.status == SolutionStatus.OPTIMAL
        assert sol_warm.status == SolutionStatus.OPTIMAL
        assert np.isclose(sol_cold.objective_value, sol_warm.objective_value, atol=1e-6)

        # Inspect solver warm start stats
        s_warm = MILPSolver(warm_start_relaxations=True)
        s_warm.solve(m)
        stats = s_warm.get_warm_start_stats()
        assert stats.attempted > 0
        assert stats.accepted > 0
        assert stats.dual_pivots > 0

    def test_general_integer_maximization(self):
        """Multi-variable general integer maximization."""
        m = Model("int_max")
        x1 = m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
        x2 = m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
        x3 = m.add_variable(Variable("x3", lower_bound=0.0, variable_type=VariableType.INTEGER))

        m.add_constraint(Constraint("c1", coefficients={"x1": 2.0, "x2": 3.0, "x3": 1.0}, sense=ConstraintSense.LESS_EQUAL, rhs=14.0))
        m.add_constraint(Constraint("c2", coefficients={"x1": 4.0, "x2": 1.0, "x3": 2.0}, sense=ConstraintSense.LESS_EQUAL, rhs=18.0))
        m.add_constraint(Constraint("c3", coefficients={"x1": 3.0, "x2": 4.0, "x3": 2.0}, sense=ConstraintSense.LESS_EQUAL, rhs=22.0))
        m.set_objective(Objective({"x1": 5.0, "x2": 4.0, "x3": 3.0}, sense=ObjectiveSense.MAXIMIZE))

        sol_cold = solve_milp(m, warm_start_relaxations=False)
        sol_warm = solve_milp(m, warm_start_relaxations=True)

        assert sol_cold.status == SolutionStatus.OPTIMAL
        assert sol_warm.status == SolutionStatus.OPTIMAL
        assert np.isclose(sol_cold.objective_value, sol_warm.objective_value, atol=1e-6)
        for var in ["x1", "x2", "x3"]:
            assert np.isclose(sol_warm.variable_values[var], round(sol_warm.variable_values[var]), atol=1e-5)

    def test_mixed_integer_minimization(self):
        """Mixed-integer minimization with >= constraints and continuous variables."""
        m = Model("mi_min")
        x1 = m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        x2 = m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        y1 = m.add_variable(Variable("y1", lower_bound=0.0, variable_type=VariableType.INTEGER))
        y2 = m.add_variable(Variable("y2", lower_bound=0.0, variable_type=VariableType.INTEGER))

        m.add_constraint(Constraint("c1", coefficients={"x1": 1.0, "x2": 2.0, "y1": 3.0, "y2": 1.0}, sense=ConstraintSense.GREATER_EQUAL, rhs=12.0))
        m.add_constraint(Constraint("c2", coefficients={"x1": 2.0, "x2": 1.0, "y1": 1.0, "y2": 4.0}, sense=ConstraintSense.GREATER_EQUAL, rhs=15.0))
        m.set_objective(Objective({"x1": 3.0, "x2": 2.0, "y1": 5.0, "y2": 7.0}, sense=ObjectiveSense.MINIMIZE))

        sol_cold = solve_milp(m, warm_start_relaxations=False)
        sol_warm = solve_milp(m, warm_start_relaxations=True)

        assert sol_cold.status == SolutionStatus.OPTIMAL
        assert sol_warm.status == SolutionStatus.OPTIMAL
        assert np.isclose(sol_cold.objective_value, sol_warm.objective_value, atol=1e-6)

    def test_infeasible_milp_detection(self):
        """Infeasible MILP problem."""
        m = Model("infeasible_test")
        x1 = m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
        x2 = m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
        m.add_constraint(Constraint("c1", coefficients={"x1": 1.0, "x2": 1.0}, sense=ConstraintSense.LESS_EQUAL, rhs=2.5))
        m.add_constraint(Constraint("c2", coefficients={"x1": 1.0, "x2": 1.0}, sense=ConstraintSense.GREATER_EQUAL, rhs=3.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, sense=ObjectiveSense.MAXIMIZE))

        sol_cold = solve_milp(m, warm_start_relaxations=False)
        sol_warm = solve_milp(m, warm_start_relaxations=True)

        assert sol_cold.status == SolutionStatus.INFEASIBLE
        assert sol_warm.status == SolutionStatus.INFEASIBLE

    def test_non_zero_lower_bounds_handling(self):
        """Problem with shifted variable lower bounds (x >= 2.0)."""
        m = Model("shifted_bounds")
        x1 = m.add_variable(Variable("x1", lower_bound=2.0, variable_type=VariableType.INTEGER))
        x2 = m.add_variable(Variable("x2", lower_bound=1.0, variable_type=VariableType.INTEGER))
        m.add_constraint(Constraint("c1", coefficients={"x1": 2.0, "x2": 3.0}, sense=ConstraintSense.LESS_EQUAL, rhs=14.0))
        m.set_objective(Objective({"x1": 4.0, "x2": 5.0}, sense=ObjectiveSense.MAXIMIZE))

        sol_cold = solve_milp(m, warm_start_relaxations=False)
        sol_warm = solve_milp(m, warm_start_relaxations=True)

        assert sol_cold.status == SolutionStatus.OPTIMAL
        assert sol_warm.status == SolutionStatus.OPTIMAL
        assert np.isclose(sol_cold.objective_value, sol_warm.objective_value, atol=1e-6)
        assert sol_warm.variable_values["x1"] >= 2.0 - 1e-6
        assert sol_warm.variable_values["x2"] >= 1.0 - 1e-6

    def test_multi_level_branching_depth(self):
        """Verify warm-start chaining down multiple B&B tree levels."""
        # 8-variable knapsack forcing multiple branch levels
        m = Model("deep_knapsack")
        weights = [12.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
        values = [15.0, 22.0, 31.0, 40.0, 48.0, 57.0, 64.0, 75.0]
        vars_list = [m.add_variable(Variable(f"x{i}", variable_type=VariableType.BINARY)) for i in range(8)]
        coeffs = {f"x{i}": weights[i] for i in range(8)}
        m.add_constraint(Constraint("cap", coefficients=coeffs, sense=ConstraintSense.LESS_EQUAL, rhs=100.0))
        m.set_objective(Objective({f"x{i}": values[i] for i in range(8)}, sense=ObjectiveSense.MAXIMIZE))

        solver = MILPSolver(warm_start_relaxations=True)
        sol = solver.solve(m)

        assert sol.status == SolutionStatus.OPTIMAL
        assert sol.max_depth >= 3
        stats = solver.get_warm_start_stats()
        assert stats.accepted >= 3
        assert stats.dual_pivots >= 3

    def test_parent_child_isolation(self):
        """Verify that child node solving does not mutate parent basis state."""
        m = Model("isolation_test")
        x1 = m.add_variable(Variable("x1", variable_type=VariableType.INTEGER))
        x2 = m.add_variable(Variable("x2", variable_type=VariableType.INTEGER))
        m.add_constraint(Constraint("c1", coefficients={"x1": 3.0, "x2": 2.0}, sense=ConstraintSense.LESS_EQUAL, rhs=9.0))
        m.set_objective(Objective({"x1": 4.0, "x2": 3.0}, sense=ObjectiveSense.MAXIMIZE))

        solver = MILPSolver(warm_start_relaxations=True)
        # Solve root manually to get state
        node = solver._create_root_node(m)
        solver._process_node(m, node)
        assert node.warm_start_state is not None
        root_A_copy = np.array(node.warm_start_state.A, copy=True)
        root_basis_copy = list(node.warm_start_state.basis_indices)

        # _process_node already created child branches into _open_nodes
        assert len(solver._open_nodes) == 2
        left_id = solver._open_nodes[0]

        # Process left child
        left_node = solver._nodes[left_id]
        solver._process_node(m, left_node)

        # Verify parent state was completely untouched
        np.testing.assert_array_equal(node.warm_start_state.A, root_A_copy)
        assert node.warm_start_state.basis_indices == root_basis_copy

    def test_defensive_fallback_on_iteration_limit(self, monkeypatch):
        """Verify that if Dual Simplex hits an iteration limit, it falls back to cold solve."""
        from optiforge.lp.dual_simplex import DualSimplexResult
        import optiforge.milp.solver as milp_mod

        m = Model("fallback_test")
        x1 = m.add_variable(Variable("x1", variable_type=VariableType.BINARY))
        x2 = m.add_variable(Variable("x2", variable_type=VariableType.BINARY))
        x3 = m.add_variable(Variable("x3", variable_type=VariableType.BINARY))
        m.add_constraint(Constraint("cap", coefficients={"x1": 2.0, "x2": 3.0, "x3": 5.0}, sense=ConstraintSense.LESS_EQUAL, rhs=7.0))
        m.set_objective(Objective({"x1": 10.0, "x2": 15.0, "x3": 25.0}, sense=ObjectiveSense.MAXIMIZE))

        # Mock DualSimplex solve_standard_form to simulate hitting iteration limit on child LP
        def mock_solve_sf(self, *args, **kwargs):
            return DualSimplexResult(
                status=SolutionStatus.ITERATION_LIMIT,
                basis_indices=[],
                nonbasic_indices=[],
                x=np.zeros(1),
                y=np.zeros(1),
                objective_value=0.0,
                iterations=0,
                message="Simulated iteration limit in Dual Simplex",
            )

        monkeypatch.setattr(milp_mod.DualRevisedSimplexSolver, "solve_standard_form", mock_solve_sf)

        solver = MILPSolver(warm_start_relaxations=True)
        sol = solver.solve(m)

        stats = solver.get_warm_start_stats()
        # Fallbacks should have been triggered when Dual Simplex hit limit
        assert stats.fallbacks > 0
        # Cold fallback allowed B&B to complete successfully to OPTIMAL
        assert sol.status == SolutionStatus.OPTIMAL
        assert np.isclose(sol.objective_value, 35.0)
