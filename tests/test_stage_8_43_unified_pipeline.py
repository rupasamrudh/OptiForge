"""Tests for OptiForge Stage 8.43: Unified Model Pipeline & Problem Fingerprint Integration.

Verifies:
1. Existing Model.solve() behavior remains valid and backward compatible.
2. Continuous LP reaches a valid LP solver (RevisedSimplexSolver or DualRevisedSimplexSolver).
3. MILP reaches MILPSolver.
4. ProblemFingerprint is generated and attached to Solution and Model.
5. Solver-selection decisions are deterministic.
6. Primal/dual selection equivalence where both apply.
7. MAXIMIZE objective works correctly.
8. MINIMIZE objective works correctly.
9. Infeasible models return SolutionStatus.INFEASIBLE.
10. Unbounded models return SolutionStatus.UNBOUNDED.
11. Presolved models correctly reconstruct original variable values via postsolve.
12. Existing solver configuration (tolerances, stop_event, pricing, presolve flag) remains compatible.
13. Edge cases: empty model, no objective, invalid solver string, type mismatch.
"""

from __future__ import annotations

import math
import threading
import pytest

from optiforge.core.solution import Solution, SolutionStatus
from optiforge.lp.simplex import SimplexTolerances
from optiforge.milp.fingerprint import ProblemFingerprint, fingerprint_model
from optiforge.milp.solver import MILPTolerances
from optiforge.model import (
    Constraint,
    ConstraintSense,
    Model,
    Objective,
    ObjectiveSense,
    Variable,
    VariableType,
    execute_pipeline,
    select_solver,
)


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def _build_canonical_lp() -> Model:
    """Build canonical 2D LP:
    max 3x1 + 2x2
    s.t. x1 + x2 <= 4
         2x1 + x2 <= 5
         x1, x2 >= 0
    Optimum: x1=1, x2=3, obj=9.0
    """
    m = Model(name="canonical_max")
    m.add_variable(Variable("x1", lower_bound=0.0))
    m.add_variable(Variable("x2", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.add_constraint(Constraint("c2", {"x1": 2.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.set_objective(Objective({"x1": 3.0, "x2": 2.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_simple_milp() -> Model:
    """Build simple integer LP:
    max x1 + x2
    s.t. x1 + x2 <= 3.5
         x1, x2 in {0, 1, 2, ...}
    Optimum: x1=3, x2=0 (or any integer sum = 3), obj=3.0
    """
    m = Model(name="simple_milp")
    m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 3.5))
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_dual_feasible_lp() -> Model:
    """Build an LP whose slack basis is naturally dual-feasible:
    min x1 + 2x2
    s.t. x1 + x2 <= 4
         x1, x2 >= 0
    All c >= 0, all senses <=, all b >= 0.
    Optimum: x1=0, x2=0, obj=0.0
    """
    m = Model(name="dual_feasible_lp")
    m.add_variable(Variable("x1", lower_bound=0.0))
    m.add_variable(Variable("x2", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.set_objective(Objective({"x1": 1.0, "x2": 2.0}, ObjectiveSense.MINIMIZE))
    return m


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

class TestStage843UnifiedPipeline:
    """Stage 8.43 Unified Pipeline verification suite."""

    def test_item_1_existing_model_solve_backward_compatibility(self):
        """1. Verify existing Model.solve() behavior remains valid and backward compatible."""
        m = _build_canonical_lp()
        sol = m.solve()

        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.variable_values["x1"] == pytest.approx(1.0, abs=1e-6)
        assert sol.variable_values["x2"] == pytest.approx(3.0, abs=1e-6)
        assert sol.objective_value == pytest.approx(9.0, abs=1e-6)
        assert sol.iterations == 2

    def test_item_2_continuous_lp_reaches_lp_solver(self):
        """2. Continuous LP reaches a valid LP solver."""
        m = _build_canonical_lp()
        sol = m.solve()

        assert sol.solver_selected in ("revised_simplex", "dual_simplex")
        assert sol.solver_selected == "revised_simplex"
        assert m.last_solver_selected == "revised_simplex"
        assert m.last_solver_reason == "primal_simplex_default"

    def test_item_3_milp_reaches_milp_solver(self):
        """3. MILP reaches MILPSolver."""
        m = _build_simple_milp()
        sol = m.solve()

        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.solver_selected == "milp"
        assert m.last_solver_selected == "milp"
        assert m.last_solver_reason == "integer_variables_detected"
        assert sol.objective_value == pytest.approx(3.0, abs=1e-6)
        # Check integer feasibility
        assert abs(sol.variable_values["x1"] - round(sol.variable_values["x1"])) < 1e-6
        assert abs(sol.variable_values["x2"] - round(sol.variable_values["x2"])) < 1e-6

    def test_item_4_fingerprint_generated_and_attached(self):
        """4. Fingerprint is actually generated and used by the pipeline."""
        m = _build_canonical_lp()
        sol = m.solve()

        assert sol.problem_fingerprint is not None
        assert isinstance(sol.problem_fingerprint, ProblemFingerprint)
        assert sol.problem_fingerprint.variable_count == 2
        assert sol.problem_fingerprint.constraint_count == 2
        assert sol.problem_fingerprint.continuous_variable_count == 2
        assert sol.problem_fingerprint.integer_variable_count == 0
        assert sol.problem_fingerprint.binary_variable_count == 0
        assert m.last_fingerprint is sol.problem_fingerprint

    def test_item_5_solver_selection_deterministic(self):
        """5. Solver-selection decisions are deterministic."""
        m_lp = _build_canonical_lp()
        fp_lp = fingerprint_model(m_lp)

        # Multiple evaluations on the same LP yield identical decision
        choice1, reason1 = select_solver(m_lp, fp_lp)
        choice2, reason2 = select_solver(m_lp, fp_lp)
        assert choice1 == "revised_simplex"
        assert reason1 == "primal_simplex_default"
        assert choice1 == choice2
        assert reason1 == reason2

        # Dual-feasible LP
        m_dual = _build_dual_feasible_lp()
        fp_dual = fingerprint_model(m_dual)
        choice_d, reason_d = select_solver(m_dual, fp_dual)
        assert choice_d == "dual_simplex"
        assert reason_d == "dual_feasible_slack_basis"

        # MILP
        m_milp = _build_simple_milp()
        fp_milp = fingerprint_model(m_milp)
        choice_m, reason_m = select_solver(m_milp, fp_milp)
        assert choice_m == "milp"
        assert reason_m == "integer_variables_detected"

    def test_item_6_primal_dual_mathematical_equivalence(self):
        """6. Primal/dual selection does not alter mathematical correctness."""
        m_primal = _build_dual_feasible_lp()
        sol_primal = m_primal.solve(solver="primal")

        m_dual = _build_dual_feasible_lp()
        sol_dual = m_dual.solve(solver="dual")

        assert sol_primal.status is SolutionStatus.OPTIMAL
        assert sol_dual.status is SolutionStatus.OPTIMAL
        assert sol_primal.objective_value == pytest.approx(sol_dual.objective_value, abs=1e-7)
        for var in ("x1", "x2"):
            assert sol_primal.variable_values[var] == pytest.approx(sol_dual.variable_values[var], abs=1e-7)

    def test_item_7_maximize_objective(self):
        """7. MAXIMIZE works."""
        m = Model(name="max_test")
        m.add_variable(Variable("x", lower_bound=0.0, upper_bound=10.0))
        m.add_variable(Variable("y", lower_bound=0.0, upper_bound=10.0))
        m.add_constraint(Constraint("c1", {"x": 1.0, "y": 2.0}, ConstraintSense.LESS_EQUAL, 14.0))
        m.add_constraint(Constraint("c2", {"x": 3.0, "y": -1.0}, ConstraintSense.GREATER_EQUAL, 0.0))
        m.set_objective(Objective({"x": 2.0, "y": 3.0}, ObjectiveSense.MAXIMIZE))

        sol = m.solve()
        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value > 0.0
        assert sol.variable_values["x"] + 2 * sol.variable_values["y"] <= 14.0 + 1e-6

    def test_item_8_minimize_objective(self):
        """8. MINIMIZE works."""
        m = Model(name="min_test")
        m.add_variable(Variable("x1", lower_bound=0.0))
        m.add_variable(Variable("x2", lower_bound=0.0))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, 3.0))
        m.add_constraint(Constraint("c2", {"x1": 1.0, "x2": 3.0}, ConstraintSense.GREATER_EQUAL, 6.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 2.0}, ObjectiveSense.MINIMIZE))

        sol = m.solve()
        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.variable_values["x1"] == pytest.approx(1.5, abs=1e-6)
        assert sol.variable_values["x2"] == pytest.approx(1.5, abs=1e-6)
        assert sol.objective_value == pytest.approx(4.5, abs=1e-6)

    def test_item_9_infeasible_models_return_correct_status(self):
        """9. Infeasible models return SolutionStatus.INFEASIBLE."""
        m = Model(name="infeasible_test")
        m.add_variable(Variable("x", lower_bound=0.0))
        m.add_constraint(Constraint("c_le", {"x": 1.0}, ConstraintSense.LESS_EQUAL, 2.0))
        m.add_constraint(Constraint("c_ge", {"x": 1.0}, ConstraintSense.GREATER_EQUAL, 5.0))
        m.set_objective(Objective({"x": 1.0}, ObjectiveSense.MINIMIZE))

        sol = m.solve()
        assert sol.status is SolutionStatus.INFEASIBLE

    def test_item_10_unbounded_models_return_correct_status(self):
        """10. Unbounded models return SolutionStatus.UNBOUNDED."""
        m = Model(name="unbounded_test")
        m.add_variable(Variable("x1", lower_bound=0.0))
        m.add_variable(Variable("x2", lower_bound=0.0))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": -1.0}, ConstraintSense.LESS_EQUAL, 5.0))
        m.set_objective(Objective({"x1": 2.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))

        sol = m.solve()
        assert sol.status is SolutionStatus.UNBOUNDED

    def test_item_11_presolved_models_reconstruct_original_variables(self):
        """11. Presolved models correctly reconstruct original variables via postsolve."""
        m = Model(name="presolve_reconstruction")
        # x1 is fixed at 3.0
        m.add_variable(Variable("x1", lower_bound=3.0, upper_bound=3.0))
        m.add_variable(Variable("x2", lower_bound=0.0, upper_bound=10.0))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 7.0))
        m.set_objective(Objective({"x1": 2.0, "x2": 5.0}, ObjectiveSense.MAXIMIZE))

        sol = m.solve()
        assert sol.status is SolutionStatus.OPTIMAL
        # x1 must be recovered as 3.0, x2 can be up to 4.0
        assert sol.variable_values["x1"] == pytest.approx(3.0, abs=1e-6)
        assert sol.variable_values["x2"] == pytest.approx(4.0, abs=1e-6)
        assert sol.objective_value == pytest.approx(2.0 * 3.0 + 5.0 * 4.0, abs=1e-6)

    def test_item_12_existing_solver_configuration_compatible(self):
        """12. Existing solver configuration remains compatible."""
        m = _build_canonical_lp()
        tol = SimplexTolerances(pivot_tolerance=1e-8, max_iterations=50)
        sol = m.solve(tolerances=tol, pricing="bland", presolve=True)

        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(9.0, abs=1e-6)

        # Stop event cancellation
        ev = threading.Event()
        ev.set()
        sol_cancel = m.solve(stop_event=ev)
        assert sol_cancel.status is SolutionStatus.ITERATION_LIMIT

    def test_item_13_edge_cases_and_error_handling(self):
        """13. Edge cases: empty model, no objective, invalid solver, solver-type mismatch."""
        # Empty model without objective
        m_no_obj = Model(name="no_obj")
        m_no_obj.add_variable(Variable("x", lower_bound=0.0))
        sol_no_obj = m_no_obj.solve()
        assert sol_no_obj.status is SolutionStatus.UNKNOWN
        assert "no objective" in sol_no_obj.message.lower()

        # Model with invalid solver string
        m_lp = _build_canonical_lp()
        with pytest.raises(ValueError, match="Unknown solver preference"):
            m_lp.solve(solver="quantum_accelerator")

        # Requesting LP solver on MILP model
        m_milp = _build_simple_milp()
        with pytest.raises(ValueError, match="cannot solve MILP"):
            m_milp.solve(solver="primal")

        with pytest.raises(ValueError, match="cannot solve MILP"):
            m_milp.solve(solver="dual")
