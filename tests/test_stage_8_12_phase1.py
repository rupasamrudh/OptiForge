"""Tests for Stage 8.12: Profiling-Guided Phase-I / Root LP Optimization.

Validates the 10 required invariants:
1. Phase-I required
2. Phase-I not required
3. Artificial variables removed correctly
4. Phase-I -> Phase-II basis validity and identity
5. Equality-heavy model
6. Degenerate model with basic artificial variables
7. Infeasible model correctly identified
8. Ill-conditioned model numerical safety
9. Original objective restoration with bounds and offsets
10. MILP root relaxation Phase-I warm-start state creation
"""

import numpy as np
import pytest

from optiforge.core.solution import SolutionStatus
from optiforge.lp.simplex import RevisedSimplexSolver, SimplexTolerances
from optiforge.lp.standard_form import to_standard_form
from optiforge.milp.solver import MILPSolver
from optiforge.model import (
    Constraint,
    ConstraintSense,
    Model,
    Objective,
    ObjectiveSense,
    Variable,
    VariableType,
)
from optiforge.numerics.basis import BasisFactorization


class TestStage812Phase1Optimization:
    """Comprehensive test suite for Phase-I and Phase-II transition optimizations."""

    def test_01_phase1_required(self) -> None:
        """1. Model with >= and == constraints requiring Phase-I."""
        model = Model("Test_Phase1_Required")
        x0 = model.add_variable(Variable("x0", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        x1 = model.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        model.add_constraint(Constraint("c0", {"x0": 1.0, "x1": 2.0}, ConstraintSense.GREATER_EQUAL, 6.0))
        model.add_constraint(Constraint("c1", {"x0": 2.0, "x1": 1.0}, ConstraintSense.EQUAL, 9.0))
        model.set_objective(Objective({"x0": 3.0, "x1": 2.0}, ObjectiveSense.MINIMIZE))

        solver = RevisedSimplexSolver()
        sol = solver.solve(model)

        assert sol.status == SolutionStatus.OPTIMAL
        assert pytest.approx(sol.objective_value, rel=1e-6) == 14.0
        assert pytest.approx(sol.variable_values["x0"], rel=1e-6) == 4.0
        assert pytest.approx(sol.variable_values["x1"], rel=1e-6) == 1.0

    def test_02_phase1_not_required(self) -> None:
        """2. Pure <= constraints with positive RHS; Phase-I not needed."""
        model = Model("Test_Phase1_Not_Required")
        x0 = model.add_variable(Variable("x0", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        x1 = model.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        model.add_constraint(Constraint("c0", {"x0": 1.0, "x1": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
        model.add_constraint(Constraint("c1", {"x0": 2.0, "x1": 1.0}, ConstraintSense.LESS_EQUAL, 6.0))
        model.set_objective(Objective({"x0": 3.0, "x1": 2.0}, ObjectiveSense.MAXIMIZE))

        sf = to_standard_form(model)
        assert not sf.needs_phase1

        solver = RevisedSimplexSolver()
        sol = solver.solve(model)

        assert sol.status == SolutionStatus.OPTIMAL
        assert pytest.approx(sol.objective_value, rel=1e-6) == 10.0
        assert pytest.approx(sol.variable_values["x0"], rel=1e-6) == 2.0
        assert pytest.approx(sol.variable_values["x1"], rel=1e-6) == 2.0

    def test_03_artificial_variables_removed_correctly(self) -> None:
        """3. Artificial columns are dropped and mapping excludes artificial variables."""
        model = Model("Test_Artificial_Removal")
        x0 = model.add_variable(Variable("x0", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        x1 = model.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        model.add_constraint(Constraint("c0", {"x0": 1.0, "x1": 1.0}, ConstraintSense.GREATER_EQUAL, 5.0))
        model.set_objective(Objective({"x0": 1.0, "x1": 1.0}, ObjectiveSense.MINIMIZE))

        sf = to_standard_form(model)
        assert sf.needs_phase1
        assert len(sf.artificial_indices) > 0

        solver = RevisedSimplexSolver()
        tol = solver.tolerances
        p1_res = solver._simplex_loop(
            A=sf.A,
            b=sf.b,
            c=sf.phase1_c,
            basis_indices=list(sf.phase1_basis_indices),
            nonbasic_indices=list(sf.phase1_nonbasic_indices),
            sense=ObjectiveSense.MINIMIZE,
            tolerances=tol,
        )
        assert p1_res.status == SolutionStatus.OPTIMAL

        (A_p2, b_p2, c_p2, b_p2_idx, nb_p2_idx, col_map) = solver._cleanup_phase1_basis(
            model_sf=sf,
            final_basis=p1_res.basis_indices,
            final_nonbasic=p1_res.nonbasic_indices,
            tolerances=tol,
        )

        assert A_p2.shape[1] == sf.A.shape[1] - len(sf.artificial_indices)
        for art_idx in sf.artificial_indices:
            assert art_idx not in col_map
        for new_idx in b_p2_idx:
            assert 0 <= new_idx < A_p2.shape[1]
        for new_idx in nb_p2_idx:
            assert 0 <= new_idx < A_p2.shape[1]

    def test_04_phase1_to_phase2_basis_validity(self) -> None:
        """4. Basis matrix from Phase-I is exactly identical to Phase-II starting basis."""
        model = Model("Test_Basis_Validity")
        for i in range(4):
            model.add_variable(Variable(f"x{i}", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        model.add_constraint(Constraint("c0", {"x0": 1.0, "x1": 2.0}, ConstraintSense.GREATER_EQUAL, 4.0))
        model.add_constraint(Constraint("c1", {"x2": 1.0, "x3": 3.0}, ConstraintSense.EQUAL, 6.0))
        model.set_objective(Objective({"x0": 1.0, "x1": 1.0, "x2": 2.0, "x3": 1.0}, ObjectiveSense.MINIMIZE))

        sf = to_standard_form(model)
        solver = RevisedSimplexSolver()
        tol = solver.tolerances
        p1_res = solver._simplex_loop(
            A=sf.A,
            b=sf.b,
            c=sf.phase1_c,
            basis_indices=list(sf.phase1_basis_indices),
            nonbasic_indices=list(sf.phase1_nonbasic_indices),
            sense=ObjectiveSense.MINIMIZE,
            tolerances=tol,
        )

        (A_p2, b_p2, c_p2, b_p2_idx, nb_p2_idx, col_map) = solver._cleanup_phase1_basis(
            model_sf=sf,
            final_basis=p1_res.basis_indices,
            final_nonbasic=p1_res.nonbasic_indices,
            tolerances=tol,
        )

        B_p1 = sf.A[:, p1_res.basis_indices]
        B_p2 = A_p2[:, b_p2_idx]

        assert np.array_equal(B_p1, B_p2)
        fact = BasisFactorization(B_p2)
        assert fact.condition_number < 1e10

    def test_05_equality_heavy_model(self) -> None:
        """5. Equality-heavy model with multiple constraints."""
        model = Model("Test_Equality_Heavy")
        for i in range(6):
            model.add_variable(Variable(f"x{i}", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        model.add_constraint(Constraint("c0", {"x0": 1.0, "x1": 1.0, "x2": 1.0}, ConstraintSense.EQUAL, 6.0))
        model.add_constraint(Constraint("c1", {"x1": 2.0, "x3": 1.0}, ConstraintSense.EQUAL, 4.0))
        model.add_constraint(Constraint("c2", {"x2": 1.0, "x4": 2.0, "x5": 1.0}, ConstraintSense.EQUAL, 8.0))
        model.set_objective(Objective({"x0": 1.0, "x1": 2.0, "x2": 1.0, "x3": 3.0, "x4": 2.0, "x5": 1.0}, ObjectiveSense.MINIMIZE))

        solver = RevisedSimplexSolver()
        sol = solver.solve(model)

        assert sol.status == SolutionStatus.OPTIMAL
        assert pytest.approx(sol.objective_value, rel=1e-6) == 12.0
        # Check constraints satisfied
        assert pytest.approx(sol.variable_values["x0"] + sol.variable_values["x1"] + sol.variable_values["x2"], rel=1e-6) == 6.0

    def test_06_degenerate_model(self) -> None:
        """6. Degenerate Phase-I with redundant constraints and zero RHS."""
        model = Model("Test_Degenerate")
        x0 = model.add_variable(Variable("x0", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        x1 = model.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        model.add_constraint(Constraint("c0", {"x0": 1.0, "x1": -1.0}, ConstraintSense.GREATER_EQUAL, 0.0))
        model.add_constraint(Constraint("c1", {"x0": 2.0, "x1": -2.0}, ConstraintSense.GREATER_EQUAL, 0.0))
        model.add_constraint(Constraint("c2", {"x0": 1.0, "x1": 1.0}, ConstraintSense.EQUAL, 4.0))
        model.set_objective(Objective({"x0": 2.0, "x1": 1.0}, ObjectiveSense.MINIMIZE))

        solver = RevisedSimplexSolver()
        sol = solver.solve(model)

        assert sol.status == SolutionStatus.OPTIMAL
        assert pytest.approx(sol.objective_value, rel=1e-6) == 6.0
        assert pytest.approx(sol.variable_values["x0"], rel=1e-6) == 2.0
        assert pytest.approx(sol.variable_values["x1"], rel=1e-6) == 2.0

    def test_07_infeasible_model(self) -> None:
        """7. Infeasible model correctly detected in Phase-I."""
        model = Model("Test_Infeasible")
        x0 = model.add_variable(Variable("x0", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        model.add_constraint(Constraint("c0", {"x0": 1.0}, ConstraintSense.LESS_EQUAL, 2.0))
        model.add_constraint(Constraint("c1", {"x0": 1.0}, ConstraintSense.GREATER_EQUAL, 5.0))
        model.set_objective(Objective({"x0": 1.0}, ObjectiveSense.MINIMIZE))

        solver = RevisedSimplexSolver()
        sol = solver.solve(model)

        assert sol.status == SolutionStatus.INFEASIBLE

    def test_08_ill_conditioned_model(self) -> None:
        """8. Model with small coefficients (near singular) solved stably."""
        model = Model("Test_Ill_Conditioned")
        x0 = model.add_variable(Variable("x0", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        x1 = model.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        eps = 1e-4
        model.add_constraint(Constraint("c0", {"x0": 1.0, "x1": 1.0}, ConstraintSense.EQUAL, 2.0))
        model.add_constraint(Constraint("c1", {"x0": 1.0, "x1": 1.0 + eps}, ConstraintSense.EQUAL, 2.0 + eps))
        model.set_objective(Objective({"x0": 1.0, "x1": 2.0}, ObjectiveSense.MINIMIZE))

        solver = RevisedSimplexSolver()
        sol = solver.solve(model)

        assert sol.status == SolutionStatus.OPTIMAL
        assert pytest.approx(sol.variable_values["x0"], abs=1e-4) == 1.0
        assert pytest.approx(sol.variable_values["x1"], abs=1e-4) == 1.0

    def test_09_original_objective_restoration(self) -> None:
        """9. Restoration of objective with shifted variable bounds."""
        model = Model("Test_Objective_Restoration")
        x0 = model.add_variable(Variable("x0", lower_bound=2.0, upper_bound=10.0, variable_type=VariableType.CONTINUOUS))
        x1 = model.add_variable(Variable("x1", lower_bound=3.0, upper_bound=10.0, variable_type=VariableType.CONTINUOUS))
        model.add_constraint(Constraint("c0", {"x0": 1.0, "x1": 1.0}, ConstraintSense.GREATER_EQUAL, 7.0))
        model.set_objective(Objective({"x0": 2.0, "x1": 3.0}, sense=ObjectiveSense.MINIMIZE))

        solver = RevisedSimplexSolver()
        sol = solver.solve(model)

        assert sol.status == SolutionStatus.OPTIMAL
        # Min at x0 = 4, x1 = 3 -> 2(4) + 3(3) = 8 + 9 = 17
        assert pytest.approx(sol.objective_value, rel=1e-6) == 17.0
        assert pytest.approx(sol.variable_values["x0"], rel=1e-6) == 4.0
        assert pytest.approx(sol.variable_values["x1"], rel=1e-6) == 3.0

    def test_10_milp_root_relaxation_warm_start_state(self) -> None:
        """10. Phase-I root relaxation produces valid WarmStartBasisState for MILP."""
        model = Model("Test_MILP_Phase1_Root")
        x0 = model.add_variable(Variable("x0", lower_bound=0, upper_bound=5, variable_type=VariableType.INTEGER))
        x1 = model.add_variable(Variable("x1", lower_bound=0, upper_bound=5, variable_type=VariableType.INTEGER))
        # Requires Phase-I and produces fractional root solution
        model.add_constraint(Constraint("c0", {"x0": 1.0, "x1": 2.0}, ConstraintSense.GREATER_EQUAL, 5.0))
        model.set_objective(Objective({"x0": 2.0, "x1": 3.0}, ObjectiveSense.MINIMIZE))

        solver = MILPSolver(warm_start_relaxations=True, inherit_basis_factorization=True)
        sol, warm_state = solver._solve_root_or_cold(model)

        assert sol.status == SolutionStatus.OPTIMAL
        assert warm_state is not None
        assert warm_state.basis_factorization is not None
        assert len(warm_state.basis_indices) == warm_state.A.shape[0]

        # Solve full MILP and verify warm-start stats
        full_sol = solver.solve(model)
        assert full_sol.status == SolutionStatus.OPTIMAL
        assert full_sol.objective_value == 8.0
        assert solver.warm_start_stats.attempted > 0
        assert solver.warm_start_stats.accepted > 0
