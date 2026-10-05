"""Tests for Stage 8.45: LP Sensitivity Analysis & Presolve-Inverted Sensitivity.

Verifies:
RHS Sensitivity:
1. Simple <= RHS increase/decrease.
2. Simple >= RHS increase/decrease.
3. Equality constraint RHS.
4. Multiple constraints.
5. Negative RHS.
6. Row sign-flipped constraint.
7. Binding constraint.
8. Non-binding constraint.
9. Degenerate optimal basis.
10. Infinite allowable direction.

Objective Coefficient Sensitivity:
11. Nonbasic variable coefficient.
12. Basic variable coefficient.
13. Multiple nonbasic variables affected by basic-cost change.
14. MAXIMIZE objective sensitivity.
15. MINIMIZE objective sensitivity.
16. Zero reduced-cost variable.
17. Degenerate optimum objective sensitivity.

Presolve Inversion:
18. Fixed-variable elimination.
19. Redundant constraint handling.
20. Surviving constraint mapping.
21. Surviving variable mapping.
22. Bound transformation mapping.

API & Safety:
23. MILP rejected.
24. Infeasible rejected.
25. Unbounded rejected.
26. Missing basis rejected.
27. Sensitivity report references original model names.
28. Infinite ranges represented correctly.
29. Existing Stage 8.44 warm start remains functional.
30. Existing Stage 8.43 solver-selection metadata remains intact.

Cross-Check Testing:
31. Perturbation inside range preserves optimal basis.
32. Perturbation outside range changes optimal basis.
"""

from __future__ import annotations

import math
import pytest

from optiforge.core import SolutionStatus
from optiforge.model import (
    Constraint,
    ConstraintSense,
    Model,
    Objective,
    ObjectiveSense,
    Variable,
    VariableType,
)
from optiforge.sensitivity import SensitivityRange, SensitivityReport, compute_sensitivity


def build_hand_computable_lp(maximize: bool = True) -> Model:
    """Standard 2-variable, 2-constraint LP with known analytical sensitivity:
    Max 3x1 + 2x2
    s.t.
      x1 + x2 <= 4  (c1)
      2x1 + x2 <= 5 (c2)
      x1, x2 >= 0
    Optimal: x1=1, x2=3, z=9.
    Basis: [x1, x2].
    RHS allowable:
      c1 (curr=4): [-Δ=1.5, +Δ=1.0] -> [2.5, 5.0]
      c2 (curr=5): [-Δ=1.0, +Δ=3.0] -> [4.0, 8.0]
    Obj allowable:
      x1 (curr=3): [-Δ=1.0, +Δ=1.0] -> [2.0, 4.0]
      x2 (curr=2): [-Δ=0.5, +Δ=1.0] -> [1.5, 3.0]
    """
    model = Model(name="hand_computable_lp")
    model.add_variable(Variable("x1", lower_bound=0.0))
    model.add_variable(Variable("x2", lower_bound=0.0))
    sense = ObjectiveSense.MAXIMIZE if maximize else ObjectiveSense.MINIMIZE
    model.set_objective(Objective({"x1": 3.0, "x2": 2.0}, sense=sense))
    model.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    model.add_constraint(Constraint("c2", {"x1": 2.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    return model


class TestStage845Sensitivity:
    """Stage 8.45 verification suite."""

    # -------------------------------------------------------------------------
    # RHS Sensitivity
    # -------------------------------------------------------------------------

    def test_01_simple_le_rhs(self):
        """1. Simple <= RHS increase/decrease matches analytical range."""
        model = build_hand_computable_lp(maximize=True)
        sol = model.solve(presolve=False)
        report = sol.sensitivity
        assert report is not None

        r1 = report.get_rhs_range("c1")
        assert r1.current_value == 4.0
        assert r1.allowable_decrease == pytest.approx(1.5, abs=1e-4)
        assert r1.allowable_increase == pytest.approx(1.0, abs=1e-4)
        assert r1.lower_bound == pytest.approx(2.5, abs=1e-4)
        assert r1.upper_bound == pytest.approx(5.0, abs=1e-4)
        assert r1.is_unbounded_below is False
        assert r1.is_unbounded_above is False

    def test_02_simple_ge_rhs(self):
        """2. Simple >= RHS increase/decrease."""
        # Min 2x1 + 3x2 s.t. x1 + x2 >= 3 (c1), 2x1 + x2 >= 4 (c2), x >= 0
        model = Model(name="ge_lp")
        model.add_variable(Variable("x1", lower_bound=0.0))
        model.add_variable(Variable("x2", lower_bound=0.0))
        model.set_objective(Objective({"x1": 2.0, "x2": 3.0}, sense=ObjectiveSense.MINIMIZE))
        model.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, 3.0))
        model.add_constraint(Constraint("c2", {"x1": 2.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, 4.0))

        sol = model.solve(presolve=False)
        report = sol.sensitivity
        assert report is not None

        r1 = report.get_rhs_range("c1")
        assert r1.current_value == 3.0
        assert r1.allowable_decrease >= 0.0
        assert r1.allowable_increase >= 0.0

    def test_03_equality_constraint_rhs(self):
        """3. Equality constraint RHS sensitivity."""
        model = Model(name="eq_lp")
        model.add_variable(Variable("x1", lower_bound=0.0))
        model.add_variable(Variable("x2", lower_bound=0.0))
        model.set_objective(Objective({"x1": 1.0, "x2": 2.0}, sense=ObjectiveSense.MINIMIZE))
        model.add_constraint(Constraint("c_eq", {"x1": 1.0, "x2": 1.0}, ConstraintSense.EQUAL, 3.0))
        model.add_constraint(Constraint("c_ge", {"x1": 1.0, "x2": 2.0}, ConstraintSense.GREATER_EQUAL, 4.0))

        sol = model.solve(presolve=False)
        report = sol.sensitivity
        assert report is not None

        r_eq = report.get_rhs_range("c_eq")
        assert r_eq.current_value == 3.0
        assert r_eq.allowable_decrease >= 0.0
        assert r_eq.allowable_increase >= 0.0

    def test_04_multiple_constraints(self):
        """4. Multiple constraints all expose valid ranges."""
        model = build_hand_computable_lp(maximize=True)
        sol = model.solve(presolve=False)
        report = sol.sensitivity
        assert len(report.rhs) == 2
        assert "c1" in report.rhs
        assert "c2" in report.rhs
        r2 = report.get_rhs_range("c2")
        assert r2.current_value == 5.0
        assert r2.lower_bound == pytest.approx(4.0, abs=1e-4)
        assert r2.upper_bound == pytest.approx(8.0, abs=1e-4)

    def test_05_negative_rhs(self):
        """5. Negative RHS correctly normalizes and maps back."""
        # Max x1 + x2 s.t. -x1 + x2 <= -2 (i.e. x1 - x2 >= 2)
        # Add bounding constraint: x1 <= 5, x2 <= 5
        model = Model(name="neg_rhs_lp")
        model.add_variable(Variable("x1", lower_bound=0.0, upper_bound=5.0))
        model.add_variable(Variable("x2", lower_bound=0.0, upper_bound=5.0))
        model.set_objective(Objective({"x1": 1.0, "x2": 1.0}, sense=ObjectiveSense.MAXIMIZE))
        model.add_constraint(Constraint("c_neg", {"x1": -1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, -2.0))

        sol = model.solve(presolve=False)
        assert sol.status == SolutionStatus.OPTIMAL
        report = sol.sensitivity
        assert report is not None
        r = report.get_rhs_range("c_neg")
        assert r.current_value == -2.0
        assert r.allowable_decrease >= 0.0
        assert r.allowable_increase >= 0.0

    def test_06_row_sign_flipped_constraint(self):
        """6. Row sign-flipped constraint preserves valid range."""
        model = Model(name="flipped_lp")
        model.add_variable(Variable("x1", lower_bound=0.0))
        model.add_variable(Variable("x2", lower_bound=0.0))
        model.set_objective(Objective({"x1": 2.0, "x2": 1.0}, sense=ObjectiveSense.MINIMIZE))
        model.add_constraint(Constraint("c1", {"x1": -1.0, "x2": -1.0}, ConstraintSense.GREATER_EQUAL, -4.0))
        model.add_constraint(Constraint("c2", {"x1": 1.0}, ConstraintSense.GREATER_EQUAL, 1.0))

        sol = model.solve(presolve=False)
        assert sol.status == SolutionStatus.OPTIMAL
        report = sol.sensitivity
        assert report is not None
        assert "c1" in report.rhs

    def test_07_binding_constraint(self):
        """7. Binding constraint has non-zero dual and finite range."""
        model = build_hand_computable_lp(maximize=True)
        sol = model.solve(presolve=False)
        report = sol.sensitivity
        r1 = report.get_rhs_range("c1")
        assert r1.metadata["binding"] is True
        assert r1.allowable_decrease < math.inf

    def test_08_non_binding_constraint(self):
        """8. Non-binding constraint allows increase to infinity for <=."""
        # Max x1 s.t. x1 <= 2 (c1 tight), x1 + x2 <= 10 (c2 slack)
        model = Model(name="non_binding_lp")
        model.add_variable(Variable("x1", lower_bound=0.0))
        model.add_variable(Variable("x2", lower_bound=0.0))
        model.set_objective(Objective({"x1": 1.0}, sense=ObjectiveSense.MAXIMIZE))
        model.add_constraint(Constraint("c1", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 2.0))
        model.add_constraint(Constraint("c2", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 10.0))

        sol = model.solve(presolve=False)
        report = sol.sensitivity
        r2 = report.get_rhs_range("c2")
        # Increasing RHS of non-binding <= constraint never violates optimality
        assert r2.is_unbounded_above is True
        assert r2.allowable_increase == math.inf
        # Can decrease until it hits activity 2.0 (allowable decrease = 10 - 2 = 8)
        assert r2.allowable_decrease == pytest.approx(8.0, abs=1e-4)

    def test_09_degenerate_optimal_basis_rhs(self):
        """9. Degenerate optimal basis handles one-sided RHS interval cleanly."""
        model = Model(name="degenerate_lp")
        model.add_variable(Variable("x1", lower_bound=0.0))
        model.add_variable(Variable("x2", lower_bound=0.0))
        model.set_objective(Objective({"x1": 1.0, "x2": 1.0}, sense=ObjectiveSense.MINIMIZE))
        model.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, 0.0))
        model.add_constraint(Constraint("c2", {"x1": 2.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, 0.0))

        sol = model.solve(presolve=False)
        report = sol.sensitivity
        assert report is not None
        assert report.metadata["degenerate_basis"] is True

    def test_10_infinite_allowable_direction(self):
        """10. Unbounded direction explicitly marked with math.inf."""
        # Min 2x1 + 3x2 s.t. x1 + x2 >= 3 (c1), 2x1 + x2 >= 4 (c2), x >= 0
        model = Model(name="unbounded_rhs_lp")
        model.add_variable(Variable("x1", lower_bound=0.0))
        model.add_variable(Variable("x2", lower_bound=0.0))
        model.set_objective(Objective({"x1": 2.0, "x2": 3.0}, sense=ObjectiveSense.MINIMIZE))
        model.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, 3.0))
        model.add_constraint(Constraint("c2", {"x1": 2.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, 4.0))

        sol = model.solve(presolve=False)
        report = sol.sensitivity
        r1 = report.get_rhs_range("c1")
        assert r1.is_unbounded_above is True
        assert r1.allowable_increase == math.inf

        r2 = report.get_rhs_range("c2")
        assert r2.is_unbounded_below is True
        assert r2.allowable_decrease == math.inf


    # -------------------------------------------------------------------------
    # Objective Coefficients
    # -------------------------------------------------------------------------

    def test_11_nonbasic_variable_coefficient(self):
        """11. Nonbasic variable coefficient allows one-sided infinite variation."""
        # Min x1 + 5x2 s.t. x1 >= 2, x >= 0 -> x1 basic (x1=2), x2 nonbasic (x2=0, rc=5.0)
        model = Model(name="nonbasic_obj_lp")
        model.add_variable(Variable("x1", lower_bound=0.0))
        model.add_variable(Variable("x2", lower_bound=0.0))
        model.set_objective(Objective({"x1": 1.0, "x2": 5.0}, sense=ObjectiveSense.MINIMIZE))
        model.add_constraint(Constraint("c1", {"x1": 1.0}, ConstraintSense.GREATER_EQUAL, 2.0))

        sol = model.solve(presolve=False)
        report = sol.sensitivity
        r_x2 = report.get_objective_range("x2")
        assert r_x2.metadata["is_basic"] is False
        # Cost can increase to +inf without entering basis
        assert r_x2.is_unbounded_above is True
        assert r_x2.allowable_increase == math.inf
        # Cost can decrease by reduced cost (5.0) before becoming attractive
        assert r_x2.allowable_decrease == pytest.approx(5.0, abs=1e-4)

    def test_12_basic_variable_coefficient(self):
        """12. Basic variable coefficient matches analytical interval."""
        model = build_hand_computable_lp(maximize=True)
        sol = model.solve(presolve=False)
        report = sol.sensitivity
        r_x1 = report.get_objective_range("x1")
        assert r_x1.metadata["is_basic"] is True
        # Analytical for x1 (curr=3.0): [-Δ=1.0, +Δ=1.0] -> [2.0, 4.0]
        assert r_x1.allowable_decrease == pytest.approx(1.0, abs=1e-4)
        assert r_x1.allowable_increase == pytest.approx(1.0, abs=1e-4)
        assert r_x1.lower_bound == pytest.approx(2.0, abs=1e-4)
        assert r_x1.upper_bound == pytest.approx(4.0, abs=1e-4)

    def test_13_multiple_nonbasics_affected(self):
        """13. Basic cost change checks all affected nonbasic reduced costs."""
        model = build_hand_computable_lp(maximize=True)
        sol = model.solve(presolve=False)
        report = sol.sensitivity
        r_x2 = report.get_objective_range("x2")
        # Analytical for x2 (curr=2.0): [-Δ=0.5, +Δ=1.0] -> [1.5, 3.0]
        assert r_x2.allowable_decrease == pytest.approx(0.5, abs=1e-4)
        assert r_x2.allowable_increase == pytest.approx(1.0, abs=1e-4)
        assert r_x2.lower_bound == pytest.approx(1.5, abs=1e-4)
        assert r_x2.upper_bound == pytest.approx(3.0, abs=1e-4)

    def test_14_maximize_objective_sensitivity(self):
        """14. MAXIMIZE objective sensitivity preserves sign conventions."""
        model = build_hand_computable_lp(maximize=True)
        sol = model.solve(presolve=False)
        assert sol.sensitivity.objective["x1"].basis_remains_optimal is True

    def test_15_minimize_objective_sensitivity(self):
        """15. MINIMIZE objective sensitivity preserves sign conventions."""
        # Min 2x1 + 3x2 s.t. x1 + x2 >= 3, 2x1 + x2 >= 4, x >= 0
        model = Model(name="min_obj_lp")
        model.add_variable(Variable("x1", lower_bound=0.0))
        model.add_variable(Variable("x2", lower_bound=0.0))
        model.set_objective(Objective({"x1": 2.0, "x2": 3.0}, sense=ObjectiveSense.MINIMIZE))
        model.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, 3.0))
        model.add_constraint(Constraint("c2", {"x1": 2.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, 4.0))

        sol = model.solve(presolve=False)
        report = sol.sensitivity
        assert report is not None
        assert report.objective["x1"].allowable_decrease >= 0.0
        assert report.objective["x1"].allowable_increase >= 0.0

    def test_16_zero_reduced_cost_variable(self):
        """16. Alternate optimal / zero reduced cost variable handles zero bound."""
        # LP with multiple optima: Max x1 + x2 s.t. x1 + x2 <= 4
        model = Model(name="alt_opt_lp")
        model.add_variable(Variable("x1", lower_bound=0.0))
        model.add_variable(Variable("x2", lower_bound=0.0))
        model.set_objective(Objective({"x1": 1.0, "x2": 1.0}, sense=ObjectiveSense.MAXIMIZE))
        model.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))

        sol = model.solve(presolve=False)
        report = sol.sensitivity
        assert report is not None

    def test_17_degenerate_optimum_obj(self):
        """17. Degenerate optimum objective sensitivity."""
        model = Model(name="degen_opt_lp")
        model.add_variable(Variable("x1", lower_bound=0.0))
        model.add_variable(Variable("x2", lower_bound=0.0))
        model.set_objective(Objective({"x1": 1.0, "x2": 1.0}, sense=ObjectiveSense.MINIMIZE))
        model.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, 0.0))

        sol = model.solve(presolve=False)
        report = sol.sensitivity
        assert report is not None

    # -------------------------------------------------------------------------
    # Presolve Inversion
    # -------------------------------------------------------------------------

    def test_18_fixed_variable_elimination(self):
        """18. Fixed variable has (-inf, +inf) sensitivity with presolved_fixed metadata."""
        model = Model(name="fixed_var_lp")
        model.add_variable(Variable("x1", lower_bound=5.0, upper_bound=5.0))
        model.add_variable(Variable("x2", lower_bound=0.0))
        model.add_variable(Variable("x3", lower_bound=0.0))
        model.set_objective(Objective({"x1": 2.0, "x2": 3.0, "x3": 4.0}, sense=ObjectiveSense.MINIMIZE))
        model.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0, "x3": 1.0}, ConstraintSense.GREATER_EQUAL, 8.0))
        model.add_constraint(Constraint("c2", {"x2": 1.0, "x3": 2.0}, ConstraintSense.GREATER_EQUAL, 6.0))

        sol = model.solve(presolve=True)
        report = sol.sensitivity
        assert report is not None
        r_x1 = report.get_objective_range("x1")
        assert r_x1.is_unbounded_below is True
        assert r_x1.is_unbounded_above is True
        assert r_x1.metadata.get("presolved_fixed") is True

    def test_19_redundant_constraint_handling(self):
        """19. Redundant constraint removed by presolve has valid allowable decrease and infinite increase."""
        model = Model(name="redundant_lp")
        model.add_variable(Variable("x1", lower_bound=0.0))
        model.add_variable(Variable("x2", lower_bound=0.0))
        model.set_objective(Objective({"x1": 1.0, "x2": 1.0}, sense=ObjectiveSense.MAXIMIZE))
        model.add_constraint(Constraint("c_tight", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
        model.add_constraint(Constraint("c_redundant", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 100.0))

        sol = model.solve(presolve=True)
        report = sol.sensitivity
        assert report is not None
        assert "c_redundant" in report.rhs
        r_red = report.get_rhs_range("c_redundant")
        assert r_red.is_unbounded_above is True
        assert r_red.allowable_increase == math.inf
        # At optimum x1+x2=4, slack is 100 - 4 = 96
        assert r_red.allowable_decrease == pytest.approx(96.0, abs=1e-4)
        assert r_red.lower_bound == pytest.approx(4.0, abs=1e-4)

    def test_20_surviving_constraint_mapping(self):
        """20. Surviving constraints map accurately through presolve."""
        model = build_hand_computable_lp(maximize=True)
        sol = model.solve(presolve=True)
        report = sol.sensitivity
        assert "c1" in report.rhs
        assert "c2" in report.rhs

    def test_21_surviving_variable_mapping(self):
        """21. Surviving variables map accurately through presolve."""
        model = build_hand_computable_lp(maximize=True)
        sol = model.solve(presolve=True)
        report = sol.sensitivity
        assert "x1" in report.objective
        assert "x2" in report.objective

    def test_22_bound_transformation_mapping(self):
        """22. Variable lower bounds shifted in presolve preserve sensitivity."""
        model = Model(name="shifted_lp")
        model.add_variable(Variable("x1", lower_bound=2.0))
        model.add_variable(Variable("x2", lower_bound=3.0))
        model.set_objective(Objective({"x1": 1.0, "x2": 1.0}, sense=ObjectiveSense.MINIMIZE))
        model.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, 10.0))

        sol = model.solve(presolve=True)
        report = sol.sensitivity
        assert report is not None
        assert "x1" in report.objective
        assert "x2" in report.objective

    # -------------------------------------------------------------------------
    # API & Safety
    # -------------------------------------------------------------------------

    def test_23_milp_rejected(self):
        """23. MILP rejects sensitivity analysis without fabricating fake values."""
        model = Model(name="milp_model")
        model.add_variable(Variable("x1", variable_type=VariableType.INTEGER, lower_bound=0.0))
        model.add_variable(Variable("x2", variable_type=VariableType.CONTINUOUS, lower_bound=0.0))
        model.set_objective(Objective({"x1": 5.0, "x2": 4.0}, sense=ObjectiveSense.MAXIMIZE))
        model.add_constraint(Constraint("c1", {"x1": 2.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))

        sol = model.solve()
        assert sol.status == SolutionStatus.OPTIMAL
        # Solution attribute is None
        assert sol.sensitivity is None
        # Calling get_sensitivity() raises ValueError
        with pytest.raises(ValueError, match="Sensitivity report is not available"):
            sol.get_sensitivity()
        # Direct compute_sensitivity raises ValueError
        with pytest.raises(ValueError, match="continuous LPs only"):
            compute_sensitivity(model=model, solution=sol)

    def test_24_infeasible_rejected(self):
        """24. Infeasible solve rejects sensitivity analysis."""
        model = Model(name="infeas_lp")
        model.add_variable(Variable("x1", lower_bound=0.0))
        model.set_objective(Objective({"x1": 1.0}, sense=ObjectiveSense.MINIMIZE))
        model.add_constraint(Constraint("c1", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 2.0))
        model.add_constraint(Constraint("c2", {"x1": 1.0}, ConstraintSense.GREATER_EQUAL, 5.0))

        sol = model.solve()
        assert sol.status == SolutionStatus.INFEASIBLE
        assert sol.sensitivity is None
        with pytest.raises(ValueError):
            sol.get_sensitivity()

    def test_25_unbounded_rejected(self):
        """25. Unbounded solve rejects sensitivity analysis."""
        model = Model(name="unbounded_lp")
        model.add_variable(Variable("x1", lower_bound=0.0))
        model.set_objective(Objective({"x1": 1.0}, sense=ObjectiveSense.MAXIMIZE))

        sol = model.solve()
        assert sol.status == SolutionStatus.UNBOUNDED
        assert sol.sensitivity is None
        with pytest.raises(ValueError):
            sol.get_sensitivity()

    def test_26_missing_basis_rejected(self):
        """26. Missing basis raises ValueError."""
        model = build_hand_computable_lp()
        with pytest.raises(ValueError, match="No optimal basis information"):
            compute_sensitivity(model=model)

    def test_27_references_original_model_names(self):
        """27. Sensitivity report references original constraint and variable names."""
        model = build_hand_computable_lp(maximize=True)
        sol = model.solve(presolve=False)
        report = sol.sensitivity
        assert set(report.rhs.keys()) == {"c1", "c2"}
        assert set(report.objective.keys()) == {"x1", "x2"}

    def test_28_infinite_ranges_representation(self):
        """28. Infinite ranges represented cleanly using math.inf and booleans."""
        model = Model(name="inf_test")
        model.add_variable(Variable("x1", lower_bound=0.0))
        model.add_variable(Variable("x2", lower_bound=0.0))
        model.set_objective(Objective({"x1": 2.0, "x2": 3.0}, sense=ObjectiveSense.MINIMIZE))
        model.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, 3.0))
        model.add_constraint(Constraint("c2", {"x1": 2.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, 4.0))

        sol = model.solve(presolve=False)
        r1 = sol.sensitivity.rhs["c1"]
        assert r1.is_unbounded_above is True
        assert r1.allowable_increase == math.inf
        assert r1.upper_bound == math.inf

        r2 = sol.sensitivity.rhs["c2"]
        assert r2.is_unbounded_below is True
        assert r2.allowable_decrease == math.inf
        assert r2.lower_bound == -math.inf


    def test_29_existing_stage_8_44_warm_start_functional(self):
        """29. Existing Stage 8.44 warm start remains fully functional."""
        model = build_hand_computable_lp(maximize=True)
        sol1 = model.solve(presolve=False)
        basis = sol1.basis_info
        assert basis is not None

        sol2 = model.solve(basis=basis, presolve=False)
        assert sol2.status == SolutionStatus.OPTIMAL
        assert sol2.objective_value == pytest.approx(sol1.objective_value, abs=1e-5)

    def test_30_stage_8_43_metadata_intact(self):
        """30. Existing Stage 8.43 solver-selection metadata is preserved."""
        model = build_hand_computable_lp(maximize=True)
        sol = model.solve()
        assert sol.solver_selected is not None
        assert sol.problem_fingerprint is not None
        assert model.last_solver_selected is not None

    # -------------------------------------------------------------------------
    # Cross-Check Testing (Perturbations Inside vs Outside Range)
    # -------------------------------------------------------------------------

    def test_31_cross_check_inside_interval(self):
        """31. Perturbing RHS and objective strictly inside interval preserves optimal basis."""
        model = build_hand_computable_lp(maximize=True)
        sol1 = model.solve(presolve=False)
        binfo1 = sol1.basis_info
        orig_basis = binfo1.basic_variable_indices
        report = sol1.sensitivity

        # Perturb c1 RHS by +0.5 (within allowable increase of 1.0)
        m_pert = build_hand_computable_lp(maximize=True)
        m_pert.constraints["c1"].rhs = 4.0 + 0.5
        sol_pert = m_pert.solve(presolve=False)
        assert sol_pert.basis_info.basic_variable_indices == orig_basis

        # Perturb x1 objective by +0.5 (within allowable increase of 1.0)
        m_pert2 = build_hand_computable_lp(maximize=True)
        m_pert2.set_objective(Objective({"x1": 3.0 + 0.5, "x2": 2.0}, sense=ObjectiveSense.MAXIMIZE))
        sol_pert2 = m_pert2.solve(presolve=False)
        assert sol_pert2.basis_info.basic_variable_indices == orig_basis

    def test_32_cross_check_outside_interval(self):
        """32. Perturbing RHS and objective outside interval changes optimal basis."""
        model = build_hand_computable_lp(maximize=True)
        sol1 = model.solve(presolve=False)
        orig_basis = sol1.basis_info.basic_variable_indices

        # Allowable increase for c1 RHS is 1.0 (range upper bound is 5.0).
        # Perturb to 6.0 (> 5.0).
        m_out = build_hand_computable_lp(maximize=True)
        m_out.constraints["c1"].rhs = 6.0
        sol_out = m_out.solve(presolve=False)
        # Optimal basis should no longer be identical:
        assert sol_out.basis_info.basic_variable_indices != orig_basis

        # Allowable increase for x1 cost is 1.0 (range upper bound is 4.0).
        # Perturb to 5.0 (> 4.0).
        m_out2 = build_hand_computable_lp(maximize=True)
        m_out2.set_objective(Objective({"x1": 5.0, "x2": 2.0}, sense=ObjectiveSense.MAXIMIZE))
        sol_out2 = m_out2.solve(presolve=False)
        assert sol_out2.basis_info.basic_variable_indices != orig_basis
