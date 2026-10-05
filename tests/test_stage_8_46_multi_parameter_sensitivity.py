"""Tests for Stage 8.46: Multi-Parameter Sensitivity, 100% Rule & Scenario/What-If Readiness."""

from __future__ import annotations

import math
import pytest

from optiforge.core.solution import SolutionStatus
from optiforge.model.constraint import Constraint, ConstraintSense
from optiforge.model.model import Model
from optiforge.model.objective import Objective, ObjectiveSense
from optiforge.model.variable import Variable, VariableType
from optiforge.sensitivity.calculator import compute_sensitivity
from optiforge.sensitivity.multi import (
    MultiParameterSensitivityResult,
    RuleStatus,
    evaluate_100_percent_rule,
)
from optiforge.sensitivity.range import SensitivityRange
from optiforge.sensitivity.report import SensitivityReport
from optiforge.sensitivity.scenario import (
    BreakEvenResult,
    Scenario,
    ScenarioComparison,
    ScenarioResult,
    compare_scenarios,
    compute_break_even,
    evaluate_scenario,
)


def _build_standard_lp() -> Model:
    """Build canonical 2-variable LP:
    max 3 x1 + 2 x2
    s.t.
      labor:    2 x1 +   x2 <= 100
      material:   x1 +   x2 <= 80
      x1 <= 40
      x1, x2 >= 0
    Optimal solution: x1 = 20, x2 = 60, obj = 180.
    labor shadow price = 1.0, material shadow price = 1.0.
    """
    m = Model(name="standard_lp")
    x1 = Variable("x1", lower_bound=0.0, upper_bound=40.0)
    x2 = Variable("x2", lower_bound=0.0, upper_bound=100.0)
    m.add_variable(x1)
    m.add_variable(x2)

    m.set_objective(Objective({"x1": 3.0, "x2": 2.0}, sense=ObjectiveSense.MAXIMIZE))
    m.add_constraint(Constraint("labor", {"x1": 2.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 100.0))
    m.add_constraint(Constraint("material", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 80.0))
    return m


# -----------------------------------------------------------------------------
# Test 1: Single RHS Scenario
# -----------------------------------------------------------------------------
def test_01_single_rhs_scenario():
    model = _build_standard_lp()
    sol = model.solve(presolve=False)
    assert sol.status is SolutionStatus.OPTIMAL

    scenario = Scenario(name="single_rhs", rhs_changes={"labor": 10.0})
    res = model.evaluate_scenario(scenario)

    assert res.scenario_name == "single_rhs"
    assert res.basis_preserved is True
    assert res.primal_feasible is True
    assert res.dual_feasible is True
    # labor shadow price is 1.0, so obj increases by 10.0 (from 180 to 190)
    assert pytest.approx(res.objective_value, rel=1e-5) == 190.0
    assert pytest.approx(res.objective_delta, rel=1e-5) == 10.0
    assert res.hundred_percent_rule_status == "within_region"
    assert res.solver_resolved is False


# -----------------------------------------------------------------------------
# Test 2: Multiple RHS Simultaneous Changes
# -----------------------------------------------------------------------------
def test_02_multiple_rhs_simultaneous_changes():
    model = _build_standard_lp()
    model.solve(presolve=False)

    # labor +5, material -5
    scenario = Scenario(name="multi_rhs", rhs_changes={"labor": 5.0, "material": -5.0})
    res = model.evaluate_scenario(scenario)

    assert res.basis_preserved is True
    # labor shadow price is 1.0, material shadow price is 1.0
    # delta = 1.0*(5.0) + 1.0*(-5.0) = 0.0
    assert pytest.approx(res.objective_delta, abs=1e-5) == 0.0
    assert pytest.approx(res.objective_value, rel=1e-5) == 180.0
    # Check variable values:
    # 2 x1 + x2 = 105, x1 + x2 = 75 => x1 = 30, x2 = 45
    assert pytest.approx(res.variable_values["x1"], rel=1e-5) == 30.0
    assert pytest.approx(res.variable_values["x2"], rel=1e-5) == 45.0


# -----------------------------------------------------------------------------
# Test 3: 100% Rule Within Region
# -----------------------------------------------------------------------------
def test_03_hundred_percent_rule_within_region():
    model = _build_standard_lp()
    model.solve(presolve=False)
    report = model.sensitivity_analysis()

    # Get allowable increase/decrease
    labor_inc = report.rhs["labor"].allowable_increase
    mat_dec = report.rhs["material"].allowable_decrease

    # Perturb each by 40% of allowable range -> total sum = 0.8 <= 1.0
    delta_labor = 0.4 * labor_inc
    delta_mat = -0.4 * mat_dec

    rule_res = evaluate_100_percent_rule(
        report=report,
        rhs_changes={"labor": delta_labor, "material": delta_mat},
    )
    assert "rhs" in rule_res
    assert rule_res["rhs"].status == RuleStatus.WITHIN_REGION
    assert rule_res["rhs"].is_guaranteed_optimal is True
    assert pytest.approx(rule_res["rhs"].total_percentage_sum, rel=1e-5) == 0.8

    scenario = Scenario("safe_rhs", rhs_changes={"labor": delta_labor, "material": delta_mat})
    res = model.evaluate_scenario(scenario)
    assert res.hundred_percent_rule_status == "within_region"
    assert res.basis_preserved is True


# -----------------------------------------------------------------------------
# Test 4: 100% Rule Outside Region
# -----------------------------------------------------------------------------
def test_04_hundred_percent_rule_outside_region():
    model = _build_standard_lp()
    model.solve(presolve=False)
    report = model.sensitivity_analysis()

    labor_inc = report.rhs["labor"].allowable_increase
    mat_dec = report.rhs["material"].allowable_decrease

    # Perturb each by 70% -> total sum = 1.4 > 1.0
    delta_labor = 0.7 * labor_inc
    delta_mat = -0.7 * mat_dec

    rule_res = evaluate_100_percent_rule(
        report=report,
        rhs_changes={"labor": delta_labor, "material": delta_mat},
    )
    assert rule_res["rhs"].status == RuleStatus.OUTSIDE_REGION
    assert rule_res["rhs"].is_guaranteed_optimal is False
    assert pytest.approx(rule_res["rhs"].total_percentage_sum, rel=1e-5) == 1.4

    scenario = Scenario("outside_rhs", rhs_changes={"labor": delta_labor, "material": delta_mat})
    res = model.evaluate_scenario(scenario)
    assert res.hundred_percent_rule_status == "outside_region"


# -----------------------------------------------------------------------------
# Test 5: Infinite Allowable Direction
# -----------------------------------------------------------------------------
def test_05_infinite_allowable_direction():
    # Build LP where one constraint has infinite allowable increase (slack > 0)
    m = Model(name="inf_lp")
    x1 = Variable("x1", lower_bound=0.0)
    m.add_variable(x1)
    m.set_objective(Objective({"x1": 1.0}, sense=ObjectiveSense.MAXIMIZE))
    m.add_constraint(Constraint("c1", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 10.0))
    m.add_constraint(Constraint("c2_loose", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 20.0))
    m.solve(presolve=False)

    report = m.sensitivity_analysis()
    assert report.rhs["c2_loose"].allowable_increase == math.inf

    rule_res = evaluate_100_percent_rule(
        report=report,
        rhs_changes={"c2_loose": 100.0},
    )
    assert rule_res["rhs"].status == RuleStatus.WITHIN_REGION
    assert rule_res["rhs"].individual_ratios["c2_loose"] == 0.0
    assert rule_res["rhs"].metadata["has_infinite_direction"] is True


# -----------------------------------------------------------------------------
# Test 6: Zero Allowable Direction
# -----------------------------------------------------------------------------
def test_06_zero_allowable_direction():
    # Create degenerate model
    m = Model(name="degenerate_lp")
    x1 = Variable("x1", lower_bound=0.0)
    x2 = Variable("x2", lower_bound=0.0)
    m.add_variable(x1)
    m.add_variable(x2)
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, sense=ObjectiveSense.MAXIMIZE))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 10.0))
    m.add_constraint(Constraint("c2", {"x1": 1.0, "x2": 0.0}, ConstraintSense.LESS_EQUAL, 10.0))
    m.add_constraint(Constraint("c3", {"x1": 1.0, "x2": 2.0}, ConstraintSense.LESS_EQUAL, 10.0))
    m.solve(presolve=False)

    report = m.sensitivity_analysis()
    # At least one constraint will have 0 allowable range in a direction
    has_zero = any(
        r.allowable_increase <= 1e-7 or r.allowable_decrease <= 1e-7
        for r in report.rhs.values()
    )
    assert has_zero

    # Evaluate with a change in that zero-direction constraint
    zero_con = next(
        c for c, r in report.rhs.items()
        if r.allowable_increase <= 1e-7 or r.allowable_decrease <= 1e-7
    )
    delta = 1.0 if report.rhs[zero_con].allowable_increase <= 1e-7 else -1.0
    rule_res = evaluate_100_percent_rule(report=report, rhs_changes={zero_con: delta})
    assert rule_res["rhs"].status == RuleStatus.OUTSIDE_REGION
    assert rule_res["rhs"].metadata["has_zero_direction"] is True


# -----------------------------------------------------------------------------
# Test 7: Exact RHS Basis-Feasibility Verification
# -----------------------------------------------------------------------------
def test_07_exact_rhs_basis_feasibility_verification():
    model = _build_standard_lp()
    model.solve(presolve=False)

    # Exceed allowable region so that basic variables become negative
    scenario_infeasible_basis = Scenario(
        name="huge_rhs_cut",
        rhs_changes={"labor": -90.0},  # b_labor drops from 100 to 10
    )
    res = model.evaluate_scenario(scenario_infeasible_basis)
    assert res.primal_feasible is False
    assert res.basis_preserved is False
    assert res.status == "basis_infeasible"
    assert res.objective_value is None


# -----------------------------------------------------------------------------
# Test 8: Single Objective Coefficient Change
# -----------------------------------------------------------------------------
def test_08_single_objective_coefficient_change():
    model = _build_standard_lp()
    model.solve(presolve=False)

    # In baseline, x1=20, x2=60, obj = 3*20 + 2*60 = 180
    # Increase x1 obj coeff by 0.5 (from 3.0 to 3.5)
    scenario = Scenario(name="obj_x1_up", objective_changes={"x1": 0.5})
    res = model.evaluate_scenario(scenario)

    assert res.basis_preserved is True
    assert res.primal_feasible is True
    assert res.dual_feasible is True
    # Since x1 = 20, delta obj = 0.5 * 20 = 10 -> new obj = 190
    assert pytest.approx(res.objective_value, rel=1e-5) == 190.0
    assert pytest.approx(res.objective_delta, rel=1e-5) == 10.0


# -----------------------------------------------------------------------------
# Test 9: Multiple Nonbasic Objective Changes
# -----------------------------------------------------------------------------
def test_09_multiple_nonbasic_objective_changes():
    # LP with nonbasic variables
    m = Model(name="nonbasic_lp")
    x1 = Variable("x1", lower_bound=0.0)
    x2 = Variable("x2", lower_bound=0.0)
    x3 = Variable("x3", lower_bound=0.0)
    m.add_variable(x1)
    m.add_variable(x2)
    m.add_variable(x3)
    # x3 has unattractive profit 0.5
    m.set_objective(Objective({"x1": 5.0, "x2": 4.0, "x3": 0.5}, sense=ObjectiveSense.MAXIMIZE))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0, "x3": 1.0}, ConstraintSense.LESS_EQUAL, 10.0))
    m.solve(presolve=False)

    # In optimal solution, x1=10, x2=0 (nonbasic), x3=0 (nonbasic)
    scenario = Scenario(
        name="nonbasic_shifts",
        objective_changes={"x2": -1.0, "x3": 0.1},
    )
    res = m.evaluate_scenario(scenario)
    assert res.basis_preserved is True
    # Nonbasic variables have value 0, so obj does not change
    assert pytest.approx(res.objective_delta, abs=1e-5) == 0.0


# -----------------------------------------------------------------------------
# Test 10: Multiple Basic Objective Changes
# -----------------------------------------------------------------------------
def test_10_multiple_basic_objective_changes():
    model = _build_standard_lp()
    model.solve(presolve=False)

    # Both x1 and x2 are basic (x1=20, x2=60)
    scenario = Scenario(
        name="both_basic_obj",
        objective_changes={"x1": 0.2, "x2": -0.1},
    )
    res = model.evaluate_scenario(scenario)
    assert res.basis_preserved is True
    # Delta = 0.2*20 + (-0.1)*60 = 4.0 - 6.0 = -2.0
    assert pytest.approx(res.objective_delta, rel=1e-5) == -2.0
    assert pytest.approx(res.objective_value, rel=1e-5) == 178.0


# -----------------------------------------------------------------------------
# Test 11: Mixed Basic/Nonbasic Objective Changes
# -----------------------------------------------------------------------------
def test_11_mixed_basic_nonbasic_objective_changes():
    m = Model(name="mixed_lp")
    x1 = Variable("x1", lower_bound=0.0)
    x2 = Variable("x2", lower_bound=0.0)
    x3 = Variable("x3", lower_bound=0.0)
    m.add_variable(x1)
    m.add_variable(x2)
    m.add_variable(x3)
    m.set_objective(Objective({"x1": 5.0, "x2": 2.0, "x3": 1.0}, sense=ObjectiveSense.MAXIMIZE))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0, "x3": 1.0}, ConstraintSense.LESS_EQUAL, 10.0))
    m.solve(presolve=False)

    # x1 is basic (x1=10), x2 and x3 nonbasic
    scenario = Scenario(
        name="mixed_obj",
        objective_changes={"x1": 0.5, "x2": -0.5, "x3": -0.2},
    )
    res = m.evaluate_scenario(scenario)
    assert res.basis_preserved is True
    assert pytest.approx(res.objective_delta, rel=1e-5) == 5.0


# -----------------------------------------------------------------------------
# Test 12: Exact Reduced-Cost Verification
# -----------------------------------------------------------------------------
def test_12_exact_reduced_cost_verification():
    model = _build_standard_lp()
    model.solve(presolve=False)

    # Increase x1 cost massively so that dual feasibility fails
    scenario = Scenario(name="huge_obj_jump", objective_changes={"x1": 50.0})
    res = model.evaluate_scenario(scenario)
    assert res.dual_feasible is False
    assert res.basis_preserved is False
    assert res.status == "basis_infeasible"


# -----------------------------------------------------------------------------
# Test 13: Combined RHS + Objective Changes
# -----------------------------------------------------------------------------
def test_13_combined_rhs_plus_objective_changes():
    model = _build_standard_lp()
    model.solve(presolve=False)

    scenario = Scenario(
        name="combined",
        rhs_changes={"labor": 2.0, "material": -1.0},
        objective_changes={"x1": 0.1, "x2": 0.05},
    )
    res = model.evaluate_scenario(scenario)
    assert res.basis_preserved is True
    assert res.primal_feasible is True
    assert res.dual_feasible is True
    assert res.objective_value is not None


# -----------------------------------------------------------------------------
# Test 14: Degenerate Basis
# -----------------------------------------------------------------------------
def test_14_degenerate_basis():
    m = Model(name="degen_basis")
    x1 = Variable("x1", lower_bound=0.0)
    x2 = Variable("x2", lower_bound=0.0)
    m.add_variable(x1)
    m.add_variable(x2)
    m.set_objective(Objective({"x1": 2.0, "x2": 1.0}, sense=ObjectiveSense.MAXIMIZE))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 10.0))
    m.add_constraint(Constraint("c2", {"x1": 1.0, "x2": 0.0}, ConstraintSense.LESS_EQUAL, 10.0))
    m.add_constraint(Constraint("c3", {"x1": 2.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 20.0))
    m.solve(presolve=False)

    scenario = Scenario("small_shift", rhs_changes={"c1": 0.01})
    res = m.evaluate_scenario(scenario)
    # Does not crash, cleanly evaluates primal/dual feasibility
    assert isinstance(res, ScenarioResult)


# -----------------------------------------------------------------------------
# Test 15: Alternate Optimum
# -----------------------------------------------------------------------------
def test_15_alternate_optimum():
    m = Model(name="alt_opt")
    x1 = Variable("x1", lower_bound=0.0)
    x2 = Variable("x2", lower_bound=0.0)
    m.add_variable(x1)
    m.add_variable(x2)
    # Parallel objective: x1 + x2 <= 10, max x1 + x2
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, sense=ObjectiveSense.MAXIMIZE))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 10.0))
    m.solve(presolve=False)

    scenario = Scenario("alt_check", objective_changes={"x1": -0.01})
    res = m.evaluate_scenario(scenario)
    assert isinstance(res, ScenarioResult)


# -----------------------------------------------------------------------------
# Test 16: Presolve-Fixed Variable
# -----------------------------------------------------------------------------
def test_16_presolve_fixed_variable():
    m = Model(name="presolve_fixed")
    x1 = Variable("x1", lower_bound=5.0, upper_bound=5.0)
    x2 = Variable("x2", lower_bound=0.0, upper_bound=10.0)
    m.add_variable(x1)
    m.add_variable(x2)
    m.set_objective(Objective({"x1": 2.0, "x2": 3.0}, sense=ObjectiveSense.MAXIMIZE))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 12.0))
    # Solve with presolve=True
    m.solve(presolve=True)

    scenario = Scenario("presolve_test", rhs_changes={"c1": 1.0}, objective_changes={"x2": 0.1})
    res = m.evaluate_scenario(scenario)
    assert res.basis_preserved is True
    assert "x1" in res.variable_values
    assert pytest.approx(res.variable_values["x1"], rel=1e-5) == 5.0


# -----------------------------------------------------------------------------
# Test 17: Presolve-Redundant Constraint
# -----------------------------------------------------------------------------
def test_17_presolve_redundant_constraint():
    m = Model(name="redundant_con")
    x1 = Variable("x1", lower_bound=0.0, upper_bound=10.0)
    m.add_variable(x1)
    m.set_objective(Objective({"x1": 1.0}, sense=ObjectiveSense.MAXIMIZE))
    m.add_constraint(Constraint("tight", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.add_constraint(Constraint("redundant", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 100.0))
    m.solve(presolve=True)

    # Perturb redundant constraint slightly
    scenario = Scenario("perturb_red", rhs_changes={"redundant": 10.0})
    res = m.evaluate_scenario(scenario)
    assert res.basis_preserved is True


# -----------------------------------------------------------------------------
# Test 18: Negative RHS / Sign-Flipped Row
# -----------------------------------------------------------------------------
def test_18_negative_rhs_sign_flipped_row():
    m = Model(name="neg_rhs")
    x1 = Variable("x1", lower_bound=0.0)
    x2 = Variable("x2", lower_bound=0.0)
    m.add_variable(x1)
    m.add_variable(x2)
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, sense=ObjectiveSense.MINIMIZE))
    # -x1 - x2 <= -5 is equivalent to x1 + x2 >= 5
    m.add_constraint(Constraint("neg_c", {"x1": -1.0, "x2": -1.0}, ConstraintSense.LESS_EQUAL, -5.0))
    m.add_constraint(Constraint("pos_c", {"x1": 1.0, "x2": 2.0}, ConstraintSense.LESS_EQUAL, 10.0))
    m.solve(presolve=False)

    scenario = Scenario("shift_neg", rhs_changes={"neg_c": -1.0})
    res = m.evaluate_scenario(scenario)
    assert isinstance(res, ScenarioResult)


# -----------------------------------------------------------------------------
# Test 19: Variable Bound Transformation
# -----------------------------------------------------------------------------
def test_19_variable_bound_transformation():
    m = Model(name="bound_shift")
    x1 = Variable("x1", lower_bound=10.0, upper_bound=50.0)
    x2 = Variable("x2", lower_bound=5.0, upper_bound=30.0)
    m.add_variable(x1)
    m.add_variable(x2)
    m.set_objective(Objective({"x1": 2.0, "x2": 1.0}, sense=ObjectiveSense.MAXIMIZE))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 30.0))
    m.solve(presolve=False)

    scenario = Scenario("test_bounds", rhs_changes={"c1": 2.0}, objective_changes={"x1": 0.1})
    res = m.evaluate_scenario(scenario)
    assert res.basis_preserved is True
    assert res.variable_values["x1"] >= 10.0 - 1e-7
    assert res.variable_values["x2"] >= 5.0 - 1e-7


# -----------------------------------------------------------------------------
# Test 20: Scenario Object Validation
# -----------------------------------------------------------------------------
def test_20_scenario_object_validation():
    model = _build_standard_lp()
    model.solve(presolve=False)

    # Unknown constraint
    bad_rhs = Scenario("bad_rhs", rhs_changes={"nonexistent_constraint": 5.0})
    with pytest.raises(KeyError, match="Constraint 'nonexistent_constraint'"):
        model.evaluate_scenario(bad_rhs)

    # Unknown variable
    bad_obj = Scenario("bad_obj", objective_changes={"nonexistent_variable": 1.0})
    with pytest.raises(KeyError, match="Variable 'nonexistent_variable'"):
        model.evaluate_scenario(bad_obj)


# -----------------------------------------------------------------------------
# Test 21: Scenario Comparison
# -----------------------------------------------------------------------------
def test_21_scenario_comparison():
    model = _build_standard_lp()
    model.solve(presolve=False)

    scenarios = [
        Scenario("base", rhs_changes={}),
        Scenario("demand_up", rhs_changes={"labor": 5.0}),
        Scenario("cost_up", objective_changes={"x1": -0.5}),
    ]
    comparison = model.compare_scenarios(scenarios)

    assert isinstance(comparison, ScenarioComparison)
    assert pytest.approx(comparison.baseline_objective, rel=1e-5) == 180.0
    assert len(comparison.scenarios) == 3
    assert "demand_up" in comparison.scenarios
    assert "cost_up" in comparison.scenarios

    summary = comparison.summary()
    assert "SCENARIO COMPARISON REPORT" in summary
    assert "demand_up" in summary
    assert "cost_up" in summary


# -----------------------------------------------------------------------------
# Test 22: Break-Even Inside Sensitivity Interval
# -----------------------------------------------------------------------------
def test_22_break_even_inside_sensitivity_interval():
    model = _build_standard_lp()
    model.solve(presolve=False)

    # Baseline obj = 180. labor shadow price = 1.0.
    # Target obj = 190 -> requires delta_labor = +10.
    be = model.break_even(parameter="labor", target_value=190.0)

    assert isinstance(be, BreakEvenResult)
    assert be.parameter_name == "labor"
    assert be.is_reachable_within_basis is True
    assert pytest.approx(be.required_delta, rel=1e-5) == 10.0
    assert pytest.approx(be.required_parameter_value, rel=1e-5) == 110.0

    # Also test with direction="increase"
    be_dir = model.break_even(parameter="labor", target_value=190.0, direction="increase")
    assert be_dir.is_reachable_within_basis is True

    # Test conflicting direction="decrease"
    be_conflict = model.break_even(parameter="labor", target_value=190.0, direction="decrease")
    assert be_conflict.is_reachable_within_basis is False


# -----------------------------------------------------------------------------
# Test 23: Break-Even Outside Sensitivity Interval
# -----------------------------------------------------------------------------
def test_23_break_even_outside_sensitivity_interval():
    model = _build_standard_lp()
    model.solve(presolve=False)

    # Baseline obj = 180. Labor allowable increase is 20 (max RHS = 120, obj = 200).
    # Target obj = 300 -> exceeds current basis range!
    be = model.break_even(parameter="labor", target_value=300.0)

    assert be.is_reachable_within_basis is False
    assert "exceeds allowable basis range" in be.message


# -----------------------------------------------------------------------------
# Test 24: MILP Rejection
# -----------------------------------------------------------------------------
def test_24_milp_rejection():
    m = Model(name="milp_model")
    x1 = Variable("x1", variable_type=VariableType.INTEGER, lower_bound=0.0)
    m.add_variable(x1)
    m.set_objective(Objective({"x1": 1.0}, sense=ObjectiveSense.MAXIMIZE))
    m.add_constraint(Constraint("c1", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))

    scenario = Scenario("milp_scenario", rhs_changes={"c1": 1.0})

    with pytest.raises(ValueError, match="Sensitivity analysis is only supported for continuous LP models."):
        m.evaluate_scenario(scenario)

    with pytest.raises(ValueError, match="Sensitivity analysis is only supported for continuous LP models."):
        m.compare_scenarios([scenario])

    with pytest.raises(ValueError, match="Sensitivity analysis is only supported for continuous LP models."):
        m.break_even(parameter="c1", target_value=10.0)


# -----------------------------------------------------------------------------
# Test 25: Infeasible Model Rejection
# -----------------------------------------------------------------------------
def test_25_infeasible_model_rejection():
    m = Model(name="infeasible_lp")
    x1 = Variable("x1", lower_bound=10.0)
    m.add_variable(x1)
    m.set_objective(Objective({"x1": 1.0}, sense=ObjectiveSense.MAXIMIZE))
    m.add_constraint(Constraint("c1", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.solve(presolve=False)

    scenario = Scenario("inf_check", rhs_changes={"c1": 1.0})
    with pytest.raises(ValueError, match="optimal basis"):
        m.evaluate_scenario(scenario)


# -----------------------------------------------------------------------------
# Test 26: Unbounded Model Rejection
# -----------------------------------------------------------------------------
def test_26_unbounded_model_rejection():
    m = Model(name="unbounded_lp")
    x1 = Variable("x1", lower_bound=0.0)
    x2 = Variable("x2", lower_bound=0.0)
    m.add_variable(x1)
    m.add_variable(x2)
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, sense=ObjectiveSense.MAXIMIZE))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": -1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    sol = m.solve(presolve=False)
    assert sol.status is SolutionStatus.UNBOUNDED

    scenario = Scenario("unb_check", rhs_changes={})
    with pytest.raises(ValueError, match="optimal basis"):
        m.evaluate_scenario(scenario)


# -----------------------------------------------------------------------------
# Test 27: Missing Basis Rejection
# -----------------------------------------------------------------------------
def test_27_missing_basis_rejection():
    m = Model(name="unsolved_lp")
    x1 = Variable("x1", lower_bound=0.0)
    m.add_variable(x1)
    m.set_objective(Objective({"x1": 1.0}, sense=ObjectiveSense.MAXIMIZE))
    m.add_constraint(Constraint("c1", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 10.0))

    scenario = Scenario("no_basis", rhs_changes={"c1": 1.0})
    with pytest.raises(ValueError, match="No optimal basis available"):
        m.evaluate_scenario(scenario)


# -----------------------------------------------------------------------------
# Test 28: Original Model Remains Unchanged After Scenario Evaluation
# -----------------------------------------------------------------------------
def test_28_original_model_remains_unchanged():
    model = _build_standard_lp()
    model.solve(presolve=False)

    orig_labor_rhs = model.constraints["labor"].rhs
    orig_x1_obj = model.objective.coefficients["x1"]

    scenario = Scenario(
        name="mutate_attempt",
        rhs_changes={"labor": 20.0},
        objective_changes={"x1": 15.0},
    )
    # Evaluate with re-solve enabled
    res = model.evaluate_scenario(scenario, resolve_if_needed=True)

    # Verify original model is completely unmutated
    assert model.constraints["labor"].rhs == orig_labor_rhs
    assert model.objective.coefficients["x1"] == orig_x1_obj


# -----------------------------------------------------------------------------
# Test 29: Explicit Re-Solve Behavior
# -----------------------------------------------------------------------------
def test_29_explicit_resolve_behavior():
    model = _build_standard_lp()
    model.solve(presolve=False)

    # Force basis change with huge RHS change
    scenario = Scenario("extreme_scenario", rhs_changes={"labor": -80.0})

    # Without re-solve
    res_no_resolve = model.evaluate_scenario(scenario, resolve_if_needed=False)
    assert res_no_resolve.basis_preserved is False
    assert res_no_resolve.solver_resolved is False
    assert res_no_resolve.objective_value is None

    # With re-solve
    res_with_resolve = model.evaluate_scenario(scenario, resolve_if_needed=True)
    assert res_with_resolve.basis_preserved is False
    assert res_with_resolve.solver_resolved is True
    assert res_with_resolve.re_solve_solution is not None
    assert res_with_resolve.objective_value is not None
    assert res_with_resolve.status == "re_solved"


# -----------------------------------------------------------------------------
# Test 30: Backward Compatibility with All Stage 8.45 APIs
# -----------------------------------------------------------------------------
def test_30_backward_compatibility_with_stage_8_45():
    model = _build_standard_lp()
    model.solve(presolve=False)

    # 1D sensitivity analysis
    report = model.sensitivity_analysis()
    assert isinstance(report, SensitivityReport)
    assert "labor" in report.rhs
    assert "x1" in report.objective

    labor_rng = report.rhs["labor"]
    assert isinstance(labor_rng, SensitivityRange)
    assert labor_rng.current_value == 100.0
    assert labor_rng.contains(105.0)

    direct_report = compute_sensitivity(model)
    assert isinstance(direct_report, SensitivityReport)
