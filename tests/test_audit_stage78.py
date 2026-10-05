import pytest
from optiforge.core.solution import SolutionStatus
from optiforge.milp import solve_milp, AdaptivePresolveStrategy, PresolveDecision, PresolveOperation
from optiforge.milp.adaptive_presolve_executor import adaptive_presolve_model
from optiforge.milp.fingerprint import fingerprint_model
from optiforge.milp.solver import MILPSolver
from optiforge.model import Constraint, ConstraintSense, Model, Objective, ObjectiveSense, Variable, VariableType
from optiforge.presolve.presolve import PresolveOperation as Stage5AOp, _presolve_model_impl, presolve_model, postsolve_solution

# ===== F-01 OBJECTIVE RECONSTRUCTION =====
def test_f01_max_single_fixed_obj_constant():
    m = Model('f01a'); m.add_variable(Variable('x1', lower_bound=5.0, upper_bound=5.0))
    m.set_objective(Objective({'x1': 10.0}, ObjectiveSense.MAXIMIZE))
    r = adaptive_presolve_model(m, AdaptivePresolveStrategy().decide(fingerprint_model(m)))
    assert r.status is SolutionStatus.FEASIBLE
    assert r.fixed_variables.get('x1') == pytest.approx(5.0)
    assert r.objective_constant == pytest.approx(50.0), f'Got {r.objective_constant}'

def test_f01_min_single_fixed_obj_constant():
    m = Model('f01b'); m.add_variable(Variable('x1', lower_bound=3.0, upper_bound=3.0))
    m.set_objective(Objective({'x1': 7.0}, ObjectiveSense.MINIMIZE))
    r = adaptive_presolve_model(m, AdaptivePresolveStrategy().decide(fingerprint_model(m)))
    assert r.objective_constant == pytest.approx(21.0), f'Got {r.objective_constant}'

def test_f01_negative_coeff():
    m = Model('f01c'); m.add_variable(Variable('x1', lower_bound=2.0, upper_bound=2.0))
    m.set_objective(Objective({'x1': -3.0}, ObjectiveSense.MAXIMIZE))
    r = adaptive_presolve_model(m, AdaptivePresolveStrategy().decide(fingerprint_model(m)))
    assert r.objective_constant == pytest.approx(-6.0), f'Got {r.objective_constant}'

def test_f01_mixed_coeff():
    m = Model('f01d')
    m.add_variable(Variable('x1', lower_bound=4.0, upper_bound=4.0))
    m.add_variable(Variable('x2', lower_bound=3.0, upper_bound=3.0))
    m.set_objective(Objective({'x1': 5.0, 'x2': -2.0}, ObjectiveSense.MAXIMIZE))
    r = adaptive_presolve_model(m, AdaptivePresolveStrategy().decide(fingerprint_model(m)))
    assert r.objective_constant == pytest.approx(14.0), f'Got {r.objective_constant}'

def test_f01_prior_failure_55_vs_5():
    '''Prior bug: solver returned 5 instead of 55 (forgot objective_constant).'''
    m = Model('f01e')
    m.add_variable(Variable('x_fixed', lower_bound=5.0, upper_bound=5.0))
    m.add_variable(Variable('x_remain', lower_bound=0.0))
    m.add_constraint(Constraint('c1', {'x_remain': 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.set_objective(Objective({'x_fixed': 10.0, 'x_remain': 1.0}, ObjectiveSense.MAXIMIZE))
    sol = MILPSolver(adaptive_presolve=AdaptivePresolveStrategy()).solve(m)
    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.objective_value == pytest.approx(55.0, abs=1e-4), f'Got {sol.objective_value} (expected 55, bug returns 5)'
    assert sol.variable_values.get('x_fixed') == pytest.approx(5.0, abs=1e-4)

def test_f01_prior_failure_54_vs_4():
    '''Prior bug: solver returned 4 instead of 54.'''
    m = Model('f01f')
    m.add_variable(Variable('x_fixed1', lower_bound=5.0, upper_bound=5.0))
    m.add_variable(Variable('x_remain', lower_bound=0.0))
    m.add_constraint(Constraint('c1', {'x_remain': 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.set_objective(Objective({'x_fixed1': 10.0, 'x_remain': 1.0}, ObjectiveSense.MAXIMIZE))
    sol = MILPSolver(adaptive_presolve=AdaptivePresolveStrategy()).solve(m)
    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.objective_value == pytest.approx(54.0, abs=1e-4), f'Got {sol.objective_value} (expected 54, bug returns 4)'

def test_f01_direct_eval_matches_reported():
    '''Independently compute obj from variable values and compare to reported.'''
    m = Model('f01g')
    m.add_variable(Variable('x1', lower_bound=0.0, upper_bound=10.0))
    m.add_variable(Variable('x2', lower_bound=0.0, upper_bound=5.0))
    m.add_variable(Variable('x3', lower_bound=0.0, upper_bound=3.0))
    m.add_constraint(Constraint('c1', {'x1': 1.0, 'x2': 1.0, 'x3': 1.0}, ConstraintSense.LESS_EQUAL, 12.0))
    m.set_objective(Objective({'x1': 2.0, 'x2': 3.0, 'x3': 5.0}, ObjectiveSense.MAXIMIZE))
    sol = MILPSolver(adaptive_presolve=AdaptivePresolveStrategy()).solve(m)
    assert sol.status is SolutionStatus.OPTIMAL
    v = sol.variable_values
    direct = 2.0*v.get('x1',0) + 3.0*v.get('x2',0) + 5.0*v.get('x3',0)
    assert sol.objective_value == pytest.approx(direct, abs=1e-4), f'Reported {sol.objective_value} != direct {direct}'

def test_f01_lp_with_fixed_vars():
    '''Pure LP: presolve fixes x1, LP solves x2. obj = 10*3 + 1*5 = 35.'''
    m = Model('f01h')
    m.add_variable(Variable('x1', lower_bound=3.0, upper_bound=3.0))
    m.add_variable(Variable('x2', lower_bound=0.0, upper_bound=5.0))
    m.add_constraint(Constraint('c1', {'x2': 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.set_objective(Objective({'x1': 10.0, 'x2': 1.0}, ObjectiveSense.MAXIMIZE))
    sol = MILPSolver(adaptive_presolve=AdaptivePresolveStrategy()).solve(m)
    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.objective_value == pytest.approx(35.0, abs=1e-4), f'Got {sol.objective_value}'

def test_f01_milp_fixed_plus_binary():
    '''MILP: x1 fixed=4 (coeff=3), y binary=1 (coeff=5) => 17.'''
    m = Model('f01i')
    m.add_variable(Variable('x1', lower_bound=4.0, upper_bound=4.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable('y', lower_bound=0.0, upper_bound=1.0, variable_type=VariableType.BINARY))
    m.add_constraint(Constraint('c1', {'y': 1.0}, ConstraintSense.LESS_EQUAL, 1.0))
    m.set_objective(Objective({'x1': 3.0, 'y': 5.0}, ObjectiveSense.MAXIMIZE))
    sol = MILPSolver(adaptive_presolve=AdaptivePresolveStrategy()).solve(m)
    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.objective_value == pytest.approx(17.0, abs=1e-4), f'Got {sol.objective_value}'

# ===== F-02 FULLY REDUCED MODEL =====
def test_f02_feasible_fully_reduced_not_infeasible():
    '''CRITICAL: feasible fully-reduced model must not be INFEASIBLE.'''
    m = Model('f02a')
    m.add_variable(Variable('x1', lower_bound=2.0, upper_bound=2.0))
    m.add_variable(Variable('x2', lower_bound=3.0, upper_bound=3.0))
    m.set_objective(Objective({'x1': 1.0, 'x2': 1.0}, ObjectiveSense.MAXIMIZE))
    sol = MILPSolver(adaptive_presolve=AdaptivePresolveStrategy()).solve(m)
    assert sol.status is not SolutionStatus.INFEASIBLE, 'CRITICAL: feasible fully-reduced model incorrectly INFEASIBLE'
    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.objective_value == pytest.approx(5.0, abs=1e-4)

def test_f02_original_vars_reconstructed():
    '''All original variables must appear in solution after postsolve.'''
    m = Model('f02b')
    m.add_variable(Variable('a', lower_bound=1.0, upper_bound=1.0))
    m.add_variable(Variable('b', lower_bound=2.0, upper_bound=2.0))
    m.add_variable(Variable('c', lower_bound=3.0, upper_bound=3.0))
    m.set_objective(Objective({'a': 1.0, 'b': 1.0, 'c': 1.0}, ObjectiveSense.MAXIMIZE))
    sol = MILPSolver(adaptive_presolve=AdaptivePresolveStrategy()).solve(m)
    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.variable_values.get('a') == pytest.approx(1.0, abs=1e-4)
    assert sol.variable_values.get('b') == pytest.approx(2.0, abs=1e-4)
    assert sol.variable_values.get('c') == pytest.approx(3.0, abs=1e-4)
    assert sol.objective_value == pytest.approx(6.0, abs=1e-4)

def test_f02_infeasible_model_detected():
    m = Model('f02c')
    m.add_variable(Variable('x1', lower_bound=0.0, upper_bound=5.0))
    m.add_constraint(Constraint('c1', {'x1': 1.0}, ConstraintSense.GREATER_EQUAL, 6.0))
    m.set_objective(Objective({'x1': 1.0}, ObjectiveSense.MAXIMIZE))
    sol = MILPSolver(adaptive_presolve=AdaptivePresolveStrategy()).solve(m)
    assert sol.status is SolutionStatus.INFEASIBLE

def test_f02_max_fully_reduced_obj():
    '''MAX fully reduced: 3*4 + 5*6 = 42.'''
    m = Model('f02d')
    m.add_variable(Variable('x1', lower_bound=4.0, upper_bound=4.0))
    m.add_variable(Variable('x2', lower_bound=6.0, upper_bound=6.0))
    m.set_objective(Objective({'x1': 3.0, 'x2': 5.0}, ObjectiveSense.MAXIMIZE))
    sol = MILPSolver(adaptive_presolve=AdaptivePresolveStrategy()).solve(m)
    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.objective_value == pytest.approx(42.0, abs=1e-4)

def test_f02_min_fully_reduced_obj():
    '''MIN fully reduced: 4*2 + 7*3 = 29.'''
    m = Model('f02e')
    m.add_variable(Variable('x1', lower_bound=2.0, upper_bound=2.0))
    m.add_variable(Variable('x2', lower_bound=3.0, upper_bound=3.0))
    m.set_objective(Objective({'x1': 4.0, 'x2': 7.0}, ObjectiveSense.MINIMIZE))
    sol = MILPSolver(adaptive_presolve=AdaptivePresolveStrategy()).solve(m)
    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.objective_value == pytest.approx(29.0, abs=1e-4)

def test_f02_negative_coeff_fully_reduced():
    '''(-3)*5 + 4*2 = -7.'''
    m = Model('f02f')
    m.add_variable(Variable('x1', lower_bound=5.0, upper_bound=5.0))
    m.add_variable(Variable('x2', lower_bound=2.0, upper_bound=2.0))
    m.set_objective(Objective({'x1': -3.0, 'x2': 4.0}, ObjectiveSense.MAXIMIZE))
    sol = MILPSolver(adaptive_presolve=AdaptivePresolveStrategy()).solve(m)
    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.objective_value == pytest.approx(-7.0, abs=1e-4)

def test_f02_constraints_satisfied_after_postsolve():
    m = Model('f02g')
    m.add_variable(Variable('x1', lower_bound=0.0, upper_bound=5.0))
    m.add_variable(Variable('x2', lower_bound=0.0, upper_bound=3.0))
    m.add_constraint(Constraint('c1', {'x1': 1.0, 'x2': 1.0}, ConstraintSense.LESS_EQUAL, 8.0))
    m.add_constraint(Constraint('eq1', {'x1': 1.0}, ConstraintSense.EQUAL, 4.0))
    m.set_objective(Objective({'x1': 2.0, 'x2': 1.0}, ObjectiveSense.MAXIMIZE))
    sol = MILPSolver(adaptive_presolve=AdaptivePresolveStrategy()).solve(m)
    assert sol.status is SolutionStatus.OPTIMAL
    x1, x2 = sol.variable_values.get('x1', 0.0), sol.variable_values.get('x2', 0.0)
    assert x1 + x2 <= 8.0 + 1e-6
    assert abs(x1 - 4.0) <= 1e-5

# ===== F-03 GENUINE ADAPTIVITY =====
def test_f03_empty_model_no_operations():
    m = Model('f03a')
    dec = AdaptivePresolveStrategy().decide(fingerprint_model(m))
    assert len(dec.enabled_operations) == 0
    assert 'no variables' in dec.reason.lower()

def test_f03_zero_var_branch_before_min_size():
    '''Zero-variable branch fires BEFORE min_problem_size check.'''
    m = Model('f03b')
    dec = AdaptivePresolveStrategy(min_problem_size_for_adaptive=5).decide(fingerprint_model(m))
    assert len(dec.enabled_operations) == 0
    assert 'no variables' in dec.reason.lower(), f'Got: {dec.reason}'

def test_f03_no_constraints_limited_ops():
    m = Model('f03c')
    for i in range(15): m.add_variable(Variable(f'x{i}', lower_bound=0.0, upper_bound=5.0))
    m.set_objective(Objective({f'x{i}': 1.0 for i in range(15)}, ObjectiveSense.MAXIMIZE))
    dec = AdaptivePresolveStrategy().decide(fingerprint_model(m))
    assert PresolveOperation.BOUND_VALIDATION in dec.enabled_operations
    assert PresolveOperation.ZERO_COLUMN_ELIMINATION in dec.enabled_operations
    assert PresolveOperation.ACTIVITY_CHECKS not in dec.enabled_operations, 'Activity checks wasteful for no-constraint model'

def test_f03_large_model_enables_fixed_point():
    m = Model('f03d')
    for i in range(60): m.add_variable(Variable(f'x{i}', lower_bound=0.0))
    m.add_constraint(Constraint('c1', {f'x{i}': 1.0 for i in range(60)}, ConstraintSense.LESS_EQUAL, 100.0))
    m.set_objective(Objective({f'x{i}': 1.0 for i in range(60)}, ObjectiveSense.MAXIMIZE))
    dec = AdaptivePresolveStrategy().decide(fingerprint_model(m))
    assert PresolveOperation.FIXED_POINT_ITERATION in dec.enabled_operations

def test_f03_different_structures_different_ops():
    '''CRITICAL F-03: two structurally different models must produce different op sets.'''
    ma = Model('f03e')
    for i in range(15): ma.add_variable(Variable(f'x{i}', lower_bound=0.0, upper_bound=5.0))
    ma.set_objective(Objective({f'x{i}': 1.0 for i in range(15)}, ObjectiveSense.MAXIMIZE))

    mb = Model('f03f')
    for i in range(60): mb.add_variable(Variable(f'x{i}', lower_bound=0.0))
    for i in range(60): mb.add_constraint(Constraint(f'c{i}', {f'x{i}': 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    mb.set_objective(Objective({f'x{i}': 1.0 for i in range(60)}, ObjectiveSense.MAXIMIZE))

    strategy = AdaptivePresolveStrategy()
    dec_a = strategy.decide(fingerprint_model(ma))
    dec_b = strategy.decide(fingerprint_model(mb))
    assert dec_a.enabled_operations != dec_b.enabled_operations, (
        f'CRITICAL F-03: Identical ops for structurally different models. A={dec_a.enabled_operations} B={dec_b.enabled_operations}'
    )

def test_f03_determinism():
    m = Model('f03g')
    for i in range(15): m.add_variable(Variable(f'x{i}', lower_bound=0.0))
    m.add_constraint(Constraint('c1', {f'x{i}': 1.0 for i in range(15)}, ConstraintSense.LESS_EQUAL, 100.0))
    m.set_objective(Objective({f'x{i}': 1.0 for i in range(15)}, ObjectiveSense.MAXIMIZE))
    strategy = AdaptivePresolveStrategy()
    fp = fingerprint_model(m)
    decisions = [strategy.decide(fp) for _ in range(5)]
    for i in range(1, 5):
        assert decisions[0].enabled_operations == decisions[i].enabled_operations

# ===== F-05 PURE LP PATH =====
def test_f05_lp_without_adaptive():
    m = Model('f05a')
    m.add_variable(Variable('x1', lower_bound=0.0)); m.add_variable(Variable('x2', lower_bound=0.0))
    m.add_constraint(Constraint('c1', {'x1': 1.0, 'x2': 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.set_objective(Objective({'x1': 2.0, 'x2': 1.0}, ObjectiveSense.MAXIMIZE))
    sol = MILPSolver().solve(m)
    assert sol.status is SolutionStatus.OPTIMAL and sol.objective_value == pytest.approx(8.0, abs=1e-4)

def test_f05_lp_with_adaptive_same_result():
    m = Model('f05b')
    m.add_variable(Variable('x1', lower_bound=0.0)); m.add_variable(Variable('x2', lower_bound=0.0))
    m.add_constraint(Constraint('c1', {'x1': 1.0, 'x2': 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.set_objective(Objective({'x1': 2.0, 'x2': 1.0}, ObjectiveSense.MAXIMIZE))
    sol = MILPSolver(adaptive_presolve=AdaptivePresolveStrategy()).solve(m)
    assert sol.status is SolutionStatus.OPTIMAL and sol.objective_value == pytest.approx(8.0, abs=1e-4)

def test_f05_lp_consistency_with_and_without():
    m = Model('f05c')
    m.add_variable(Variable('x1', lower_bound=0.0)); m.add_variable(Variable('x2', lower_bound=0.0))
    m.add_constraint(Constraint('c1', {'x1': 1.0, 'x2': 1.0}, ConstraintSense.LESS_EQUAL, 6.0))
    m.add_constraint(Constraint('c2', {'x1': 2.0, 'x2': 1.0}, ConstraintSense.LESS_EQUAL, 10.0))
    m.set_objective(Objective({'x1': 3.0, 'x2': 1.0}, ObjectiveSense.MAXIMIZE))
    s1 = MILPSolver().solve(m); s2 = MILPSolver(adaptive_presolve=AdaptivePresolveStrategy()).solve(m)
    assert s1.status == s2.status
    assert s1.objective_value == pytest.approx(s2.objective_value, abs=1e-4)

def test_f05_lp_fixed_vars_obj_correct():
    '''LP with presolve-fixed x1: 2*5 + 1*3 = 13.'''
    m = Model('f05d')
    m.add_variable(Variable('x_fixed', lower_bound=5.0, upper_bound=5.0))
    m.add_variable(Variable('x_free', lower_bound=0.0))
    m.add_constraint(Constraint('c1', {'x_free': 1.0}, ConstraintSense.LESS_EQUAL, 3.0))
    m.set_objective(Objective({'x_fixed': 2.0, 'x_free': 1.0}, ObjectiveSense.MAXIMIZE))
    sol = MILPSolver(adaptive_presolve=AdaptivePresolveStrategy()).solve(m)
    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.objective_value == pytest.approx(13.0, abs=1e-4), f'Got {sol.objective_value}'

def test_f05_lp_all_fixed():
    m = Model('f05e')
    m.add_variable(Variable('x1', lower_bound=3.0, upper_bound=3.0))
    m.add_variable(Variable('x2', lower_bound=4.0, upper_bound=4.0))
    m.set_objective(Objective({'x1': 2.0, 'x2': 1.0}, ObjectiveSense.MAXIMIZE))
    sol = MILPSolver(adaptive_presolve=AdaptivePresolveStrategy()).solve(m)
    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.objective_value == pytest.approx(10.0, abs=1e-4)

def test_f05_lp_maximize():
    m = Model('f05f'); m.add_variable(Variable('x', lower_bound=0.0))
    m.add_constraint(Constraint('c1', {'x': 1.0}, ConstraintSense.LESS_EQUAL, 10.0))
    m.set_objective(Objective({'x': 5.0}, ObjectiveSense.MAXIMIZE))
    sol = MILPSolver(adaptive_presolve=AdaptivePresolveStrategy()).solve(m)
    assert sol.status is SolutionStatus.OPTIMAL and sol.objective_value == pytest.approx(50.0, abs=1e-4)

def test_f05_lp_minimize():
    m = Model('f05g'); m.add_variable(Variable('x', lower_bound=0.0))
    m.add_constraint(Constraint('c1', {'x': 1.0}, ConstraintSense.GREATER_EQUAL, 2.0))
    m.set_objective(Objective({'x': 3.0}, ObjectiveSense.MINIMIZE))
    sol = MILPSolver(adaptive_presolve=AdaptivePresolveStrategy()).solve(m)
    assert sol.status is SolutionStatus.OPTIMAL and sol.objective_value == pytest.approx(6.0, abs=1e-4)

# ===== F-06 SIZE BOUNDARIES =====
def test_f06_zero_vars_empty_ops_before_min_size():
    m = Model('f06a')
    dec = AdaptivePresolveStrategy(min_problem_size_for_adaptive=5).decide(fingerprint_model(m))
    assert len(dec.enabled_operations) == 0 and 'no variables' in dec.reason.lower()

def test_f06_one_var_small_path():
    m = Model('f06b'); m.add_variable(Variable('x', lower_bound=0.0))
    m.add_constraint(Constraint('c1', {'x': 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.set_objective(Objective({'x': 1.0}, ObjectiveSense.MAXIMIZE))
    dec = AdaptivePresolveStrategy(min_problem_size_for_adaptive=10).decide(fingerprint_model(m))
    assert 'too small' in dec.reason.lower() or 'problem too small' in dec.reason.lower()

def test_f06_nine_vars_all_ops():
    m = Model('f06c')
    for i in range(9): m.add_variable(Variable(f'x{i}', lower_bound=0.0))
    m.add_constraint(Constraint('c1', {f'x{i}': 1.0 for i in range(9)}, ConstraintSense.LESS_EQUAL, 50.0))
    m.set_objective(Objective({f'x{i}': 1.0 for i in range(9)}, ObjectiveSense.MAXIMIZE))
    dec = AdaptivePresolveStrategy(min_problem_size_for_adaptive=10).decide(fingerprint_model(m))
    assert dec.enabled_operations == frozenset(PresolveOperation), f'9-var should enable all. Got {dec.enabled_operations}'

def test_f06_eleven_vars_adaptive_path():
    m = Model('f06d')
    for i in range(11): m.add_variable(Variable(f'x{i}', lower_bound=0.0))
    m.add_constraint(Constraint('c1', {f'x{i}': 1.0 for i in range(11)}, ConstraintSense.LESS_EQUAL, 50.0))
    m.set_objective(Objective({f'x{i}': 1.0 for i in range(11)}, ObjectiveSense.MAXIMIZE))
    dec = AdaptivePresolveStrategy(min_problem_size_for_adaptive=10).decide(fingerprint_model(m))
    assert 'problem too small' not in dec.reason.lower()

def test_f06_empty_model_end_to_end_no_crash():
    m = Model('f06e')
    try:
        sol = MILPSolver(adaptive_presolve=AdaptivePresolveStrategy()).solve(m)
        assert sol.status in (SolutionStatus.INFEASIBLE, SolutionStatus.OPTIMAL)
    except Exception as e:
        pytest.fail(f'F-06: Empty model raised {e}')

# ===== F-07 OPERATION ORDER =====
def test_f07_selected_ops_in_stage5a_order():
    stage5a_order = [
        PresolveOperation.BOUND_VALIDATION, PresolveOperation.ZERO_ROW_REMOVAL,
        PresolveOperation.ZERO_COLUMN_ELIMINATION, PresolveOperation.SINGLETON_CONSTRAINTS,
        PresolveOperation.ACTIVITY_CHECKS, PresolveOperation.BOUND_PROPAGATION,
        PresolveOperation.FIXED_POINT_ITERATION,
    ]
    m = Model('f07a')
    for i in range(15): m.add_variable(Variable(f'x{i}', lower_bound=0.0))
    m.add_constraint(Constraint('c1', {f'x{i}': 1.0 for i in range(15)}, ConstraintSense.LESS_EQUAL, 100.0))
    m.set_objective(Objective({f'x{i}': 1.0 for i in range(15)}, ObjectiveSense.MAXIMIZE))
    dec = AdaptivePresolveStrategy().decide(fingerprint_model(m))
    last = -1
    for op in dec.selected_operations:
        idx = stage5a_order.index(op)
        assert idx > last, f'F-07: {op.value} out of Stage5A order in selected_operations'
        last = idx

def test_f07_enabled_selected_match():
    m = Model('f07b')
    for i in range(15): m.add_variable(Variable(f'x{i}', lower_bound=0.0))
    m.add_constraint(Constraint('c1', {f'x{i}': 1.0 for i in range(15)}, ConstraintSense.LESS_EQUAL, 100.0))
    m.set_objective(Objective({f'x{i}': 1.0 for i in range(15)}, ObjectiveSense.MAXIMIZE))
    dec = AdaptivePresolveStrategy().decide(fingerprint_model(m))
    assert set(dec.enabled_operations) == set(dec.selected_operations)

def test_f07_stage5a_respects_enabled_operations():
    m = Model('f07c')
    m.add_variable(Variable('x1', lower_bound=0.0, upper_bound=5.0))
    m.add_variable(Variable('x2', lower_bound=0.0, upper_bound=5.0))
    m.add_constraint(Constraint('c1', {'x1': 1.0, 'x2': 1.0}, ConstraintSense.LESS_EQUAL, 8.0))
    m.set_objective(Objective({'x1': 1.0, 'x2': 1.0}, ObjectiveSense.MAXIMIZE))
    r1 = _presolve_model_impl(m, enabled_operations={Stage5AOp.BOUND_VALIDATION})
    r2 = _presolve_model_impl(m, enabled_operations={Stage5AOp.BOUND_VALIDATION, Stage5AOp.ACTIVITY_CHECKS})
    assert r1.status is SolutionStatus.FEASIBLE
    assert r2.status is SolutionStatus.FEASIBLE

# ===== STAGE 5A SINGLE SOURCE =====
def test_sst_executor_delegates_to_stage5a():
    m = Model('sst_a')
    m.add_variable(Variable('x', lower_bound=0.0, upper_bound=3.0))
    m.add_constraint(Constraint('c1', {'x': 1.0}, ConstraintSense.LESS_EQUAL, 2.0))
    m.set_objective(Objective({'x': 5.0}, ObjectiveSense.MAXIMIZE))
    r_stage5a = presolve_model(m)
    dec = AdaptivePresolveStrategy(min_problem_size_for_adaptive=100).decide(fingerprint_model(m))
    r_exec = adaptive_presolve_model(m, dec)
    assert r_stage5a.status == r_exec.status
    assert r_stage5a.objective_constant == pytest.approx(r_exec.objective_constant)
    assert r_stage5a.fixed_variables == r_exec.fixed_variables

def test_sst_adaptive_none_uses_stage5a():
    m = Model('sst_b')
    m.add_variable(Variable('x1', lower_bound=0.0)); m.add_variable(Variable('x2', lower_bound=0.0))
    m.add_constraint(Constraint('c1', {'x1': 1.0, 'x2': 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.set_objective(Objective({'x1': 2.0, 'x2': 1.0}, ObjectiveSense.MAXIMIZE))
    s1 = MILPSolver(adaptive_presolve=None).solve(m)
    s2 = solve_milp(m, adaptive_presolve=None)
    assert s1.status == s2.status
    assert s1.objective_value == pytest.approx(s2.objective_value, abs=1e-6)

def test_sst_mapping_after_multi_reductions():
    m = Model('sst_c')
    m.add_variable(Variable('x1', lower_bound=0.0)); m.add_variable(Variable('x2', lower_bound=0.0)); m.add_variable(Variable('x3', lower_bound=0.0))
    m.add_constraint(Constraint('c1', {'x2': 1.0}, ConstraintSense.EQUAL, 3.0))
    m.add_constraint(Constraint('c2', {'x1': 1.0, 'x2': 1.0, 'x3': 1.0}, ConstraintSense.LESS_EQUAL, 10.0))
    m.set_objective(Objective({'x1': 1.0, 'x2': 2.0, 'x3': 3.0}, ObjectiveSense.MAXIMIZE))
    dec = AdaptivePresolveStrategy().decide(fingerprint_model(m))
    r = adaptive_presolve_model(m, dec)
    assert r.status is SolutionStatus.FEASIBLE
    assert r.fixed_variables.get('x2') == pytest.approx(3.0)
    assert r.objective_constant == pytest.approx(6.0, abs=1e-6), f'Got {r.objective_constant}'

# ===== CROSS-STAGE COMPATIBILITY =====
def test_cs_stage5a_regression():
    m = Model('cs_a')
    m.add_variable(Variable('x1', lower_bound=3.0, upper_bound=3.0))
    m.add_variable(Variable('x2', lower_bound=0.0))
    m.add_constraint(Constraint('c1', {'x1': 2.0, 'x2': 1.0}, ConstraintSense.LESS_EQUAL, 10.0))
    m.set_objective(Objective({'x1': 4.0, 'x2': 1.0}, ObjectiveSense.MAXIMIZE))
    r = presolve_model(m)
    assert r.status is SolutionStatus.FEASIBLE
    assert r.fixed_variables == {'x1': 3.0, 'x2': 4.0}
    assert r.objective_constant == pytest.approx(16.0)

def test_cs_bnb_binary_with_adaptive():
    m = Model('cs_b')
    m.add_variable(Variable('x1', variable_type=VariableType.BINARY))
    m.add_variable(Variable('x2', variable_type=VariableType.BINARY))
    m.add_constraint(Constraint('c1', {'x1': 1.0, 'x2': 1.0}, ConstraintSense.LESS_EQUAL, 1.0))
    m.set_objective(Objective({'x1': 3.0, 'x2': 5.0}, ObjectiveSense.MAXIMIZE))
    sol = MILPSolver(adaptive_presolve=AdaptivePresolveStrategy()).solve(m)
    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.objective_value == pytest.approx(5.0, abs=1e-4)

def test_cs_max_min_direction_preserved():
    m1 = Model('cs_c')
    m1.add_variable(Variable('x', lower_bound=0.0, upper_bound=10.0))
    m1.add_constraint(Constraint('c1', {'x': 1.0}, ConstraintSense.LESS_EQUAL, 8.0))
    m1.set_objective(Objective({'x': 1.0}, ObjectiveSense.MAXIMIZE))
    m2 = Model('cs_d')
    m2.add_variable(Variable('x', lower_bound=2.0, upper_bound=10.0))
    m2.add_constraint(Constraint('c1', {'x': 1.0}, ConstraintSense.GREATER_EQUAL, 3.0))
    m2.set_objective(Objective({'x': 1.0}, ObjectiveSense.MINIMIZE))
    solver = MILPSolver(adaptive_presolve=AdaptivePresolveStrategy())
    s1 = solver.solve(m1); s2 = solver.solve(m2)
    assert s1.objective_value == pytest.approx(8.0, abs=1e-4)
    assert s2.objective_value == pytest.approx(3.0, abs=1e-4)

def test_cs_infeasibility_through_adaptive():
    m = Model('cs_e')
    m.add_variable(Variable('x1', lower_bound=0.0, upper_bound=3.0))
    m.add_constraint(Constraint('c1', {'x1': 1.0}, ConstraintSense.GREATER_EQUAL, 5.0))
    m.set_objective(Objective({'x1': 1.0}, ObjectiveSense.MAXIMIZE))
    sol = MILPSolver(adaptive_presolve=AdaptivePresolveStrategy()).solve(m)
    assert sol.status is SolutionStatus.INFEASIBLE

def test_cs_all_fixed_milp():
    '''All-fixed MILP with INTEGER vars: 5*2 + 4*3 = 22.'''
    m = Model('cs_f')
    m.add_variable(Variable('x1', lower_bound=2.0, upper_bound=2.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable('x2', lower_bound=3.0, upper_bound=3.0, variable_type=VariableType.INTEGER))
    m.set_objective(Objective({'x1': 5.0, 'x2': 4.0}, ObjectiveSense.MAXIMIZE))
    sol = MILPSolver(adaptive_presolve=AdaptivePresolveStrategy()).solve(m)
    assert sol.status is SolutionStatus.OPTIMAL
    assert sol.objective_value == pytest.approx(22.0, abs=1e-4)

def test_imports_clean():
    from optiforge.milp import solve_milp, AdaptivePresolveStrategy, PresolveDecision, PresolveOperation
    from optiforge.milp.adaptive_presolve_executor import adaptive_presolve_model
    from optiforge.milp.fingerprint import fingerprint_model
    from optiforge.milp.solver import MILPSolver
    from optiforge.presolve.presolve import _presolve_model_impl, presolve_model, postsolve_solution
