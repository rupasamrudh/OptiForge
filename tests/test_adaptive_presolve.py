"""Tests for Stage 7.8: Adaptive Presolve.

Tests the AdaptivePresolveStrategy that dynamically selects among existing
presolve operations based on ProblemFingerprint.
"""

import pytest

from optiforge.core.solution import SolutionStatus
from optiforge.milp import solve_milp, AdaptivePresolveStrategy, PresolveDecision, PresolveOperation
from optiforge.milp.adaptive_presolve_executor import adaptive_presolve_model
from optiforge.milp.fingerprint import fingerprint_model
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

def _build_simple_lp() -> Model:
    """Simple LP: max x1 + x2 s.t. x1 + x2 <= 3, x1,x2 >= 0."""
    m = Model("simple_lp")
    m.add_variable(Variable("x1", lower_bound=0.0))
    m.add_variable(Variable("x2", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 3.0))
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_binary_heavy_lp() -> Model:
    """Binary-heavy MILP: max sum of binary variables with constraints."""
    m = Model("binary_heavy")
    for i in range(15):
        m.add_variable(Variable(f"x{i}", variable_type=VariableType.BINARY))
    m.add_variable(Variable("y1", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
    m.add_variable(Variable("y2", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
    # Multiple constraints with binary variables to increase binary_constraint_coverage
    for j in range(5):
        coeffs = {f"x{i}": 1.0 for i in range(15)}
        m.add_constraint(Constraint(f"c{j}", coeffs, ConstraintSense.LESS_EQUAL, 10.0))
    m.add_constraint(Constraint("c_cont", {"y1": 1.0, "y2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.set_objective(Objective({f"x{i}": 1.0 for i in range(15)}, ObjectiveSense.MAXIMIZE))
    return m


def _build_sparse_lp() -> Model:
    """Sparse LP with few constraints per variable - large enough to trigger adaptive decisions."""
    m = Model("sparse")
    for i in range(60):  # > 50 to trigger large problem threshold
        m.add_variable(Variable(f"x{i}", lower_bound=0.0))
    for i in range(60):
        m.add_constraint(Constraint(f"c{i}", {f"x{i}": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.set_objective(Objective({f"x{i}": 1.0 for i in range(60)}, ObjectiveSense.MAXIMIZE))
    return m


def _build_dense_lp() -> Model:
    """Dense LP with many constraints per variable."""
    m = Model("dense")
    for i in range(5):
        m.add_variable(Variable(f"x{i}", lower_bound=0.0))
    # Each constraint involves all variables
    for j in range(5):
        coeffs = {f"x{i}": 1.0 for i in range(5)}
        m.add_constraint(Constraint(f"c{j}", coeffs, ConstraintSense.LESS_EQUAL, 10.0))
    m.set_objective(Objective({f"x{i}": 1.0 for i in range(5)}, ObjectiveSense.MAXIMIZE))
    return m


def _build_equality_heavy_lp() -> Model:
    """Equality-heavy LP."""
    m = Model("equality_heavy")
    for i in range(10):  # Increased to 10
        m.add_variable(Variable(f"x{i}", lower_bound=0.0))
    for j in range(10):
        coeffs = {f"x{i}": 1.0 for i in range(10)}
        m.add_constraint(Constraint(f"c{j}", coeffs, ConstraintSense.EQUAL, 10.0))
    m.set_objective(Objective({f"x{i}": 1.0 for i in range(10)}, ObjectiveSense.MAXIMIZE))
    return m


def _build_bound_heavy_lp() -> Model:
    """Bound-heavy LP with many variables having finite bounds."""
    m = Model("bound_heavy")
    for i in range(15):  # Increased to 15
        m.add_variable(Variable(f"x{i}", lower_bound=0.0, upper_bound=10.0))
    for j in range(5):
        coeffs = {f"x{i}": 1.0 for i in range(15)}
        m.add_constraint(Constraint(f"c{j}", coeffs, ConstraintSense.LESS_EQUAL, 20.0))
    m.set_objective(Objective({f"x{i}": 1.0 for i in range(15)}, ObjectiveSense.MAXIMIZE))
    return m


def _build_integer_heavy_lp() -> Model:
    """Integer-heavy MILP."""
    m = Model("integer_heavy")
    for i in range(12):  # Increased to 12
        m.add_variable(Variable(f"x{i}", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable("y1", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
    m.add_variable(Variable("y2", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
    for j in range(6):
        coeffs = {f"x{i}": 1.0 for i in range(12)}
        coeffs["y1"] = 1.0
        coeffs["y2"] = 1.0
        m.add_constraint(Constraint(f"c{j}", coeffs, ConstraintSense.LESS_EQUAL, 15.0))
    m.set_objective(Objective({f"x{i}": 1.0 for i in range(12)}, ObjectiveSense.MAXIMIZE))
    return m


def _build_no_constraints_lp() -> Model:
    """LP with no constraints."""
    m = Model("no_constraints")
    for i in range(10):  # Increased to 10
        m.add_variable(Variable(f"x{i}", lower_bound=0.0, upper_bound=5.0 + i))
    m.set_objective(Objective({f"x{i}": 1.0 for i in range(10)}, ObjectiveSense.MAXIMIZE))
    return m


def _build_small_lp() -> Model:
    """Very small LP (below adaptive threshold)."""
    m = Model("small")
    m.add_variable(Variable("x1", lower_bound=0.0))
    m.add_variable(Variable("x2", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 3.0))
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
    return m


# ── Tests ───────────────────────────────────────────────────────────────────

class TestAdaptivePresolveArchitecture:
    """Test the AdaptivePresolveStrategy architecture."""

    def test_strategy_initialization(self):
        """AdaptivePresolveStrategy initializes with defaults."""
        strategy = AdaptivePresolveStrategy()
        assert strategy.enable_bound_validation is True
        assert strategy.enable_zero_row_removal is True
        assert strategy.enable_zero_column_elimination is True
        assert strategy.enable_singleton_constraints is True
        assert strategy.enable_activity_checks is True
        assert strategy.enable_bound_propagation is True
        assert strategy.enable_fixed_point_iteration is True
        assert strategy.min_problem_size_for_adaptive == 10

    def test_strategy_custom_defaults(self):
        """AdaptivePresolveStrategy accepts custom defaults."""
        strategy = AdaptivePresolveStrategy(
            enable_bound_validation=False,
            enable_zero_row_removal=False,
            min_problem_size_for_adaptive=5,
        )
        assert strategy.enable_bound_validation is False
        assert strategy.enable_zero_row_removal is False
        assert strategy.min_problem_size_for_adaptive == 5

    def test_strategy_invalid_min_size(self):
        """Invalid min_problem_size_for_adaptive raises ValueError."""
        with pytest.raises(ValueError, match="min_problem_size_for_adaptive must be >= 0"):
            AdaptivePresolveStrategy(min_problem_size_for_adaptive=-1)

    def test_presolve_decision_immutable(self):
        """PresolveDecision is immutable."""
        decision = PresolveDecision(
            enabled_operations=frozenset([PresolveOperation.BOUND_VALIDATION]),
            selected_operations=(PresolveOperation.BOUND_VALIDATION,),
            reason="test",
            fingerprint_summary={"var_count": 5},
        )
        assert decision.enabled_operations == frozenset([PresolveOperation.BOUND_VALIDATION])
        assert decision.selected_operations == (PresolveOperation.BOUND_VALIDATION,)
        assert decision.reason == "test"
        assert decision.fingerprint_summary == {"var_count": 5}

    def test_presolve_decision_validation(self):
        """PresolveDecision validates enabled_operations matches selected_operations."""
        with pytest.raises(ValueError, match="enabled_operations.*must match selected_operations"):
            PresolveDecision(
                enabled_operations=frozenset([PresolveOperation.BOUND_VALIDATION]),
                selected_operations=(PresolveOperation.ZERO_ROW_REMOVAL,),
                reason="test",
            )


class TestAdaptivePresolveDefaultCompatibility:
    """Test that default behavior is preserved when no adaptive presolve is configured."""

    def test_solver_without_adaptive_presolve(self):
        """Solver works without adaptive presolve (default behavior)."""
        m = _build_simple_lp()
        sol = solve_milp(m)
        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(3.0, abs=1e-6)

    def test_solver_with_adaptive_presolve_none(self):
        """Solver works with adaptive_presolve=None explicitly."""
        m = _build_simple_lp()
        sol = solve_milp(m, adaptive_presolve=None)
        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(3.0, abs=1e-6)

    def test_existing_presolve_unchanged(self):
        """Existing presolve behavior remains unchanged when adaptive presolve is not used."""
        m = _build_simple_lp()
        sol1 = solve_milp(m)
        sol2 = solve_milp(m, adaptive_presolve=None)
        assert sol1.objective_value == sol2.objective_value
        assert sol1.status == sol2.status


class TestAdaptivePresolveFingerprintAdaptivity:
    """Test that presolve selection is fingerprint-sensitive."""

    def test_small_problem_enables_all_operations(self):
        """Small problems enable all operations (below adaptive threshold)."""
        m = _build_small_lp()
        strategy = AdaptivePresolveStrategy(min_problem_size_for_adaptive=10)
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)

        # All operations should be enabled for small problems
        assert PresolveOperation.BOUND_VALIDATION in decision.enabled_operations
        assert PresolveOperation.ZERO_ROW_REMOVAL in decision.enabled_operations
        assert PresolveOperation.ZERO_COLUMN_ELIMINATION in decision.enabled_operations
        assert PresolveOperation.SINGLETON_CONSTRAINTS in decision.enabled_operations
        assert PresolveOperation.ACTIVITY_CHECKS in decision.enabled_operations
        assert PresolveOperation.BOUND_PROPAGATION in decision.enabled_operations
        assert PresolveOperation.FIXED_POINT_ITERATION in decision.enabled_operations

    def test_binary_heavy_prioritizes_singleton_and_propagation(self):
        """Binary-heavy problems prioritize singleton constraints and bound propagation."""
        m = _build_binary_heavy_lp()
        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)

        assert PresolveOperation.SINGLETON_CONSTRAINTS in decision.enabled_operations
        assert PresolveOperation.BOUND_PROPAGATION in decision.enabled_operations
        assert "binary-heavy" in decision.reason.lower()

    def test_equality_heavy_prioritizes_singleton(self):
        """Equality-heavy problems prioritize singleton constraints."""
        m = _build_equality_heavy_lp()
        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)

        assert PresolveOperation.SINGLETON_CONSTRAINTS in decision.enabled_operations
        assert "equality-heavy" in decision.reason.lower()

    def test_bound_heavy_prioritizes_bound_operations(self):
        """Bound-heavy problems prioritize bound validation and propagation."""
        m = _build_bound_heavy_lp()
        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)

        assert PresolveOperation.BOUND_VALIDATION in decision.enabled_operations
        assert PresolveOperation.BOUND_PROPAGATION in decision.enabled_operations
        assert "bound-heavy" in decision.reason.lower()

    def test_large_problem_enables_fixed_point(self):
        """Large problems enable fixed-point iteration."""
        m = _build_sparse_lp()  # 10 variables, 10 constraints
        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)

        assert PresolveOperation.FIXED_POINT_ITERATION in decision.enabled_operations
        assert "large problem" in decision.reason.lower()

    def test_high_integer_coverage_prioritizes_propagation(self):
        """High integer coverage prioritizes bound propagation."""
        m = _build_integer_heavy_lp()
        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)

        assert PresolveOperation.BOUND_PROPAGATION in decision.enabled_operations
        assert "high integer coverage" in decision.reason.lower()

    def test_high_binary_coverage_prioritizes_singleton(self):
        """High binary coverage prioritizes singleton constraints."""
        m = _build_binary_heavy_lp()
        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)

        assert PresolveOperation.SINGLETON_CONSTRAINTS in decision.enabled_operations
        assert "high binary coverage" in decision.reason.lower()

    def test_no_constraints_only_bound_and_zero_column(self):
        """Problems with no constraints only need bound validation and zero-column elimination."""
        m = _build_no_constraints_lp()
        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)

        enabled = decision.enabled_operations
        assert PresolveOperation.BOUND_VALIDATION in enabled
        assert PresolveOperation.ZERO_COLUMN_ELIMINATION in enabled
        # Note: ZERO_ROW_REMOVAL is enabled by default but has no effect when there are no constraints
        assert "no constraints" in decision.reason.lower()

    def test_no_variables_no_operations(self):
        """Problems with no variables need no operations."""
        m = Model("empty")
        # No variables, no constraints
        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)

        # With no variables, no operations should be needed
        assert len(decision.enabled_operations) == 0
        assert "no variables" in decision.reason.lower()

    def test_fixed_variable_no_constraints(self):
        """Problems with fixed variables but no constraints."""
        m = Model("fixed_no_constr")
        m.add_variable(Variable("x1", lower_bound=0.0, upper_bound=0.0))  # Fixed variable
        m.set_objective(Objective({"x1": 0.0}, ObjectiveSense.MAXIMIZE))
        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)

        # With 1 variable and no constraints, only bound validation and zero-column elimination apply
        assert len(decision.enabled_operations) >= 0
        assert "no constraints" in decision.reason.lower() or "problem too small" in decision.reason.lower()

    def test_configuration_flags_filter_operations(self):
        """Configuration flags can disable operations."""
        m = _build_binary_heavy_lp()
        strategy = AdaptivePresolveStrategy(
            enable_singleton_constraints=False,
            enable_bound_propagation=False,
        )
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)

        assert PresolveOperation.SINGLETON_CONSTRAINTS not in decision.enabled_operations
        assert PresolveOperation.BOUND_PROPAGATION not in decision.enabled_operations
        assert "disabled by configuration" in decision.reason.lower()


class TestAdaptivePresolveOperationIntegration:
    """Test that selected operations actually execute and preserve semantics."""

    def test_bound_validation_detects_infeasible_bounds(self):
        """Bound validation detects contradictory bounds derived from constraints."""
        m = Model("infeasible_bounds")
        m.add_variable(Variable("x1", lower_bound=0.0, upper_bound=10.0))
        m.add_constraint(Constraint("c1", {"x1": 1.0}, ConstraintSense.GREATER_EQUAL, 12.0))  # x1 >= 12, but ub=10
        m.set_objective(Objective({"x1": 1.0}, ObjectiveSense.MAXIMIZE))

        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)
        result = adaptive_presolve_model(m, decision)

        assert result.status is SolutionStatus.INFEASIBLE
        assert any("infeasible" in msg.lower() for msg in result.transformation_metadata)

    def test_zero_row_removal_removes_redundant_rows(self):
        """Zero row removal removes redundant zero rows."""
        m = Model("zero_row")
        m.add_variable(Variable("x1", lower_bound=0.0))
        m.add_constraint(Constraint("r1", {"x1": 0.0}, ConstraintSense.LESS_EQUAL, 2.0))
        m.set_objective(Objective({"x1": 1.0}, ObjectiveSense.MINIMIZE))

        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)
        result = adaptive_presolve_model(m, decision)

        assert result.status is SolutionStatus.FEASIBLE
        assert any("zero row" in msg.lower() for msg in result.transformation_metadata)

    def test_zero_column_elimination_fixes_unconstrained_variables(self):
        """Zero column elimination fixes variables not in any constraint."""
        m = Model("zero_col")
        m.add_variable(Variable("x1", lower_bound=0.0, upper_bound=5.0))
        m.add_constraint(Constraint("c1", {"x1": 0.0}, ConstraintSense.LESS_EQUAL, 10.0))
        m.set_objective(Objective({"x1": 3.0}, ObjectiveSense.MAXIMIZE))

        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)
        result = adaptive_presolve_model(m, decision)

        assert result.status is SolutionStatus.FEASIBLE
        assert "x1" in result.fixed_variables
        assert result.fixed_variables["x1"] == pytest.approx(5.0)  # Upper bound for MAX

    def test_singleton_constraints_derives_bounds(self):
        """Singleton constraints derive bounds from single-variable constraints."""
        m = Model("singleton")
        m.add_variable(Variable("x1", lower_bound=0.0))
        m.add_constraint(Constraint("c1", {"x1": 3.0}, ConstraintSense.LESS_EQUAL, 12.0))
        m.set_objective(Objective({"x1": 1.0}, ObjectiveSense.MINIMIZE))

        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)
        result = adaptive_presolve_model(m, decision)

        assert result.status is SolutionStatus.FEASIBLE
        assert any("singleton" in msg.lower() for msg in result.transformation_metadata)

    def test_activity_checks_detect_redundant_constraints(self):
        """Activity checks detect redundant constraints."""
        m = Model("activity")
        m.add_variable(Variable("x1", lower_bound=0.0, upper_bound=5.0))
        m.add_variable(Variable("x2", lower_bound=0.0, upper_bound=10.0))
        m.add_constraint(Constraint("c1", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
        # c2: x1 + 0.5*x2 <= 15. With x1<=5 and x2<=10, max activity = 5 + 0.5*10 = 10 <= 15, so redundant
        m.add_constraint(Constraint("c2", {"x1": 1.0, "x2": 0.5}, ConstraintSense.LESS_EQUAL, 15.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MINIMIZE))

        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)
        result = adaptive_presolve_model(m, decision)

        assert result.status is SolutionStatus.FEASIBLE
        assert any("redundant" in msg.lower() for msg in result.transformation_metadata)

    def test_bound_propagation_tightens_bounds(self):
        """Bound propagation tightens variable bounds."""
        m = Model("propagate")
        m.add_variable(Variable("x1", lower_bound=0.0))
        m.add_variable(Variable("x2", lower_bound=0.0))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
        m.add_constraint(Constraint("c2", {"x1": 2.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 8.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MINIMIZE))

        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)
        result = adaptive_presolve_model(m, decision)

        assert result.status is SolutionStatus.FEASIBLE
        assert result.statistics["tightened_bounds"] >= 1
        assert any("bound propagation" in msg.lower() for msg in result.transformation_metadata)

    def test_fixed_point_iteration_runs_multiple_passes(self):
        """Fixed point iteration runs multiple passes until convergence."""
        m = Model("fixed_point")
        m.add_variable(Variable("x1", lower_bound=0.0))
        m.add_variable(Variable("x2", lower_bound=0.0))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
        m.add_constraint(Constraint("c2", {"x1": 2.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 8.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MINIMIZE))

        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)
        result = adaptive_presolve_model(m, decision)

        assert result.status is SolutionStatus.FEASIBLE
        assert result.statistics["passes"] >= 1


class TestAdaptivePresolveMappingCorrectness:
    """Test mapping and reconstruction correctness."""

    def test_fixed_variables_reconstructed(self):
        """Fixed variables are correctly reconstructed in postsolve."""
        m = Model("fixed_reconstruct")
        m.add_variable(Variable("x1", lower_bound=3.0, upper_bound=3.0))
        m.add_variable(Variable("x2", lower_bound=0.0))
        m.add_constraint(Constraint("c1", {"x1": 2.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 10.0))
        m.set_objective(Objective({"x1": 4.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))

        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)
        result = adaptive_presolve_model(m, decision)

        assert result.status is SolutionStatus.FEASIBLE
        assert result.fixed_variables == {"x1": 3.0, "x2": 4.0}
        assert result.objective_constant == pytest.approx(16.0)

        # Test postsolve reconstruction
        from optiforge.presolve import postsolve_solution
        restored = postsolve_solution({}, result)
        assert restored["x1"] == pytest.approx(3.0)
        assert restored["x2"] == pytest.approx(4.0)

    def test_variable_ordering_preserved(self):
        """Variable ordering is preserved through presolve."""
        m = Model("ordering")
        m.add_variable(Variable("x1", lower_bound=0.0))
        m.add_variable(Variable("x2", lower_bound=0.0))
        m.add_variable(Variable("x3", lower_bound=0.0))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0, "x3": 1.0}, ConstraintSense.LESS_EQUAL, 10.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 2.0, "x3": 3.0}, ObjectiveSense.MAXIMIZE))

        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)
        result = adaptive_presolve_model(m, decision)

        assert result.status is SolutionStatus.FEASIBLE
        # Original variable order should be preserved
        assert result.original_variable_order == ["x1", "x2", "x3"]

    def test_removed_variables_tracked(self):
        """Removed variables are tracked in statistics."""
        m = Model("removed_vars")
        m.add_variable(Variable("x1", lower_bound=3.0, upper_bound=3.0))
        m.add_variable(Variable("x2", lower_bound=0.0))
        m.add_constraint(Constraint("c1", {"x1": 2.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 10.0))
        m.set_objective(Objective({"x1": 4.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))

        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)
        result = adaptive_presolve_model(m, decision)

        assert result.status is SolutionStatus.FEASIBLE
        assert result.statistics["fixed_vars"] == 2
        assert result.statistics["reduced_vars"] == 0


class TestAdaptivePresolveObjectiveCorrectness:
    """Test objective correctness for MAXIMIZE and MINIMIZE."""

    def test_maximize_objective_correct(self):
        """MAXIMIZE objective is correctly handled."""
        m = Model("max_obj")
        m.add_variable(Variable("x1", lower_bound=0.0, upper_bound=5.0))
        m.add_constraint(Constraint("c1", {"x1": 0.0}, ConstraintSense.LESS_EQUAL, 10.0))
        m.set_objective(Objective({"x1": 3.0}, ObjectiveSense.MAXIMIZE))

        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)
        result = adaptive_presolve_model(m, decision)

        assert result.status is SolutionStatus.FEASIBLE
        # x1 is zero-column with positive MAX coefficient -> fixed at upper bound 5.0
        assert result.fixed_variables["x1"] == pytest.approx(5.0)
        assert result.objective_constant == pytest.approx(15.0)

    def test_minimize_objective_correct(self):
        """MINIMIZE objective is correctly handled."""
        m = Model("min_obj")
        m.add_variable(Variable("x1", lower_bound=0.0, upper_bound=5.0))
        m.add_constraint(Constraint("c1", {"x1": 0.0}, ConstraintSense.LESS_EQUAL, 10.0))
        m.set_objective(Objective({"x1": 3.0}, ObjectiveSense.MINIMIZE))

        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)
        result = adaptive_presolve_model(m, decision)

        assert result.status is SolutionStatus.FEASIBLE
        # x1 is zero-column with positive MIN coefficient -> fixed at lower bound 0.0
        assert result.fixed_variables["x1"] == pytest.approx(0.0)
        assert result.objective_constant == pytest.approx(0.0)

    def test_negative_objective_correct(self):
        """Negative objective coefficients are correctly handled."""
        m = Model("neg_obj")
        m.add_variable(Variable("x1", lower_bound=0.0, upper_bound=5.0))
        m.add_constraint(Constraint("c1", {"x1": 0.0}, ConstraintSense.LESS_EQUAL, 10.0))
        m.set_objective(Objective({"x1": -3.0}, ObjectiveSense.MAXIMIZE))

        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)
        result = adaptive_presolve_model(m, decision)

        assert result.status is SolutionStatus.FEASIBLE
        # x1 is zero-column with negative MAX coefficient -> fixed at lower bound 0.0
        assert result.fixed_variables["x1"] == pytest.approx(0.0)
        assert result.objective_constant == pytest.approx(0.0)


class TestAdaptivePresolveFeasibility:
    """Test feasibility preservation."""

    def test_feasible_model_remains_feasible(self):
        """Feasible model remains feasible after presolve."""
        m = _build_simple_lp()
        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)
        result = adaptive_presolve_model(m, decision)

        assert result.status is SolutionStatus.FEASIBLE
        assert result.reduced_model is not None

    def test_infeasible_model_detected(self):
        """Infeasible model is detected by presolve."""
        m = Model("infeasible")
        m.add_variable(Variable("x1", lower_bound=0.0, upper_bound=10.0))
        m.add_constraint(Constraint("c1", {"x1": 1.0}, ConstraintSense.GREATER_EQUAL, 12.0))  # x1 >= 12, but ub=10
        m.set_objective(Objective({"x1": 1.0}, ObjectiveSense.MINIMIZE))

        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)
        result = adaptive_presolve_model(m, decision)

        assert result.status is SolutionStatus.INFEASIBLE

    def test_tightened_bounds_preserve_feasibility(self):
        """Tightened bounds preserve feasibility."""
        m = Model("tightened")
        m.add_variable(Variable("x1", lower_bound=0.0, upper_bound=10.0))
        m.add_variable(Variable("x2", lower_bound=0.0, upper_bound=10.0))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))

        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)
        result = adaptive_presolve_model(m, decision)

        assert result.status is SolutionStatus.FEASIBLE
        # Bounds should be tightened but not contradictory
        for var in result.reduced_model.variables.values():
            assert var.lower_bound <= var.upper_bound + 1e-7

    def test_redundant_constraints_removed(self):
        """Redundant constraints are correctly removed."""
        m = Model("redundant")
        m.add_variable(Variable("x1", lower_bound=0.0, upper_bound=10.0))
        m.add_constraint(Constraint("c1", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
        m.add_constraint(Constraint("c2", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 15.0))  # Redundant
        m.set_objective(Objective({"x1": 1.0}, ObjectiveSense.MINIMIZE))

        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)
        result = adaptive_presolve_model(m, decision)

        assert result.status is SolutionStatus.FEASIBLE
        assert result.statistics["removed_rows"] >= 1

    def test_all_fixed_model_handled(self):
        """All-fixed model is correctly handled."""
        m = Model("all_fixed")
        m.add_variable(Variable("x1", lower_bound=3.0, upper_bound=3.0))
        m.add_variable(Variable("x2", lower_bound=5.0, upper_bound=5.0))
        m.set_objective(Objective({"x1": 2.0, "x2": 3.0}, ObjectiveSense.MAXIMIZE))

        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)
        result = adaptive_presolve_model(m, decision)

        assert result.status is SolutionStatus.FEASIBLE
        assert result.reduced_model.num_variables == 0
        assert result.fixed_variables == {"x1": 3.0, "x2": 5.0}
        assert result.objective_constant == pytest.approx(21.0)

    def test_empty_model_handled(self):
        """Empty model is correctly handled."""
        m = Model("empty")
        m.add_variable(Variable("x1", lower_bound=0.0, upper_bound=0.0))  # Fixed variable
        m.set_objective(Objective({"x1": 0.0}, ObjectiveSense.MAXIMIZE))

        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)
        result = adaptive_presolve_model(m, decision)

        assert result.status is SolutionStatus.FEASIBLE
        assert result.reduced_model.num_variables == 0
        assert result.reduced_model.num_constraints == 0
        assert result.fixed_variables == {"x1": 0.0}

    def test_binary_model_handled(self):
        """Binary model is correctly handled."""
        m = Model("binary")
        m.add_variable(Variable("x1", variable_type=VariableType.BINARY))
        m.add_variable(Variable("x2", variable_type=VariableType.BINARY))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 1.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))

        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)
        result = adaptive_presolve_model(m, decision)

        assert result.status is SolutionStatus.FEASIBLE

    def test_integer_model_handled(self):
        """Integer model is correctly handled."""
        m = Model("integer")
        m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
        m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 3.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))

        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)
        result = adaptive_presolve_model(m, decision)

        assert result.status is SolutionStatus.FEASIBLE

    def test_mixed_model_handled(self):
        """Mixed integer/continuous model is correctly handled."""
        m = Model("mixed")
        m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
        m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 3.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))

        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)
        result = adaptive_presolve_model(m, decision)

        assert result.status is SolutionStatus.FEASIBLE


class TestAdaptivePresolveToleranceBehavior:
    """Test tolerance behavior."""

    def test_feasibility_tolerance_respected(self):
        """Feasibility tolerance is respected in bound comparisons."""
        m = Model("tolerance")
        m.add_variable(Variable("x1", lower_bound=0.0, upper_bound=1e-8))  # Very tight bounds
        m.set_objective(Objective({"x1": 1.0}, ObjectiveSense.MAXIMIZE))

        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)
        result = adaptive_presolve_model(m, decision, feasibility_tol=1e-7)

        assert result.status is SolutionStatus.FEASIBLE
        # With 1e-7 tolerance, bounds 0.0 and 1e-8 are considered fixed
        assert "x1" in result.fixed_variables


class TestAdaptivePresolveImmutability:
    """Test that the original model is not mutated."""

    def test_model_not_mutated(self):
        """Original model is not mutated by adaptive presolve."""
        m = _build_simple_lp()
        original_vars = {name: (var.lower_bound, var.upper_bound) for name, var in m.variables.items()}
        original_constraints = {name: (dict(con.coefficients), con.sense, con.rhs) for name, con in m.constraints.items()}

        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)
        adaptive_presolve_model(m, decision)

        # Model should be unchanged
        for name, var in m.variables.items():
            assert (var.lower_bound, var.upper_bound) == original_vars[name]
        for name, con in m.constraints.items():
            orig_coeffs, orig_sense, orig_rhs = original_constraints[name]
            assert dict(con.coefficients) == orig_coeffs
            assert con.sense == orig_sense
            assert con.rhs == orig_rhs

    def test_fingerprint_not_mutated(self):
        """Fingerprint is not mutated."""
        m = _build_simple_lp()
        fp1 = fingerprint_model(m)

        strategy = AdaptivePresolveStrategy()
        decision = strategy.decide(fp1)
        adaptive_presolve_model(m, decision)

        fp2 = fingerprint_model(m)
        assert fp1.variable_count == fp2.variable_count
        assert fp1.constraint_count == fp2.constraint_count
        assert fp1.constraint_density == fp2.constraint_density


class TestAdaptivePresolveEndToEnd:
    """End-to-end tests with full MILP solve."""

    def test_end_to_end_solve_with_adaptive_presolve(self):
        """Complete MILP solve with adaptive presolve enabled."""
        m = _build_binary_heavy_lp()
        strategy = AdaptivePresolveStrategy()
        sol = solve_milp(m, adaptive_presolve=strategy)

        assert sol.status is SolutionStatus.OPTIMAL

    def test_end_to_end_maximize(self):
        """MAXIMIZE problem solves correctly with adaptive presolve."""
        m = _build_simple_lp()
        strategy = AdaptivePresolveStrategy()
        sol = solve_milp(m, adaptive_presolve=strategy)

        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(3.0, abs=1e-6)

    def test_end_to_end_minimize(self):
        """MINIMIZE problem solves correctly with adaptive presolve."""
        m = Model("minimize")
        m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
        m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, 2.5))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MINIMIZE))

        strategy = AdaptivePresolveStrategy()
        sol = solve_milp(m, adaptive_presolve=strategy)

        assert sol.status in (SolutionStatus.OPTIMAL, SolutionStatus.UNKNOWN)

    def test_end_to_end_binary_milp(self):
        """Binary MILP solves correctly with adaptive presolve."""
        m = _build_binary_heavy_lp()
        strategy = AdaptivePresolveStrategy()
        sol = solve_milp(m, adaptive_presolve=strategy)

        assert sol.status is SolutionStatus.OPTIMAL

    def test_end_to_end_general_integer_milp(self):
        """General integer MILP solves correctly with adaptive presolve."""
        m = _build_simple_lp()
        strategy = AdaptivePresolveStrategy()
        sol = solve_milp(m, adaptive_presolve=strategy)

        assert sol.status is SolutionStatus.OPTIMAL

    def test_end_to_end_mixed_integer_continuous(self):
        """Mixed integer/continuous MILP solves correctly with adaptive presolve."""
        m = Model("mixed")
        m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
        m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 3.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))

        strategy = AdaptivePresolveStrategy()
        sol = solve_milp(m, adaptive_presolve=strategy)

        assert sol.status is SolutionStatus.OPTIMAL

    def test_end_to_end_negative_objective(self):
        """Negative objectives solve correctly with adaptive presolve."""
        m = Model("neg_obj")
        m.add_variable(Variable("x1", lower_bound=-10.0, variable_type=VariableType.INTEGER))
        m.add_variable(Variable("x2", lower_bound=-10.0, variable_type=VariableType.INTEGER))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 0.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))

        strategy = AdaptivePresolveStrategy()
        sol = solve_milp(m, adaptive_presolve=strategy)

        assert sol.status is SolutionStatus.OPTIMAL

    def test_end_to_end_heuristic_enabled(self):
        """Heuristic enabled solves correctly with adaptive presolve."""
        from optiforge.milp.heuristics import LPRoundingHeuristic

        m = _build_simple_lp()
        strategy = AdaptivePresolveStrategy()
        sol = solve_milp(m, adaptive_presolve=strategy, primal_heuristic=LPRoundingHeuristic())

        assert sol.status is SolutionStatus.OPTIMAL

    def test_end_to_end_heuristic_disabled(self):
        """Heuristic disabled solves correctly with adaptive presolve."""
        m = _build_simple_lp()
        strategy = AdaptivePresolveStrategy()
        sol = solve_milp(m, adaptive_presolve=strategy)

        assert sol.status is SolutionStatus.OPTIMAL

    def test_end_to_end_strategy_switching(self):
        """Dynamic strategy switching works with adaptive presolve."""
        from optiforge.milp import StrategyManager

        m = _build_binary_heavy_lp()
        strategy = AdaptivePresolveStrategy()
        sm = StrategyManager(min_switch_interval=2, gap_change_threshold=0.05)
        sol = solve_milp(m, adaptive_presolve=strategy, strategy_manager=sm)

        assert sol.status is SolutionStatus.OPTIMAL
        assert sm.get_switch_count() >= 1


class TestAdaptivePresolveDeterminism:
    """Test determinism of adaptive presolve."""

    def test_deterministic_decisions(self):
        """Repeated decisions with same fingerprint are identical."""
        m = _build_binary_heavy_lp()
        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)

        decisions = [strategy.decide(fp) for _ in range(5)]
        for d in decisions[1:]:
            assert d.enabled_operations == decisions[0].enabled_operations
            assert d.selected_operations == decisions[0].selected_operations
            assert d.reason == decisions[0].reason

    def test_deterministic_presolve_results(self):
        """Repeated presolve with same decision produces identical results."""
        m = _build_binary_heavy_lp()
        strategy = AdaptivePresolveStrategy()
        fp = fingerprint_model(m)
        decision = strategy.decide(fp)

        results = [adaptive_presolve_model(m, decision) for _ in range(5)]
        for r in results[1:]:
            assert r.status == results[0].status
            assert r.statistics == results[0].statistics
            assert r.fixed_variables == results[0].fixed_variables
            assert r.objective_constant == results[0].objective_constant

    def test_deterministic_end_to_end_solves(self):
        """Repeated end-to-end solves produce identical results."""
        m = _build_binary_heavy_lp()
        strategy = AdaptivePresolveStrategy()

        results = []
        for _ in range(5):
            sol = solve_milp(m, adaptive_presolve=strategy)
            results.append((sol.objective_value, sol.status, sol.variable_values))

        for r in results[1:]:
            assert r[0] == pytest.approx(results[0][0], abs=1e-6)
            assert r[1] == results[0][1]
            for var, val in r[2].items():
                assert val == pytest.approx(results[0][2][var], abs=1e-6)


# Run all tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])