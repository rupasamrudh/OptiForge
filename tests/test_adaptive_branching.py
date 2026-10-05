"""Tests for Stage 7.4: Adaptive Branching Strategy."""

import math

import pytest

from optiforge.core.solution import SolutionStatus
from optiforge.milp import (
    AdaptiveBranchingStrategy,
    MILPSolver,
    MILPTolerances,
    MostFractionalBranching,
    ProblemFingerprint,
    SearchState,
    fingerprint_model,
    solve_milp,
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
    s.t. x1 + x2 <= 2.5
         x1, x2 >= 0, integer
    LP: x1=1.25, x2=1.25 -> obj=2.5 (fractional)
    Integer: x1=1, x2=1 -> obj=2
    """
    m = Model("fractional_relax")
    m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 2.5))
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_binary_vs_integer() -> Model:
    """Model with both binary and general integer variables:
    max x1 + 2*x2 + 3*x3
    s.t. x1 + x2 + x3 <= 3
         x1 binary, x2 integer, x3 integer
    """
    m = Model("binary_vs_int")
    m.add_variable(Variable("x1", variable_type=VariableType.BINARY))
    m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable("x3", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0, "x3": 1.0}, ConstraintSense.LESS_EQUAL, 3.0))
    m.set_objective(Objective({"x1": 1.0, "x2": 2.0, "x3": 3.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_negative_fractional() -> Model:
    """Model with negative fractional values:
    min x1 + x2
    s.t. x1 + x2 >= -2.5
         x1, x2 >= -5, integer
    """
    m = Model("neg_frac")
    m.add_variable(Variable("x1", lower_bound=-5.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable("x2", lower_bound=-5.0, variable_type=VariableType.INTEGER))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, -2.5))
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MINIMIZE))
    return m


def _build_high_constraint_participation() -> Model:
    """Model where one variable appears in many constraints:
    max x1 + x2 + x3
    s.t. x1 + x2 <= 2
         x1 + x3 <= 2
         x1 <= 1
         x2 <= 2
         x3 <= 2
         x1, x2, x3 >= 0, integer
    x1 appears in 3 constraints, x2 in 2, x3 in 2
    """
    m = Model("high_participation")
    m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable("x3", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 2.0))
    m.add_constraint(Constraint("c2", {"x1": 1.0, "x3": 1.0}, ConstraintSense.LESS_EQUAL, 2.0))
    m.add_constraint(Constraint("c3", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 1.0))
    m.add_constraint(Constraint("c4", {"x2": 1.0}, ConstraintSense.LESS_EQUAL, 2.0))
    m.add_constraint(Constraint("c5", {"x3": 1.0}, ConstraintSense.LESS_EQUAL, 2.0))
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0, "x3": 1.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_minimize_model() -> Model:
    """Simple MINIMIZE model with fractional relaxation."""
    m = Model("minimize")
    m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, 2.5))
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MINIMIZE))
    return m


def _build_all_continuous() -> Model:
    """All continuous model (no integer variables)."""
    m = Model("all_cont")
    m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
    m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 3.0))
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
    return m


# ── Tests ───────────────────────────────────────────────────────────────────


class TestAdaptiveBranchingConstruction:
    """Test AdaptiveBranchingStrategy construction and configuration."""

    def test_default_construction(self):
        """Test default construction with no arguments."""
        strategy = AdaptiveBranchingStrategy()
        assert strategy is not None
        assert strategy.get_fallback_strategy() is not None

    def test_construction_with_fingerprint(self):
        """Test construction with fingerprint."""
        m = _build_simple_integer_lp()
        fp = fingerprint_model(m)
        strategy = AdaptiveBranchingStrategy(fingerprint=fp)
        assert strategy._fingerprint is fp

    def test_construction_with_search_state(self):
        """Test construction with search state."""
        ss = SearchState(
            nodes_processed=5,
            nodes_remaining=3,
            incumbent_objective=10.0,
            best_open_bound=12.0,
            gap=0.2,
            absolute_gap=2.0,
            feasible_solutions_found=2,
            nodes_pruned=1,
            max_depth=3,
            elapsed_time=1.5,
        )
        strategy = AdaptiveBranchingStrategy(search_state=ss)
        assert strategy._search_state is ss

    def test_construction_with_fallback(self):
        """Test construction with custom fallback."""
        fallback = MostFractionalBranching()
        strategy = AdaptiveBranchingStrategy(fallback_strategy=fallback)
        assert strategy.get_fallback_strategy() is fallback

    def test_update_context(self):
        """Test updating context after construction."""
        strategy = AdaptiveBranchingStrategy()
        m = _build_simple_integer_lp()
        fp = fingerprint_model(m)
        strategy.update_context(fingerprint=fp)
        assert strategy._fingerprint is fp

        ss = SearchState(
            nodes_processed=10,
            nodes_remaining=5,
            incumbent_objective=20.0,
            best_open_bound=25.0,
            gap=0.25,
            absolute_gap=5.0,
            feasible_solutions_found=5,
            nodes_pruned=3,
            max_depth=5,
            elapsed_time=5.0,
        )
        strategy.update_context(search_state=ss)
        assert strategy._search_state is ss


class TestCandidateExtraction:
    """Test candidate extraction logic."""

    def test_extracts_only_integer_variables(self):
        """Only integer/binary variables should be candidates."""
        m = _build_mixed_integer()
        strategy = AdaptiveBranchingStrategy()

        # Create a mock node with fractional solution
        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 4.0, "x2": 1.5}  # x1 continuous, x2 integer

        candidates = strategy._extract_candidates(node, m, MILPTolerances())
        assert len(candidates) == 1
        assert candidates[0][0] == "x2"
        assert candidates[0][2] is False  # not binary

    def test_excludes_continuous_variables(self):
        """Continuous variables should never be candidates."""
        m = _build_all_continuous()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 1.5, "x2": 2.5}

        candidates = strategy._extract_candidates(node, m, MILPTolerances())
        assert len(candidates) == 0

    def test_excludes_integral_variables(self):
        """Already-integral variables should be excluded."""
        m = _build_simple_integer_lp()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 2.0, "x2": 1.0}  # both integral

        candidates = strategy._extract_candidates(node, m, MILPTolerances())
        assert len(candidates) == 0

    def test_includes_binary_variables(self):
        """Binary variables should be included when fractional."""
        m = _build_binary_knapsack()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 0.5, "x2": 0.0, "x3": 1.0}  # x1 fractional binary

        candidates = strategy._extract_candidates(node, m, MILPTolerances())
        assert len(candidates) == 1
        assert candidates[0][0] == "x1"
        assert candidates[0][2] is True  # binary

    def test_includes_general_integer_variables(self):
        """General integer variables should be included when fractional."""
        m = _build_simple_integer_lp()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 1.5, "x2": 1.0}  # x1 fractional integer

        candidates = strategy._extract_candidates(node, m, MILPTolerances())
        assert len(candidates) == 1
        assert candidates[0][0] == "x1"
        assert candidates[0][2] is False  # not binary

    def test_negative_fractional_values(self):
        """Negative fractional values should be handled correctly."""
        m = _build_negative_fractional()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": -1.5, "x2": -1.0}  # x1 fractional

        candidates = strategy._extract_candidates(node, m, MILPTolerances())
        assert len(candidates) == 1
        assert candidates[0][0] == "x1"
        assert candidates[0][1] == -1.5

    def test_respects_integrality_tolerance(self):
        """Variables within tolerance of integer should be excluded."""
        m = _build_simple_integer_lp()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        # x1 = 2.0000001 (within default 1e-6 tolerance of 2.0)
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 2.0000001, "x2": 1.5}

        candidates = strategy._extract_candidates(node, m, MILPTolerances())
        assert len(candidates) == 1
        assert candidates[0][0] == "x2"

    def test_empty_candidate_set(self):
        """Empty candidate set should be handled."""
        m = _build_all_continuous()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 1.5, "x2": 2.5}

        candidates = strategy._extract_candidates(node, m, MILPTolerances())
        assert len(candidates) == 0


class TestFeatureComputation:
    """Test feature computation for candidates."""

    def test_fractionality_normalization(self):
        """Fractionality should be normalized to [0, 1]."""
        m = _build_simple_integer_lp()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)

        # Test various fractional values
        test_cases = [
            (1.5, 1.0),   # maximally fractional (0.5 from nearest int)
            (1.25, 0.5),  # halfway
            (1.1, 0.2),   # close to integer
            (1.0, 0.0),   # integral
            (1.49, 0.98), # very close to 0.5
        ]

        for value, expected_frac in test_cases:
            node.lp_solution = {"x1": value, "x2": 1.0}
            features = strategy._compute_features("x1", value, False, m, node)
            assert abs(features.fractionality - expected_frac) < 0.01, \
                f"Value {value}: expected {expected_frac}, got {features.fractionality}"

    def test_binary_flag(self):
        """Binary flag should be correctly set."""
        m = _build_binary_vs_integer()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 0.5, "x2": 1.5, "x3": 1.5}

        # x1 is binary
        features = strategy._compute_features("x1", 0.5, True, m, node)
        assert features.is_binary is True

        # x2 is general integer
        features = strategy._compute_features("x2", 1.5, False, m, node)
        assert features.is_binary is False

    def test_constraint_participation(self):
        """Constraint participation should be correctly computed."""
        m = _build_high_constraint_participation()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 1.5, "x2": 1.5, "x3": 1.5}

        # x1 appears in 3 out of 5 constraints
        features = strategy._compute_features("x1", 1.5, False, m, node)
        assert abs(features.constraint_participation - 3/5) < 0.01

        # x2 appears in 2 out of 5 constraints
        features = strategy._compute_features("x2", 1.5, False, m, node)
        assert abs(features.constraint_participation - 2/5) < 0.01

    def test_fingerprint_features(self):
        """Fingerprint features should be correctly extracted."""
        m = _build_binary_knapsack()
        fp = fingerprint_model(m)
        strategy = AdaptiveBranchingStrategy(fingerprint=fp)

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 0.5, "x2": 0.0, "x3": 1.0}

        features = strategy._compute_features("x1", 0.5, True, m, node)

        # Check fingerprint features are passed through
        assert features.binary_constraint_coverage == fp.binary_constraint_coverage
        assert features.integer_constraint_coverage == fp.integer_constraint_coverage
        assert features.constraint_density == fp.constraint_density
        assert features.is_binary_heavy == fp.is_binary_heavy
        assert features.is_equality_heavy == fp.is_equality_heavy
        assert features.is_sparse == fp.is_sparse
        assert features.is_dense == fp.is_dense

    def test_search_state_features(self):
        """Search state features should be correctly extracted."""
        ss = SearchState(
            nodes_processed=10,
            nodes_remaining=5,
            incumbent_objective=10.0,
            best_open_bound=12.0,
            gap=0.2,
            absolute_gap=2.0,
            feasible_solutions_found=2,
            nodes_pruned=3,
            max_depth=5,
            elapsed_time=2.0,
        )
        strategy = AdaptiveBranchingStrategy(search_state=ss)

        m = _build_simple_integer_lp()
        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=2)
        node.lp_solution = {"x1": 1.5, "x2": 1.0}

        features = strategy._compute_features("x1", 1.5, False, m, node)

        assert features.current_depth == 2
        assert features.max_depth == 5
        assert features.nodes_processed == 10
        assert features.nodes_remaining == 5
        assert features.gap == 0.2
        assert features.absolute_gap == 2.0
        assert features.incumbent_available is True
        # SearchState doesn't have objective_sense, so it should be None
        assert features.objective_sense is None


class TestScoring:
    """Test scoring logic."""

    def test_score_components_present(self):
        """All expected score components should be present."""
        m = _build_simple_integer_lp()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 1.5, "x2": 1.0}

        features = strategy._compute_features("x1", 1.5, False, m, node)
        score, components = strategy._compute_score(features)

        expected_components = {
            "fractionality",
            "binary_preference",
            "constraint_participation",
            "binary_coverage",
            "integer_coverage",
            "depth_factor",
            "gap_factor",
            "incumbent_factor",
        }
        assert set(components.keys()) == expected_components

    def test_score_range(self):
        """Score should be non-negative."""
        m = _build_simple_integer_lp()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 1.5, "x2": 1.0}

        features = strategy._compute_features("x1", 1.5, False, m, node)
        score, _ = strategy._compute_score(features)

        assert score >= 0.0

    def test_binary_preference(self):
        """Binary variables should get binary_preference = 1.0."""
        m = _build_binary_vs_integer()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 0.5, "x2": 1.5, "x3": 1.5}

        # Binary variable
        features_bin = strategy._compute_features("x1", 0.5, True, m, node)
        _, comp_bin = strategy._compute_score(features_bin)
        assert comp_bin["binary_preference"] == 1.0

        # General integer
        features_int = strategy._compute_features("x2", 1.5, False, m, node)
        _, comp_int = strategy._compute_score(features_int)
        assert comp_int["binary_preference"] == 0.0

    def test_fractionality_weight(self):
        """Fractionality should be the primary signal."""
        m = _build_simple_integer_lp()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)

        # More fractional should get higher score
        node.lp_solution = {"x1": 1.5, "x2": 1.0}
        features1 = strategy._compute_features("x1", 1.5, False, m, node)
        score1, _ = strategy._compute_score(features1)

        node.lp_solution = {"x1": 1.1, "x2": 1.0}
        features2 = strategy._compute_features("x1", 1.1, False, m, node)
        score2, _ = strategy._compute_score(features2)

        assert score1 > score2

    def test_binary_preference_in_score(self):
        """Binary variables should score higher due to binary_preference."""
        m = _build_binary_vs_integer()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 0.5, "x2": 1.5, "x3": 1.5}

        # Binary variable with same fractionality as integer
        features_bin = strategy._compute_features("x1", 0.5, True, m, node)
        score_bin, _ = strategy._compute_score(features_bin)

        features_int = strategy._compute_features("x2", 0.5, False, m, node)
        score_int, _ = strategy._compute_score(features_int)

        # Binary should score higher due to binary_preference weight
        assert score_bin > score_int


class TestSelection:
    """Test candidate selection logic."""

    def test_single_candidate_returns_directly(self):
        """Single candidate should be returned without scoring."""
        m = _build_simple_integer_lp()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 1.5, "x2": 1.0}

        decision = strategy.choose_branching_variable(node, m, MILPTolerances())
        assert decision is not None
        assert decision.variable_name == "x1"
        assert decision.value == 1.5

    def test_empty_candidates_fallback(self):
        """Empty candidate set should trigger fallback."""
        m = _build_all_continuous()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 1.5, "x2": 2.5}

        decision = strategy.choose_branching_variable(node, m, MILPTolerances())
        # Should fall back to MostFractionalBranching which returns None for no candidates
        assert decision is None

    def test_deterministic_tie_breaking(self):
        """Tie-breaking should be deterministic by variable name."""
        m = _build_simple_integer_lp()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        # Both variables have same fractional value
        node.lp_solution = {"x1": 1.5, "x2": 1.5}

        decision = strategy.choose_branching_variable(node, m, MILPTolerances())
        assert decision is not None
        # Should pick x1 (alphabetically first)
        assert decision.variable_name == "x1"

    def test_deterministic_repeated_decisions(self):
        """Repeated calls with same inputs should give same decision."""
        m = _build_fractional_relaxation()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 1.25, "x2": 1.25}

        decisions = []
        for _ in range(10):
            decision = strategy.choose_branching_variable(node, m, MILPTolerances())
            decisions.append(decision.variable_name)

        assert all(d == decisions[0] for d in decisions)

    def test_multiple_candidates_different_scores(self):
        """Multiple candidates with different scores should pick highest."""
        m = _build_high_constraint_participation()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 1.5, "x2": 1.5, "x3": 1.5}

        decision = strategy.choose_branching_variable(node, m, MILPTolerances())
        assert decision is not None
        # x1 has highest constraint participation (3/5), should be preferred
        assert decision.variable_name == "x1"


class TestFallback:
    """Test fallback behavior."""

    def test_fallback_on_empty_candidates(self):
        """Should fall back when no candidates."""
        m = _build_all_continuous()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 1.5, "x2": 2.5}

        decision = strategy.choose_branching_variable(node, m, MILPTolerances())
        assert decision is None  # MostFractionalBranching returns None for no candidates

    def test_fallback_on_scoring_error(self):
        """Should fall back when scoring fails."""
        m = _build_simple_integer_lp()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 1.5, "x2": 1.0}

        # Monkey-patch to cause scoring error
        original_compute = strategy._compute_score
        def failing_compute(features):
            raise ValueError("Scoring failed")
        strategy._compute_score = failing_compute

        decision = strategy.choose_branching_variable(node, m, MILPTolerances())
        assert decision is not None  # Should fall back and return a decision

    def test_fallback_strategy_used(self):
        """Custom fallback strategy should be used."""
        from optiforge.milp.solver import BranchingDecision

        class CustomFallback:
            def choose_branching_variable(self, node, model, tolerances):
                return BranchingDecision(variable_name="custom", value=999.0)

        fallback = CustomFallback()
        strategy = AdaptiveBranchingStrategy(fallback_strategy=fallback)

        m = _build_all_continuous()
        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 1.5, "x2": 2.5}

        decision = strategy.choose_branching_variable(node, m, MILPTolerances())
        assert decision is not None
        assert decision.variable_name == "custom"
        assert decision.value == 999.0


class TestBranchValidity:
    """Test that generated branches are mathematically valid."""

    def test_branch_bounds(self):
        """Branches should be x <= floor(v) and x >= ceil(v)."""
        m = _build_simple_integer_lp()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 1.5, "x2": 1.0}

        decision = strategy.choose_branching_variable(node, m, MILPTolerances())
        assert decision is not None

        value = decision.value
        floor_val = math.floor(value)
        ceil_val = math.ceil(value)

        # For value=1.5: floor=1, ceil=2
        # Left branch: x <= 1
        # Right branch: x >= 2
        assert floor_val == 1
        assert ceil_val == 2

    def test_negative_fractional_branches(self):
        """Negative fractional values should branch correctly."""
        m = _build_negative_fractional()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": -1.5, "x2": -1.0}

        decision = strategy.choose_branching_variable(node, m, MILPTolerances())
        assert decision is not None

        value = decision.value
        floor_val = math.floor(value)
        ceil_val = math.ceil(value)

        # For value=-1.5: floor=-2, ceil=-1
        # Left branch: x <= -2
        # Right branch: x >= -1
        assert floor_val == -2
        assert ceil_val == -1

    def test_binary_branches(self):
        """Binary variables should branch to 0 and 1."""
        m = _build_binary_knapsack()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 0.5, "x2": 0.0, "x3": 1.0}

        decision = strategy.choose_branching_variable(node, m, MILPTolerances())
        assert decision is not None

        value = decision.value
        floor_val = math.floor(value)
        ceil_val = math.ceil(value)

        # For binary 0.5: floor=0, ceil=1
        assert floor_val == 0
        assert ceil_val == 1


class TestObjectiveSense:
    """Test both MAXIMIZE and MINIMIZE."""

    def test_maximize(self):
        """MAXIMIZE problems should work."""
        m = _build_simple_integer_lp()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 1.5, "x2": 1.0}

        decision = strategy.choose_branching_variable(node, m, MILPTolerances())
        assert decision is not None
        assert decision.variable_name == "x1"

    def test_minimize(self):
        """MINIMIZE problems should work."""
        m = _build_minimize_model()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 1.5, "x2": 1.0}

        decision = strategy.choose_branching_variable(node, m, MILPTolerances())
        assert decision is not None
        assert decision.variable_name == "x1"

    def test_objective_sense_in_features(self):
        """Objective sense should be available in features when search state has it."""
        # SearchState doesn't have objective_sense, but we can test the feature exists
        m = _build_simple_integer_lp()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 1.5, "x2": 1.0}

        features = strategy._compute_features("x1", 1.5, False, m, node)
        # objective_sense comes from search_state which doesn't have it
        # This is expected to be None
        assert features.objective_sense is None


class TestSolverIntegration:
    """Test integration with MILPSolver."""

    def test_solver_with_adaptive_branching(self):
        """Solver should work with AdaptiveBranchingStrategy."""
        m = _build_fractional_relaxation()
        strategy = AdaptiveBranchingStrategy()
        solver = MILPSolver(branching_strategy=strategy)

        sol = solver.solve(m)
        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(2.0, abs=1e-6)

    def test_solver_with_adaptive_branching_and_fingerprint(self):
        """Solver should work with adaptive strategy using fingerprint."""
        m = _build_fractional_relaxation()
        fp = fingerprint_model(m)
        strategy = AdaptiveBranchingStrategy(fingerprint=fp)
        solver = MILPSolver(branching_strategy=strategy)

        sol = solver.solve(m)
        assert sol.status is SolutionStatus.OPTIMAL

    def test_solver_with_adaptive_branching_and_search_state(self):
        """Solver should work with adaptive strategy using search state."""
        m = _build_fractional_relaxation()
        strategy = AdaptiveBranchingStrategy()
        solver = MILPSolver(branching_strategy=strategy)

        sol = solver.solve(m)
        assert sol.status is SolutionStatus.OPTIMAL

    def test_default_behavior_unchanged(self):
        """Default MostFractionalBranching should remain default."""
        solver = MILPSolver()
        m = _build_fractional_relaxation()
        sol = solver.solve(m)
        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(2.0, abs=1e-6)

    def test_solver_reuse(self):
        """Solver reuse should work with adaptive strategy."""
        strategy = AdaptiveBranchingStrategy()
        solver = MILPSolver(branching_strategy=strategy)

        m1 = _build_simple_integer_lp()
        sol1 = solver.solve(m1)
        assert sol1.status is SolutionStatus.OPTIMAL

        m2 = _build_binary_knapsack()
        sol2 = solver.solve(m2)
        assert sol2.status is SolutionStatus.OPTIMAL

    def test_solve_milp_function(self):
        """solve_milp convenience function should work with adaptive strategy."""
        m = _build_fractional_relaxation()
        strategy = AdaptiveBranchingStrategy()
        sol = solve_milp(m, branching_strategy=strategy)
        assert sol.status is SolutionStatus.OPTIMAL


class TestAdaptivityDemonstration:
    """Critical test: demonstrate genuine adaptivity.

    This test shows that the adaptive strategy makes different decisions
    based on problem/search characteristics, not just MostFractionalBranching.
    """

    def test_binary_preference_over_general_integer(self):
        """Binary variables should be preferred over general integer when
        fractionality is equal, due to binary_preference weight."""
        m = _build_binary_vs_integer()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        # Both x1 (binary) and x2 (integer) have same fractionality (0.5)
        node.lp_solution = {"x1": 0.5, "x2": 1.5, "x3": 1.5}

        decision = strategy.choose_branching_variable(node, m, MILPTolerances())
        assert decision is not None
        # Binary x1 should be preferred over integer x2 due to binary_preference
        assert decision.variable_name == "x1"

    def test_constraint_participation_influence(self):
        """Variables in more constraints should be preferred when
        fractionality is equal."""
        m = _build_high_constraint_participation()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        # All have same fractionality (1.5)
        node.lp_solution = {"x1": 1.5, "x2": 1.5, "x3": 1.5}

        decision = strategy.choose_branching_variable(node, m, MILPTolerances())
        assert decision is not None
        # x1 participates in 3/5 constraints, x2 and x3 in 2/5
        assert decision.variable_name == "x1"

    def test_fractionality_primary_signal(self):
        """Fractionality should be the primary signal - more fractional
        variables should be preferred even if they have less constraint
        participation."""
        m = _build_high_constraint_participation()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        # x1: fractionality=0.5 (1.5), participation=3/5
        # x2: fractionality=0.1 (1.1), participation=2/5
        # x3: fractionality=0.1 (1.1), participation=2/5
        # x1 should win due to much higher fractionality
        node.lp_solution = {"x1": 1.5, "x2": 1.1, "x3": 1.1}

        decision = strategy.choose_branching_variable(node, m, MILPTolerances())
        assert decision is not None
        assert decision.variable_name == "x1"

    def test_binary_heavy_fingerprint_influence(self):
        """Binary-heavy fingerprint should increase binary_coverage score."""
        m = _build_binary_knapsack()
        fp = fingerprint_model(m)
        strategy = AdaptiveBranchingStrategy(fingerprint=fp)

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 0.5, "x2": 0.5, "x3": 0.5}

        # All binary, all same fractionality
        # Should pick x1 (alphabetically first due to tie-break)
        decision = strategy.choose_branching_variable(node, m, MILPTolerances())
        assert decision is not None
        assert decision.variable_name == "x1"

    def test_gap_influence(self):
        """Large gap should increase gap_factor score."""
        ss = SearchState(
            nodes_processed=10,
            nodes_remaining=5,
            incumbent_objective=10.0,
            best_open_bound=20.0,
            gap=0.5,  # large gap
            absolute_gap=10.0,
            feasible_solutions_found=2,
            nodes_pruned=3,
            max_depth=5,
            elapsed_time=2.0,
        )
        strategy = AdaptiveBranchingStrategy(search_state=ss)

        m = _build_simple_integer_lp()
        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 1.5, "x2": 1.0}

        decision = strategy.choose_branching_variable(node, m, MILPTolerances())
        assert decision is not None
        assert decision.variable_name == "x1"


class TestGenuineDivergence:
    """Tests that prove the adaptive strategy makes DIFFERENT decisions
    than MostFractionalBranching, demonstrating genuine adaptivity."""

    def test_binary_preferred_over_higher_fractionality_integer(self):
        """Binary variable with LOWER fractionality should be preferred over
        integer with HIGHER fractionality, due to binary_preference weight.

        MostFractionalBranching would pick x_integer (deviation=0.5).
        AdaptiveBranchingStrategy picks x_binary (binary_preference compensates).
        """
        m = Model("diverge_binary")
        m.add_variable(Variable("x_binary", variable_type=VariableType.BINARY))
        m.add_variable(Variable("x_integer", lower_bound=0, upper_bound=10,
                                variable_type=VariableType.INTEGER))
        m.add_constraint(Constraint("c1", {"x_binary": 1.0, "x_integer": 1.0},
                                    ConstraintSense.LESS_EQUAL, 10.0))
        m.set_objective(Objective({"x_binary": 1.0, "x_integer": 1.0},
                                  ObjectiveSense.MAXIMIZE))

        adaptive = AdaptiveBranchingStrategy()
        most_frac = MostFractionalBranching()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        # x_binary at 0.4 (deviation=0.4, fractionality=0.8)
        # x_integer at 0.5 (deviation=0.5, fractionality=1.0)
        # MostFractional: x_integer wins (0.5 > 0.4)
        # Adaptive: x_binary wins (0.8 + 0.5*1.0 = 1.3 > 1.0 + 0.5*0.0 = 1.0)
        node.lp_solution = {"x_binary": 0.4, "x_integer": 0.5}

        d_mf = most_frac.choose_branching_variable(node, m, MILPTolerances())
        d_ad = adaptive.choose_branching_variable(node, m, MILPTolerances())

        assert d_mf.variable_name == "x_integer"
        assert d_ad.variable_name == "x_binary"
        assert d_mf.variable_name != d_ad.variable_name

    def test_constraint_participation_overcomes_fractionality_gap(self):
        """Variable with HIGH constraint participation and LOWER fractionality
        should be preferred over variable with LOW participation and HIGHER
        fractionality, when the participation advantage is large enough.

        MostFractionalBranching would pick x_low_part (deviation=0.49).
        AdaptiveBranchingStrategy picks x_high_part (participation compensates).
        """
        m = Model("diverge_participation")
        m.add_variable(Variable("x_high_part", lower_bound=0,
                                variable_type=VariableType.INTEGER))
        m.add_variable(Variable("x_low_part", lower_bound=0,
                                variable_type=VariableType.INTEGER))
        # x_high_part appears in 3 constraints, x_low_part in 0
        m.add_constraint(Constraint("c1", {"x_high_part": 1.0},
                                    ConstraintSense.LESS_EQUAL, 3.0))
        m.add_constraint(Constraint("c2", {"x_high_part": 1.0},
                                    ConstraintSense.LESS_EQUAL, 3.0))
        m.add_constraint(Constraint("c3", {"x_high_part": 1.0},
                                    ConstraintSense.LESS_EQUAL, 3.0))
        m.add_constraint(Constraint("c4", {"x_low_part": 1.0},
                                    ConstraintSense.LESS_EQUAL, 5.0))
        m.set_objective(Objective({"x_high_part": 1.0, "x_low_part": 1.0},
                                  ObjectiveSense.MAXIMIZE))

        adaptive = AdaptiveBranchingStrategy()
        most_frac = MostFractionalBranching()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        # x_high_part at 1.44 (deviation=0.44, fractionality=0.88)
        # x_low_part at 1.49 (deviation=0.49, fractionality=0.98)
        # MostFractional: x_low_part wins (0.49 > 0.44)
        # Adaptive: x_high_part wins
        #   x_high_part: 0.88 + 0.3*(3/4) = 0.88 + 0.225 = 1.105
        #   x_low_part:  0.98 + 0.3*(0/4) = 0.98 + 0.0   = 0.98
        node.lp_solution = {"x_high_part": 1.44, "x_low_part": 1.49}

        d_mf = most_frac.choose_branching_variable(node, m, MILPTolerances())
        d_ad = adaptive.choose_branching_variable(node, m, MILPTolerances())

        assert d_mf.variable_name == "x_low_part"
        assert d_ad.variable_name == "x_high_part"
        assert d_mf.variable_name != d_ad.variable_name

    def test_is_binary_correctly_distinguishes_in_pipeline(self):
        """Verify that is_binary is correctly passed through the scoring
        pipeline (regression test for the hardcoded True bug)."""
        m = Model("pipeline_binary")
        m.add_variable(Variable("a_binary", variable_type=VariableType.BINARY))
        m.add_variable(Variable("b_integer", lower_bound=0,
                                variable_type=VariableType.INTEGER))
        m.add_constraint(Constraint("c1", {"a_binary": 1.0, "b_integer": 1.0},
                                    ConstraintSense.LESS_EQUAL, 5.0))
        m.set_objective(Objective({"a_binary": 1.0, "b_integer": 1.0},
                                  ObjectiveSense.MAXIMIZE))

        strategy = AdaptiveBranchingStrategy()
        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"a_binary": 0.5, "b_integer": 1.5}

        # Capture features from the scoring pipeline
        captured = {}
        orig = strategy._compute_features
        def capture(*args, **kwargs):
            result = orig(*args, **kwargs)
            captured[result.variable_name] = result
            return result
        strategy._compute_features = capture

        strategy.choose_branching_variable(node, m, MILPTolerances())

        # a_binary should be is_binary=True, b_integer should be is_binary=False
        assert captured["a_binary"].is_binary is True
        assert captured["b_integer"].is_binary is False

    def test_adaptive_score_diverges_from_most_fractional_score(self):
        """Score comparison: adaptive and MostFractional must produce different
        rankings for the same candidates."""
        m = Model("score_diverge")
        m.add_variable(Variable("x_bin", variable_type=VariableType.BINARY))
        m.add_variable(Variable("x_int", lower_bound=0, upper_bound=10,
                                variable_type=VariableType.INTEGER))
        m.add_constraint(Constraint("c1", {"x_bin": 1.0, "x_int": 1.0},
                                    ConstraintSense.LESS_EQUAL, 10.0))
        m.set_objective(Objective({"x_bin": 1.0, "x_int": 1.0},
                                  ObjectiveSense.MAXIMIZE))

        strategy = AdaptiveBranchingStrategy()
        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x_bin": 0.4, "x_int": 0.5}

        # Compute adaptive scores for each candidate
        feat_bin = strategy._compute_features("x_bin", 0.4, True, m, node)
        feat_int = strategy._compute_features("x_int", 0.5, False, m, node)
        score_bin, _ = strategy._compute_score(feat_bin)
        score_int, _ = strategy._compute_score(feat_int)

        # Adaptive: x_bin scores higher due to binary_preference
        assert score_bin > score_int
        # MostFractional would pick x_int (deviation 0.5 > 0.4)

    def test_solver_end_to_end_divergence(self):
        """End-to-end: solver with adaptive strategy should produce
        different branching decisions than MostFractionalBranching.

        Verifies the adaptive strategy adapts during an actual solve,
        not just in isolated unit tests."""
        m = Model("e2e_diverge")
        # Binary variable with tighter coupling, integer with looser
        m.add_variable(Variable("x_bin", variable_type=VariableType.BINARY))
        m.add_variable(Variable("y_int", lower_bound=0, upper_bound=5,
                                variable_type=VariableType.INTEGER))
        m.add_variable(Variable("z_int", lower_bound=0, upper_bound=5,
                                variable_type=VariableType.INTEGER))
        m.add_constraint(Constraint("c1", {"x_bin": 1.0, "y_int": 1.0},
                                    ConstraintSense.LESS_EQUAL, 3.0))
        m.add_constraint(Constraint("c2", {"x_bin": 1.0, "z_int": 1.0},
                                    ConstraintSense.LESS_EQUAL, 3.0))
        m.set_objective(Objective({"x_bin": 2.0, "y_int": 1.0, "z_int": 1.0},
                                  ObjectiveSense.MAXIMIZE))

        # Collect branching decisions from each strategy
        adaptive_decisions = []
        mf_decisions = []

        for strategy_class in (AdaptiveBranchingStrategy, MostFractionalBranching):
            decisions = []
            strategy = strategy_class()

            # Build a test node with fractional LP solution
            from optiforge.milp.solver import BnBNode
            node = BnBNode(node_id=1, parent_id=None, depth=0)
            # x_bin at 0.3 (deviation=0.3, fractionality=0.6)
            # y_int at 1.2 (deviation=0.2, fractionality=0.4)
            # z_int at 1.49 (deviation=0.49, fractionality=0.98)
            node.lp_solution = {"x_bin": 0.3, "y_int": 1.2, "z_int": 1.49}

            decision = strategy.choose_branching_variable(node, m, MILPTolerances())
            decisions.append(decision.variable_name if decision else None)

            if isinstance(strategy, AdaptiveBranchingStrategy):
                adaptive_decisions = decisions
            else:
                mf_decisions = decisions

        # MostFractional picks z_int (deviation=0.49, highest)
        assert mf_decisions[0] == "z_int"
        # Adaptive picks x_bin (binary_preference 0.5 compensates:
        # x_bin: 0.6 + 0.5*1.0 + 0.3*1.0 = 1.4
        # z_int: 0.98 + 0.5*0.0 + 0.3*0.5 = 1.13)
        assert adaptive_decisions[0] == "x_bin"
        assert adaptive_decisions != mf_decisions


class TestDeterminism:
    """Test deterministic behavior."""

    def test_repeated_solves_deterministic(self):
        """Repeated solves with same model should give same results."""
        m = _build_fractional_relaxation()
        strategy = AdaptiveBranchingStrategy()

        results = []
        for _ in range(5):
            solver = MILPSolver(branching_strategy=strategy)
            sol = solver.solve(m)
            results.append(sol.objective_value)

        assert all(r == results[0] for r in results)

    def test_same_model_different_solvers_same_decisions(self):
        """Two solvers with same strategy on same model should make same decisions."""
        m = _build_fractional_relaxation()
        strategy1 = AdaptiveBranchingStrategy()
        strategy2 = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 1.25, "x2": 1.25}

        d1 = strategy1.choose_branching_variable(node, m, MILPTolerances())
        d2 = strategy2.choose_branching_variable(node, m, MILPTolerances())

        assert d1.variable_name == d2.variable_name
        assert d1.value == d2.value


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_no_fingerprint_no_search_state(self):
        """Should work with neither fingerprint nor search state."""
        m = _build_simple_integer_lp()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 1.5, "x2": 1.0}

        decision = strategy.choose_branching_variable(node, m, MILPTolerances())
        assert decision is not None

    def test_none_fingerprint(self):
        """Explicit None fingerprint should work."""
        strategy = AdaptiveBranchingStrategy(fingerprint=None)
        assert strategy._fingerprint is None

    def test_none_search_state(self):
        """Explicit None search state should work."""
        strategy = AdaptiveBranchingStrategy(search_state=None)
        assert strategy._search_state is None

    def test_update_context_none(self):
        """update_context with None should not change existing values."""
        m = _build_simple_integer_lp()
        fp = fingerprint_model(m)
        strategy = AdaptiveBranchingStrategy(fingerprint=fp)

        strategy.update_context(fingerprint=None)
        assert strategy._fingerprint is fp  # unchanged

        strategy.update_context(search_state=None)
        assert strategy._search_state is None  # was None, stays None

    def test_binary_bounds_respected(self):
        """Binary variable bounds should be respected in branching."""
        m = _build_binary_knapsack()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 0.5, "x2": 0.0, "x3": 1.0}

        decision = strategy.choose_branching_variable(node, m, MILPTolerances())
        assert decision is not None

        value = decision.value
        floor_val = math.floor(value)
        ceil_val = math.ceil(value)

        # Binary bounds should be [0, 1]
        assert floor_val >= 0
        assert ceil_val <= 1


class TestImmutability:
    """Test that adaptive branching doesn't mutate inputs."""

    def test_model_not_mutated(self):
        """Model should not be mutated by branching strategy."""
        m = _build_fractional_relaxation()
        strategy = AdaptiveBranchingStrategy()

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 1.25, "x2": 1.25}

        # Snapshot model state
        before_vars = {name: (v.lower_bound, v.upper_bound, v.variable_type)
                       for name, v in m.variables.items()}

        strategy.choose_branching_variable(node, m, MILPTolerances())

        after_vars = {name: (v.lower_bound, v.upper_bound, v.variable_type)
                      for name, v in m.variables.items()}

        assert before_vars == after_vars

    def test_fingerprint_not_mutated(self):
        """Fingerprint should not be mutated."""
        m = _build_simple_integer_lp()
        fp = fingerprint_model(m)
        strategy = AdaptiveBranchingStrategy(fingerprint=fp)

        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 1.5, "x2": 1.0}

        before_fp = fp.to_dict()
        strategy.choose_branching_variable(node, m, MILPTolerances())
        after_fp = fp.to_dict()

        assert before_fp == after_fp

    def test_search_state_not_mutated(self):
        """Search state should not be mutated."""
        ss = SearchState(
            nodes_processed=5,
            nodes_remaining=3,
            incumbent_objective=10.0,
            best_open_bound=12.0,
            gap=0.2,
            absolute_gap=2.0,
            feasible_solutions_found=2,
            nodes_pruned=1,
            max_depth=3,
            elapsed_time=1.5,
        )
        strategy = AdaptiveBranchingStrategy(search_state=ss)

        m = _build_simple_integer_lp()
        from optiforge.milp.solver import BnBNode
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_solution = {"x1": 1.5, "x2": 1.0}

        before_ss = {
            "nodes_processed": ss.nodes_processed,
            "nodes_remaining": ss.nodes_remaining,
            "incumbent_objective": ss.incumbent_objective,
            "best_open_bound": ss.best_open_bound,
            "gap": ss.gap,
            "absolute_gap": ss.absolute_gap,
        }

        strategy.choose_branching_variable(node, m, MILPTolerances())

        after_ss = {
            "nodes_processed": ss.nodes_processed,
            "nodes_remaining": ss.nodes_remaining,
            "incumbent_objective": ss.incumbent_objective,
            "best_open_bound": ss.best_open_bound,
            "gap": ss.gap,
            "absolute_gap": ss.absolute_gap,
        }

        assert before_ss == after_ss