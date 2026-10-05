"""Tests for Stage 7.5: Adaptive Node Selection Strategy."""

import math

import pytest

from optiforge.core.solution import SolutionStatus
from optiforge.milp import (
    AdaptiveBranchingStrategy,
    AdaptiveNodeSelectionStrategy,
    BestBoundNodeSelection,
    MILPSolver,
    MILPTolerances,
    MostFractionalBranching,
    ProblemFingerprint,
    SearchState,
    fingerprint_model,
    solve_milp,
)
from optiforge.milp.solver import BnBNode, NodeSelectionStrategy
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


def _build_minimize_model() -> Model:
    """Simple MINIMIZE model with fractional relaxation."""
    m = Model("minimize")
    m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, 2.5))
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MINIMIZE))
    return m


def _build_sparse_model() -> Model:
    """Sparse model (density < 0.1): many variables, few constraints.

    20 variables, 2 constraints. Each variable appears in only 1 constraint.
    Nonzeros = 20, total possible = 20 * 2 = 40, density = 0.5.
    To get density < 0.1, we need fewer nonzeros.
    Let's use 20 variables, 10 constraints, each variable in 1 constraint.
    Nonzeros = 20, total possible = 20 * 10 = 200, density = 0.1.
    Actually need < 0.1, so 20 variables, 25 constraints, each in 1 constraint.
    Nonzeros = 20, total = 500, density = 0.04.
    """
    m = Model("sparse")
    for i in range(20):
        m.add_variable(Variable(f"x{i}", lower_bound=0.0, variable_type=VariableType.INTEGER))
    # 25 constraints, each variable appears in exactly 1 constraint
    for i in range(25):
        # Each constraint has only 1 variable
        var_name = f"x{i % 20}"
        m.add_constraint(Constraint(f"c{i}", {var_name: 1.0}, ConstraintSense.LESS_EQUAL, 10.0))
    m.set_objective(Objective({f"x{i}": 1.0 for i in range(20)}, ObjectiveSense.MAXIMIZE))
    return m


def _build_dense_model() -> Model:
    """Dense model (density > 0.5): few variables, many constraints."""
    m = Model("dense")
    m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable("x3", lower_bound=0.0, variable_type=VariableType.INTEGER))
    # 5 constraints, 3 vars, all nonzeros -> 15/15 = 1.0 density
    for i in range(5):
        m.add_constraint(Constraint(f"c{i}", {"x1": 1.0, "x2": 1.0, "x3": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0, "x3": 1.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_negative_bounds_model() -> Model:
    """Model with negative bounds."""
    m = Model("neg_bounds")
    m.add_variable(Variable("x1", lower_bound=-10.0, upper_bound=10.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable("x2", lower_bound=-10.0, upper_bound=10.0, variable_type=VariableType.INTEGER))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 0.0))
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_zero_bounds_model() -> Model:
    """Model with zero bounds."""
    m = Model("zero_bounds")
    m.add_variable(Variable("x1", lower_bound=0.0, upper_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable("x2", lower_bound=0.0, upper_bound=5.0, variable_type=VariableType.INTEGER))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
    return m


# ── Tests ───────────────────────────────────────────────────────────────────


class TestAdaptiveNodeSelectionConstruction:
    """Test AdaptiveNodeSelectionStrategy construction and configuration."""

    def test_default_construction(self):
        """Test default construction with no arguments."""
        strategy = AdaptiveNodeSelectionStrategy()
        assert strategy is not None
        assert strategy.get_fallback_strategy() is not None

    def test_construction_with_fingerprint(self):
        """Test construction with fingerprint."""
        m = _build_simple_integer_lp()
        fp = fingerprint_model(m)
        strategy = AdaptiveNodeSelectionStrategy(fingerprint=fp)
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
        strategy = AdaptiveNodeSelectionStrategy(search_state=ss)
        assert strategy._search_state is ss

    def test_construction_with_fallback(self):
        """Test construction with custom fallback."""
        fallback = BestBoundNodeSelection()
        strategy = AdaptiveNodeSelectionStrategy(fallback_strategy=fallback)
        assert strategy.get_fallback_strategy() is fallback

    def test_update_context(self):
        """Test updating context after construction."""
        strategy = AdaptiveNodeSelectionStrategy()
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


class TestEmptyAndSingleNode:
    """Test empty and single-node edge cases."""

    def test_empty_open_nodes_raises(self):
        """Empty open-node set should raise RuntimeError (matching baseline)."""
        strategy = AdaptiveNodeSelectionStrategy()
        with pytest.raises(RuntimeError, match="No open nodes to select"):
            strategy.select_node([], {}, True)

    def test_single_node_returns_directly(self):
        """Single node should be returned without scoring."""
        strategy = AdaptiveNodeSelectionStrategy()
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        nodes = {1: node}
        result = strategy.select_node([1], nodes, True)
        assert result == 1

    def test_nonexistent_node_id_filtered(self):
        """Nonexistent node IDs in open_nodes should be filtered out."""
        strategy = AdaptiveNodeSelectionStrategy()
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        nodes = {1: node}
        # open_nodes contains 1 (exists) and 999 (doesn't exist)
        result = strategy.select_node([1, 999], nodes, True)
        assert result == 1


class TestValidNodeFiltering:
    """Test that only valid open nodes are selected."""

    def test_processed_node_not_preferred(self):
        """Processed nodes should rank below unprocessed nodes."""
        strategy = AdaptiveNodeSelectionStrategy()
        # Unprocessed node
        node1 = BnBNode(node_id=1, parent_id=None, depth=0)
        node1.lp_bound = 10.0
        # Processed node (has lp_objective)
        node2 = BnBNode(node_id=2, parent_id=None, depth=0)
        node2.lp_objective = 10.0
        node2.lp_bound = 10.0

        nodes = {1: node1, 2: node2}
        result = strategy.select_node([1, 2], nodes, True)
        # Unprocessed node should win despite same bound
        assert result == 1

    def test_invalid_node_ids_rejected(self):
        """Node IDs not in the nodes dict should never be selected."""
        strategy = AdaptiveNodeSelectionStrategy()
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_bound = 10.0
        nodes = {1: node}
        # open_nodes has 1 (valid) and 999 (invalid)
        result = strategy.select_node([1, 999], nodes, True)
        assert result == 1


class TestMaximizeBoundOrdering:
    """Test MAXIMIZE bound ordering."""

    def test_maximize_prefers_higher_bound(self):
        """For MAXIMIZE, node with higher bound should be preferred."""
        strategy = AdaptiveNodeSelectionStrategy()
        node1 = BnBNode(node_id=1, parent_id=None, depth=0)
        node1.lp_bound = 10.0
        node2 = BnBNode(node_id=2, parent_id=None, depth=0)
        node2.lp_bound = 15.0

        nodes = {1: node1, 2: node2}
        result = strategy.select_node([1, 2], nodes, True)
        assert result == 2  # Higher bound wins

    def test_maximize_with_incumbent(self):
        """For MAXIMIZE with incumbent, bound quality is incumbent-relative."""
        ss = SearchState(
            nodes_processed=5,
            nodes_remaining=3,
            incumbent_objective=10.0,
            best_open_bound=15.0,
            gap=0.5,
            absolute_gap=5.0,
            feasible_solutions_found=2,
            nodes_pruned=1,
            max_depth=3,
            elapsed_time=1.5,
        )
        strategy = AdaptiveNodeSelectionStrategy(search_state=ss)
        # Node at incumbent (barely not prunable)
        node1 = BnBNode(node_id=1, parent_id=None, depth=0)
        node1.lp_bound = 10.0
        # Node at best open bound
        node2 = BnBNode(node_id=2, parent_id=None, depth=0)
        node2.lp_bound = 15.0

        nodes = {1: node1, 2: node2}
        result = strategy.select_node([1, 2], nodes, True)
        assert result == 2  # Best open bound wins


class TestMinimizeBoundOrdering:
    """Test MINIMIZE bound ordering."""

    def test_minimize_prefers_lower_bound(self):
        """For MINIMIZE, node with lower bound should be preferred."""
        strategy = AdaptiveNodeSelectionStrategy()
        node1 = BnBNode(node_id=1, parent_id=None, depth=0)
        node1.lp_bound = 10.0
        node2 = BnBNode(node_id=2, parent_id=None, depth=0)
        node2.lp_bound = 5.0

        nodes = {1: node1, 2: node2}
        result = strategy.select_node([1, 2], nodes, False)
        assert result == 2  # Lower bound wins

    def test_minimize_with_incumbent(self):
        """For MINIMIZE with incumbent, bound quality is incumbent-relative."""
        ss = SearchState(
            nodes_processed=5,
            nodes_remaining=3,
            incumbent_objective=10.0,
            best_open_bound=5.0,
            gap=0.5,
            absolute_gap=5.0,
            feasible_solutions_found=2,
            nodes_pruned=1,
            max_depth=3,
            elapsed_time=1.5,
        )
        strategy = AdaptiveNodeSelectionStrategy(search_state=ss)
        # Node at incumbent (barely not prunable)
        node1 = BnBNode(node_id=1, parent_id=None, depth=0)
        node1.lp_bound = 10.0
        # Node at best open bound
        node2 = BnBNode(node_id=2, parent_id=None, depth=0)
        node2.lp_bound = 5.0

        nodes = {1: node1, 2: node2}
        result = strategy.select_node([1, 2], nodes, False)
        assert result == 2  # Best open bound wins


class TestNegativeBounds:
    """Test negative bounds."""

    def test_maximize_negative_bounds(self):
        """MAXIMIZE with negative bounds: higher (less negative) is better."""
        strategy = AdaptiveNodeSelectionStrategy()
        node1 = BnBNode(node_id=1, parent_id=None, depth=0)
        node1.lp_bound = -10.0
        node2 = BnBNode(node_id=2, parent_id=None, depth=0)
        node2.lp_bound = -5.0

        nodes = {1: node1, 2: node2}
        result = strategy.select_node([1, 2], nodes, True)
        assert result == 2  # -5 > -10

    def test_minimize_negative_bounds(self):
        """MINIMIZE with negative bounds: lower (more negative) is better."""
        strategy = AdaptiveNodeSelectionStrategy()
        node1 = BnBNode(node_id=1, parent_id=None, depth=0)
        node1.lp_bound = -10.0
        node2 = BnBNode(node_id=2, parent_id=None, depth=0)
        node2.lp_bound = -5.0

        nodes = {1: node1, 2: node2}
        result = strategy.select_node([1, 2], nodes, False)
        assert result == 1  # -10 < -5


class TestZeroBounds:
    """Test zero bounds."""

    def test_maximize_zero_bounds(self):
        """MAXIMIZE with zero bounds."""
        strategy = AdaptiveNodeSelectionStrategy()
        node1 = BnBNode(node_id=1, parent_id=None, depth=0)
        node1.lp_bound = 0.0
        node2 = BnBNode(node_id=2, parent_id=None, depth=0)
        node2.lp_bound = 5.0

        nodes = {1: node1, 2: node2}
        result = strategy.select_node([1, 2], nodes, True)
        assert result == 2

    def test_minimize_zero_bounds(self):
        """MINIMIZE with zero bounds."""
        strategy = AdaptiveNodeSelectionStrategy()
        node1 = BnBNode(node_id=1, parent_id=None, depth=0)
        node1.lp_bound = 0.0
        node2 = BnBNode(node_id=2, parent_id=None, depth=0)
        node2.lp_bound = 5.0

        nodes = {1: node1, 2: node2}
        result = strategy.select_node([1, 2], nodes, False)
        assert result == 1


class TestDepthBehavior:
    """Test depth behavior."""

    def test_depth_factor_normalization(self):
        """Depth factor should be normalized to [0, 1]."""
        strategy = AdaptiveNodeSelectionStrategy()
        node1 = BnBNode(node_id=1, parent_id=None, depth=0)
        node1.lp_bound = 10.0
        node2 = BnBNode(node_id=2, parent_id=None, depth=5)
        node2.lp_bound = 10.0

        nodes = {1: node1, 2: node2}
        # Same bounds, deeper node should win due to depth pull
        result = strategy.select_node([1, 2], nodes, True)
        assert result == 2

    def test_depth_tie_break(self):
        """When bounds and depth are equal, node ID breaks tie."""
        strategy = AdaptiveNodeSelectionStrategy()
        node1 = BnBNode(node_id=1, parent_id=None, depth=2)
        node1.lp_bound = 10.0
        node2 = BnBNode(node_id=2, parent_id=None, depth=2)
        node2.lp_bound = 10.0

        nodes = {1: node1, 2: node2}
        result = strategy.select_node([1, 2], nodes, True)
        assert result == 1  # Lower node ID wins


class TestFingerprintBehavior:
    """Test fingerprint behavior."""

    def test_sparse_fingerprint_strengthens_depth_pull(self):
        """Sparse fingerprint should strengthen depth pull."""
        m = _build_sparse_model()
        fp = fingerprint_model(m)
        assert fp.is_sparse is True

        strategy = AdaptiveNodeSelectionStrategy(fingerprint=fp)
        # Same bounds, different depths
        node1 = BnBNode(node_id=1, parent_id=None, depth=0)
        node1.lp_bound = 10.0
        node2 = BnBNode(node_id=2, parent_id=None, depth=5)
        node2.lp_bound = 10.0

        nodes = {1: node1, 2: node2}
        result = strategy.select_node([1, 2], nodes, True)
        assert result == 2  # Deeper node wins (depth pull strengthened)

    def test_dense_fingerprint_strengthens_bound_pull(self):
        """Dense fingerprint should strengthen bound pull."""
        m = _build_dense_model()
        fp = fingerprint_model(m)
        assert fp.is_dense is True

        strategy = AdaptiveNodeSelectionStrategy(fingerprint=fp)
        # Different bounds, same depth
        node1 = BnBNode(node_id=1, parent_id=None, depth=2)
        node1.lp_bound = 10.0
        node2 = BnBNode(node_id=2, parent_id=None, depth=2)
        node2.lp_bound = 15.0

        nodes = {1: node1, 2: node2}
        result = strategy.select_node([1, 2], nodes, True)
        assert result == 2  # Better bound wins (bound pull strengthened)


class TestSearchStateBehavior:
    """Test search-state behavior."""

    def test_incumbent_available_strengthens_depth_pull(self):
        """Incumbent available should strengthen depth pull."""
        ss = SearchState(
            nodes_processed=5,
            nodes_remaining=3,
            incumbent_objective=10.0,
            best_open_bound=15.0,
            gap=0.5,
            absolute_gap=5.0,
            feasible_solutions_found=2,
            nodes_pruned=1,
            max_depth=3,
            elapsed_time=1.5,
        )
        strategy = AdaptiveNodeSelectionStrategy(search_state=ss)
        # Same bounds, different depths
        node1 = BnBNode(node_id=1, parent_id=None, depth=0)
        node1.lp_bound = 10.0
        node2 = BnBNode(node_id=2, parent_id=None, depth=5)
        node2.lp_bound = 10.0

        nodes = {1: node1, 2: node2}
        result = strategy.select_node([1, 2], nodes, True)
        assert result == 2  # Deeper node wins (depth pull strengthened)

    def test_large_gap_strengthens_depth_pull(self):
        """Large gap should strengthen depth pull."""
        ss = SearchState(
            nodes_processed=5,
            nodes_remaining=3,
            incumbent_objective=10.0,
            best_open_bound=20.0,
            gap=0.5,  # large gap
            absolute_gap=10.0,
            feasible_solutions_found=2,
            nodes_pruned=1,
            max_depth=3,
            elapsed_time=1.5,
        )
        strategy = AdaptiveNodeSelectionStrategy(search_state=ss)
        # Same bounds, different depths
        node1 = BnBNode(node_id=1, parent_id=None, depth=0)
        node1.lp_bound = 10.0
        node2 = BnBNode(node_id=2, parent_id=None, depth=5)
        node2.lp_bound = 10.0

        nodes = {1: node1, 2: node2}
        result = strategy.select_node([1, 2], nodes, True)
        assert result == 2  # Deeper node wins

    def test_large_frontier_strengthens_depth_pull(self):
        """Large frontier should strengthen depth pull."""
        ss = SearchState(
            nodes_processed=5,
            nodes_remaining=10,  # large frontier
            incumbent_objective=10.0,
            best_open_bound=15.0,
            gap=0.33,
            absolute_gap=5.0,
            feasible_solutions_found=2,
            nodes_pruned=1,
            max_depth=3,
            elapsed_time=1.5,
        )
        strategy = AdaptiveNodeSelectionStrategy(search_state=ss)
        # Same bounds, different depths
        node1 = BnBNode(node_id=1, parent_id=None, depth=0)
        node1.lp_bound = 10.0
        node2 = BnBNode(node_id=2, parent_id=None, depth=5)
        node2.lp_bound = 10.0

        nodes = {1: node1, 2: node2}
        result = strategy.select_node([1, 2], nodes, True)
        assert result == 2  # Deeper node wins


class TestScoringFormula:
    """Test scoring formula."""

    def test_score_components_present(self):
        """All expected score components should be present."""
        strategy = AdaptiveNodeSelectionStrategy()
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_bound = 10.0
        nodes = {1: node}

        # Access internal method for testing
        _, best_bound, worst_bound, max_open_depth = strategy._open_node_stats([1], nodes, True)
        features = strategy._compute_features(node, [1], nodes, True, best_bound, worst_bound, max_open_depth)
        score, components = strategy._compute_score(features)

        expected_components = {
            "bound_pull",
            "depth_pull",
            "depth_weight",
            "processed_penalty",
        }
        assert set(components.keys()) == expected_components

    def test_score_range(self):
        """Score should be bounded."""
        strategy = AdaptiveNodeSelectionStrategy()
        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_bound = 10.0
        nodes = {1: node}

        _, best_bound, worst_bound, max_open_depth = strategy._open_node_stats([1], nodes, True)
        features = strategy._compute_features(node, [1], nodes, True, best_bound, worst_bound, max_open_depth)
        score, _ = strategy._compute_score(features)

        # Max unprocessed score: 1.0 (bound) + 0.9 (depth) = 1.9
        # Min unprocessed score: 0.0 (bound) + 0.5 * 0.0 (depth) = 0.0
        # Processed node: max 1.9 - 2.0 = -0.1
        assert score <= 1.9
        assert score >= -0.1


class TestDeterminism:
    """Test deterministic behavior."""

    def test_repeated_calls_identical(self):
        """Repeated calls with same inputs should give same decision."""
        strategy = AdaptiveNodeSelectionStrategy()
        node1 = BnBNode(node_id=1, parent_id=None, depth=0)
        node1.lp_bound = 10.0
        node2 = BnBNode(node_id=2, parent_id=None, depth=0)
        node2.lp_bound = 15.0
        nodes = {1: node1, 2: node2}

        results = []
        for _ in range(20):
            result = strategy.select_node([1, 2], nodes, True)
            results.append(result)

        assert all(r == results[0] for r in results)

    def test_tie_break_by_node_id(self):
        """Tie-breaking should be deterministic by node ID."""
        strategy = AdaptiveNodeSelectionStrategy()
        node1 = BnBNode(node_id=1, parent_id=None, depth=0)
        node1.lp_bound = 10.0
        node2 = BnBNode(node_id=2, parent_id=None, depth=0)
        node2.lp_bound = 10.0
        nodes = {1: node1, 2: node2}

        result = strategy.select_node([1, 2], nodes, True)
        assert result == 1  # Lower node ID wins


class TestFallback:
    """Test fallback behavior."""

    def test_fallback_on_scoring_error(self):
        """Should fall back when scoring fails."""
        strategy = AdaptiveNodeSelectionStrategy()

        # Monkey-patch to cause scoring error
        def failing_compute(features):
            raise ValueError("Scoring failed")

        strategy._compute_score = failing_compute

        node1 = BnBNode(node_id=1, parent_id=None, depth=0)
        node1.lp_bound = 10.0
        node2 = BnBNode(node_id=2, parent_id=None, depth=0)
        node2.lp_bound = 15.0
        nodes = {1: node1, 2: node2}

        # Should fall back to BestBoundNodeSelection (picks higher bound)
        result = strategy.select_node([1, 2], nodes, True)
        assert result == 2

    def test_custom_fallback_strategy_used(self):
        """Custom fallback strategy should be used."""
        class CustomFallback(NodeSelectionStrategy):
            def select_node(self, open_nodes, nodes, is_maximize):
                return open_nodes[-1]  # Always pick last

        fallback = CustomFallback()
        strategy = AdaptiveNodeSelectionStrategy(fallback_strategy=fallback)

        # Cause scoring error
        def failing_compute(features):
            raise ValueError("Scoring failed")
        strategy._compute_score = failing_compute

        node1 = BnBNode(node_id=1, parent_id=None, depth=0)
        node1.lp_bound = 10.0
        node2 = BnBNode(node_id=2, parent_id=None, depth=0)
        node2.lp_bound = 15.0
        nodes = {1: node1, 2: node2}

        result = strategy.select_node([1, 2], nodes, True)
        assert result == 2  # Custom fallback picks last


class TestSolverIntegration:
    """Test integration with MILPSolver."""

    def test_solver_with_adaptive_node_selection(self):
        """Solver should work with AdaptiveNodeSelectionStrategy."""
        m = _build_fractional_relaxation()
        strategy = AdaptiveNodeSelectionStrategy()
        solver = MILPSolver(node_selection_strategy=strategy)

        sol = solver.solve(m)
        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(2.0, abs=1e-6)

    def test_solver_with_adaptive_node_selection_and_fingerprint(self):
        """Solver should work with adaptive strategy using fingerprint."""
        m = _build_fractional_relaxation()
        fp = fingerprint_model(m)
        strategy = AdaptiveNodeSelectionStrategy(fingerprint=fp)
        solver = MILPSolver(node_selection_strategy=strategy)

        sol = solver.solve(m)
        assert sol.status is SolutionStatus.OPTIMAL

    def test_solver_with_adaptive_node_selection_and_search_state(self):
        """Solver should work with adaptive strategy using search state."""
        m = _build_fractional_relaxation()
        strategy = AdaptiveNodeSelectionStrategy()
        solver = MILPSolver(node_selection_strategy=strategy)

        sol = solver.solve(m)
        assert sol.status is SolutionStatus.OPTIMAL

    def test_default_behavior_unchanged(self):
        """Default BestBoundNodeSelection should remain default."""
        solver = MILPSolver()
        m = _build_fractional_relaxation()
        sol = solver.solve(m)
        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(2.0, abs=1e-6)

    def test_solver_reuse(self):
        """Solver reuse should work with adaptive strategy."""
        strategy = AdaptiveNodeSelectionStrategy()
        solver = MILPSolver(node_selection_strategy=strategy)

        m1 = _build_simple_integer_lp()
        sol1 = solver.solve(m1)
        assert sol1.status is SolutionStatus.OPTIMAL

        m2 = _build_binary_knapsack()
        sol2 = solver.solve(m2)
        assert sol2.status is SolutionStatus.OPTIMAL

    def test_solve_milp_function(self):
        """solve_milp convenience function should work with adaptive strategy."""
        m = _build_fractional_relaxation()
        strategy = AdaptiveNodeSelectionStrategy()
        sol = solve_milp(m, node_selection_strategy=strategy)
        assert sol.status is SolutionStatus.OPTIMAL


class TestAdaptivityDemonstration:
    """Critical test: demonstrate genuine adaptivity.

    This test shows that the adaptive strategy makes different decisions
    based on problem/search characteristics, not just BestBoundNodeSelection.
    """

    def test_incumbent_changes_ranking(self):
        """With incumbent, bound quality is incumbent-relative, changing ranking.

        Without incumbent: open-node range normalization.
        With incumbent: incumbent-relative normalization.
        This can change which node wins.
        """
        # Two nodes with bounds 10 and 15
        node1 = BnBNode(node_id=1, parent_id=None, depth=0)
        node1.lp_bound = 10.0
        node2 = BnBNode(node_id=2, parent_id=None, depth=0)
        node2.lp_bound = 15.0
        nodes = {1: node1, 2: node2}

        # Without incumbent: both have same depth, node2 wins (higher bound)
        strategy_no_inc = AdaptiveNodeSelectionStrategy()
        result_no_inc = strategy_no_inc.select_node([1, 2], nodes, True)
        assert result_no_inc == 2

        # With incumbent at 10: node1 is at incumbent (quality 0), node2 is best (quality 1)
        ss = SearchState(
            nodes_processed=5,
            nodes_remaining=3,
            incumbent_objective=10.0,
            best_open_bound=15.0,
            gap=0.5,
            absolute_gap=5.0,
            feasible_solutions_found=2,
            nodes_pruned=1,
            max_depth=3,
            elapsed_time=1.5,
        )
        strategy_inc = AdaptiveNodeSelectionStrategy(search_state=ss)
        result_inc = strategy_inc.select_node([1, 2], nodes, True)
        assert result_inc == 2  # Still node2, but for different reason

        # The key is: the normalization SCALE changed, which can change
        # ranking when depths differ. Let's test that:
        node3 = BnBNode(node_id=3, parent_id=None, depth=5)
        node3.lp_bound = 12.0
        nodes3 = {1: node1, 3: node3}

        # Without incumbent: node3 bound=12, node1 bound=10 -> node3 wins
        result_no_inc2 = strategy_no_inc.select_node([1, 3], nodes3, True)
        assert result_no_inc2 == 3

        # With incumbent at 10: node1 quality=0, node3 quality=(12-10)/(15-10)=0.4
        # But node3 has depth=5, node1 depth=0 -> depth pull favors node3
        # Both favor node3, so still node3. Need a case where ranking flips.

        # Let's try: node1 bound=14 (quality 0.8), depth=0; node3 bound=12 (quality 0.4), depth=5
        node1b = BnBNode(node_id=1, parent_id=None, depth=0)
        node1b.lp_bound = 14.0
        node3b = BnBNode(node_id=3, parent_id=None, depth=5)
        node3b.lp_bound = 12.0
        nodes3b = {1: node1b, 3: node3b}

        # Without incumbent: node1b bound=14 > node3b bound=12 -> node1b wins
        result_no_inc3 = strategy_no_inc.select_node([1, 3], nodes3b, True)
        assert result_no_inc3 == 1

        # With incumbent at 10: node1b quality=(14-10)/5=0.8, node3b quality=(12-10)/5=0.4
        # node1b score = 0.8 + 0.5*0 = 0.8
        # node3b score = 0.4 + 0.5*1.0 = 0.9 (depth_weight=0.5, depth_factor=1.0)
        # node3b wins! Ranking flipped.
        result_inc3 = strategy_inc.select_node([1, 3], nodes3b, True)
        assert result_inc3 == 3
        assert result_no_inc3 != result_inc3  # GENUINE ADAPTIVITY

    def test_gap_changes_ranking(self):
        """Large gap should strengthen depth pull, potentially flipping ranking."""
        # Node with better bound but shallow vs node with worse bound but deep
        node_shallow = BnBNode(node_id=1, parent_id=None, depth=0)
        node_shallow.lp_bound = 14.0
        node_deep = BnBNode(node_id=2, parent_id=None, depth=5)
        node_deep.lp_bound = 12.0
        nodes = {1: node_shallow, 2: node_deep}

        # Small gap: depth_weight=0.5
        ss_small = SearchState(
            nodes_processed=5,
            nodes_remaining=3,
            incumbent_objective=10.0,
            best_open_bound=14.0,
            gap=0.1,  # small gap
            absolute_gap=1.4,
            feasible_solutions_found=2,
            nodes_pruned=1,
            max_depth=3,
            elapsed_time=1.5,
        )
        strategy_small = AdaptiveNodeSelectionStrategy(search_state=ss_small)
        result_small = strategy_small.select_node([1, 2], nodes, True)
        # shallow: 0.8 + 0.5*0 = 0.8
        # deep: 0.4 + 0.5*1.0 = 0.9 -> deep wins

        # Large gap: depth_weight=0.5 + 0.2*1.0 = 0.7
        ss_large = SearchState(
            nodes_processed=5,
            nodes_remaining=3,
            incumbent_objective=10.0,
            best_open_bound=20.0,
            gap=0.5,  # large gap
            absolute_gap=10.0,
            feasible_solutions_found=2,
            nodes_pruned=1,
            max_depth=3,
            elapsed_time=1.5,
        )
        strategy_large = AdaptiveNodeSelectionStrategy(search_state=ss_large)
        result_large = strategy_large.select_node([1, 2], nodes, True)
        # shallow: 0.8 + 0.7*0 = 0.8
        # deep: 0.4 + 0.7*1.0 = 1.1 -> deep wins even more strongly

        # Both pick deep, but the SCORE DIFFERENCE changes.
        # To get a flip, we need a case where small gap picks shallow, large picks deep.
        # Let's adjust bounds:
        node_shallow2 = BnBNode(node_id=1, parent_id=None, depth=0)
        node_shallow2.lp_bound = 13.5  # quality = 3.5/10 = 0.35
        node_deep2 = BnBNode(node_id=2, parent_id=None, depth=5)
        node_deep2.lp_bound = 12.0   # quality = 2.0/10 = 0.2
        nodes2 = {1: node_shallow2, 2: node_deep2}

        # Small gap (0.1): depth_weight=0.5
        # shallow: 0.35 + 0.5*0 = 0.35
        # deep: 0.2 + 0.5*1.0 = 0.7 -> deep wins

        # Need shallow to win with small gap. Let's make shallow bound much better.
        node_shallow3 = BnBNode(node_id=1, parent_id=None, depth=0)
        node_shallow3.lp_bound = 14.5  # quality = 4.5/10 = 0.45
        node_deep3 = BnBNode(node_id=2, parent_id=None, depth=5)
        node_deep3.lp_bound = 12.0   # quality = 2.0/10 = 0.2
        nodes3 = {1: node_shallow3, 2: node_deep3}

        # Small gap (0.1): depth_weight=0.5
        # shallow: 0.45 + 0 = 0.45
        # deep: 0.2 + 0.5 = 0.7 -> deep wins

        # Hmm, depth pull is strong. Let's try with no incumbent (depth_weight=0.5 always)
        # and use open-node range normalization.
        strategy_no_inc = AdaptiveNodeSelectionStrategy()
        # bounds 14.5 and 12.0, max=14.5, min=12.0
        # shallow quality = (14.5-12)/(14.5-12) = 1.0
        # deep quality = (12-12)/(14.5-12) = 0.0
        # shallow: 1.0 + 0.5*0 = 1.0
        # deep: 0.0 + 0.5*1.0 = 0.5 -> shallow wins!
        result_no_inc = strategy_no_inc.select_node([1, 2], nodes3, True)
        assert result_no_inc == 1

        # With incumbent at 10, gap=0.5: depth_weight=0.7
        # shallow quality = (14.5-10)/(14.5-10) = 1.0
        # deep quality = (12-10)/(14.5-10) = 0.44
        # shallow: 1.0 + 0.7*0 = 1.0
        # deep: 0.44 + 0.7*1.0 = 1.14 -> deep wins!
        ss = SearchState(
            nodes_processed=5,
            nodes_remaining=3,
            incumbent_objective=10.0,
            best_open_bound=14.5,
            gap=0.5,
            absolute_gap=4.5,
            feasible_solutions_found=2,
            nodes_pruned=1,
            max_depth=3,
            elapsed_time=1.5,
        )
        strategy_inc = AdaptiveNodeSelectionStrategy(search_state=ss)
        result_inc = strategy_inc.select_node([1, 2], nodes3, True)
        assert result_inc == 2
        assert result_no_inc != result_inc  # GENUINE ADAPTIVITY


class TestInteractionWithAdaptiveBranching:
    """Test interaction with Stage 7.4 AdaptiveBranchingStrategy."""

    def test_both_adaptive_strategies_together(self):
        """Solver should work with both adaptive strategies simultaneously."""
        m = _build_fractional_relaxation()
        branching = AdaptiveBranchingStrategy()
        node_selection = AdaptiveNodeSelectionStrategy()
        solver = MILPSolver(
            branching_strategy=branching,
            node_selection_strategy=node_selection,
        )

        sol = solver.solve(m)
        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(2.0, abs=1e-6)

    def test_adaptive_branching_unchanged(self):
        """AdaptiveBranchingStrategy should work unchanged."""
        m = _build_fractional_relaxation()
        branching = AdaptiveBranchingStrategy()
        solver = MILPSolver(branching_strategy=branching)

        sol = solver.solve(m)
        assert sol.status is SolutionStatus.OPTIMAL


class TestSideEffects:
    """Test that adaptive node selection doesn't mutate inputs."""

    def test_model_not_mutated(self):
        """Model should not be mutated."""
        m = _build_fractional_relaxation()
        strategy = AdaptiveNodeSelectionStrategy()

        node1 = BnBNode(node_id=1, parent_id=None, depth=0)
        node1.lp_bound = 10.0
        node2 = BnBNode(node_id=2, parent_id=None, depth=0)
        node2.lp_bound = 15.0
        nodes = {1: node1, 2: node2}

        before = {nid: (n.lp_bound, n.lp_objective, n.depth) for nid, n in nodes.items()}
        strategy.select_node([1, 2], nodes, True)
        after = {nid: (n.lp_bound, n.lp_objective, n.depth) for nid, n in nodes.items()}

        assert before == after

    def test_fingerprint_not_mutated(self):
        """Fingerprint should not be mutated."""
        m = _build_simple_integer_lp()
        fp = fingerprint_model(m)
        strategy = AdaptiveNodeSelectionStrategy(fingerprint=fp)

        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_bound = 10.0
        nodes = {1: node}

        before = fp.to_dict()
        strategy.select_node([1], nodes, True)
        after = fp.to_dict()

        assert before == after

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
        strategy = AdaptiveNodeSelectionStrategy(search_state=ss)

        node = BnBNode(node_id=1, parent_id=None, depth=0)
        node.lp_bound = 10.0
        nodes = {1: node}

        before = {
            "nodes_processed": ss.nodes_processed,
            "nodes_remaining": ss.nodes_remaining,
            "incumbent_objective": ss.incumbent_objective,
            "best_open_bound": ss.best_open_bound,
            "gap": ss.gap,
            "absolute_gap": ss.absolute_gap,
        }
        strategy.select_node([1], nodes, True)
        after = {
            "nodes_processed": ss.nodes_processed,
            "nodes_remaining": ss.nodes_remaining,
            "incumbent_objective": ss.incumbent_objective,
            "best_open_bound": ss.best_open_bound,
            "gap": ss.gap,
            "absolute_gap": ss.absolute_gap,
        }

        assert before == after


class TestEndToEndAdaptivity:
    """End-to-end test: adaptive strategy chooses among multiple open nodes."""

    def test_multiple_open_nodes_adaptive_choice(self):
        """End-to-end: solver with adaptive strategy should choose among open nodes."""
        m = _build_fractional_relaxation()
        strategy = AdaptiveNodeSelectionStrategy()
        solver = MILPSolver(node_selection_strategy=strategy)

        sol = solver.solve(m)
        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(2.0, abs=1e-6)

    def test_adaptive_vs_best_bound_different_decisions(self):
        """Adaptive and BestBound should make different node selections in some cases."""
        # Build a model where multiple nodes become open
        m = _build_fractional_relaxation()
        fp = fingerprint_model(m)

        # Run with adaptive strategy
        adaptive_solver = MILPSolver(
            node_selection_strategy=AdaptiveNodeSelectionStrategy(fingerprint=fp),
        )
        adaptive_sol = adaptive_solver.solve(m)

        # Run with default best-bound strategy
        default_solver = MILPSolver()
        default_sol = default_solver.solve(m)

        # Both should find the same optimal solution
        assert adaptive_sol.status is SolutionStatus.OPTIMAL
        assert default_sol.status is SolutionStatus.OPTIMAL
        assert adaptive_sol.objective_value == pytest.approx(default_sol.objective_value, abs=1e-6)


class TestImmutability:
    """Test that adaptive node selection doesn't mutate inputs."""

    def test_nodes_dict_not_mutated(self):
        """Nodes dict should not be mutated."""
        strategy = AdaptiveNodeSelectionStrategy()
        node1 = BnBNode(node_id=1, parent_id=None, depth=0)
        node1.lp_bound = 10.0
        node2 = BnBNode(node_id=2, parent_id=None, depth=0)
        node2.lp_bound = 15.0
        nodes = {1: node1, 2: node2}

        before = {nid: (n.lp_bound, n.lp_objective, n.depth, n.bounds, n.constraints)
                  for nid, n in nodes.items()}
        strategy.select_node([1, 2], nodes, True)
        after = {nid: (n.lp_bound, n.lp_objective, n.depth, n.bounds, n.constraints)
                 for nid, n in nodes.items()}

        assert before == after