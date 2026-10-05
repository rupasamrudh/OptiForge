"""Adaptive Node Selection Strategy for MILP Branch-and-Bound.

This module implements a conservative, deterministic, explainable adaptive
node selection strategy that uses ProblemFingerprint and SearchState
information to choose WHICH OPEN node is processed next.

The strategy is designed to be:
- Conservative: only uses well-understood, explainable features
- Deterministic: identical inputs always produce identical decisions
- Explainable: scoring formula is explicit and inspectable
- Safe: falls back to BestBoundNodeSelection when adaptive logic cannot decide

Scope boundary
--------------
This strategy ONLY decides which open node is processed next. It does NOT
change branching variable selection, branch generation, pruning logic,
termination conditions, solver tolerances, LP solver behavior, presolve,
or incumbent heuristics. Those belong to other stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from optiforge.milp.fingerprint import ProblemFingerprint
from optiforge.milp.search_state import SearchState
from optiforge.milp.solver import BnBNode, BestBoundNodeSelection, NodeSelectionStrategy


@dataclass(frozen=True)
class NodeFeatures:
    """Features computed for a candidate open node.

    All features are normalized to [0, 1] where possible for fair comparison.
    Each feature documents whether it is node-specific (can change which node
    wins) or global (identical for every open node at a given decision point,
    so it cannot change the ranking by itself).
    """

    node_id: int
    """Unique identifier of the node (used only for deterministic tie-breaking)."""

    bound_quality: float
    """Node-specific. Normalized quality of the node's effective bound.

    Meaning: how good the node's relaxation bound is relative to the pruning
    threshold (the incumbent) when one exists, otherwise relative to the
    best/worst bounds among open nodes. For MAXIMIZE, a larger upper bound is
    better; for MINIMIZE, a smaller lower bound is better.
    Range: [0, 1].
    Normalization (incumbent scale, when an incumbent exists):
        MAX: (bound - incumbent) / (best_open_bound - incumbent)
        MIN: (incumbent - bound) / (incumbent - best_open_bound)
      clamped to [0, 1]. A node at the incumbent scores 0.0 (barely not
      prunable); the best open node scores 1.0.
    Normalization (open-node range, when no incumbent exists):
        (bound - worst) / (best - worst), with 1.0 when all bounds are equal
        and 0.0 when the bound is unknown.
    Influence: higher score -> node processed sooner (best-bound-first pull).
    """

    depth_factor: float
    """Node-specific. Normalized depth of the node in the B&B tree.

    Meaning: how deep the node is relative to the deepest open node.
    Range: [0, 1]. 1.0 = deepest open node, 0.0 = shallowest.
    Normalization: depth / max_open_depth (0.0 when max_open_depth == 0).
    Influence: higher score -> deeper nodes preferred (depth-first pull).
    """

    is_processed: bool
    """Node-specific. True if the node has already been processed.

    Meaning: whether the node's LP relaxation has been solved. Open nodes are
    normally unprocessed; a processed node can only appear in the open set
    through caller error.
    Range: {False, True}.
    Normalization: none (boolean).
    Influence: processed nodes are never preferred; they rank below all
    unprocessed nodes regardless of bound/depth.
    """

    incumbent_available: bool
    """Global. True if an incumbent integer-feasible solution exists.

    Meaning: whether the search has a bound to prune against.
    Range: {False, True}.
    Normalization: none (boolean).
    Influence: when True, the depth pull is strengthened (dive toward a
    feasible solution); when False, the bound pull dominates (explore).
    Identical for every open node at a decision point -> cannot change
    which node wins by itself.
    """

    gap: float | None
    """Global. Relative gap between incumbent and best open bound.

    Meaning: how far the search is from proving optimality.
    Range: [0, inf) or None when no incumbent/best bound exists.
    Normalization: min(gap, 1.0) -> [0, 1].
    Influence: larger gap -> stronger depth pull (dive to find feasible
    solutions faster); smaller gap -> stronger bound pull (prove
    optimality). Identical for every open node -> cannot change ranking.
    """

    nodes_remaining: int
    """Global. Number of open (unprocessed) nodes in the queue.

    Meaning: size of the open-node frontier.
    Range: [0, inf).
    Normalization: none (used only as a threshold signal).
    Influence: when the frontier is large, the depth pull is strengthened
    (avoid exploding the tree); when small, bound pull dominates.
    Identical for every open node -> cannot change ranking.
    """

    is_sparse: bool
    """Global. From ProblemFingerprint: True if constraint density < 0.1.

    Meaning: the constraint matrix is structurally sparse.
    Range: {False, True}.
    Normalization: none (boolean).
    Influence: when True, the depth pull is strengthened (sparse problems
    tend to have weak LP bounds, so diving finds feasible solutions).
    Identical for every open node -> cannot change ranking.
    """

    is_dense: bool
    """Global. From ProblemFingerprint: True if constraint density > 0.5.

    Meaning: the constraint matrix is structurally dense.
    Range: {False, True}.
    Normalization: none (boolean).
    Influence: when True, the bound pull is strengthened (dense problems
    tend to have strong LP bounds, so best-bound-first is effective).
    Identical for every open node -> cannot change ranking.
    """

    def to_dict(self) -> dict[str, Any]:
        """Convert features to dictionary for inspection/debugging."""
        return {
            "node_id": self.node_id,
            "bound_quality": self.bound_quality,
            "depth_factor": self.depth_factor,
            "is_processed": self.is_processed,
            "incumbent_available": self.incumbent_available,
            "gap": self.gap,
            "nodes_remaining": self.nodes_remaining,
            "is_sparse": self.is_sparse,
            "is_dense": self.is_dense,
        }


@dataclass(frozen=True)
class NodeSelectionScore:
    """Result of scoring a candidate open node.

    Contains the computed score and the feature values used for transparency.
    """

    node_id: int
    """Node ID of the scored candidate."""

    score: float
    """Final computed score (higher is better)."""

    features: NodeFeatures
    """Features used to compute the score."""

    component_scores: dict[str, float]
    """Breakdown of score components for transparency."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for inspection/debugging."""
        return {
            "node_id": self.node_id,
            "score": self.score,
            "features": self.features.to_dict(),
            "component_scores": self.component_scores,
        }


class AdaptiveNodeSelectionStrategy(NodeSelectionStrategy):
    """Adaptive node selection strategy using ProblemFingerprint and SearchState.

    This strategy evaluates open B&B nodes using a deterministic scoring
    formula that combines:
    - Bound quality (best-bound-first pull, sense-aware)
    - Depth (depth-first pull, stronger when diving is beneficial)
    - Search state (gap, incumbent availability, frontier size)
    - Problem structure (sparse/dense fingerprint)

    The strategy falls back to BestBoundNodeSelection when:
    - No open nodes exist (raises RuntimeError, matching the baseline)
    - Feature calculation fails
    - Scoring produces invalid results

    Context management
    ------------------
    The fingerprint and search state are CALLER-MANAGED. The solver does not
    automatically feed live search state into this strategy; the caller must
    pass them via the constructor or ``update_context``. When not provided,
    global features use safe defaults (no incumbent, no gap, neutral
    structure), which makes the strategy behave like best-bound-first with a
    depth tie-break.
    """

    def __init__(
        self,
        fingerprint: ProblemFingerprint | None = None,
        search_state: SearchState | None = None,
        fallback_strategy: NodeSelectionStrategy | None = None,
    ) -> None:
        """Initialize the adaptive node selection strategy.

        Parameters
        ----------
        fingerprint : ProblemFingerprint | None
            Problem fingerprint for structural features. If None, structural
            features will not be used (neutral defaults).
        search_state : SearchState | None
            Current search state for dynamic features. If None, dynamic
            features will not be used (neutral defaults).
        fallback_strategy : NodeSelectionStrategy | None
            Fallback strategy when adaptive logic cannot decide.
            Defaults to BestBoundNodeSelection.
        """
        self._fingerprint = fingerprint
        self._search_state = search_state
        self._fallback = fallback_strategy or BestBoundNodeSelection()

    def update_context(
        self,
        fingerprint: ProblemFingerprint | None = None,
        search_state: SearchState | None = None,
    ) -> None:
        """Update the context for subsequent node selection decisions.

        This allows the strategy to adapt as the search progresses.

        Parameters
        ----------
        fingerprint : ProblemFingerprint | None
            Updated problem fingerprint (typically unchanged during solve).
        search_state : SearchState | None
            Updated search state (changes after each node).
        """
        if fingerprint is not None:
            self._fingerprint = fingerprint
        if search_state is not None:
            self._search_state = search_state

    def _effective_bound(self, node: BnBNode) -> float | None:
        """Return the node's effective bound, consistent with the solver.

        For a processed node this is the node's own LP objective. For an
        unprocessed node it is the bound inherited from the parent's
        relaxation (``lp_bound``). Returns ``None`` when no bound is known
        (e.g. the root node before it is processed).

        This mirrors ``MILPSolver._node_effective_bound`` so the adaptive
        strategy never invents a second, incompatible bound definition.
        """
        if node.lp_objective is not None:
            return node.lp_objective
        return node.lp_bound

    def _open_node_stats(
        self,
        open_nodes: list[int],
        nodes: dict[int, BnBNode],
        is_maximize: bool,
    ) -> tuple[list[float], float | None, float | None, int]:
        """Compute shared statistics across the open nodes.

        Returns
        -------
        tuple[list[float], float | None, float | None, int]
            (known_bounds, best_bound, worst_bound, max_depth).
            best/worst are None when no open node has a known bound.
        """
        bounds: list[float] = []
        max_depth = 0
        for nid in open_nodes:
            node = nodes.get(nid)
            if node is None:
                continue
            bound = self._effective_bound(node)
            if bound is not None:
                bounds.append(bound)
            max_depth = max(max_depth, node.depth)

        if not bounds:
            return bounds, None, None, max_depth

        if is_maximize:
            best_bound = max(bounds)
            worst_bound = min(bounds)
        else:
            best_bound = min(bounds)
            worst_bound = max(bounds)

        return bounds, best_bound, worst_bound, max_depth

    def _compute_features(
        self,
        node: BnBNode,
        open_nodes: list[int],
        nodes: dict[int, BnBNode],
        is_maximize: bool,
        best_bound: float | None,
        worst_bound: float | None,
        max_open_depth: int,
    ) -> NodeFeatures:
        """Compute all features for a candidate open node."""
        # --- Node-specific features -------------------------------------
        node_bound = self._effective_bound(node)

        # Bound quality: normalize against the incumbent (pruning threshold)
        # when one exists, otherwise against the open-node bound range.
        ss = self._search_state
        incumbent = ss.incumbent_objective if ss else None

        if node_bound is None:
            bound_quality = 0.0
        elif incumbent is not None and best_bound is not None:
            if is_maximize:
                denom = best_bound - incumbent
                if denom > 1e-12:
                    bound_quality = min(max((node_bound - incumbent) / denom, 0.0), 1.0)
                else:
                    bound_quality = 1.0  # all open bounds at incumbent
            else:
                denom = incumbent - best_bound
                if denom > 1e-12:
                    bound_quality = min(max((incumbent - node_bound) / denom, 0.0), 1.0)
                else:
                    bound_quality = 1.0
        elif best_bound is not None and worst_bound is not None:
            if abs(best_bound - worst_bound) < 1e-12:
                bound_quality = 1.0
            elif is_maximize:
                bound_quality = (node_bound - worst_bound) / (best_bound - worst_bound)
            else:
                bound_quality = (worst_bound - node_bound) / (worst_bound - best_bound)
        else:
            bound_quality = 1.0  # single node with a known bound

        # Depth factor: normalize depth against the deepest open node.
        if max_open_depth > 0:
            depth_factor = node.depth / max_open_depth
        else:
            depth_factor = 0.0

        is_processed = node.lp_objective is not None

        # --- Global features (identical for every open node) ------------
        incumbent_available = incumbent is not None
        gap = ss.gap if ss else None
        nodes_remaining = ss.nodes_remaining if ss else len(open_nodes)

        fp = self._fingerprint
        is_sparse = fp.is_sparse if fp else False
        is_dense = fp.is_dense if fp else False

        return NodeFeatures(
            node_id=node.node_id,
            bound_quality=bound_quality,
            depth_factor=depth_factor,
            is_processed=is_processed,
            incumbent_available=incumbent_available,
            gap=gap,
            nodes_remaining=nodes_remaining,
            is_sparse=is_sparse,
            is_dense=is_dense,
        )

    def _compute_score(self, features: NodeFeatures) -> tuple[float, dict[str, float]]:
        """Compute the adaptive node selection score for a candidate.

        The score combines two pulls:
        1. Bound pull (best-bound-first): prefer nodes with better bounds.
        2. Depth pull (depth-first): prefer deeper nodes.

        The relative strength of the two pulls adapts to the search context:
        - When an incumbent exists and the gap is large, the depth pull is
          strengthened (dive to find feasible solutions faster).
        - When the frontier is large, the depth pull is strengthened
          (avoid exploding the tree).
        - When the problem is sparse (weak LP bounds), the depth pull is
          strengthened (diving finds feasible solutions).
        - When the problem is dense (strong LP bounds), the bound pull is
          strengthened (best-bound-first is effective).

        Without an incumbent there is nothing to dive toward, so the bound
        pull dominates and the strategy behaves like best-bound-first.

        All components are normalized to [0, 1].

        Returns
        -------
        tuple[float, dict[str, float]]
            (total_score, component_scores_dict)
        """
        components: dict[str, float] = {}

        # 1. Bound pull (weight 1.0) - sense-aware bound quality
        components["bound_pull"] = features.bound_quality

        # 2. Depth pull - deeper nodes preferred. Base weight 0.5; the
        #    adaptive context can raise it up to 0.9 (dive mode) or keep it
        #    at 0.5 (explore mode). The bound pull always has weight 1.0.
        depth_weight = 0.5

        # Incumbent + large gap -> dive (find feasible solutions faster)
        if features.incumbent_available and features.gap is not None:
            normalized_gap = min(features.gap, 1.0)
            depth_weight += 0.2 * normalized_gap

        # Large frontier -> dive (avoid exploding the tree)
        if features.nodes_remaining >= 4:
            depth_weight += 0.1

        # Sparse problem -> dive (weak LP bounds)
        if features.is_sparse:
            depth_weight += 0.1

        # Dense problem -> bound pull dominates (strong LP bounds)
        if features.is_dense:
            depth_weight -= 0.1

        # Clamp to [0.5, 0.9]
        depth_weight = min(max(depth_weight, 0.5), 0.9)

        components["depth_pull"] = features.depth_factor
        components["depth_weight"] = depth_weight

        # Processed nodes are never preferred: a penalty of 2.0 guarantees
        # they rank below every unprocessed node (max unprocessed score is
        # 1.0 + 0.9 = 1.9, so a processed node scores at most -0.1).
        processed_penalty = 2.0 if features.is_processed else 0.0
        components["processed_penalty"] = processed_penalty

        total_score = (
            components["bound_pull"]
            + depth_weight * components["depth_pull"]
            - processed_penalty
        )

        return total_score, components

    def select_node(
        self,
        open_nodes: list[int],
        nodes: dict[int, BnBNode],
        is_maximize: bool,
    ) -> int:
        """Select the next node to process from the open nodes.

        Parameters
        ----------
        open_nodes : list[int]
            List of node IDs that are available for processing.
        nodes : dict[int, BnBNode]
            Dictionary mapping node IDs to BnBNode objects.
        is_maximize : bool
            True if the problem is a maximization, False for minimization.

        Returns
        -------
        int
            The node ID to process next.

        Raises
        ------
        RuntimeError
            If there are no open nodes (matching the baseline contract).
        """
        if not open_nodes:
            raise RuntimeError("No open nodes to select")

        # Single valid node: return it directly (no scoring needed).
        if len(open_nodes) == 1 and open_nodes[0] in nodes:
            return open_nodes[0]

        # Shared open-node statistics (computed once, not per candidate).
        _, best_bound, worst_bound, max_open_depth = self._open_node_stats(
            open_nodes, nodes, is_maximize
        )

        # Score all open nodes. Nonexistent node IDs are never selected.
        scored: list[tuple[int, float]] = []
        for nid in open_nodes:
            if nid not in nodes:
                continue
            try:
                features = self._compute_features(
                    nodes[nid], open_nodes, nodes, is_maximize,
                    best_bound, worst_bound, max_open_depth,
                )
                score, _ = self._compute_score(features)
                scored.append((nid, score))
            except Exception:
                # Fallback contract: any feature/scoring error degrades
                # gracefully to the baseline strategy rather than crashing
                # the solve. This mirrors the Stage 7.4 adaptive branching
                # fallback contract.
                return self._fallback.select_node(open_nodes, nodes, is_maximize)

        if not scored:
            return self._fallback.select_node(open_nodes, nodes, is_maximize)

        # Deterministic selection: highest score, tie-break by node ID.
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[0][0]

    def get_fallback_strategy(self) -> NodeSelectionStrategy:
        """Get the fallback strategy."""
        return self._fallback