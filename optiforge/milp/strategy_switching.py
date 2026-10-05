"""Dynamic Strategy Switching for MILP Branch-and-Bound (Stage 7.7).

This module provides a lightweight, deterministic Strategy Manager that
observes the current ProblemFingerprint and SearchState and selects among
ALREADY-EXISTING strategies:

BRANCHING:
- MostFractionalBranching (default)
- AdaptiveBranchingStrategy (Stage 7.4)

NODE SELECTION:
- BestBoundNodeSelection (default)
- AdaptiveNodeSelectionStrategy (Stage 7.5)

HEURISTICS:
- no heuristic (None)
- configured PrimalHeuristic (Stage 7.6)

The manager does NOT implement any new branching or node-selection
algorithms. It only decides WHICH existing strategy is used at each
decision point, based on deterministic, explainable context signals.

Design guarantees
-----------------
- Deterministic: identical (fingerprint, search state, config) always
  yields the same decision.
- Anti-thrashing: a minimum switch interval (cooldown) prevents
  unnecessary strategy flapping at every node.
- State lifecycle: the manager resets its internal state at the start of
  each solve; no decisions/counters carry between independent solves.
- Observability: current selections, switch counts, and recent decisions
  are inspectable.
- Default behavior preserved: if no manager is configured, the solver
  uses its existing defaults exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from optiforge.milp.fingerprint import ProblemFingerprint
from optiforge.milp.search_state import SearchState


# ---------------------------------------------------------------------------
# Strategy identifiers
# ---------------------------------------------------------------------------

# Branching strategy identifiers
BRANCHING_MOST_FRACTIONAL = "most_fractional"
BRANCHING_ADAPTIVE = "adaptive_branching"

# Node selection strategy identifiers
NODE_SELECTION_BEST_BOUND = "best_bound"
NODE_SELECTION_ADAPTIVE = "adaptive_node_selection"

# Heuristic identifiers
HEURISTIC_NONE = "none"
HEURISTIC_CONFIGURED = "configured"


@dataclass(frozen=True)
class StrategyDecision:
    """Immutable decision produced by the StrategyManager.

    Attributes
    ----------
    branching_strategy : str
        Identifier of the selected branching strategy.
    node_selection_strategy : str
        Identifier of the selected node-selection strategy.
    heuristic : str
        Identifier of the selected heuristic ("none" or "configured").
    reason : str
        Human-readable explanation of why this decision was made.
    """

    branching_strategy: str
    node_selection_strategy: str
    heuristic: str
    reason: str = ""


@dataclass(frozen=True)
class StrategySwitchRecord:
    """Record of a single strategy switch event (for observability)."""

    nodes_processed: int
    """Number of nodes processed when the switch occurred."""

    decision: StrategyDecision
    """The new decision after the switch."""

    reason: str
    """Why the switch occurred."""


class StrategyManager:
    """Deterministic manager that selects among existing strategies.

    The manager observes the current ProblemFingerprint and SearchState and
    selects which existing branching / node-selection / heuristic strategy
    the solver should use. It is purely a *selector* — it never implements
    strategy logic itself.

    Parameters
    ----------
    enable_branching_switching : bool
        Whether branching strategy switching is enabled. If False, the
        branching strategy is fixed to the configured default.
    enable_node_selection_switching : bool
        Whether node-selection strategy switching is enabled. If False,
        the node-selection strategy is fixed to the configured default.
    enable_heuristic_switching : bool
        Whether heuristic switching is enabled. If False, the heuristic
        selection is fixed to the configured default.
    default_branching : str
        Default branching strategy identifier when switching is disabled
        or before the first decision.
    default_node_selection : str
        Default node-selection strategy identifier when switching is
        disabled or before the first decision.
    default_heuristic : str
        Default heuristic identifier when switching is disabled or before
        the first decision.
    min_switch_interval : int
        Minimum number of nodes that must be processed between strategy
        switches (anti-thrashing cooldown).
    gap_change_threshold : float
        Minimum relative change in the optimality gap required to justify
        a switch (hysteresis).
    """

    def __init__(
        self,
        enable_branching_switching: bool = True,
        enable_node_selection_switching: bool = True,
        enable_heuristic_switching: bool = True,
        default_branching: str = BRANCHING_MOST_FRACTIONAL,
        default_node_selection: str = NODE_SELECTION_BEST_BOUND,
        default_heuristic: str = HEURISTIC_NONE,
        min_switch_interval: int = 5,
        gap_change_threshold: float = 0.05,
    ) -> None:
        self.enable_branching_switching = enable_branching_switching
        self.enable_node_selection_switching = enable_node_selection_switching
        self.enable_heuristic_switching = enable_heuristic_switching
        self.default_branching = default_branching
        self.default_node_selection = default_node_selection
        self.default_heuristic = default_heuristic
        self.min_switch_interval = min_switch_interval
        self.gap_change_threshold = gap_change_threshold

        # Validate configuration
        self._validate_config()

        # Internal state (reset per solve)
        self._reset_state()

    def _validate_config(self) -> None:
        """Validate the manager configuration, raising on invalid input."""
        valid_branching = {BRANCHING_MOST_FRACTIONAL, BRANCHING_ADAPTIVE}
        valid_node = {NODE_SELECTION_BEST_BOUND, NODE_SELECTION_ADAPTIVE}
        valid_heuristic = {HEURISTIC_NONE, HEURISTIC_CONFIGURED}

        if self.default_branching not in valid_branching:
            raise ValueError(
                f"Invalid default_branching '{self.default_branching}'. "
                f"Must be one of {sorted(valid_branching)}."
            )
        if self.default_node_selection not in valid_node:
            raise ValueError(
                f"Invalid default_node_selection '{self.default_node_selection}'. "
                f"Must be one of {sorted(valid_node)}."
            )
        if self.default_heuristic not in valid_heuristic:
            raise ValueError(
                f"Invalid default_heuristic '{self.default_heuristic}'. "
                f"Must be one of {sorted(valid_heuristic)}."
            )
        if self.min_switch_interval < 0:
            raise ValueError(
                f"min_switch_interval must be >= 0, got {self.min_switch_interval}."
            )
        if self.gap_change_threshold < 0:
            raise ValueError(
                f"gap_change_threshold must be >= 0, got {self.gap_change_threshold}."
            )

    def _reset_state(self) -> None:
        """Reset all per-solve internal state."""
        self._current_branching = self.default_branching
        self._current_node_selection = self.default_node_selection
        self._current_heuristic = self.default_heuristic
        self._last_switch_node = -self.min_switch_interval - 1
        self._last_gap: float | None = None
        self._switch_count = 0
        self._recent_decisions: list[StrategySwitchRecord] = []
        self._solve_active = False

    # -----------------------------------------------------------------------
    # Solve lifecycle
    # -----------------------------------------------------------------------

    def start_solve(self) -> None:
        """Reset manager state at the start of a new solve.

        This guarantees no decisions or counters carry between independent
        solves (no stale state).
        """
        self._reset_state()
        self._solve_active = True

    def end_solve(self) -> None:
        """Mark the end of a solve."""
        self._solve_active = False

    # -----------------------------------------------------------------------
    # Decision making
    # -----------------------------------------------------------------------

    def decide(
        self,
        fingerprint: ProblemFingerprint | None,
        search_state: SearchState | None,
    ) -> StrategyDecision:
        """Make a deterministic strategy decision based on context.

        Parameters
        ----------
        fingerprint : ProblemFingerprint | None
            The problem fingerprint, or None if unavailable.
        search_state : SearchState | None
            The current search state, or None if unavailable.

        Returns
        -------
        StrategyDecision
            The selected strategy decision.
        """
        # If no context is available, keep current selections.
        if fingerprint is None and search_state is None:
            return self._current_decision("no context available")

        # Compute the desired selections based on context.
        desired_branching = self._select_branching(fingerprint, search_state)
        desired_node = self._select_node_selection(fingerprint, search_state)
        desired_heuristic = self._select_heuristic(fingerprint, search_state)

        # Apply anti-thrashing: only switch if enough nodes have passed
        # since the last switch AND the state change is meaningful.
        nodes_processed = search_state.nodes_processed if search_state is not None else 0
        can_switch = (
            nodes_processed - self._last_switch_node >= self.min_switch_interval
        )

        changed = (
            desired_branching != self._current_branching
            or desired_node != self._current_node_selection
            or desired_heuristic != self._current_heuristic
        )

        if changed and can_switch:
            # Hysteresis: require a meaningful gap change before switching
            # when an incumbent exists.
            gap = search_state.gap if search_state is not None else None
            gap_changed = self._gap_changed_meaningfully(gap)

            if gap_changed or self._last_gap is None:
                self._apply_switch(
                    desired_branching,
                    desired_node,
                    desired_heuristic,
                    nodes_processed,
                    self._build_reason(
                        desired_branching,
                        desired_node,
                        desired_heuristic,
                        fingerprint,
                        search_state,
                    ),
                )

        # Track gap for hysteresis on the next decision.
        if search_state is not None:
            self._last_gap = search_state.gap

        return self._current_decision("current selection")

    def _gap_changed_meaningfully(self, gap: float | None) -> bool:
        """Return True if the gap changed meaningfully since the last decision.

        This implements hysteresis: a switch is only justified when the
        search state has changed in a meaningful way (gap moved by more
        than the threshold), preventing thrashing on tiny fluctuations.

        Behavior with None gap:
        - If current gap is None (no incumbent yet) and last gap was None:
          Returns True (allow switch, no prior gap to compare)
        - If current gap is None but last gap was not None:
          Returns True (incumbent was lost or first incumbent appeared)
        - If current gap is not None but last gap was None:
          Returns True (incumbent first appeared, gap became available)
        - If both gaps are not None:
          Returns True only if absolute difference >= gap_change_threshold
        """
        if gap is None or self._last_gap is None:
            return True  # No prior gap to compare against -> allow switch.
        return abs(gap - self._last_gap) >= self.gap_change_threshold

    def _apply_switch(
        self,
        branching: str,
        node_selection: str,
        heuristic: str,
        nodes_processed: int,
        reason: str,
    ) -> None:
        """Apply a strategy switch and record it."""
        self._current_branching = branching
        self._current_node_selection = node_selection
        self._current_heuristic = heuristic
        self._last_switch_node = nodes_processed
        self._switch_count += 1

        decision = StrategyDecision(
            branching_strategy=branching,
            node_selection_strategy=node_selection,
            heuristic=heuristic,
            reason=reason,
        )
        self._recent_decisions.append(
            StrategySwitchRecord(
                nodes_processed=nodes_processed,
                decision=decision,
                reason=reason,
            )
        )

    def _current_decision(self, reason: str) -> StrategyDecision:
        """Build a StrategyDecision from the current selections."""
        return StrategyDecision(
            branching_strategy=self._current_branching,
            node_selection_strategy=self._current_node_selection,
            heuristic=self._current_heuristic,
            reason=reason,
        )

    def _build_reason(
        self,
        branching: str,
        node_selection: str,
        heuristic: str,
        fingerprint: ProblemFingerprint | None,
        search_state: SearchState | None,
    ) -> str:
        """Build a human-readable reason for a switch."""
        parts = []
        if fingerprint is not None:
            parts.append(
                f"vars={fingerprint.variable_count}, "
                f"int_frac={fingerprint.integer_variable_fraction:.2f}, "
                f"bin_frac={fingerprint.binary_variable_fraction:.2f}"
            )
        if search_state is not None:
            parts.append(
                f"nodes={search_state.nodes_processed}, "
                f"gap={search_state.gap if search_state.gap is not None else 'N/A'}"
            )
        context = "; ".join(parts) if parts else "no context"
        return (
            f"switched to branching={branching}, node={node_selection}, "
            f"heuristic={heuristic} [{context}]"
        )

    # -----------------------------------------------------------------------
    # Selection policies
    # -----------------------------------------------------------------------

    def _select_branching(
        self,
        fingerprint: ProblemFingerprint | None,
        search_state: SearchState | None,
    ) -> str:
        """Select the branching strategy based on context.

        Policy (deterministic, explainable):
        - If branching switching is disabled, return the default.
        - If the problem is binary-heavy (>= 50% binary variables) and the
          search is still early (no incumbent or large gap), use the
          adaptive branching strategy (it can exploit binary structure).
        - Otherwise use MostFractionalBranching (the safe default).
        """
        if not self.enable_branching_switching:
            return self.default_branching

        if fingerprint is None:
            return self.default_branching

        binary_heavy = fingerprint.binary_variable_fraction >= 0.5

        # Early search with binary-heavy structure -> adaptive branching.
        if binary_heavy:
            if search_state is None:
                return BRANCHING_ADAPTIVE
            if search_state.incumbent_objective is None:
                return BRANCHING_ADAPTIVE
            if search_state.gap is not None and search_state.gap > 0.1:
                return BRANCHING_ADAPTIVE

        return BRANCHING_MOST_FRACTIONAL

    def _select_node_selection(
        self,
        fingerprint: ProblemFingerprint | None,
        search_state: SearchState | None,
    ) -> str:
        """Select the node-selection strategy based on context.

        Policy (deterministic, explainable):
        - If node-selection switching is disabled, return the default.
        - If an incumbent exists and the gap is small (< 5%), the search
          is close to proving optimality -> use BestBound (proven,
          safe, tight bound-driven).
        - If the search is deep (max_depth >= 10) and the frontier is
          large, use AdaptiveNodeSelection (it can balance bound and
          depth pulls).
        - Otherwise use BestBound (the safe default).
        """
        if not self.enable_node_selection_switching:
            return self.default_node_selection

        if search_state is None:
            return self.default_node_selection

        # Small gap with incumbent -> BestBound (tight bound-driven).
        if (
            search_state.incumbent_objective is not None
            and search_state.gap is not None
            and search_state.gap < 0.05
        ):
            return NODE_SELECTION_BEST_BOUND

        # Deep search with large frontier -> adaptive node selection.
        if (
            search_state.max_depth >= 10
            and search_state.nodes_remaining > 20
        ):
            return NODE_SELECTION_ADAPTIVE

        return NODE_SELECTION_BEST_BOUND

    def _select_heuristic(
        self,
        fingerprint: ProblemFingerprint | None,
        search_state: SearchState | None,
    ) -> str:
        """Select the heuristic strategy based on context.

        Policy (deterministic, explainable):
        - If heuristic switching is disabled, return the default.
        - If no incumbent exists yet, enable the heuristic (it can find
          a first feasible solution quickly).
        - If an incumbent exists and the gap is large (> 10%), keep the
          heuristic active (it may improve the incumbent).
        - If the gap is small (< 1%), disable the heuristic (the search
          is nearly done; heuristic effort is wasted).
        - Otherwise keep the current heuristic selection.
        """
        if not self.enable_heuristic_switching:
            return self.default_heuristic

        if search_state is None:
            return self.default_heuristic

        if search_state.incumbent_objective is None:
            return HEURISTIC_CONFIGURED

        if search_state.gap is not None:
            if search_state.gap > 0.1:
                return HEURISTIC_CONFIGURED
            if search_state.gap < 0.01:
                return HEURISTIC_NONE

        return self._current_heuristic

    # -----------------------------------------------------------------------
    # Observability
    # -----------------------------------------------------------------------

    def get_current_branching(self) -> str:
        """Get the currently selected branching strategy identifier."""
        return self._current_branching

    def get_current_node_selection(self) -> str:
        """Get the currently selected node-selection strategy identifier."""
        return self._current_node_selection

    def get_current_heuristic(self) -> str:
        """Get the currently selected heuristic identifier."""
        return self._current_heuristic

    def get_switch_count(self) -> int:
        """Get the total number of strategy switches made."""
        return self._switch_count

    def get_recent_decisions(self) -> list[StrategySwitchRecord]:
        """Get the recent strategy switch records (observational)."""
        return list(self._recent_decisions)

    def get_current_decision(self) -> StrategyDecision:
        """Get the current strategy decision."""
        return self._current_decision("current selection")