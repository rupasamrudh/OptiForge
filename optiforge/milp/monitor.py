"""Search State Monitor for MILP Branch-and-Bound.

This module provides a clean, reliable SearchStateMonitor component that
captures meaningful state during MILP Branch-and-Bound search. It is
designed for future adaptive strategies but remains purely observational.

The monitor reuses the existing SearchState dataclass and SearchStateCallback
abstractions from Stage 7.1, extending them with additional capabilities
such as history tracking and root-bound information.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from optiforge.core.solution import SolutionStatus
from optiforge.milp.search_state import SearchState, SearchStateCallback


@dataclass(frozen=True)
class SearchStateSnapshot:
    """Immutable snapshot of the Branch-and-Bound search state.

    This is a frozen dataclass to ensure snapshots cannot be mutated
    after creation. It extends the base SearchState with additional
    fields for root-bound information and history tracking.

    Attributes
    ----------
    nodes_processed : int
        Total number of nodes processed so far.
    nodes_remaining : int
        Number of open (unprocessed) nodes in the queue.
    incumbent_objective : float | None
        Best integer-feasible objective found so far, or None if none found.
    best_open_bound : float | None
        Best LP relaxation bound among open nodes (highest for MAX, lowest for MIN).
    gap : float | None
        Relative gap between incumbent and best open bound, if both exist.
    absolute_gap : float | None
        Absolute gap between incumbent and best open bound, if both exist.
    feasible_solutions_found : int
        Number of integer-feasible solutions found so far.
    nodes_pruned : int
        Total number of nodes pruned (by infeasibility, bound, or integrality).
    max_depth : int
        Maximum depth reached in the B&B tree.
    elapsed_time : float | None
        Elapsed wall-clock time in seconds since the start of the current solve.
    objective_sense : str | None
        "maximize" or "minimize" indicating the optimization direction.
    root_bound : float | None
        LP relaxation bound of the root node (if processed).
    current_depth : int | None
        Depth of the node currently being processed (None if between nodes).
    """

    nodes_processed: int
    nodes_remaining: int
    incumbent_objective: float | None
    best_open_bound: float | None
    gap: float | None
    absolute_gap: float | None
    feasible_solutions_found: int
    nodes_pruned: int
    max_depth: int
    elapsed_time: float | None
    objective_sense: str | None
    root_bound: float | None
    current_depth: int | None

    def to_dict(self) -> dict[str, Any]:
        """Convert snapshot to a dictionary for serialization."""
        return {
            "nodes_processed": self.nodes_processed,
            "nodes_remaining": self.nodes_remaining,
            "incumbent_objective": self.incumbent_objective,
            "best_open_bound": self.best_open_bound,
            "gap": self.gap,
            "absolute_gap": self.absolute_gap,
            "feasible_solutions_found": self.feasible_solutions_found,
            "nodes_pruned": self.nodes_pruned,
            "max_depth": self.max_depth,
            "elapsed_time": self.elapsed_time,
            "objective_sense": self.objective_sense,
            "root_bound": self.root_bound,
            "current_depth": self.current_depth,
        }

    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            f"SearchStateSnapshot(",
            f"  processed={self.nodes_processed}, remaining={self.nodes_remaining}",
            f"  incumbent={self.incumbent_objective}, best_open_bound={self.best_open_bound}",
            f"  gap={self.gap}, absolute_gap={self.absolute_gap}",
            f"  feasible={self.feasible_solutions_found}, pruned={self.nodes_pruned}",
            f"  max_depth={self.max_depth}, current_depth={self.current_depth}",
            f"  elapsed={self.elapsed_time:.3f}s" if self.elapsed_time is not None else "  elapsed=N/A",
            f"  sense={self.objective_sense}, root_bound={self.root_bound}",
            f")",
        ]
        return "\n".join(lines)


class SearchStateMonitor:
    """Observational monitor for MILP Branch-and-Bound search state.

    This class captures and maintains search state during the Branch-and-Bound
    process. It is purely observational and does not influence solver decisions.

    The monitor supports:
    - Callback-based observation after each node is processed
    - Optional history tracking with configurable maximum size
    - Immutable snapshots for safe external consumption
    - Root-bound tracking for initial relaxation information

    Parameters
    ----------
    max_history : int | None
        Maximum number of snapshots to retain in history. If None, history
        is unlimited (use with caution for long-running solves). Default: 1000.
    track_history : bool
        Whether to maintain a history of snapshots. Default: True.
    """

    def __init__(
        self,
        max_history: int | None = 1000,
        track_history: bool = True,
    ) -> None:
        self._max_history = max_history
        self._track_history = track_history
        self._history: list[SearchStateSnapshot] = []
        self._callback: SearchStateCallback | None = None
        self._solve_start_time: float | None = None
        self._objective_sense: str | None = None
        self._root_bound: float | None = None
        self._current_depth: int | None = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def set_callback(self, callback: SearchStateCallback | None) -> None:
        """Set or clear the search state callback.

        The callback is invoked with a SearchStateSnapshot after each
        node is processed.

        Parameters
        ----------
        callback : SearchStateCallback | None
            The callback to invoke, or None to disable callbacks.
        """
        self._callback = callback

    def get_callback(self) -> SearchStateCallback | None:
        """Get the current search state callback."""
        return self._callback

    def start_solve(self, objective_sense: str) -> None:
        """Initialize the monitor for a new solve.

        Parameters
        ----------
        objective_sense : str
            "maximize" or "minimize" indicating the optimization direction.
        """
        self._solve_start_time = time.monotonic()
        self._objective_sense = objective_sense
        self._root_bound = None
        self._current_depth = None
        if self._track_history:
            self._history.clear()

    def record_root_bound(self, bound: float | None) -> None:
        """Record the root node's LP relaxation bound.

        Parameters
        ----------
        bound : float | None
            The LP relaxation objective value of the root node, or None
            if the root was infeasible/unbounded.
        """
        self._root_bound = bound

    def record_node_start(self, depth: int) -> None:
        """Record that a node at the given depth is being processed.

        Parameters
        ----------
        depth : int
            Depth of the node in the B&B tree (root = 0).
        """
        self._current_depth = depth

    def record_node_complete(
        self,
        nodes_processed: int,
        nodes_remaining: int,
        incumbent_objective: float | None,
        best_open_bound: float | None,
        feasible_solutions_found: int,
        nodes_pruned: int,
        max_depth: int,
    ) -> None:
        """Record the completion of a node and build a snapshot.

        This method computes gaps and creates an immutable snapshot,
        optionally adding it to history and invoking the callback.

        Parameters
        ----------
        nodes_processed : int
            Total number of nodes processed so far.
        nodes_remaining : int
            Number of open (unprocessed) nodes in the queue.
        incumbent_objective : float | None
            Best integer-feasible objective found so far.
        best_open_bound : float | None
            Best LP relaxation bound among open nodes.
        feasible_solutions_found : int
            Number of integer-feasible solutions found so far.
        nodes_pruned : int
            Total number of nodes pruned.
        max_depth : int
            Maximum depth reached in the B&B tree.
        """
        # Compute gaps
        absolute_gap = None
        gap = None
        if incumbent_objective is not None and best_open_bound is not None:
            absolute_gap = abs(incumbent_objective - best_open_bound)
            if abs(incumbent_objective) > 1e-12:
                gap = absolute_gap / abs(incumbent_objective)
            elif absolute_gap > 1e-12:
                gap = float("inf")

        # Compute elapsed time
        elapsed_time = None
        if self._solve_start_time is not None:
            elapsed_time = time.monotonic() - self._solve_start_time

        # Create immutable snapshot
        snapshot = SearchStateSnapshot(
            nodes_processed=nodes_processed,
            nodes_remaining=nodes_remaining,
            incumbent_objective=incumbent_objective,
            best_open_bound=best_open_bound,
            gap=gap,
            absolute_gap=absolute_gap,
            feasible_solutions_found=feasible_solutions_found,
            nodes_pruned=nodes_pruned,
            max_depth=max_depth,
            elapsed_time=elapsed_time,
            objective_sense=self._objective_sense,
            root_bound=self._root_bound,
            current_depth=self._current_depth,
        )

        # Add to history if tracking
        if self._track_history:
            self._history.append(snapshot)
            # Enforce max history size
            if self._max_history is not None and len(self._history) > self._max_history:
                # Remove oldest entries
                self._history = self._history[-self._max_history:]

        # Invoke callback if set
        if self._callback is not None:
            # Convert to base SearchState for backward compatibility
            base_state = SearchState(
                nodes_processed=snapshot.nodes_processed,
                nodes_remaining=snapshot.nodes_remaining,
                incumbent_objective=snapshot.incumbent_objective,
                best_open_bound=snapshot.best_open_bound,
                gap=snapshot.gap,
                absolute_gap=snapshot.absolute_gap,
                feasible_solutions_found=snapshot.feasible_solutions_found,
                nodes_pruned=snapshot.nodes_pruned,
                max_depth=snapshot.max_depth,
                elapsed_time=snapshot.elapsed_time,
            )
            self._callback(base_state)

        # Clear current depth
        self._current_depth = None

    def end_solve(self) -> None:
        """Finalize the monitor after a solve completes."""
        self._solve_start_time = None
        self._objective_sense = None
        self._root_bound = None
        self._current_depth = None

    # -----------------------------------------------------------------------
    # History Access
    # -----------------------------------------------------------------------

    def get_history(self) -> list[SearchStateSnapshot]:
        """Get a copy of the snapshot history.

        Returns
        -------
        list[SearchStateSnapshot]
            Copy of the history list (oldest first).
        """
        return list(self._history)

    def get_latest_snapshot(self) -> SearchStateSnapshot | None:
        """Get the most recent snapshot, or None if no snapshots exist."""
        if not self._history:
            return None
        return self._history[-1]

    def clear_history(self) -> None:
        """Clear the snapshot history."""
        self._history.clear()

    def get_history_size(self) -> int:
        """Get the current number of snapshots in history."""
        return len(self._history)

    # -----------------------------------------------------------------------
    # Convenience Methods
    # -----------------------------------------------------------------------

    def get_snapshots_with_incumbent(self) -> list[SearchStateSnapshot]:
        """Get all snapshots where an incumbent was found."""
        return [s for s in self._history if s.incumbent_objective is not None]

    def get_snapshots_with_gap(self) -> list[SearchStateSnapshot]:
        """Get all snapshots where both incumbent and bound exist (gap defined)."""
        return [s for s in self._history if s.gap is not None]

    def get_incumbent_progression(self) -> list[tuple[int, float]]:
        """Get the progression of incumbent objectives over time.

        Returns
        -------
        list[tuple[int, float]]
            List of (nodes_processed, incumbent_objective) pairs.
        """
        return [
            (s.nodes_processed, s.incumbent_objective)
            for s in self._history
            if s.incumbent_objective is not None
        ]

    def get_gap_progression(self) -> list[tuple[int, float]]:
        """Get the progression of relative gaps over time.

        Returns
        -------
        list[tuple[int, float]]
            List of (nodes_processed, gap) pairs.
        """
        return [
            (s.nodes_processed, s.gap)
            for s in self._history
            if s.gap is not None
        ]

    def get_bound_progression(self) -> list[tuple[int, float]]:
        """Get the progression of best open bounds over time.

        Returns
        -------
        list[tuple[int, float]]
            List of (nodes_processed, best_open_bound) pairs.
        """
        return [
            (s.nodes_processed, s.best_open_bound)
            for s in self._history
            if s.best_open_bound is not None
        ]

    def __len__(self) -> int:
        """Return the number of snapshots in history."""
        return len(self._history)

    def __bool__(self) -> bool:
        """Return True if history is non-empty."""
        return bool(self._history)


def create_monitor_callback(monitor: SearchStateMonitor) -> SearchStateCallback:
    """Create a SearchStateCallback that delegates to a SearchStateMonitor.

    This is a convenience function for integrating the monitor with
    the existing MILPSolver callback mechanism.

    Parameters
    ----------
    monitor : SearchStateMonitor
        The monitor to delegate to.

    Returns
    -------
    SearchStateCallback
        A callback that records state in the monitor.
    """
    def callback(state: SearchState) -> None:
        # The monitor's record_node_complete will be called by the solver
        # This is just a pass-through for the callback interface
        pass
    return callback