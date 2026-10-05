"""Search State types for MILP Branch-and-Bound.

This module contains the core search state types that are shared between
the solver and the monitor, avoiding circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class SearchState:
    """Snapshot of the Branch-and-Bound search state for observation.

    This is an observation mechanism only — no adaptive decisions are made
    based on this state within the core solver.
    """

    nodes_processed: int
    """Total number of nodes processed so far."""

    nodes_remaining: int
    """Number of open (unprocessed) nodes in the queue."""

    incumbent_objective: float | None
    """Best integer-feasible objective found so far, or None if none found."""

    best_open_bound: float | None
    """Best LP relaxation bound among open nodes (highest for MAX, lowest for MIN)."""

    gap: float | None
    """Relative gap between incumbent and best open bound, if both exist."""

    absolute_gap: float | None
    """Absolute gap between incumbent and best open bound, if both exist."""

    feasible_solutions_found: int
    """Number of integer-feasible solutions found so far."""

    nodes_pruned: int
    """Total number of nodes pruned (by infeasibility, bound, or integrality)."""

    max_depth: int
    """Maximum depth reached in the B&B tree."""

    elapsed_time: float | None = None
    """Elapsed wall-clock time in seconds since the start of the current
    solve, or None if timing is not available (e.g. before the first node
    is processed)."""


SearchStateCallback = Callable[[SearchState], None]
"""Callback type for observing search state.

The callback receives a SearchState snapshot after each node is processed.
It is called for observation only — no adaptive decisions should be made
within the callback.
"""