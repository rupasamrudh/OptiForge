"""Numerical Stability Monitoring for OptiForge.

This module provides an observational monitoring layer for tracking numerical
behavior during LP and MILP solves. It is purely observational and does not
influence solver decisions.

The monitor tracks metrics where the underlying algorithms expose the required
data, and uses explicit None/unknown representations for unavailable metrics.

Limitations:
- Integrality violation is only meaningful for MILP integer-feasibility checks,
  not during LP relaxation iterations where all variables are continuous.
- Constraint violation and bound violation are computed from the current
  solution vector and constraint matrix; they may not reflect the original
  model after presolve transformations.
- Bound violation specifically measures standard-form non-negativity (x >= 0)
  violations only. It does NOT measure general upper-bound violations (x <= u)
  from the original model, as standard form represents bounds differently.
- Condition number estimation (np.linalg.cond) is O(n³) and is only computed
  when explicitly enabled via compute_condition_every parameter.
- Residual growth detection tracks residuals within a single phase; use
  reset_phase_state() when transitioning between Phase I and Phase II.
- Refactorization tracking is available but not automatically triggered by
  the current solver implementation (which refactorizes every iteration).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Callable, Optional

import numpy as np

from optiforge.numerics.diagnostics import NumericalDiagnostics
from optiforge.numerics.lu import DenseLU


# ---------------------------------------------------------------------------
# Warning Levels
# ---------------------------------------------------------------------------

class NumericalWarningLevel(IntEnum):
    """Severity levels for numerical warnings.

    Uses IntEnum to ensure proper ordering: NONE < INFO < WARNING < CRITICAL
    """

    NONE = 0
    """No numerical concerns detected."""

    INFO = 1
    """Informational: numerical behavior is within expected ranges but notable."""

    WARNING = 2
    """Warning: numerical behavior may affect solution quality or reliability."""

    CRITICAL = 3
    """Critical: severe numerical issues likely affecting correctness."""


# ---------------------------------------------------------------------------
# Event Types
# ---------------------------------------------------------------------------

class NumericalEventType(Enum):
    """Types of numerical events that can be recorded."""

    SMALL_PIVOT = "small_pivot"
    """Pivot element magnitude below warning threshold."""

    LARGE_PRIMAL_RESIDUAL = "large_primal_residual"
    """Primal feasibility residual exceeds warning threshold."""

    LARGE_DUAL_RESIDUAL = "large_dual_residual"
    """Dual feasibility residual exceeds warning threshold."""

    LARGE_CONSTRAINT_VIOLATION = "large_constraint_violation"
    """Constraint violation magnitude exceeds warning threshold."""

    LARGE_BOUND_VIOLATION = "large_bound_violation"
    """Bound violation magnitude exceeds warning threshold."""

    LARGE_INTEGRALITY_VIOLATION = "large_integrality_violation"
    """Integrality violation magnitude exceeds warning threshold."""

    LU_FACTORIZATION_FAILURE = "lu_factorization_failure"
    """LU factorization failed (singular or near-singular basis)."""

    NEAR_SINGULAR_BASIS = "near_singular_basis"
    """Basis matrix is near-singular (condition number exceeds threshold)."""

    RESIDUAL_GROWTH = "residual_growth"
    """Residual has grown significantly compared to previous iterations."""

    REFACTORIZATION = "refactorization"
    """Basis refactorization occurred (if supported by solver)."""

    PIVOT_MAGNITUDE = "pivot_magnitude"
    """Pivot element magnitude recorded for tracking."""


# ---------------------------------------------------------------------------
# Numerical Event
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NumericalEvent:
    """Immutable record of a numerical event.

    Attributes
    ----------
    event_type : NumericalEventType
        The type of numerical event.
    iteration : int | None
        Simplex iteration number (LP) or node ID (MILP), if available.
    node_id : int | None
        MILP node ID, if applicable.
    measured_value : float
        The measured numerical value that triggered the event.
    threshold : float
        The threshold that was exceeded (or compared against).
    warning_level : NumericalWarningLevel
        The severity level of this event.
    message : str
        Human-readable explanation of the event.
    context : dict[str, Any]
        Additional context (e.g., pivot row/col, basis indices).
    """

    event_type: NumericalEventType
    iteration: int | None = None
    node_id: int | None = None
    measured_value: float = 0.0
    threshold: float = 0.0
    warning_level: NumericalWarningLevel = NumericalWarningLevel.INFO
    message: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [
            f"NumericalEvent({self.event_type.value}",
            f"level={self.warning_level.name}",
            f"value={self.measured_value:.3e}",
            f"threshold={self.threshold:.3e}",
        ]
        if self.iteration is not None:
            parts.append(f"iter={self.iteration}")
        if self.node_id is not None:
            parts.append(f"node={self.node_id}")
        if self.message:
            parts.append(f"msg={self.message}")
        return ", ".join(parts) + ")"


# ---------------------------------------------------------------------------
# Numerical Snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NumericalSnapshot:
    """Immutable snapshot of numerical state at a specific iteration/node.

    Attributes
    ----------
    iteration : int | None
        Simplex iteration number (LP) or node ID (MILP), if available.
    node_id : int | None
        MILP node ID, if applicable.
    primal_residual : float | None
        Primal feasibility residual ||Ax - b|| (or max violation).
    dual_residual : float | None
        Dual feasibility residual ||A^T y - c||_∞ over ALL variables (basic and nonbasic).
        For basic variables, reduced cost is 0 by construction (B^T y = c_B).
        For nonbasic variables, reduced cost is c_j - A[:, j]^T y.
        This is the true analytical dual residual, not just max reduced cost magnitude.
    constraint_violation : float | None
        Maximum constraint violation magnitude.
    bound_violation : float | None
        Maximum bound violation magnitude.
        Note: This measures standard-form non-negativity violations (x >= 0) only.
        It does NOT measure general upper-bound violations (x <= u) from the
        original model, as standard form represents bounds differently.
    integrality_violation : float | None
        Maximum integrality violation magnitude (MILP).
        Note: Only meaningful during MILP integer-feasibility checks.
        During LP relaxation iterations (where all variables are continuous),
        this will be None.
    pivot_magnitude : float | None
        Magnitude of the pivot element used in this iteration.
    condition_estimate : float | None
        Estimated condition number of the basis matrix.
    warning_level : NumericalWarningLevel
        Overall warning level for this snapshot.
    event_count : int
        Number of events recorded at this iteration/node.
    diagnostics : NumericalDiagnostics | None
        Full numerical diagnostics from basis factorization, if available.
    """

    iteration: int | None = None
    node_id: int | None = None
    primal_residual: float | None = None
    dual_residual: float | None = None
    constraint_violation: float | None = None
    bound_violation: float | None = None
    integrality_violation: float | None = None
    pivot_magnitude: float | None = None
    condition_estimate: float | None = None
    warning_level: NumericalWarningLevel = NumericalWarningLevel.NONE
    event_count: int = 0
    diagnostics: NumericalDiagnostics | None = None

    def __str__(self) -> str:
        parts = [f"NumericalSnapshot("]
        if self.iteration is not None:
            parts.append(f"iter={self.iteration}")
        if self.node_id is not None:
            parts.append(f"node={self.node_id}")
        if self.primal_residual is not None:
            parts.append(f"primal_res={self.primal_residual:.3e}")
        if self.dual_residual is not None:
            parts.append(f"dual_res={self.dual_residual:.3e}")
        if self.constraint_violation is not None:
            parts.append(f"constr_viol={self.constraint_violation:.3e}")
        if self.bound_violation is not None:
            parts.append(f"bound_viol={self.bound_violation:.3e}")
        if self.integrality_violation is not None:
            parts.append(f"int_viol={self.integrality_violation:.3e}")
        if self.pivot_magnitude is not None:
            parts.append(f"pivot={self.pivot_magnitude:.3e}")
        if self.condition_estimate is not None:
            parts.append(f"cond={self.condition_estimate:.3e}")
        parts.append(f"level={self.warning_level.name}")
        parts.append(f"events={self.event_count}")
        return ", ".join(parts) + ")"


# ---------------------------------------------------------------------------
# Threshold Configuration
# ---------------------------------------------------------------------------

@dataclass
class NumericalThresholds:
    """Configurable thresholds for numerical monitoring.

    These thresholds are separate from solver tolerances and are used
    only for monitoring/warning purposes. They do not affect solver decisions.

    Attributes
    ----------
    small_pivot_warning : float
        Pivot magnitude below this triggers WARNING. Default: 1e-8.
    small_pivot_critical : float
        Pivot magnitude below this triggers CRITICAL. Default: 1e-12.
    primal_residual_warning : float
        Primal residual above this triggers WARNING. Default: 1e-6.
    primal_residual_critical : float
        Primal residual above this triggers CRITICAL. Default: 1e-4.
    dual_residual_warning : float
        Dual residual above this triggers WARNING. Default: 1e-6.
    dual_residual_critical : float
        Dual residual above this triggers CRITICAL. Default: 1e-4.
    constraint_violation_warning : float
        Constraint violation above this triggers WARNING. Default: 1e-6.
    constraint_violation_critical : float
        Constraint violation above this triggers CRITICAL. Default: 1e-4.
    bound_violation_warning : float
        Bound violation above this triggers WARNING. Default: 1e-7.
    bound_violation_critical : float
        Bound violation above this triggers CRITICAL. Default: 1e-5.
    integrality_violation_warning : float
        Integrality violation above this triggers WARNING. Default: 1e-5.
    integrality_violation_critical : float
        Integrality violation above this triggers CRITICAL. Default: 1e-3.
    condition_number_warning : float
        Condition number above this triggers WARNING. Default: 1e10.
    condition_number_critical : float
        Condition number above this triggers CRITICAL. Default: 1e14.
    residual_growth_factor_warning : float
        Residual growth factor above this triggers WARNING. Default: 100.0.
    residual_growth_factor_critical : float
        Residual growth factor above this triggers CRITICAL. Default: 1000.0.
    pivot_growth_warning : float
        Pivot growth factor above this triggers WARNING. Default: 1e8.
    pivot_growth_critical : float
        Pivot growth factor above this triggers CRITICAL. Default: 1e12.
    """

    # Pivot thresholds
    small_pivot_warning: float = 1e-8
    small_pivot_critical: float = 1e-12

    # Primal residual thresholds
    primal_residual_warning: float = 1e-6
    primal_residual_critical: float = 1e-4

    # Dual residual thresholds
    dual_residual_warning: float = 1e-6
    dual_residual_critical: float = 1e-4

    # Constraint violation thresholds
    constraint_violation_warning: float = 1e-6
    constraint_violation_critical: float = 1e-4

    # Bound violation thresholds
    bound_violation_warning: float = 1e-7
    bound_violation_critical: float = 1e-5

    # Integrality violation thresholds
    integrality_violation_warning: float = 1e-5
    integrality_violation_critical: float = 1e-3

    # Condition number thresholds
    condition_number_warning: float = 1e10
    condition_number_critical: float = 1e14

    # Residual growth thresholds
    residual_growth_factor_warning: float = 100.0
    residual_growth_factor_critical: float = 1000.0

    # Pivot growth thresholds
    pivot_growth_warning: float = 1e8
    pivot_growth_critical: float = 1e12

    def __post_init__(self) -> None:
        """Validate threshold ordering."""
        if self.small_pivot_critical >= self.small_pivot_warning:
            raise ValueError("small_pivot_critical must be < small_pivot_warning")
        if self.primal_residual_warning >= self.primal_residual_critical:
            raise ValueError("primal_residual_warning must be < primal_residual_critical")
        if self.dual_residual_warning >= self.dual_residual_critical:
            raise ValueError("dual_residual_warning must be < dual_residual_critical")
        if self.constraint_violation_warning >= self.constraint_violation_critical:
            raise ValueError("constraint_violation_warning must be < constraint_violation_critical")
        if self.bound_violation_warning >= self.bound_violation_critical:
            raise ValueError("bound_violation_warning must be < bound_violation_critical")
        if self.integrality_violation_warning >= self.integrality_violation_critical:
            raise ValueError("integrality_violation_warning must be < integrality_violation_critical")
        if self.condition_number_warning >= self.condition_number_critical:
            raise ValueError("condition_number_warning must be < condition_number_critical")
        if self.residual_growth_factor_warning >= self.residual_growth_factor_critical:
            raise ValueError("residual_growth_factor_warning must be < residual_growth_factor_critical")
        if self.pivot_growth_warning >= self.pivot_growth_critical:
            raise ValueError("pivot_growth_warning must be < pivot_growth_critical")


# ---------------------------------------------------------------------------
# Numerical Stability Monitor
# ---------------------------------------------------------------------------

class NumericalStabilityMonitor:
    """Observational monitor for numerical stability during LP/MILP solves.

    This monitor records numerical events and snapshots at key computation
    points. It is purely observational and does not influence solver decisions.

    The monitor can be attached to:
    - Revised Simplex iterations
    - LU factorization events
    - MILP node processing
    - Feasibility checks

    Parameters
    ----------
    thresholds : NumericalThresholds | None
        Custom thresholds. If None, defaults are used.
    max_history : int | None
        Maximum number of snapshots to retain. If None, unlimited.
        Default: 1000.
    track_history : bool
        Whether to maintain snapshot history. Default: True.
    callback : Callable[[NumericalSnapshot], None] | None
        Optional callback invoked after each snapshot is recorded.
    compute_condition_every : int | None
        Sampling interval for condition number estimation.
        If None (default), condition number is computed every iteration.
        If an integer N > 0, condition number is computed every N iterations.
    compute_residuals_every : int | None
        Sampling interval for residual computation.
        If None (default), residuals are computed every iteration.
        If an integer N > 0, residuals are computed every N iterations.
    """

    def __init__(
        self,
        thresholds: NumericalThresholds | None = None,
        max_history: int | None = 1000,
        track_history: bool = True,
        callback: Callable[[NumericalSnapshot], None] | None = None,
        compute_condition_every: int | None = None,
        compute_residuals_every: int | None = None,
    ) -> None:
        self._thresholds = thresholds or NumericalThresholds()
        self._max_history = max_history
        self._track_history = track_history
        self._callback = callback
        self._compute_condition_every = compute_condition_every
        self._compute_residuals_every = compute_residuals_every

        self._history: list[NumericalSnapshot] = []
        self._events: list[NumericalEvent] = []
        self._previous_primal_residual: float | None = None
        self._previous_dual_residual: float | None = None
        self._refactorization_count = 0
        self._iteration_counter = 0

    # -----------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------

    @property
    def thresholds(self) -> NumericalThresholds:
        """Get the current thresholds."""
        return self._thresholds

    @property
    def max_history(self) -> int | None:
        """Get the maximum history size."""
        return self._max_history

    @property
    def track_history(self) -> bool:
        """Whether history tracking is enabled."""
        return self._track_history

    @property
    def callback(self) -> Callable[[NumericalSnapshot], None] | None:
        """Get the current callback."""
        return self._callback

    def set_callback(self, callback: Callable[[NumericalSnapshot], None] | None) -> None:
        """Set or clear the snapshot callback."""
        self._callback = callback
    @property
    def compute_condition_every(self) -> int | None:
        """Get the condition estimation sampling interval."""
        return self._compute_condition_every

    @property
    def compute_residuals_every(self) -> int | None:
        """Get the residual computation sampling interval."""
        return self._compute_residuals_every

    def should_compute_residuals(self, iteration: int) -> bool:
        """Check if residuals should be computed at the given iteration.
        
        Parameters
        ----------
        iteration : int
            The current iteration number.
            
        Returns
        -------
        bool
            True if residuals should be computed, False otherwise.
            
        Notes
        -----
        If `compute_residuals_every` is None (default), residuals are computed
        every iteration. If set to an integer N > 0, residuals are computed
        every N iterations (i.e., when iteration % N == 0).
        """
        if self._compute_residuals_every is None:
            return True
        return iteration % self._compute_residuals_every == 0

    def should_compute_condition(self, iteration: int) -> bool:
        """Check if condition number should be computed at the given iteration.
        
        Parameters
        ----------
        iteration : int
            The current iteration number.
            
        Returns
        -------
        bool
            True if condition number should be computed, False otherwise.
            
        Notes
        -----
        If `compute_condition_every` is None (default), condition number is
        computed every iteration. If set to an integer N > 0, condition number
        is computed every N iterations (i.e., when iteration % N == 0).
        """
        if self._compute_condition_every is None:
            return True
        return iteration % self._compute_condition_every == 0

    # -----------------------------------------------------------------------
    # Event Recording
    # -----------------------------------------------------------------------

    def record_event(
        self,
        event_type: NumericalEventType,
        measured_value: float,
        threshold: float,
        iteration: int | None = None,
        node_id: int | None = None,
        message: str = "",
        context: dict[str, Any] | None = None,
    ) -> NumericalEvent:
        """Record a numerical event and return the created event.

        The warning level is determined by comparing measured_value against
        the threshold using the configured warning/critical thresholds.

        Parameters
        ----------
        event_type : NumericalEventType
            Type of event.
        measured_value : float
            The measured value.
        threshold : float
            The threshold for comparison.
        iteration : int | None
            Simplex iteration number.
        node_id : int | None
            MILP node ID.
        message : str
            Optional explanatory message.
        context : dict | None
            Additional context.

        Returns
        -------
        NumericalEvent
            The recorded event.
        """
        # Determine warning level based on event type and thresholds
        warning_level = self._classify_event(event_type, measured_value, threshold)

        event = NumericalEvent(
            event_type=event_type,
            iteration=iteration,
            node_id=node_id,
            measured_value=measured_value,
            threshold=threshold,
            warning_level=warning_level,
            message=message,
            context=context or {},
        )

        self._events.append(event)
        return event

    def _classify_event(
        self,
        event_type: NumericalEventType,
        measured_value: float,
        threshold: float,
    ) -> NumericalWarningLevel:
        """Classify event severity based on thresholds."""
        t = self._thresholds

        if event_type == NumericalEventType.SMALL_PIVOT:
            if measured_value <= t.small_pivot_critical:
                return NumericalWarningLevel.CRITICAL
            if measured_value <= t.small_pivot_warning:
                return NumericalWarningLevel.WARNING
            return NumericalWarningLevel.INFO

        if event_type == NumericalEventType.LARGE_PRIMAL_RESIDUAL:
            if measured_value >= t.primal_residual_critical:
                return NumericalWarningLevel.CRITICAL
            if measured_value >= t.primal_residual_warning:
                return NumericalWarningLevel.WARNING
            return NumericalWarningLevel.INFO

        if event_type == NumericalEventType.LARGE_DUAL_RESIDUAL:
            if measured_value >= t.dual_residual_critical:
                return NumericalWarningLevel.CRITICAL
            if measured_value >= t.dual_residual_warning:
                return NumericalWarningLevel.WARNING
            return NumericalWarningLevel.INFO

        if event_type == NumericalEventType.LARGE_CONSTRAINT_VIOLATION:
            if measured_value >= t.constraint_violation_critical:
                return NumericalWarningLevel.CRITICAL
            if measured_value >= t.constraint_violation_warning:
                return NumericalWarningLevel.WARNING
            return NumericalWarningLevel.INFO

        if event_type == NumericalEventType.LARGE_BOUND_VIOLATION:
            if measured_value >= t.bound_violation_critical:
                return NumericalWarningLevel.CRITICAL
            if measured_value >= t.bound_violation_warning:
                return NumericalWarningLevel.WARNING
            return NumericalWarningLevel.INFO

        if event_type == NumericalEventType.LARGE_INTEGRALITY_VIOLATION:
            if measured_value >= t.integrality_violation_critical:
                return NumericalWarningLevel.CRITICAL
            if measured_value >= t.integrality_violation_warning:
                return NumericalWarningLevel.WARNING
            return NumericalWarningLevel.INFO

        if event_type == NumericalEventType.NEAR_SINGULAR_BASIS:
            if measured_value >= t.condition_number_critical:
                return NumericalWarningLevel.CRITICAL
            if measured_value >= t.condition_number_warning:
                return NumericalWarningLevel.WARNING
            return NumericalWarningLevel.INFO

        if event_type == NumericalEventType.RESIDUAL_GROWTH:
            if measured_value >= t.residual_growth_factor_critical:
                return NumericalWarningLevel.CRITICAL
            if measured_value >= t.residual_growth_factor_warning:
                return NumericalWarningLevel.WARNING
            return NumericalWarningLevel.INFO

        if event_type == NumericalEventType.PIVOT_MAGNITUDE:
            # For pivot magnitude, we just record it; classification is based on small_pivot
            return NumericalWarningLevel.INFO

        if event_type == NumericalEventType.LU_FACTORIZATION_FAILURE:
            # LU factorization failure is always critical
            return NumericalWarningLevel.CRITICAL

        # Default for other event types
        return NumericalWarningLevel.INFO

    # -----------------------------------------------------------------------
    # Snapshot Recording
    # -----------------------------------------------------------------------

    def record_snapshot(
        self,
        iteration: int | None = None,
        node_id: int | None = None,
        primal_residual: float | None = None,
        dual_residual: float | None = None,
        constraint_violation: float | None = None,
        bound_violation: float | None = None,
        integrality_violation: float | None = None,
        pivot_magnitude: float | None = None,
        condition_estimate: float | None = None,
        diagnostics: NumericalDiagnostics | None = None,
    ) -> NumericalSnapshot:
        """Record a numerical snapshot at the current iteration/node.

        This method computes the overall warning level based on all provided
        metrics and records events for any threshold exceedances.

        Parameters
        ----------
        iteration : int | None
            Simplex iteration number.
        node_id : int | None
            MILP node ID.
        primal_residual : float | None
            Primal feasibility residual.
        dual_residual : float | None
            Dual feasibility residual.
        constraint_violation : float | None
            Maximum constraint violation.
        bound_violation : float | None
            Maximum bound violation.
        integrality_violation : float | None
            Maximum integrality violation.
        pivot_magnitude : float | None
            Pivot element magnitude.
        condition_estimate : float | None
            Estimated condition number.
        diagnostics : NumericalDiagnostics | None
            Full diagnostics from basis factorization.

        Returns
        -------
        NumericalSnapshot
            The recorded snapshot.
        """
        events_this_snapshot = 0
        max_warning_level = NumericalWarningLevel.NONE

        # Record events for each metric
        if primal_residual is not None:
            event = self.record_event(
                NumericalEventType.LARGE_PRIMAL_RESIDUAL,
                measured_value=primal_residual,
                threshold=self._thresholds.primal_residual_warning,
                iteration=iteration,
                node_id=node_id,
                message=f"Primal residual {primal_residual:.3e}",
            )
            if event.warning_level >= NumericalWarningLevel.WARNING:
                events_this_snapshot += 1
                max_warning_level = max(max_warning_level, event.warning_level)
            elif event.warning_level == NumericalWarningLevel.INFO:
                # INFO events are recorded but don't affect snapshot warning level
                pass

        if dual_residual is not None:
            event = self.record_event(
                NumericalEventType.LARGE_DUAL_RESIDUAL,
                measured_value=dual_residual,
                threshold=self._thresholds.dual_residual_warning,
                iteration=iteration,
                node_id=node_id,
                message=f"Dual residual {dual_residual:.3e}",
            )
            if event.warning_level >= NumericalWarningLevel.WARNING:
                events_this_snapshot += 1
                max_warning_level = max(max_warning_level, event.warning_level)
            elif event.warning_level == NumericalWarningLevel.INFO:
                # INFO events are recorded but don't affect snapshot warning level
                pass

        if constraint_violation is not None:
            event = self.record_event(
                NumericalEventType.LARGE_CONSTRAINT_VIOLATION,
                measured_value=constraint_violation,
                threshold=self._thresholds.constraint_violation_warning,
                iteration=iteration,
                node_id=node_id,
                message=f"Constraint violation {constraint_violation:.3e}",
            )
            if event.warning_level >= NumericalWarningLevel.WARNING:
                events_this_snapshot += 1
                max_warning_level = max(max_warning_level, event.warning_level)
            elif event.warning_level == NumericalWarningLevel.INFO:
                # INFO events are recorded but don't affect snapshot warning level
                pass

        if bound_violation is not None:
            event = self.record_event(
                NumericalEventType.LARGE_BOUND_VIOLATION,
                measured_value=bound_violation,
                threshold=self._thresholds.bound_violation_warning,
                iteration=iteration,
                node_id=node_id,
                message=f"Bound violation {bound_violation:.3e}",
            )
            if event.warning_level >= NumericalWarningLevel.WARNING:
                events_this_snapshot += 1
                max_warning_level = max(max_warning_level, event.warning_level)
            elif event.warning_level == NumericalWarningLevel.INFO:
                # INFO events are recorded but don't affect snapshot warning level
                pass

        if integrality_violation is not None:
            event = self.record_event(
                NumericalEventType.LARGE_INTEGRALITY_VIOLATION,
                measured_value=integrality_violation,
                threshold=self._thresholds.integrality_violation_warning,
                iteration=iteration,
                node_id=node_id,
                message=f"Integrality violation {integrality_violation:.3e}",
            )
            if event.warning_level >= NumericalWarningLevel.WARNING:
                events_this_snapshot += 1
                max_warning_level = max(max_warning_level, event.warning_level)
            elif event.warning_level == NumericalWarningLevel.INFO:
                # INFO events are recorded but don't affect snapshot warning level
                pass

        if pivot_magnitude is not None:
            # Record pivot magnitude event
            self.record_event(
                NumericalEventType.PIVOT_MAGNITUDE,
                measured_value=pivot_magnitude,
                threshold=self._thresholds.small_pivot_warning,
                iteration=iteration,
                node_id=node_id,
                message=f"Pivot magnitude {pivot_magnitude:.3e}",
            )
            # Check for small pivot
            if pivot_magnitude <= self._thresholds.small_pivot_warning:
                event = self.record_event(
                    NumericalEventType.SMALL_PIVOT,
                    measured_value=pivot_magnitude,
                    threshold=self._thresholds.small_pivot_warning,
                    iteration=iteration,
                    node_id=node_id,
                    message=f"Small pivot {pivot_magnitude:.3e}",
                )
                if event.warning_level >= NumericalWarningLevel.WARNING:
                    events_this_snapshot += 1
                    max_warning_level = max(max_warning_level, event.warning_level)
                elif event.warning_level == NumericalWarningLevel.INFO:
                    # INFO events are recorded but don't affect snapshot warning level
                    pass

        if condition_estimate is not None and not math.isnan(condition_estimate):
            if condition_estimate >= self._thresholds.condition_number_warning:
                event = self.record_event(
                    NumericalEventType.NEAR_SINGULAR_BASIS,
                    measured_value=condition_estimate,
                    threshold=self._thresholds.condition_number_warning,
                    iteration=iteration,
                    node_id=node_id,
                    message=f"Condition number {condition_estimate:.3e}",
                )
                if event.warning_level >= NumericalWarningLevel.WARNING:
                    events_this_snapshot += 1
                    max_warning_level = max(max_warning_level, event.warning_level)
                elif event.warning_level == NumericalWarningLevel.INFO:
                    # INFO events are recorded but don't affect snapshot warning level
                    pass

        # Check residual growth
        if primal_residual is not None and self._previous_primal_residual is not None:
            if self._previous_primal_residual > 1e-15:
                growth = primal_residual / self._previous_primal_residual
                if growth >= self._thresholds.residual_growth_factor_warning:
                    event = self.record_event(
                        NumericalEventType.RESIDUAL_GROWTH,
                        measured_value=growth,
                        threshold=self._thresholds.residual_growth_factor_warning,
                        iteration=iteration,
                        node_id=node_id,
                        message=f"Primal residual growth factor {growth:.3e}",
                    )
                    if event.warning_level >= NumericalWarningLevel.WARNING:
                        events_this_snapshot += 1
                        max_warning_level = max(max_warning_level, event.warning_level)
                    elif event.warning_level == NumericalWarningLevel.INFO:
                        # INFO events are recorded but don't affect snapshot warning level
                        pass

        if dual_residual is not None and self._previous_dual_residual is not None:
            if self._previous_dual_residual > 1e-15:
                growth = dual_residual / self._previous_dual_residual
                if growth >= self._thresholds.residual_growth_factor_warning:
                    event = self.record_event(
                        NumericalEventType.RESIDUAL_GROWTH,
                        measured_value=growth,
                        threshold=self._thresholds.residual_growth_factor_warning,
                        iteration=iteration,
                        node_id=node_id,
                        message=f"Dual residual growth factor {growth:.3e}",
                    )
                    if event.warning_level >= NumericalWarningLevel.WARNING:
                        events_this_snapshot += 1
                        max_warning_level = max(max_warning_level, event.warning_level)
                    elif event.warning_level == NumericalWarningLevel.INFO:
                        # INFO events are recorded but don't affect snapshot warning level
                        pass

        # Update previous residuals
        if primal_residual is not None:
            self._previous_primal_residual = primal_residual
        if dual_residual is not None:
            self._previous_dual_residual = dual_residual

        # Create snapshot
        snapshot = NumericalSnapshot(
            iteration=iteration,
            node_id=node_id,
            primal_residual=primal_residual,
            dual_residual=dual_residual,
            constraint_violation=constraint_violation,
            bound_violation=bound_violation,
            integrality_violation=integrality_violation,
            pivot_magnitude=pivot_magnitude,
            condition_estimate=condition_estimate,
            warning_level=max_warning_level,
            event_count=events_this_snapshot,
            diagnostics=diagnostics,
        )

        # Add to history
        if self._track_history:
            self._history.append(snapshot)
            if self._max_history is not None and len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        # Invoke callback
        if self._callback is not None:
            self._callback(snapshot)

        return snapshot

    def record_lu_factorization(
        self,
        lu: DenseLU,
        iteration: int | None = None,
        node_id: int | None = None,
        success: bool = True,
    ) -> NumericalSnapshot:
        """Record a snapshot from an LU factorization.

        Parameters
        ----------
        lu : DenseLU
            The LU factorization result.
        iteration : int | None
            Simplex iteration number.
        node_id : int | None
            MILP node ID.
        success : bool
            Whether factorization succeeded.

        Returns
        -------
        NumericalSnapshot
            The recorded snapshot.
        """
        if not success:
            self.record_event(
                NumericalEventType.LU_FACTORIZATION_FAILURE,
                measured_value=0.0,
                threshold=0.0,
                iteration=iteration,
                node_id=node_id,
                message="LU factorization failed (singular basis)",
                context={"pivot_tolerance": lu.pivot_tolerance},
            )
            # Create a snapshot with CRITICAL warning level for LU failure
            return NumericalSnapshot(
                iteration=iteration,
                node_id=node_id,
                warning_level=NumericalWarningLevel.CRITICAL,
                event_count=1,
            )

        # Determine if we should compute condition number based on sampling interval
        # Use the current counter value (which matches the simplex iteration number)
        compute_condition = (
            self._compute_condition_every is not None 
            and self._iteration_counter % self._compute_condition_every == 0
        )
        
        # Increment iteration counter for next iteration
        self._iteration_counter += 1

        # Compute diagnostics
        from optiforge.numerics.diagnostics import compute_diagnostics
        diagnostics = compute_diagnostics(lu.L @ lu.U, lu, estimate_condition=compute_condition)

        # Record pivot growth event if significant
        if diagnostics.pivot_growth >= self._thresholds.pivot_growth_warning:
            self.record_event(
                NumericalEventType.PIVOT_MAGNITUDE,
                measured_value=diagnostics.pivot_growth,
                threshold=self._thresholds.pivot_growth_warning,
                iteration=iteration,
                node_id=node_id,
                message=f"Pivot growth factor {diagnostics.pivot_growth:.3e}",
            )

        return self.record_snapshot(
            iteration=iteration,
            node_id=node_id,
            pivot_magnitude=diagnostics.min_abs_pivot,
            condition_estimate=diagnostics.condition_number if compute_condition else None,
            diagnostics=diagnostics,
        )

    def record_refactorization(
        self,
        iteration: int | None = None,
        node_id: int | None = None,
    ) -> None:
        """Record a basis refactorization event."""
        self._refactorization_count += 1
        self.record_event(
            NumericalEventType.REFACTORIZATION,
            measured_value=float(self._refactorization_count),
            threshold=0.0,
            iteration=iteration,
            node_id=node_id,
            message=f"Basis refactorization #{self._refactorization_count}",
        )

    # -----------------------------------------------------------------------
    # History Access
    # -----------------------------------------------------------------------

    def get_history(self) -> list[NumericalSnapshot]:
        """Get a copy of the snapshot history (oldest first)."""
        return list(self._history)

    def get_latest_snapshot(self) -> NumericalSnapshot | None:
        """Get the most recent snapshot, or None if no snapshots exist."""
        if not self._history:
            return None
        return self._history[-1]

    def get_events(self) -> list[NumericalEvent]:
        """Get a copy of all recorded events."""
        return list(self._events)

    def get_events_by_type(self, event_type: NumericalEventType) -> list[NumericalEvent]:
        """Get events filtered by type."""
        return [e for e in self._events if e.event_type == event_type]

    def get_events_by_level(self, level: NumericalWarningLevel) -> list[NumericalEvent]:
        """Get events filtered by warning level."""
        return [e for e in self._events if e.warning_level == level]

    def get_warning_summary(self) -> dict[NumericalWarningLevel, int]:
        """Get count of events by warning level."""
        counts = {level: 0 for level in NumericalWarningLevel}
        for event in self._events:
            counts[event.warning_level] += 1
        return counts

    def clear_history(self) -> None:
        """Clear snapshot history and events."""
        self._history.clear()
        self._events.clear()
        self._previous_primal_residual = None
        self._previous_dual_residual = None
        self._refactorization_count = 0

    def reset_phase_state(self) -> None:
        """Reset phase-specific state (e.g., residual growth tracking).
        
        This should be called when transitioning between Phase I and Phase II
        of the simplex algorithm, or between different solve phases, to avoid
        comparing residuals across unrelated phases.
        """
        self._previous_primal_residual = None
        self._previous_dual_residual = None
        self._iteration_counter = 0

    def get_history_size(self) -> int:
        """Get the current number of snapshots in history."""
        return len(self._history)

    def get_event_count(self) -> int:
        """Get the total number of recorded events."""
        return len(self._events)

    def get_refactorization_count(self) -> int:
        """Get the number of recorded refactorizations.
        
        Note: The current solver implementation refactorizes the basis every
        iteration, so this count will equal the iteration count. This method
        is provided for future solver implementations that may perform
        periodic refactorization.
        """
        return self._refactorization_count

    # -----------------------------------------------------------------------
    # Convenience Methods
    # -----------------------------------------------------------------------

    def has_warnings(self) -> bool:
        """Return True if any WARNING or CRITICAL events were recorded."""
        return any(
            e.warning_level in (NumericalWarningLevel.WARNING, NumericalWarningLevel.CRITICAL)
            for e in self._events
        )

    def has_critical_events(self) -> bool:
        """Return True if any CRITICAL events were recorded."""
        return any(e.warning_level == NumericalWarningLevel.CRITICAL for e in self._events)

    def get_max_warning_level(self) -> NumericalWarningLevel:
        """Get the maximum warning level across all events."""
        if not self._events:
            return NumericalWarningLevel.NONE
        return max(self._events, key=lambda e: e.warning_level).warning_level

    def __len__(self) -> int:
        """Return the number of snapshots in history."""
        return len(self._history)

    def __bool__(self) -> bool:
        """Return True if history is non-empty."""
        return bool(self._history)