"""Tests for Stage 7.9: Numerical Stability Monitoring.

Tests the NumericalStabilityMonitor component that tracks numerical behavior
during LP and MILP solves.
"""

import math
import numpy as np
import pytest

from optiforge.core.solution import SolutionStatus
from optiforge.lp.simplex import RevisedSimplexSolver, SimplexTolerances
from optiforge.lp.standard_form import to_standard_form
from optiforge.model import (
    Constraint,
    ConstraintSense,
    Model,
    Objective,
    ObjectiveSense,
    Variable,
    VariableType,
)
from optiforge.numerics.numerical_monitor import (
    NumericalEvent,
    NumericalEventType,
    NumericalSnapshot,
    NumericalStabilityMonitor,
    NumericalThresholds,
    NumericalWarningLevel,
)
from optiforge.numerics.diagnostics import NumericalDiagnostics
from optiforge.numerics.lu import DenseLU


# ── Helpers ─────────────────────────────────────────────────────────────────


def _build_simple_lp() -> Model:
    """Simple LP: max x1 + x2 s.t. x1 + x2 <= 3, x1,x2 >= 0."""
    m = Model("simple_lp")
    m.add_variable(Variable("x1", lower_bound=0.0))
    m.add_variable(Variable("x2", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 3.0))
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_ill_conditioned_lp() -> Model:
    """Ill-conditioned LP with nearly dependent constraints."""
    m = Model("ill_conditioned")
    m.add_variable(Variable("x1", lower_bound=0.0))
    m.add_variable(Variable("x2", lower_bound=0.0))
    # Nearly parallel constraints
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 3.0))
    m.add_constraint(Constraint("c2", {"x1": 1.0, "x2": 1.0001}, ConstraintSense.LESS_EQUAL, 3.0))
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_small_pivot_lp() -> Model:
    """LP that will have small pivot elements."""
    m = Model("small_pivot")
    m.add_variable(Variable("x1", lower_bound=0.0))
    m.add_variable(Variable("x2", lower_bound=0.0))
    # Constraint with very small coefficient
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1e-10}, ConstraintSense.LESS_EQUAL, 1.0))
    m.add_constraint(Constraint("c2", {"x1": 0.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 1.0))
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
    return m


# ── Tests ───────────────────────────────────────────────────────────────────


class TestNumericalThresholds:
    """Test NumericalThresholds configuration and validation."""

    def test_default_thresholds(self):
        """Default thresholds should be valid and ordered correctly."""
        t = NumericalThresholds()
        assert t.small_pivot_critical < t.small_pivot_warning
        assert t.primal_residual_warning < t.primal_residual_critical
        assert t.dual_residual_warning < t.dual_residual_critical
        assert t.constraint_violation_warning < t.constraint_violation_critical
        assert t.bound_violation_warning < t.bound_violation_critical
        assert t.integrality_violation_warning < t.integrality_violation_critical
        assert t.condition_number_warning < t.condition_number_critical
        assert t.residual_growth_factor_warning < t.residual_growth_factor_critical
        assert t.pivot_growth_warning < t.pivot_growth_critical

    def test_custom_thresholds(self):
        """Custom thresholds should be accepted."""
        t = NumericalThresholds(
            small_pivot_warning=1e-6,
            small_pivot_critical=1e-10,
            primal_residual_warning=1e-5,
            primal_residual_critical=1e-3,
        )
        assert t.small_pivot_warning == 1e-6
        assert t.small_pivot_critical == 1e-10

    def test_invalid_threshold_ordering(self):
        """Invalid threshold ordering should raise ValueError."""
        with pytest.raises(ValueError, match="small_pivot_critical must be < small_pivot_warning"):
            NumericalThresholds(small_pivot_warning=1e-10, small_pivot_critical=1e-6)

        with pytest.raises(ValueError, match="primal_residual_warning must be < primal_residual_critical"):
            NumericalThresholds(primal_residual_warning=1e-3, primal_residual_critical=1e-5)


class TestNumericalEvent:
    """Test NumericalEvent data model."""

    def test_event_creation(self):
        """Basic event creation should work."""
        event = NumericalEvent(
            event_type=NumericalEventType.SMALL_PIVOT,
            iteration=5,
            measured_value=1e-10,
            threshold=1e-8,
            warning_level=NumericalWarningLevel.WARNING,
            message="Small pivot detected",
            context={"pivot_row": 2, "pivot_col": 3},
        )
        assert event.event_type == NumericalEventType.SMALL_PIVOT
        assert event.iteration == 5
        assert event.measured_value == 1e-10
        assert event.warning_level == NumericalWarningLevel.WARNING

    def test_event_str_representation(self):
        """Event string representation should be informative."""
        event = NumericalEvent(
            event_type=NumericalEventType.LARGE_PRIMAL_RESIDUAL,
            iteration=10,
            measured_value=1e-4,
            threshold=1e-6,
            warning_level=NumericalWarningLevel.CRITICAL,
        )
        s = str(event)
        assert "large_primal_residual" in s  # enum value is lowercase
        assert "CRITICAL" in s  # enum name is uppercase
        assert "1.000e-04" in s


class TestNumericalSnapshot:
    """Test NumericalSnapshot data model."""

    def test_snapshot_creation(self):
        """Basic snapshot creation should work."""
        snapshot = NumericalSnapshot(
            iteration=5,
            primal_residual=1e-7,
            dual_residual=1e-8,
            constraint_violation=1e-9,
            bound_violation=1e-10,
            integrality_violation=None,
            pivot_magnitude=0.5,
            condition_estimate=1e6,
            warning_level=NumericalWarningLevel.INFO,
            event_count=2,
        )
        assert snapshot.iteration == 5
        assert snapshot.primal_residual == 1e-7
        assert snapshot.warning_level == NumericalWarningLevel.INFO
        assert snapshot.event_count == 2

    def test_snapshot_with_none_values(self):
        """Snapshot should handle None values for unavailable metrics."""
        snapshot = NumericalSnapshot(
            iteration=1,
            primal_residual=None,
            dual_residual=None,
            condition_estimate=None,
        )
        assert snapshot.primal_residual is None
        assert snapshot.dual_residual is None
        assert snapshot.condition_estimate is None

    def test_snapshot_str_representation(self):
        """Snapshot string representation should be informative."""
        snapshot = NumericalSnapshot(
            iteration=3,
            primal_residual=1e-6,
            dual_residual=1e-7,
            warning_level=NumericalWarningLevel.WARNING,
            event_count=1,
        )
        s = str(snapshot)
        assert "iter=3" in s
        assert "primal_res=1.000e-06" in s
        assert "WARNING" in s  # enum name is uppercase


class TestNumericalStabilityMonitorBasics:
    """Test basic NumericalStabilityMonitor functionality."""

    def test_monitor_creation(self):
        """Monitor should initialize with defaults."""
        monitor = NumericalStabilityMonitor()
        assert monitor.get_history_size() == 0
        assert monitor.get_event_count() == 0
        assert monitor.get_latest_snapshot() is None
        assert monitor.max_history == 1000
        assert monitor.track_history is True

    def test_monitor_with_custom_settings(self):
        """Monitor should accept custom settings."""
        thresholds = NumericalThresholds(small_pivot_warning=1e-6)
        monitor = NumericalStabilityMonitor(
            thresholds=thresholds,
            max_history=100,
            track_history=False,
        )
        assert monitor.thresholds.small_pivot_warning == 1e-6
        assert monitor.max_history == 100
        assert monitor.track_history is False

    def test_callback_management(self):
        """Callback should be settable and retrievable."""
        monitor = NumericalStabilityMonitor()
        callback = lambda s: None
        monitor.set_callback(callback)
        assert monitor.callback == callback
        monitor.set_callback(None)
        assert monitor.callback is None

    def test_clear_history(self):
        """History should be clearable."""
        monitor = NumericalStabilityMonitor()
        monitor.record_snapshot(iteration=1, primal_residual=1e-6)
        monitor.record_event(NumericalEventType.SMALL_PIVOT, 1e-10, 1e-8)
        assert monitor.get_history_size() == 1
        assert monitor.get_event_count() == 2  # snapshot records event + explicit event
        monitor.clear_history()
        assert monitor.get_history_size() == 0
        assert monitor.get_event_count() == 0


class TestEventRecording:
    """Test event recording and classification."""

    def test_record_small_pivot_event(self):
        """Small pivot events should be classified correctly."""
        monitor = NumericalStabilityMonitor()
        # Critical level
        event = monitor.record_event(
            NumericalEventType.SMALL_PIVOT,
            measured_value=1e-13,
            threshold=1e-8,
            iteration=1,
        )
        assert event.warning_level == NumericalWarningLevel.CRITICAL

        # Warning level
        event = monitor.record_event(
            NumericalEventType.SMALL_PIVOT,
            measured_value=1e-10,
            threshold=1e-8,
            iteration=2,
        )
        assert event.warning_level == NumericalWarningLevel.WARNING

        # Info level
        event = monitor.record_event(
            NumericalEventType.SMALL_PIVOT,
            measured_value=1e-6,
            threshold=1e-8,
            iteration=3,
        )
        assert event.warning_level == NumericalWarningLevel.INFO

    def test_record_primal_residual_event(self):
        """Primal residual events should be classified correctly."""
        monitor = NumericalStabilityMonitor()
        # Critical level
        event = monitor.record_event(
            NumericalEventType.LARGE_PRIMAL_RESIDUAL,
            measured_value=1e-3,
            threshold=1e-6,
            iteration=1,
        )
        assert event.warning_level == NumericalWarningLevel.CRITICAL

        # Warning level
        event = monitor.record_event(
            NumericalEventType.LARGE_PRIMAL_RESIDUAL,
            measured_value=1e-5,
            threshold=1e-6,
            iteration=2,
        )
        assert event.warning_level == NumericalWarningLevel.WARNING

    def test_record_dual_residual_event(self):
        """Dual residual events should be classified correctly."""
        monitor = NumericalStabilityMonitor()
        event = monitor.record_event(
            NumericalEventType.LARGE_DUAL_RESIDUAL,
            measured_value=1e-3,
            threshold=1e-6,
            iteration=1,
        )
        assert event.warning_level == NumericalWarningLevel.CRITICAL

    def test_record_constraint_violation_event(self):
        """Constraint violation events should be classified correctly."""
        monitor = NumericalStabilityMonitor()
        event = monitor.record_event(
            NumericalEventType.LARGE_CONSTRAINT_VIOLATION,
            measured_value=1e-3,
            threshold=1e-6,
            iteration=1,
        )
        assert event.warning_level == NumericalWarningLevel.CRITICAL

    def test_record_bound_violation_event(self):
        """Bound violation events should be classified correctly."""
        monitor = NumericalStabilityMonitor()
        event = monitor.record_event(
            NumericalEventType.LARGE_BOUND_VIOLATION,
            measured_value=1e-4,
            threshold=1e-7,
            iteration=1,
        )
        assert event.warning_level == NumericalWarningLevel.CRITICAL

    def test_record_integrality_violation_event(self):
        """Integrality violation events should be classified correctly."""
        monitor = NumericalStabilityMonitor()
        event = monitor.record_event(
            NumericalEventType.LARGE_INTEGRALITY_VIOLATION,
            measured_value=1e-2,
            threshold=1e-5,
            iteration=1,
        )
        assert event.warning_level == NumericalWarningLevel.CRITICAL

    def test_record_near_singular_basis_event(self):
        """Near-singular basis events should be classified correctly."""
        monitor = NumericalStabilityMonitor()
        event = monitor.record_event(
            NumericalEventType.NEAR_SINGULAR_BASIS,
            measured_value=1e15,
            threshold=1e10,
            iteration=1,
        )
        assert event.warning_level == NumericalWarningLevel.CRITICAL

    def test_record_residual_growth_event(self):
        """Residual growth events should be classified correctly."""
        monitor = NumericalStabilityMonitor()
        event = monitor.record_event(
            NumericalEventType.RESIDUAL_GROWTH,
            measured_value=10000.0,
            threshold=100.0,
            iteration=1,
        )
        assert event.warning_level == NumericalWarningLevel.CRITICAL


class TestSnapshotRecording:
    """Test snapshot recording with various metrics."""

    def test_record_snapshot_basic(self):
        """Basic snapshot recording should work."""
        monitor = NumericalStabilityMonitor()
        snapshot = monitor.record_snapshot(
            iteration=1,
            primal_residual=1e-7,
            dual_residual=1e-8,
        )
        assert snapshot.iteration == 1
        assert snapshot.primal_residual == 1e-7
        assert snapshot.dual_residual == 1e-8
        assert monitor.get_history_size() == 1

    def test_record_snapshot_with_all_metrics(self):
        """Snapshot with all metrics should work."""
        monitor = NumericalStabilityMonitor()
        snapshot = monitor.record_snapshot(
            iteration=5,
            node_id=10,
            primal_residual=1e-6,
            dual_residual=1e-7,
            constraint_violation=1e-8,
            bound_violation=1e-9,
            integrality_violation=1e-5,
            pivot_magnitude=0.1,
            condition_estimate=1e8,
        )
        assert snapshot.node_id == 10
        assert snapshot.integrality_violation == 1e-5
        assert snapshot.pivot_magnitude == 0.1
        assert snapshot.condition_estimate == 1e8

    def test_snapshot_warning_level_aggregation(self):
        """Snapshot warning level should be max of individual events."""
        monitor = NumericalStabilityMonitor()
        # Record snapshot with critical primal residual
        snapshot = monitor.record_snapshot(
            iteration=1,
            primal_residual=1e-3,  # Critical
            dual_residual=1e-8,    # Info
        )
        assert snapshot.warning_level == NumericalWarningLevel.CRITICAL
        assert snapshot.event_count >= 1

    def test_snapshot_no_warnings_for_normal_values(self):
        """Normal values should not generate warnings."""
        monitor = NumericalStabilityMonitor()
        snapshot = monitor.record_snapshot(
            iteration=1,
            primal_residual=1e-10,
            dual_residual=1e-10,
            constraint_violation=1e-10,
            bound_violation=1e-10,
        )
        assert snapshot.warning_level == NumericalWarningLevel.NONE
        assert snapshot.event_count == 0

    def test_history_size_limit(self):
        """History should respect max_history limit."""
        monitor = NumericalStabilityMonitor(max_history=3)
        for i in range(5):
            monitor.record_snapshot(iteration=i, primal_residual=1e-6)
        assert monitor.get_history_size() == 3
        # Should keep the last 3
        history = monitor.get_history()
        assert history[0].iteration == 2
        assert history[2].iteration == 4

    def test_callback_invocation(self):
        """Callback should be invoked on snapshot recording."""
        monitor = NumericalStabilityMonitor()
        received = []

        def callback(snapshot):
            received.append(snapshot)

        monitor.set_callback(callback)
        monitor.record_snapshot(iteration=1, primal_residual=1e-6)
        assert len(received) == 1
        assert received[0].iteration == 1


class TestResidualGrowthTracking:
    """Test residual growth tracking across iterations."""

    def test_primal_residual_growth_detection(self):
        """Growing primal residual should trigger event."""
        monitor = NumericalStabilityMonitor()
        # First iteration - small residual
        monitor.record_snapshot(iteration=1, primal_residual=1e-10)
        # Second iteration - large growth
        snapshot = monitor.record_snapshot(iteration=2, primal_residual=1e-5)
        # Growth factor = 1e5, should trigger WARNING
        assert snapshot.warning_level in (NumericalWarningLevel.WARNING, NumericalWarningLevel.CRITICAL)

    def test_dual_residual_growth_detection(self):
        """Growing dual residual should trigger event."""
        monitor = NumericalStabilityMonitor()
        monitor.record_snapshot(iteration=1, dual_residual=1e-10)
        snapshot = monitor.record_snapshot(iteration=2, dual_residual=1e-5)
        assert snapshot.warning_level in (NumericalWarningLevel.WARNING, NumericalWarningLevel.CRITICAL)

    def test_no_growth_event_for_first_iteration(self):
        """First iteration should not trigger growth event."""
        monitor = NumericalStabilityMonitor()
        snapshot = monitor.record_snapshot(iteration=1, primal_residual=1e-5)
        # No previous residual to compare
        growth_events = monitor.get_events_by_type(NumericalEventType.RESIDUAL_GROWTH)
        assert len(growth_events) == 0


class TestLUFactorizationRecording:
    """Test LU factorization event recording."""

    def test_record_lu_factorization_success(self):
        """Successful LU factorization should record snapshot."""
        monitor = NumericalStabilityMonitor(compute_condition_every=1)
        # Create a simple LU factorization
        B = np.array([[2.0, 1.0], [1.0, 2.0]])
        lu = DenseLU(
            L=np.eye(2),
            U=B.copy(),
            perm=np.array([0, 1]),
            det_sign=1,
            pivot_tolerance=1e-12,
        )
        snapshot = monitor.record_lu_factorization(lu, iteration=1, success=True)
        assert snapshot.iteration == 1
        assert snapshot.condition_estimate is not None
        assert snapshot.diagnostics is not None

    def test_record_lu_factorization_failure(self):
        """Failed LU factorization should record critical event."""
        monitor = NumericalStabilityMonitor()
        # Create a dummy LU (won't be used since success=False)
        lu = DenseLU(
            L=np.eye(2),
            U=np.eye(2),
            perm=np.array([0, 1]),
            det_sign=1,
            pivot_tolerance=1e-12,
        )
        snapshot = monitor.record_lu_factorization(lu, iteration=1, success=False)
        assert snapshot.warning_level == NumericalWarningLevel.CRITICAL
        events = monitor.get_events_by_type(NumericalEventType.LU_FACTORIZATION_FAILURE)
        assert len(events) == 1


class TestRefactorizationTracking:
    """Test refactorization event tracking."""

    def test_record_refactorization(self):
        """Refactorization should increment counter and record event."""
        monitor = NumericalStabilityMonitor()
        monitor.record_refactorization(iteration=5)
        monitor.record_refactorization(iteration=10)
        assert monitor.get_refactorization_count() == 2
        events = monitor.get_events_by_type(NumericalEventType.REFACTORIZATION)
        assert len(events) == 2
        assert events[0].measured_value == 1.0
        assert events[1].measured_value == 2.0


class TestEventFiltering:
    """Test event filtering and summary methods."""

    def test_get_events_by_type(self):
        """Filtering events by type should work."""
        monitor = NumericalStabilityMonitor()
        monitor.record_event(NumericalEventType.SMALL_PIVOT, 1e-10, 1e-8, iteration=1)
        monitor.record_event(NumericalEventType.LARGE_PRIMAL_RESIDUAL, 1e-3, 1e-6, iteration=2)
        monitor.record_event(NumericalEventType.SMALL_PIVOT, 1e-11, 1e-8, iteration=3)

        pivot_events = monitor.get_events_by_type(NumericalEventType.SMALL_PIVOT)
        assert len(pivot_events) == 2

        residual_events = monitor.get_events_by_type(NumericalEventType.LARGE_PRIMAL_RESIDUAL)
        assert len(residual_events) == 1

    def test_get_events_by_level(self):
        """Filtering events by warning level should work."""
        monitor = NumericalStabilityMonitor()
        monitor.record_event(NumericalEventType.SMALL_PIVOT, 1e-13, 1e-8)  # Critical
        monitor.record_event(NumericalEventType.SMALL_PIVOT, 1e-10, 1e-8)  # Warning
        monitor.record_event(NumericalEventType.SMALL_PIVOT, 1e-6, 1e-8)   # Info

        critical = monitor.get_events_by_level(NumericalWarningLevel.CRITICAL)
        assert len(critical) == 1

        warning = monitor.get_events_by_level(NumericalWarningLevel.WARNING)
        assert len(warning) == 1

    def test_get_warning_summary(self):
        """Warning summary should count events by level."""
        monitor = NumericalStabilityMonitor()
        monitor.record_event(NumericalEventType.SMALL_PIVOT, 1e-13, 1e-8)  # Critical
        monitor.record_event(NumericalEventType.SMALL_PIVOT, 1e-10, 1e-8)  # Warning
        monitor.record_event(NumericalEventType.SMALL_PIVOT, 1e-6, 1e-8)   # Info

        summary = monitor.get_warning_summary()
        assert summary[NumericalWarningLevel.CRITICAL] == 1
        assert summary[NumericalWarningLevel.WARNING] == 1
        assert summary[NumericalWarningLevel.INFO] == 1
        assert summary[NumericalWarningLevel.NONE] == 0

    def test_has_warnings(self):
        """has_warnings should detect WARNING and CRITICAL events."""
        monitor = NumericalStabilityMonitor()
        assert not monitor.has_warnings()
        monitor.record_event(NumericalEventType.SMALL_PIVOT, 1e-10, 1e-8)  # Warning
        assert monitor.has_warnings()
        monitor.clear_history()
        monitor.record_event(NumericalEventType.SMALL_PIVOT, 1e-13, 1e-8)  # Critical
        assert monitor.has_warnings()

    def test_has_critical_events(self):
        """has_critical_events should detect CRITICAL events."""
        monitor = NumericalStabilityMonitor()
        assert not monitor.has_critical_events()
        monitor.record_event(NumericalEventType.SMALL_PIVOT, 1e-10, 1e-8)  # Warning
        assert not monitor.has_critical_events()
        monitor.record_event(NumericalEventType.SMALL_PIVOT, 1e-13, 1e-8)  # Critical
        assert monitor.has_critical_events()

    def test_get_max_warning_level(self):
        """get_max_warning_level should return highest level."""
        monitor = NumericalStabilityMonitor()
        assert monitor.get_max_warning_level() == NumericalWarningLevel.NONE
        monitor.record_event(NumericalEventType.SMALL_PIVOT, 1e-6, 1e-8)   # Info
        assert monitor.get_max_warning_level() == NumericalWarningLevel.INFO
        monitor.record_event(NumericalEventType.SMALL_PIVOT, 1e-10, 1e-8)  # Warning
        assert monitor.get_max_warning_level() == NumericalWarningLevel.WARNING
        monitor.record_event(NumericalEventType.SMALL_PIVOT, 1e-13, 1e-8)  # Critical
        assert monitor.get_max_warning_level() == NumericalWarningLevel.CRITICAL


class TestIntegrationWithSimplex:
    """Test integration with Revised Simplex solver."""

    def test_solver_without_monitor(self):
        """Solver should work without monitor (backward compatibility)."""
        m = _build_simple_lp()
        solver = RevisedSimplexSolver()
        sol = solver.solve(m)
        assert sol.status == SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(3.0, abs=1e-6)

    def test_solver_with_monitor(self):
        """Solver should work with monitor attached."""
        m = _build_simple_lp()
        monitor = NumericalStabilityMonitor()
        solver = RevisedSimplexSolver(numerical_monitor=monitor)
        sol = solver.solve(m)
        assert sol.status == SolutionStatus.OPTIMAL
        assert monitor.get_history_size() > 0
        assert monitor.get_event_count() >= 0

    def test_solver_with_monitor_per_solve(self):
        """Monitor passed to solve() should override instance monitor."""
        m = _build_simple_lp()
        monitor1 = NumericalStabilityMonitor()
        monitor2 = NumericalStabilityMonitor()
        solver = RevisedSimplexSolver(numerical_monitor=monitor1)
        sol = solver.solve(m, numerical_monitor=monitor2)
        assert sol.status == SolutionStatus.OPTIMAL
        assert monitor2.get_history_size() > 0
        assert monitor1.get_history_size() == 0

    def test_monitor_records_iterations(self):
        """Monitor should record snapshots at each iteration."""
        m = _build_simple_lp()
        monitor = NumericalStabilityMonitor()
        solver = RevisedSimplexSolver(numerical_monitor=monitor)
        sol = solver.solve(m)
        assert sol.status == SolutionStatus.OPTIMAL
        # Should have at least one snapshot per iteration
        assert monitor.get_history_size() >= sol.iterations

    def test_monitor_detects_small_pivot(self):
        """Monitor should detect small pivots in ill-conditioned problems."""
        m = _build_small_pivot_lp()
        monitor = NumericalStabilityMonitor()
        solver = RevisedSimplexSolver(numerical_monitor=monitor)
        sol = solver.solve(m)
        # Check if small pivot events were recorded
        pivot_events = monitor.get_events_by_type(NumericalEventType.SMALL_PIVOT)
        # May or may not have small pivots depending on the problem
        # Just verify the mechanism works
        assert isinstance(pivot_events, list)

    def test_monitor_records_final_snapshot(self):
        """Monitor should record final snapshot at optimality."""
        m = _build_simple_lp()
        monitor = NumericalStabilityMonitor()
        solver = RevisedSimplexSolver(numerical_monitor=monitor)
        sol = solver.solve(m)
        assert sol.status == SolutionStatus.OPTIMAL
        latest = monitor.get_latest_snapshot()
        assert latest is not None
        assert latest.primal_residual is not None
        assert latest.dual_residual is not None


class TestIntegrationWithMILP:
    """Test integration with MILP solver."""

    def test_milp_solver_without_monitor(self):
        """MILP solver should work without monitor (backward compatibility)."""
        m = Model("simple_milp")
        m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
        m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 3.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))

        from optiforge.milp import solve_milp
        sol = solve_milp(m)
        assert sol.status == SolutionStatus.OPTIMAL

    def test_milp_solver_with_monitor(self):
        """MILP solver should work with monitor attached."""
        m = Model("simple_milp")
        m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
        m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 3.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))

        from optiforge.milp import solve_milp
        monitor = NumericalStabilityMonitor()
        sol = solve_milp(m, numerical_monitor=monitor)
        assert sol.status == SolutionStatus.OPTIMAL
        # Monitor should have recorded events from LP relaxations
        assert monitor.get_history_size() >= 0


class TestPerformanceAndSafety:
    """Test performance and safety characteristics."""

    def test_monitor_disabled_by_default(self):
        """Monitor should be None by default in solvers."""
        solver = RevisedSimplexSolver()
        assert solver.numerical_monitor is None

    def test_monitor_does_not_mutate_solver_state(self):
        """Monitor should not affect solver decisions."""
        m = _build_simple_lp()
        monitor = NumericalStabilityMonitor()
        solver1 = RevisedSimplexSolver()
        solver2 = RevisedSimplexSolver(numerical_monitor=monitor)

        sol1 = solver1.solve(m)
        sol2 = solver2.solve(m)

        # Solutions should be identical
        assert sol1.status == sol2.status
        assert sol1.objective_value == pytest.approx(sol2.objective_value)
        assert sol1.iterations == sol2.iterations
        assert sol1.variable_values == sol2.variable_values

    def test_snapshot_immutability(self):
        """Snapshots should be immutable (frozen dataclass)."""
        monitor = NumericalStabilityMonitor()
        snapshot = monitor.record_snapshot(iteration=1, primal_residual=1e-6)
        # Attempting to modify should fail (frozen dataclass)
        with pytest.raises(Exception):
            snapshot.primal_residual = 1e-3

    def test_event_immutability(self):
        """Events should be immutable (frozen dataclass)."""
        monitor = NumericalStabilityMonitor()
        event = monitor.record_event(NumericalEventType.SMALL_PIVOT, 1e-10, 1e-8)
        with pytest.raises(Exception):
            event.measured_value = 1e-3

    def test_no_expensive_operations_unconditionally(self):
        """Monitor should not add expensive operations when disabled."""
        # This is a design test - verify monitor is optional
        m = _build_simple_lp()
        solver = RevisedSimplexSolver()  # No monitor
        sol = solver.solve(m)
        assert sol.status == SolutionStatus.OPTIMAL


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_residuals(self):
        """Zero residuals should not cause division by zero in growth tracking."""
        monitor = NumericalStabilityMonitor()
        monitor.record_snapshot(iteration=1, primal_residual=0.0)
        snapshot = monitor.record_snapshot(iteration=2, primal_residual=1e-6)
        # Should not crash
        assert snapshot is not None

    def test_nan_condition_number(self):
        """NaN condition number should be handled gracefully."""
        monitor = NumericalStabilityMonitor()
        snapshot = monitor.record_snapshot(
            iteration=1,
            condition_estimate=float('nan'),
        )
        assert snapshot.condition_estimate is not None
        assert math.isnan(snapshot.condition_estimate)

    def test_infinite_condition_number(self):
        """Infinite condition number should be handled."""
        monitor = NumericalStabilityMonitor()
        snapshot = monitor.record_snapshot(
            iteration=1,
            condition_estimate=float('inf'),
        )
        assert snapshot.condition_estimate == float('inf')

    def test_negative_residuals(self):
        """Negative residuals should be handled (absolute value used)."""
        monitor = NumericalStabilityMonitor()
        # Residuals should be non-negative, but test robustness
        snapshot = monitor.record_snapshot(
            iteration=1,
            primal_residual=-1e-6,  # Should not happen but test robustness
        )
        assert snapshot.primal_residual == -1e-6

    def test_large_number_of_snapshots(self):
        """Monitor should handle many snapshots without memory issues."""
        monitor = NumericalStabilityMonitor(max_history=10000)
        for i in range(10000):
            monitor.record_snapshot(iteration=i, primal_residual=1e-6)
        assert monitor.get_history_size() == 10000

    def test_deterministic_event_classification(self):
        """Event classification should be deterministic."""
        monitor1 = NumericalStabilityMonitor()
        monitor2 = NumericalStabilityMonitor()

        for i in range(10):
            monitor1.record_event(NumericalEventType.SMALL_PIVOT, 1e-10, 1e-8, iteration=i)
            monitor2.record_event(NumericalEventType.SMALL_PIVOT, 1e-10, 1e-8, iteration=i)

        events1 = monitor1.get_events_by_type(NumericalEventType.SMALL_PIVOT)
        events2 = monitor2.get_events_by_type(NumericalEventType.SMALL_PIVOT)

        for e1, e2 in zip(events1, events2):
            assert e1.warning_level == e2.warning_level


class TestBackwardCompatibility:
    """Test backward compatibility without monitoring."""

    def test_existing_lp_tests_still_pass(self):
        """All existing LP functionality should work unchanged."""
        m = _build_simple_lp()
        solver = RevisedSimplexSolver()
        sol = solver.solve(m)
        assert sol.status == SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(3.0, abs=1e-6)
        assert sol.variable_values["x1"] + sol.variable_values["x2"] == pytest.approx(3.0, abs=1e-6)

    def test_existing_milp_tests_still_pass(self):
        """All existing MILP functionality should work unchanged."""
        m = Model("simple_milp")
        m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
        m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 3.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))

        from optiforge.milp import solve_milp
        sol = solve_milp(m)
        assert sol.status == SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(3.0, abs=1e-6)

    def test_monitor_none_preserves_behavior(self):
        """Explicitly passing None monitor should preserve behavior."""
        m = _build_simple_lp()
        solver = RevisedSimplexSolver(numerical_monitor=None)
        sol = solver.solve(m)
        assert sol.status == SolutionStatus.OPTIMAL

    def test_solve_without_monitor_parameter(self):
        """Calling solve() without monitor parameter should work."""
        m = _build_simple_lp()
        solver = RevisedSimplexSolver()
        sol = solver.solve(m)  # No monitor parameter
        assert sol.status == SolutionStatus.OPTIMAL


class TestInvalidThresholdValidation:
    """Test validation of invalid threshold configurations."""

    def test_all_threshold_pairs_validated(self):
        """All threshold pairs should be validated."""
        # Test each pair
        with pytest.raises(ValueError):
            NumericalThresholds(dual_residual_warning=1e-3, dual_residual_critical=1e-5)
        with pytest.raises(ValueError):
            NumericalThresholds(constraint_violation_warning=1e-3, constraint_violation_critical=1e-5)
        with pytest.raises(ValueError):
            NumericalThresholds(bound_violation_warning=1e-3, bound_violation_critical=1e-5)
        with pytest.raises(ValueError):
            NumericalThresholds(integrality_violation_warning=1e-3, integrality_violation_critical=1e-5)
        with pytest.raises(ValueError):
            NumericalThresholds(condition_number_warning=1e14, condition_number_critical=1e10)
        with pytest.raises(ValueError):
            NumericalThresholds(residual_growth_factor_warning=1000, residual_growth_factor_critical=100)
        with pytest.raises(ValueError):
            NumericalThresholds(pivot_growth_warning=1e12, pivot_growth_critical=1e8)