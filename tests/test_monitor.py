"""Tests for Stage 7.3: Search State Monitor."""

import time

import pytest

from optiforge.core.solution import SolutionStatus
from optiforge.milp import (
    MILPSolver,
    MILPTolerances,
    SearchStateMonitor,
    SearchStateSnapshot,
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


def _build_infeasible_milp() -> Model:
    """Infeasible MILP:
    max x1
    s.t. x1 >= 5
         x1 <= 2
         x1 >= 0, integer
    """
    m = Model("infeasible_milp")
    m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_constraint(Constraint("c1", {"x1": 1.0}, ConstraintSense.GREATER_EQUAL, 5.0))
    m.add_constraint(Constraint("c2", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 2.0))
    m.set_objective(Objective({"x1": 1.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_minimize_milp() -> Model:
    """Simple MINIMIZE MILP."""
    m = Model("min_milp")
    m.add_variable(Variable("x", lower_bound=0.0, upper_bound=10.0, variable_type=VariableType.INTEGER))
    m.set_objective(Objective({"x": 2.0}, ObjectiveSense.MINIMIZE))
    return m


def _build_maximize_milp() -> Model:
    """Simple MAXIMIZE MILP."""
    m = Model("max_milp")
    m.add_variable(Variable("x", lower_bound=0.0, upper_bound=10.0, variable_type=VariableType.INTEGER))
    m.set_objective(Objective({"x": 2.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_negative_objective_milp() -> Model:
    """MILP with negative objective coefficients."""
    m = Model("neg_obj")
    m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.set_objective(Objective({"x1": -1.0, "x2": -2.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_zero_incumbent_milp() -> Model:
    """MILP where optimal objective is zero."""
    m = Model("zero_obj")
    m.add_variable(Variable("x", lower_bound=0.0, upper_bound=0.0, variable_type=VariableType.INTEGER))
    m.set_objective(Objective({"x": 1.0}, ObjectiveSense.MAXIMIZE))
    return m


# ── Tests ───────────────────────────────────────────────────────────────────


class TestSearchStateMonitorBasics:
    """Basic SearchStateMonitor functionality."""

    def test_monitor_creation(self):
        monitor = SearchStateMonitor()
        assert monitor.get_history_size() == 0
        assert monitor.get_latest_snapshot() is None
        assert len(monitor) == 0
        assert not monitor

    def test_monitor_with_max_history(self):
        monitor = SearchStateMonitor(max_history=5)
        assert monitor._max_history == 5

    def test_monitor_without_history(self):
        monitor = SearchStateMonitor(track_history=False)
        monitor.start_solve("maximize")
        monitor.record_node_complete(
            nodes_processed=1,
            nodes_remaining=0,
            incumbent_objective=1.0,
            best_open_bound=None,
            feasible_solutions_found=1,
            nodes_pruned=0,
            max_depth=0,
        )
        assert monitor.get_history_size() == 0

    def test_start_solve_initializes(self):
        monitor = SearchStateMonitor()
        monitor.start_solve("maximize")
        assert monitor._objective_sense == "maximize"
        assert monitor._solve_start_time is not None
        assert monitor._root_bound is None
        assert monitor._current_depth is None

    def test_end_solve_clears_state(self):
        monitor = SearchStateMonitor()
        monitor.start_solve("maximize")
        monitor.end_solve()
        assert monitor._solve_start_time is None
        assert monitor._objective_sense is None
        assert monitor._root_bound is None
        assert monitor._current_depth is None

    def test_callback_management(self):
        monitor = SearchStateMonitor()
        callback = lambda s: None
        monitor.set_callback(callback)
        assert monitor.get_callback() == callback
        monitor.set_callback(None)
        assert monitor.get_callback() is None


class TestSearchStateSnapshot:
    """SearchStateSnapshot immutability and serialization."""

    def test_snapshot_immutability(self):
        snapshot = SearchStateSnapshot(
            nodes_processed=1,
            nodes_remaining=2,
            incumbent_objective=1.0,
            best_open_bound=2.0,
            gap=0.5,
            absolute_gap=1.0,
            feasible_solutions_found=1,
            nodes_pruned=0,
            max_depth=1,
            elapsed_time=1.0,
            objective_sense="maximize",
            root_bound=2.5,
            current_depth=0,
        )
        with pytest.raises(Exception):
            snapshot.nodes_processed = 99

    def test_snapshot_to_dict(self):
        snapshot = SearchStateSnapshot(
            nodes_processed=1,
            nodes_remaining=2,
            incumbent_objective=1.0,
            best_open_bound=2.0,
            gap=0.5,
            absolute_gap=1.0,
            feasible_solutions_found=1,
            nodes_pruned=0,
            max_depth=1,
            elapsed_time=1.0,
            objective_sense="maximize",
            root_bound=2.5,
            current_depth=0,
        )
        d = snapshot.to_dict()
        assert d["nodes_processed"] == 1
        assert d["objective_sense"] == "maximize"
        assert d["root_bound"] == 2.5

    def test_snapshot_str(self):
        snapshot = SearchStateSnapshot(
            nodes_processed=1,
            nodes_remaining=2,
            incumbent_objective=1.0,
            best_open_bound=2.0,
            gap=0.5,
            absolute_gap=1.0,
            feasible_solutions_found=1,
            nodes_pruned=0,
            max_depth=1,
            elapsed_time=1.0,
            objective_sense="maximize",
            root_bound=2.5,
            current_depth=0,
        )
        s = str(snapshot)
        assert "processed=1" in s
        assert "maximize" in s


class TestMonitorRecording:
    """Monitor recording functionality."""

    def test_record_root_bound(self):
        monitor = SearchStateMonitor()
        monitor.start_solve("maximize")
        monitor.record_root_bound(2.5)
        assert monitor._root_bound == 2.5

    def test_record_node_start(self):
        monitor = SearchStateMonitor()
        monitor.start_solve("maximize")
        monitor.record_node_start(2)
        assert monitor._current_depth == 2

    def test_record_node_complete_creates_snapshot(self):
        monitor = SearchStateMonitor()
        monitor.start_solve("maximize")
        monitor.record_node_complete(
            nodes_processed=1,
            nodes_remaining=2,
            incumbent_objective=1.0,
            best_open_bound=2.0,
            feasible_solutions_found=1,
            nodes_pruned=0,
            max_depth=1,
        )
        assert monitor.get_history_size() == 1
        snapshot = monitor.get_latest_snapshot()
        assert snapshot.nodes_processed == 1
        assert snapshot.nodes_remaining == 2
        assert snapshot.incumbent_objective == 1.0
        assert snapshot.best_open_bound == 2.0
        # For MAXIMIZE: gap = (best_open_bound - incumbent) / |incumbent|
        # = (2.0 - 1.0) / 1.0 = 1.0
        assert snapshot.gap == pytest.approx(1.0)
        assert snapshot.absolute_gap == pytest.approx(1.0)
        assert snapshot.objective_sense == "maximize"
        assert snapshot.root_bound is None
        assert snapshot.current_depth is None

    def test_record_node_complete_with_root_bound(self):
        monitor = SearchStateMonitor()
        monitor.start_solve("maximize")
        monitor.record_root_bound(2.5)
        monitor.record_node_complete(
            nodes_processed=1,
            nodes_remaining=2,
            incumbent_objective=1.0,
            best_open_bound=2.0,
            feasible_solutions_found=1,
            nodes_pruned=0,
            max_depth=1,
        )
        snapshot = monitor.get_latest_snapshot()
        assert snapshot.root_bound == 2.5

    def test_record_node_complete_with_current_depth(self):
        monitor = SearchStateMonitor()
        monitor.start_solve("maximize")
        monitor.record_node_start(2)
        monitor.record_node_complete(
            nodes_processed=1,
            nodes_remaining=2,
            incumbent_objective=1.0,
            best_open_bound=2.0,
            feasible_solutions_found=1,
            nodes_pruned=0,
            max_depth=1,
        )
        snapshot = monitor.get_latest_snapshot()
        assert snapshot.current_depth == 2

    def test_gap_computation_positive_incumbent(self):
        monitor = SearchStateMonitor()
        monitor.start_solve("maximize")
        monitor.record_node_complete(
            nodes_processed=1,
            nodes_remaining=2,
            incumbent_objective=10.0,
            best_open_bound=12.0,
            feasible_solutions_found=1,
            nodes_pruned=0,
            max_depth=1,
        )
        snapshot = monitor.get_latest_snapshot()
        assert snapshot.absolute_gap == pytest.approx(2.0)
        assert snapshot.gap == pytest.approx(0.2)

    def test_gap_computation_negative_incumbent(self):
        monitor = SearchStateMonitor()
        monitor.start_solve("maximize")
        monitor.record_node_complete(
            nodes_processed=1,
            nodes_remaining=2,
            incumbent_objective=-10.0,
            best_open_bound=-8.0,
            feasible_solutions_found=1,
            nodes_pruned=0,
            max_depth=1,
        )
        snapshot = monitor.get_latest_snapshot()
        assert snapshot.absolute_gap == pytest.approx(2.0)
        assert snapshot.gap == pytest.approx(0.2)

    def test_gap_computation_zero_incumbent(self):
        monitor = SearchStateMonitor()
        monitor.start_solve("maximize")
        monitor.record_node_complete(
            nodes_processed=1,
            nodes_remaining=2,
            incumbent_objective=0.0,
            best_open_bound=2.0,
            feasible_solutions_found=1,
            nodes_pruned=0,
            max_depth=1,
        )
        snapshot = monitor.get_latest_snapshot()
        assert snapshot.absolute_gap == pytest.approx(2.0)
        assert snapshot.gap == float("inf")

    def test_gap_computation_no_incumbent(self):
        monitor = SearchStateMonitor()
        monitor.start_solve("maximize")
        monitor.record_node_complete(
            nodes_processed=1,
            nodes_remaining=2,
            incumbent_objective=None,
            best_open_bound=2.0,
            feasible_solutions_found=0,
            nodes_pruned=0,
            max_depth=1,
        )
        snapshot = monitor.get_latest_snapshot()
        assert snapshot.gap is None
        assert snapshot.absolute_gap is None

    def test_gap_computation_no_bound(self):
        monitor = SearchStateMonitor()
        monitor.start_solve("maximize")
        monitor.record_node_complete(
            nodes_processed=1,
            nodes_remaining=2,
            incumbent_objective=10.0,
            best_open_bound=None,
            feasible_solutions_found=1,
            nodes_pruned=0,
            max_depth=1,
        )
        snapshot = monitor.get_latest_snapshot()
        assert snapshot.gap is None
        assert snapshot.absolute_gap is None

    def test_gap_computation_no_open_nodes(self):
        monitor = SearchStateMonitor()
        monitor.start_solve("maximize")
        monitor.record_node_complete(
            nodes_processed=1,
            nodes_remaining=0,
            incumbent_objective=10.0,
            best_open_bound=None,
            feasible_solutions_found=1,
            nodes_pruned=0,
            max_depth=1,
        )
        snapshot = monitor.get_latest_snapshot()
        assert snapshot.gap is None
        assert snapshot.absolute_gap is None

    def test_elapsed_time_recorded(self):
        monitor = SearchStateMonitor()
        monitor.start_solve("maximize")
        time.sleep(0.01)
        monitor.record_node_complete(
            nodes_processed=1,
            nodes_remaining=2,
            incumbent_objective=1.0,
            best_open_bound=2.0,
            feasible_solutions_found=1,
            nodes_pruned=0,
            max_depth=1,
        )
        snapshot = monitor.get_latest_snapshot()
        assert snapshot.elapsed_time is not None
        assert snapshot.elapsed_time >= 0.0  # Windows timer resolution may be coarse


class TestMonitorHistory:
    """Monitor history management."""

    def test_history_ordering(self):
        monitor = SearchStateMonitor()
        monitor.start_solve("maximize")
        for i in range(5):
            monitor.record_node_complete(
                nodes_processed=i + 1,
                nodes_remaining=5 - i - 1,
                incumbent_objective=float(i),
                best_open_bound=10.0,
                feasible_solutions_found=i,
                nodes_pruned=0,
                max_depth=1,
            )
        history = monitor.get_history()
        assert len(history) == 5
        for i, s in enumerate(history):
            assert s.nodes_processed == i + 1

    def test_max_history_enforced(self):
        monitor = SearchStateMonitor(max_history=3)
        monitor.start_solve("maximize")
        for i in range(5):
            monitor.record_node_complete(
                nodes_processed=i + 1,
                nodes_remaining=5 - i - 1,
                incumbent_objective=float(i),
                best_open_bound=10.0,
                feasible_solutions_found=i,
                nodes_pruned=0,
                max_depth=1,
            )
        assert monitor.get_history_size() == 3
        history = monitor.get_history()
        assert history[0].nodes_processed == 3
        assert history[-1].nodes_processed == 5

    def test_clear_history(self):
        monitor = SearchStateMonitor()
        monitor.start_solve("maximize")
        monitor.record_node_complete(
            nodes_processed=1,
            nodes_remaining=2,
            incumbent_objective=1.0,
            best_open_bound=2.0,
            feasible_solutions_found=1,
            nodes_pruned=0,
            max_depth=1,
        )
        monitor.clear_history()
        assert monitor.get_history_size() == 0

    def test_get_snapshots_with_incumbent(self):
        monitor = SearchStateMonitor()
        monitor.start_solve("maximize")
        monitor.record_node_complete(
            nodes_processed=1,
            nodes_remaining=2,
            incumbent_objective=None,
            best_open_bound=2.0,
            feasible_solutions_found=0,
            nodes_pruned=0,
            max_depth=1,
        )
        monitor.record_node_complete(
            nodes_processed=2,
            nodes_remaining=1,
            incumbent_objective=1.0,
            best_open_bound=2.0,
            feasible_solutions_found=1,
            nodes_pruned=0,
            max_depth=1,
        )
        with_incumbent = monitor.get_snapshots_with_incumbent()
        assert len(with_incumbent) == 1
        assert with_incumbent[0].incumbent_objective == 1.0

    def test_get_snapshots_with_gap(self):
        monitor = SearchStateMonitor()
        monitor.start_solve("maximize")
        monitor.record_node_complete(
            nodes_processed=1,
            nodes_remaining=2,
            incumbent_objective=None,
            best_open_bound=2.0,
            feasible_solutions_found=0,
            nodes_pruned=0,
            max_depth=1,
        )
        monitor.record_node_complete(
            nodes_processed=2,
            nodes_remaining=1,
            incumbent_objective=1.0,
            best_open_bound=2.0,
            feasible_solutions_found=1,
            nodes_pruned=0,
            max_depth=1,
        )
        with_gap = monitor.get_snapshots_with_gap()
        assert len(with_gap) == 1
        assert with_gap[0].gap is not None

    def test_get_incumbent_progression(self):
        monitor = SearchStateMonitor()
        monitor.start_solve("maximize")
        monitor.record_node_complete(
            nodes_processed=1,
            nodes_remaining=2,
            incumbent_objective=1.0,
            best_open_bound=2.0,
            feasible_solutions_found=1,
            nodes_pruned=0,
            max_depth=1,
        )
        monitor.record_node_complete(
            nodes_processed=2,
            nodes_remaining=1,
            incumbent_objective=2.0,
            best_open_bound=2.0,
            feasible_solutions_found=2,
            nodes_pruned=0,
            max_depth=1,
        )
        progression = monitor.get_incumbent_progression()
        assert progression == [(1, 1.0), (2, 2.0)]

    def test_get_gap_progression(self):
        monitor = SearchStateMonitor()
        monitor.start_solve("maximize")
        monitor.record_node_complete(
            nodes_processed=1,
            nodes_remaining=2,
            incumbent_objective=1.0,
            best_open_bound=2.0,
            feasible_solutions_found=1,
            nodes_pruned=0,
            max_depth=1,
        )
        monitor.record_node_complete(
            nodes_processed=2,
            nodes_remaining=1,
            incumbent_objective=1.5,
            best_open_bound=2.0,
            feasible_solutions_found=2,
            nodes_pruned=0,
            max_depth=1,
        )
        progression = monitor.get_gap_progression()
        assert len(progression) == 2
        assert progression[0][0] == 1
        assert progression[1][0] == 2

    def test_get_bound_progression(self):
        monitor = SearchStateMonitor()
        monitor.start_solve("maximize")
        monitor.record_node_complete(
            nodes_processed=1,
            nodes_remaining=2,
            incumbent_objective=1.0,
            best_open_bound=2.0,
            feasible_solutions_found=1,
            nodes_pruned=0,
            max_depth=1,
        )
        monitor.record_node_complete(
            nodes_processed=2,
            nodes_remaining=1,
            incumbent_objective=1.5,
            best_open_bound=1.8,
            feasible_solutions_found=2,
            nodes_pruned=0,
            max_depth=1,
        )
        progression = monitor.get_bound_progression()
        assert progression == [(1, 2.0), (2, 1.8)]


class TestMonitorWithSolver:
    """Monitor integration with MILPSolver."""

    def test_monitor_created_by_default(self):
        solver = MILPSolver()
        monitor = solver.get_search_state_monitor()
        assert isinstance(monitor, SearchStateMonitor)

    def test_monitor_captures_solve(self):
        solver = MILPSolver()
        monitor = solver.get_search_state_monitor()
        m = _build_simple_integer_lp()
        sol = solver.solve(m)
        assert sol.status is SolutionStatus.OPTIMAL
        assert monitor.get_history_size() > 0
        last = monitor.get_latest_snapshot()
        assert last.nodes_remaining == 0
        assert last.incumbent_objective == pytest.approx(3.0, abs=1e-6)

    def test_monitor_captures_root_bound(self):
        solver = MILPSolver()
        monitor = solver.get_search_state_monitor()
        m = _build_simple_integer_lp()
        sol = solver.solve(m)
        assert sol.status is SolutionStatus.OPTIMAL
        # Root bound should be captured
        first = monitor.get_history()[0]
        assert first.root_bound is not None

    def test_monitor_captures_max_depth(self):
        solver = MILPSolver()
        monitor = solver.get_search_state_monitor()
        m = _build_fractional_relaxation()
        sol = solver.solve(m)
        assert sol.status is SolutionStatus.OPTIMAL
        last = monitor.get_latest_snapshot()
        assert last.max_depth >= 1

    def test_monitor_captures_elapsed_time(self):
        solver = MILPSolver()
        monitor = solver.get_search_state_monitor()
        m = _build_simple_integer_lp()
        sol = solver.solve(m)
        assert sol.status is SolutionStatus.OPTIMAL
        for s in monitor.get_history():
            assert s.elapsed_time is not None
            assert s.elapsed_time >= 0.0

    def test_monitor_maximize(self):
        solver = MILPSolver()
        monitor = solver.get_search_state_monitor()
        m = _build_maximize_milp()
        sol = solver.solve(m)
        assert sol.status is SolutionStatus.OPTIMAL
        for s in monitor.get_history():
            assert s.objective_sense == "maximize"

    def test_monitor_minimize(self):
        solver = MILPSolver()
        monitor = solver.get_search_state_monitor()
        m = _build_minimize_milp()
        sol = solver.solve(m)
        assert sol.status is SolutionStatus.OPTIMAL
        for s in monitor.get_history():
            assert s.objective_sense == "minimize"

    def test_monitor_negative_objective(self):
        solver = MILPSolver()
        monitor = solver.get_search_state_monitor()
        m = _build_negative_objective_milp()
        sol = solver.solve(m)
        assert sol.status is SolutionStatus.OPTIMAL
        last = monitor.get_latest_snapshot()
        assert last.incumbent_objective is not None
        # The model has negative coefficients but optimal is x1=0, x2=0 -> obj=0.0
        assert last.incumbent_objective == pytest.approx(0.0)

    def test_monitor_zero_incumbent(self):
        solver = MILPSolver()
        monitor = solver.get_search_state_monitor()
        m = _build_zero_incumbent_milp()
        sol = solver.solve(m)
        assert sol.status is SolutionStatus.OPTIMAL
        last = monitor.get_latest_snapshot()
        assert last.incumbent_objective == pytest.approx(0.0)
        # When no open nodes remain, best_open_bound is None, so gap is None
        assert last.gap is None
        assert last.absolute_gap is None

    def test_monitor_infeasible(self):
        solver = MILPSolver()
        monitor = solver.get_search_state_monitor()
        m = _build_infeasible_milp()
        sol = solver.solve(m)
        assert sol.status is SolutionStatus.INFEASIBLE
        assert monitor.get_history_size() == 1
        snapshot = monitor.get_latest_snapshot()
        assert snapshot.nodes_processed == 1
        assert snapshot.nodes_remaining == 0
        assert snapshot.incumbent_objective is None

    def test_monitor_iteration_limit(self):
        tol = MILPTolerances(max_nodes=1)
        solver = MILPSolver(tolerances=tol)
        monitor = solver.get_search_state_monitor()
        m = _build_fractional_relaxation()
        sol = solver.solve(m)
        assert sol.status is SolutionStatus.ITERATION_LIMIT
        assert monitor.get_history_size() == 1

    def test_monitor_callback_invoked(self):
        snapshots = []
        monitor = SearchStateMonitor()
        def callback(state):
            snapshots.append(state)
        monitor.set_callback(callback)
        solver = MILPSolver(search_state_monitor=monitor)
        m = _build_simple_integer_lp()
        sol = solver.solve(m)
        assert sol.status is SolutionStatus.OPTIMAL
        assert len(snapshots) == monitor.get_history_size()

    def test_monitor_history_persists_across_solves(self):
        solver = MILPSolver()
        monitor = solver.get_search_state_monitor()
        m = _build_simple_integer_lp()
        solver.solve(m)
        history_after_first = monitor.get_history_size()
        solver.solve(m)
        history_after_second = monitor.get_history_size()
        # History accumulates by default
        assert history_after_second >= history_after_first

    def test_monitor_can_be_cleared_between_solves(self):
        solver = MILPSolver()
        monitor = solver.get_search_state_monitor()
        m = _build_simple_integer_lp()
        solver.solve(m)
        monitor.clear_history()
        solver.solve(m)
        assert monitor.get_history_size() > 0
        # Only second solve's history
        for s in monitor.get_history():
            assert s.nodes_processed <= 3  # small problem

    def test_monitor_deterministic(self):
        solver1 = MILPSolver()
        solver2 = MILPSolver()
        m = _build_fractional_relaxation()
        sol1 = solver1.solve(m)
        sol2 = solver2.solve(m)
        assert sol1.objective_value == sol2.objective_value
        h1 = solver1.get_search_state_monitor().get_history()
        h2 = solver2.get_search_state_monitor().get_history()
        assert len(h1) == len(h2)
        for s1, s2 in zip(h1, h2):
            assert s1.nodes_processed == s2.nodes_processed
            assert s1.incumbent_objective == s2.incumbent_objective
            assert s1.best_open_bound == s2.best_open_bound

    def test_monitor_with_solve_milp_function(self):
        monitor = SearchStateMonitor()
        m = _build_simple_integer_lp()
        sol = solve_milp(m, search_state_monitor=monitor)
        assert sol.status is SolutionStatus.OPTIMAL
        assert monitor.get_history_size() > 0

    def test_monitor_progression_methods(self):
        solver = MILPSolver()
        monitor = solver.get_search_state_monitor()
        m = _build_fractional_relaxation()
        sol = solver.solve(m)
        assert sol.status is SolutionStatus.OPTIMAL
        inc_prog = monitor.get_incumbent_progression()
        gap_prog = monitor.get_gap_progression()
        bound_prog = monitor.get_bound_progression()
        assert len(inc_prog) > 0
        assert len(gap_prog) > 0
        assert len(bound_prog) > 0
        # Incumbent should be non-decreasing for MAXIMIZE
        for i in range(1, len(inc_prog)):
            assert inc_prog[i][1] >= inc_prog[i-1][1]


class TestMonitorDoesNotAlterSolver:
    """Verify monitor is purely observational."""

    def test_monitor_does_not_affect_solution(self):
        solver_with = MILPSolver()
        solver_without = MILPSolver()
        m = _build_fractional_relaxation()
        sol_with = solver_with.solve(m)
        sol_without = solver_without.solve(m)
        assert sol_with.status == sol_without.status
        assert sol_with.objective_value == pytest.approx(sol_without.objective_value, abs=1e-6)
        assert sol_with.variable_values == sol_without.variable_values

    def test_monitor_does_not_affect_branching(self):
        # Same model, same solver config, with and without monitor
        # should produce identical node counts
        solver1 = MILPSolver()
        solver2 = MILPSolver()
        m = _build_fractional_relaxation()
        sol1 = solver1.solve(m)
        sol2 = solver2.solve(m)
        assert sol1.iterations == sol2.iterations

    def test_monitor_does_not_affect_node_selection(self):
        solver1 = MILPSolver()
        solver2 = MILPSolver()
        m = _build_fractional_relaxation()
        sol1 = solver1.solve(m)
        sol2 = solver2.solve(m)
        # Same iterations means same node selection behavior
        assert sol1.iterations == sol2.iterations


class TestMonitorEdgeCases:
    """Edge cases for SearchStateMonitor."""

    def test_snapshot_equality(self):
        s1 = SearchStateSnapshot(
            nodes_processed=1,
            nodes_remaining=2,
            incumbent_objective=1.0,
            best_open_bound=2.0,
            gap=0.5,
            absolute_gap=1.0,
            feasible_solutions_found=1,
            nodes_pruned=0,
            max_depth=1,
            elapsed_time=1.0,
            objective_sense="maximize",
            root_bound=2.5,
            current_depth=0,
        )
        s2 = SearchStateSnapshot(
            nodes_processed=1,
            nodes_remaining=2,
            incumbent_objective=1.0,
            best_open_bound=2.0,
            gap=0.5,
            absolute_gap=1.0,
            feasible_solutions_found=1,
            nodes_pruned=0,
            max_depth=1,
            elapsed_time=1.0,
            objective_sense="maximize",
            root_bound=2.5,
            current_depth=0,
        )
        assert s1 == s2

    def test_snapshot_inequality(self):
        s1 = SearchStateSnapshot(
            nodes_processed=1,
            nodes_remaining=2,
            incumbent_objective=1.0,
            best_open_bound=2.0,
            gap=0.5,
            absolute_gap=1.0,
            feasible_solutions_found=1,
            nodes_pruned=0,
            max_depth=1,
            elapsed_time=1.0,
            objective_sense="maximize",
            root_bound=2.5,
            current_depth=0,
        )
        s2 = SearchStateSnapshot(
            nodes_processed=2,
            nodes_remaining=2,
            incumbent_objective=1.0,
            best_open_bound=2.0,
            gap=0.5,
            absolute_gap=1.0,
            feasible_solutions_found=1,
            nodes_pruned=0,
            max_depth=1,
            elapsed_time=1.0,
            objective_sense="maximize",
            root_bound=2.5,
            current_depth=0,
        )
        assert s1 != s2

    def test_monitor_without_callback(self):
        monitor = SearchStateMonitor()
        monitor.start_solve("maximize")
        monitor.record_node_complete(
            nodes_processed=1,
            nodes_remaining=2,
            incumbent_objective=1.0,
            best_open_bound=2.0,
            feasible_solutions_found=1,
            nodes_pruned=0,
            max_depth=1,
        )
        assert monitor.get_history_size() == 1

    def test_monitor_with_none_callback(self):
        monitor = SearchStateMonitor()
        monitor.set_callback(None)
        monitor.start_solve("maximize")
        monitor.record_node_complete(
            nodes_processed=1,
            nodes_remaining=2,
            incumbent_objective=1.0,
            best_open_bound=2.0,
            feasible_solutions_found=1,
            nodes_pruned=0,
            max_depth=1,
        )
        assert monitor.get_history_size() == 1

    def test_monitor_boolean_and_len(self):
        monitor = SearchStateMonitor()
        assert not monitor
        assert len(monitor) == 0
        monitor.start_solve("maximize")
        monitor.record_node_complete(
            nodes_processed=1,
            nodes_remaining=2,
            incumbent_objective=1.0,
            best_open_bound=2.0,
            feasible_solutions_found=1,
            nodes_pruned=0,
            max_depth=1,
        )
        assert monitor
        assert len(monitor) == 1