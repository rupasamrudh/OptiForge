"""MILP integrality data-flow test.

Tests that integrality violation is correctly computed and passed to the
numerical monitor during MILP branch-and-bound.
"""
import numpy as np
from optiforge.milp.solver import solve_milp, MILPTolerances
from optiforge.numerics.numerical_monitor import NumericalStabilityMonitor
from optiforge.model import (
    Constraint, ConstraintSense, Model, Objective, ObjectiveSense, Variable, VariableType
)


def test_milp_integrality_monitoring():
    """Test that MILP integrality violation is recorded in numerical monitor."""
    print("=" * 60)
    print("MILP INTEGRALITY MONITORING TEST")
    print("=" * 60)
    
    # Simple MILP: max x1 + x2 s.t. x1 + x2 <= 1.5, x1, x2 integer >= 0
    # LP relaxation: x1 = 0.75, x2 = 0.75, obj = 1.5
    # Integer solution: x1 = 1, x2 = 0 or x1 = 0, x2 = 1, obj = 1
    m = Model('milp_integrality_test')
    m.add_variable(Variable('x1', lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable('x2', lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_constraint(Constraint('c1', {'x1': 1.0, 'x2': 1.0}, ConstraintSense.LESS_EQUAL, 1.5))
    m.set_objective(Objective({'x1': 1.0, 'x2': 1.0}, ObjectiveSense.MAXIMIZE))
    
    monitor = NumericalStabilityMonitor()
    tolerances = MILPTolerances(integrality_tolerance=1e-6)
    
    sol = solve_milp(m, tolerances=tolerances, numerical_monitor=monitor)
    
    print(f"Status: {sol.status}")
    print(f"Solution: {sol.variable_values}")
    print(f"Objective: {sol.objective_value}")
    
    # Check monitor for integrality violation snapshots
    snapshots = monitor.get_history()
    print(f"\nMonitor history size: {len(snapshots)}")
    
    integrality_snapshots = [s for s in snapshots if s.integrality_violation is not None]
    print(f"Snapshots with integrality_violation: {len(integrality_snapshots)}")
    
    for i, snap in enumerate(integrality_snapshots):
        print(f"  Snapshot {i}: node_id={snap.node_id}, integrality_violation={snap.integrality_violation:.6e}")
    
    # Verify that integrality violation was recorded
    assert len(integrality_snapshots) > 0, "No integrality violation snapshots recorded"
    
    # The root node LP relaxation should have integrality violation
    # x1 = 1.5, x2 = 0.0 -> deviation from nearest integer = 0.5
    root_snap = integrality_snapshots[0]
    assert root_snap.node_id == 1, f"Expected root node_id=1, got {root_snap.node_id}"
    assert abs(root_snap.integrality_violation - 0.5) < 1e-6, \
        f"Expected integrality_violation=0.5, got {root_snap.integrality_violation}"
    
    print("\nMILP integrality monitoring test: PASSED")
    print("=" * 60)
    return True


def test_milp_integrality_monitoring_binary():
    """Test MILP integrality monitoring with binary variables."""
    print("=" * 60)
    print("MILP INTEGRALITY MONITORING TEST (BINARY)")
    print("=" * 60)
    
    # Binary MILP: max x1 + x2 s.t. x1 + x2 <= 1.5, x1, x2 binary
    m = Model('milp_binary_test')
    m.add_variable(Variable('x1', lower_bound=0.0, upper_bound=1.0, variable_type=VariableType.BINARY))
    m.add_variable(Variable('x2', lower_bound=0.0, upper_bound=1.0, variable_type=VariableType.BINARY))
    m.add_constraint(Constraint('c1', {'x1': 1.0, 'x2': 1.0}, ConstraintSense.LESS_EQUAL, 1.5))
    m.set_objective(Objective({'x1': 1.0, 'x2': 1.0}, ObjectiveSense.MAXIMIZE))
    
    monitor = NumericalStabilityMonitor()
    tolerances = MILPTolerances(integrality_tolerance=1e-6)
    
    sol = solve_milp(m, tolerances=tolerances, numerical_monitor=monitor)
    
    print(f"Status: {sol.status}")
    print(f"Solution: {sol.variable_values}")
    print(f"Objective: {sol.objective_value}")
    
    # Check monitor for integrality violation snapshots
    snapshots = monitor.get_history()
    print(f"\nMonitor history size: {len(snapshots)}")
    
    integrality_snapshots = [s for s in snapshots if s.integrality_violation is not None]
    print(f"Snapshots with integrality_violation: {len(integrality_snapshots)}")
    
    for i, snap in enumerate(integrality_snapshots):
        print(f"  Snapshot {i}: node_id={snap.node_id}, integrality_violation={snap.integrality_violation:.6e}")
    
    # Verify that integrality violation was recorded
    assert len(integrality_snapshots) > 0, "No integrality violation snapshots recorded"
    
    # The root node LP relaxation should have integrality violation
    # x1 = 1.5, x2 = 0.0 -> deviation from nearest integer = 0.5
    root_snap = integrality_snapshots[0]
    assert root_snap.node_id == 1, f"Expected root node_id=1, got {root_snap.node_id}"
    assert abs(root_snap.integrality_violation - 0.5) < 1e-6, \
        f"Expected integrality_violation=0.5, got {root_snap.integrality_violation}"
    
    print("\nMILP integrality monitoring test (binary): PASSED")
    print("=" * 60)
    return True


def test_milp_integrality_monitoring_integer_feasible():
    """Test MILP integrality monitoring when LP relaxation is integer-feasible."""
    print("=" * 60)
    print("MILP INTEGRALITY MONITORING TEST (INTEGER FEASIBLE)")
    print("=" * 60)
    
    # MILP where LP relaxation is already integer-feasible
    # max x1 + x2 s.t. x1 + x2 <= 2, x1, x2 integer >= 0
    # LP relaxation: x1 = 1, x2 = 1, obj = 2 (already integer)
    m = Model('milp_integer_feasible_test')
    m.add_variable(Variable('x1', lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable('x2', lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_constraint(Constraint('c1', {'x1': 1.0, 'x2': 1.0}, ConstraintSense.LESS_EQUAL, 2.0))
    m.set_objective(Objective({'x1': 1.0, 'x2': 1.0}, ObjectiveSense.MAXIMIZE))
    
    monitor = NumericalStabilityMonitor()
    tolerances = MILPTolerances(integrality_tolerance=1e-6)
    
    sol = solve_milp(m, tolerances=tolerances, numerical_monitor=monitor)
    
    print(f"Status: {sol.status}")
    print(f"Solution: {sol.variable_values}")
    print(f"Objective: {sol.objective_value}")
    
    # Check monitor for integrality violation snapshots
    snapshots = monitor.get_history()
    print(f"\nMonitor history size: {len(snapshots)}")
    
    integrality_snapshots = [s for s in snapshots if s.integrality_violation is not None]
    print(f"Snapshots with integrality_violation: {len(integrality_snapshots)}")
    
    for i, snap in enumerate(integrality_snapshots):
        print(f"  Snapshot {i}: node_id={snap.node_id}, integrality_violation={snap.integrality_violation:.6e}")
    
    # Verify that integrality violation was recorded (should be 0.0 for integer-feasible)
    assert len(integrality_snapshots) > 0, "No integrality violation snapshots recorded"
    
    root_snap = integrality_snapshots[0]
    assert root_snap.node_id == 1, f"Expected root node_id=1, got {root_snap.node_id}"
    assert abs(root_snap.integrality_violation - 0.0) < 1e-6, \
        f"Expected integrality_violation=0.0, got {root_snap.integrality_violation}"
    
    print("\nMILP integrality monitoring test (integer feasible): PASSED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_milp_integrality_monitoring()
    test_milp_integrality_monitoring_binary()
    test_milp_integrality_monitoring_integer_feasible()
    print("\nALL MILP INTEGRALITY TESTS PASSED!")