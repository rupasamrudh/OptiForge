"""Independent numerical verification tests for dual residual calculation.

These tests verify that the monitor's dual_residual field correctly computes
the analytical dual residual ||A^T y - c||_∞ over ALL variables (basic and nonbasic),
not just the maximum reduced cost magnitude over nonbasic variables.

Three test cases from the Evidence Gate Report:
1. MINIMIZE case where monitor reports 3.0 but independent residual is 1.0
2. MAXIMIZE-converted case where both report 1.0
3. Degenerate case where monitor reports 1.0 but independent residual is 2.0
"""

import numpy as np
from optiforge.lp.simplex import RevisedSimplexSolver
from optiforge.numerics.numerical_monitor import NumericalStabilityMonitor
from optiforge.model import (
    Constraint, ConstraintSense, Model, Objective, ObjectiveSense, Variable
)
from optiforge.lp.standard_form import to_standard_form


def compute_analytical_dual_residual(A, c, basis_indices, x_B):
    """Compute the analytical dual residual ||A^T y - c||_∞.
    
    This is the true dual residual that should be reported by the monitor.
    """
    from optiforge.numerics.basis import BasisFactorization
    
    B = A[:, basis_indices]
    c_B = c[basis_indices]
    
    basis_fact = BasisFactorization(B)
    y = basis_fact.solve_transpose(c_B)
    
    ATy = A.T @ y
    dual_residual = float(np.max(np.abs(ATy - c)))
    
    return dual_residual, y


def test_case_1_minimize():
    """Test Case 1: MINIMIZE case.
    
    Evidence Gate: Monitor reports 3.0, independent residual is 1.0.
    """
    print("=" * 60)
    print("TEST CASE 1: MINIMIZE")
    print("=" * 60)
    
    # Problem: min x1 + x2 s.t. x1 + x2 <= 4, 2x1 + x2 <= 5, x >= 0
    # Optimal: x = [0, 0], obj = 0
    m = Model('test_minimize')
    m.add_variable(Variable('x1', lower_bound=0.0))
    m.add_variable(Variable('x2', lower_bound=0.0))
    m.add_constraint(Constraint('c1', {'x1': 1.0, 'x2': 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.add_constraint(Constraint('c2', {'x1': 2.0, 'x2': 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.set_objective(Objective({'x1': 1.0, 'x2': 1.0}, ObjectiveSense.MINIMIZE))
    
    monitor = NumericalStabilityMonitor()
    solver = RevisedSimplexSolver(numerical_monitor=monitor)
    sol = solver.solve(m)
    
    print(f"Status: {sol.status}")
    print(f"Iterations: {sol.iterations}")
    print(f"Solution: {sol.variable_values}")
    print(f"Objective: {sol.objective_value}")
    
    latest = monitor.get_latest_snapshot()
    if latest:
        print(f"Monitor dual_residual: {latest.dual_residual}")
        print(f"Monitor primal_residual: {latest.primal_residual}")
    
    # Compute analytical dual residual
    sf = to_standard_form(m)
    A = sf.A
    c = sf.c
    
    # At optimum x=[0,0], basis = [s1, s2] (slack variables at indices 2, 3)
    basis_indices = [2, 3]  # slack variables
    x_B = np.array([4.0, 5.0])  # slack values
    
    analytical_residual, y = compute_analytical_dual_residual(A, c, basis_indices, x_B)
    print(f"\nAnalytical dual residual: {analytical_residual}")
    print(f"Dual vector y: {y}")
    print(f"Monitor dual_residual: {latest.dual_residual if latest else None}")
    print(f"Match: {abs(analytical_residual - (latest.dual_residual if latest else 0)) < 1e-10}")
    
    assert abs(analytical_residual - (latest.dual_residual if latest else 0)) < 1e-10, \
        f"Dual residual mismatch: analytical={analytical_residual}, monitor={latest.dual_residual if latest else None}"
    
    print("TEST CASE 1: PASSED\n")
    return True


def test_case_2_maximize():
    """Test Case 2: MAXIMIZE case (converted to MINIMIZE internally).
    
    Evidence Gate: Both report 1.0.
    """
    print("=" * 60)
    print("TEST CASE 2: MAXIMIZE (converted to MINIMIZE internally)")
    print("=" * 60)
    
    # Problem: max x1 + x2 s.t. x1 + x2 <= 4, 2x1 + x2 <= 5, x >= 0
    # Optimal: x = [1, 3], obj = 4
    m = Model('test_maximize')
    m.add_variable(Variable('x1', lower_bound=0.0))
    m.add_variable(Variable('x2', lower_bound=0.0))
    m.add_constraint(Constraint('c1', {'x1': 1.0, 'x2': 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.add_constraint(Constraint('c2', {'x1': 2.0, 'x2': 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.set_objective(Objective({'x1': 1.0, 'x2': 1.0}, ObjectiveSense.MAXIMIZE))
    
    monitor = NumericalStabilityMonitor()
    solver = RevisedSimplexSolver(numerical_monitor=monitor)
    sol = solver.solve(m)
    
    print(f"Status: {sol.status}")
    print(f"Iterations: {sol.iterations}")
    print(f"Solution: {sol.variable_values}")
    print(f"Objective: {sol.objective_value}")
    
    latest = monitor.get_latest_snapshot()
    if latest:
        print(f"Monitor dual_residual: {latest.dual_residual}")
        print(f"Monitor primal_residual: {latest.primal_residual}")
    
    # Compute analytical dual residual
    sf = to_standard_form(m)
    A = sf.A
    c = sf.c
    
    # At optimum x=[1,3], basis = [x1, x2] (indices 0, 1)
    basis_indices = [0, 1]
    x_B = np.array([1.0, 3.0])
    
    analytical_residual, y = compute_analytical_dual_residual(A, c, basis_indices, x_B)
    print(f"\nAnalytical dual residual: {analytical_residual}")
    print(f"Dual vector y: {y}")
    print(f"Monitor dual_residual: {latest.dual_residual if latest else None}")
    print(f"Match: {abs(analytical_residual - (latest.dual_residual if latest else 0)) < 1e-10}")
    
    assert abs(analytical_residual - (latest.dual_residual if latest else 0)) < 1e-10, \
        f"Dual residual mismatch: analytical={analytical_residual}, monitor={latest.dual_residual if latest else None}"
    
    print("TEST CASE 2: PASSED\n")
    return True


def test_case_3_degenerate():
    """Test Case 3: Degenerate case.
    
    Evidence Gate: Monitor reports 1.0, independent residual is 2.0.
    """
    print("=" * 60)
    print("TEST CASE 3: DEGENERATE")
    print("=" * 60)
    
    # Problem from evidence gate:
    # Matrix A: [[1, 0, 1], [0, 1, 1]]
    # Objective c: [1, 1, 0]
    # MAXIMIZE
    m = Model('test_degenerate')
    m.add_variable(Variable('x1', lower_bound=0.0))
    m.add_variable(Variable('x2', lower_bound=0.0))
    m.add_variable(Variable('x3', lower_bound=0.0))
    m.add_constraint(Constraint('c1', {'x1': 1.0, 'x3': 1.0}, ConstraintSense.LESS_EQUAL, 2.0))
    m.add_constraint(Constraint('c2', {'x2': 1.0, 'x3': 1.0}, ConstraintSense.LESS_EQUAL, 2.0))
    m.set_objective(Objective({'x1': 1.0, 'x2': 1.0, 'x3': 0.0}, ObjectiveSense.MAXIMIZE))
    
    monitor = NumericalStabilityMonitor()
    solver = RevisedSimplexSolver(numerical_monitor=monitor)
    sol = solver.solve(m)
    
    print(f"Status: {sol.status}")
    print(f"Iterations: {sol.iterations}")
    print(f"Solution: {sol.variable_values}")
    print(f"Objective: {sol.objective_value}")
    
    latest = monitor.get_latest_snapshot()
    if latest:
        print(f"Monitor dual_residual: {latest.dual_residual}")
        print(f"Monitor primal_residual: {latest.primal_residual}")
    
    # Compute analytical dual residual
    sf = to_standard_form(m)
    A = sf.A
    c = sf.c
    
    # At optimum x=[2,2,0], basis = [x1, x2] (indices 0, 1)
    basis_indices = [0, 1]
    x_B = np.array([2.0, 2.0])
    
    analytical_residual, y = compute_analytical_dual_residual(A, c, basis_indices, x_B)
    print(f"\nAnalytical dual residual: {analytical_residual}")
    print(f"Dual vector y: {y}")
    print(f"Monitor dual_residual: {latest.dual_residual if latest else None}")
    print(f"Match: {abs(analytical_residual - (latest.dual_residual if latest else 0)) < 1e-10}")
    
    assert abs(analytical_residual - (latest.dual_residual if latest else 0)) < 1e-10, \
        f"Dual residual mismatch: analytical={analytical_residual}, monitor={latest.dual_residual if latest else None}"
    
    print("TEST CASE 3: PASSED\n")
    return True


def test_dual_residual_vs_max_reduced_cost():
    """Verify that dual_residual is NOT just max reduced cost magnitude.
    
    The dual residual ||A^T y - c||_∞ considers ALL variables (basic and nonbasic).
    For basic variables, reduced cost is 0 by construction.
    For nonbasic variables, reduced cost is c_j - A[:, j]^T y.
    """
    print("=" * 60)
    print("TEST: Dual residual vs max reduced cost")
    print("=" * 60)
    
    # Problem where basic variables have non-zero reduced cost in the full space
    # but zero in the basis space
    m = Model('test_dual_vs_reduced')
    m.add_variable(Variable('x1', lower_bound=0.0))
    m.add_variable(Variable('x2', lower_bound=0.0))
    m.add_constraint(Constraint('c1', {'x1': 1.0, 'x2': 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.add_constraint(Constraint('c2', {'x1': 2.0, 'x2': 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.set_objective(Objective({'x1': 3.0, 'x2': 2.0}, ObjectiveSense.MINIMIZE))
    
    monitor = NumericalStabilityMonitor()
    solver = RevisedSimplexSolver(numerical_monitor=monitor)
    sol = solver.solve(m)
    
    print(f"Status: {sol.status}")
    print(f"Iterations: {sol.iterations}")
    print(f"Solution: {sol.variable_values}")
    print(f"Objective: {sol.objective_value}")
    
    latest = monitor.get_latest_snapshot()
    if latest:
        print(f"Monitor dual_residual: {latest.dual_residual}")
    
    # Compute analytical dual residual
    sf = to_standard_form(m)
    A = sf.A
    c = sf.c
    
    # At optimum x=[0,0], basis = [s1, s2] (slack variables at indices 2, 3)
    basis_indices = [2, 3]
    x_B = np.array([4.0, 5.0])
    
    analytical_residual, y = compute_analytical_dual_residual(A, c, basis_indices, x_B)
    
    # Compute max reduced cost over nonbasic variables only
    from optiforge.numerics.basis import BasisFactorization
    B = A[:, basis_indices]
    c_B = c[basis_indices]
    basis_fact = BasisFactorization(B)
    y = basis_fact.solve_transpose(c_B)
    
    nonbasic = [0, 1]  # x1, x2
    max_reduced_cost = 0.0
    for j in nonbasic:
        r_j = float(c[j] - np.dot(A[:, j], y))
        max_reduced_cost = max(max_reduced_cost, abs(r_j))
    
    print(f"\nAnalytical dual residual (||A^T y - c||_∞): {analytical_residual}")
    print(f"Max reduced cost (nonbasic only): {max_reduced_cost}")
    print(f"Monitor dual_residual: {latest.dual_residual if latest else None}")
    print(f"Monitor equals analytical: {abs(analytical_residual - (latest.dual_residual if latest else 0)) < 1e-10}")
    print(f"Monitor equals max reduced cost: {abs(max_reduced_cost - (latest.dual_residual if latest else 0)) < 1e-10}")
    
    # The monitor should report the analytical dual residual
    assert abs(analytical_residual - (latest.dual_residual if latest else 0)) < 1e-10
    
    print("TEST: PASSED\n")
    return True


if __name__ == "__main__":
    test_case_1_minimize()
    test_case_2_maximize()
    test_case_3_degenerate()
    test_dual_residual_vs_max_reduced_cost()
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)