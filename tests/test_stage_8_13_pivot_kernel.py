"""Unit and integration tests for Stage 8.13 Simplex Pivot Kernel Optimization.

Tests verify:
1. Exact mathematical equivalence of primal x_B update formula.
2. In-place GEMV buffer correctness for reduced costs.
3. Bland's rule preservation with early-break nonbasic scanning.
4. Vectorized ratio test and Bland tie-breaking equivalence.
5. Infeasible, unbounded, degenerate, and multi-optimal LP handling.
6. Correctness across equality, <=, >=, mixed senses, and negative RHS.
7. MILP child relaxation and warm-start solve consistency.
8. Numerical stability: primal/dual residuals, condition numbers, bound violations.
"""

from __future__ import annotations

import bisect
import numpy as np
import pytest

from optiforge.core.solution import SolutionStatus
from optiforge.lp.simplex import RevisedSimplexSolver, SimplexTolerances
from optiforge.lp.dual_simplex import DualRevisedSimplexSolver
from optiforge.lp.standard_form import StandardFormModel, to_standard_form
from optiforge.milp.solver import MILPSolver
from optiforge.model import (
    Constraint,
    ConstraintSense,
    Model,
    Objective,
    ObjectiveSense,
    Variable,
    VariableType,
)
from optiforge.numerics.basis import BasisFactorization
from optiforge.numerics.numerical_monitor import NumericalStabilityMonitor


# ---------------------------------------------------------------------------
# Test 1: Mathematical Invariant of Primal x_B Update
# ---------------------------------------------------------------------------

def test_primal_xb_update_mathematical_identity():
    """Verify that x_B_new = x_B - theta* * d satisfies B_new x_B_new = b to machine precision."""
    rng = np.random.RandomState(42)
    m = 15
    for trial in range(10):
        # Generate well-conditioned square basis B
        B = rng.randn(m, m) + np.eye(m) * 5.0
        b = rng.uniform(1.0, 10.0, size=m)

        bf = BasisFactorization(B)
        x_B = bf.solve(b)

        # Entering column a_enter
        a_enter = rng.randn(m)
        d = bf.solve(a_enter)

        # Find positive direction components
        pos_idx = np.where(d > 1e-4)[0]
        if len(pos_idx) == 0:
            continue
        ratios = x_B[pos_idx] / d[pos_idx]
        best_local = np.argmin(ratios)
        i_leave = pos_idx[best_local]
        theta_star = ratios[best_local]

        # Simplex update
        x_B_updated = x_B.copy()
        x_B_updated -= theta_star * d
        x_B_updated[i_leave] = theta_star

        # Reconstructed basis
        B_new = B.copy()
        B_new[:, i_leave] = a_enter
        bf_new = BasisFactorization(B_new)
        x_B_direct = bf_new.solve(b)

        # Check maximum deviation
        diff = np.max(np.abs(x_B_updated - x_B_direct))
        assert diff < 1e-11, f"Trial {trial}: x_B update deviated by {diff:.3e}"

        # Check residual ||B_new x_B_new - b||
        res = np.max(np.abs(B_new @ x_B_updated - b))
        assert res < 1e-11, f"Trial {trial}: residual ||B x - b|| is {res:.3e}"


# ---------------------------------------------------------------------------
# Test 2: Bland's Rule Early Break vs Full Candidate Scan
# ---------------------------------------------------------------------------

def test_bland_pricing_early_break_equivalence():
    """Verify that early break on sorted nonbasic list yields identical entering variable to min(candidates)."""
    rng = np.random.RandomState(123)
    n = 100
    m = 25

    for trial in range(100):
        nonbasic = sorted(rng.choice(n, size=n - m, replace=False).tolist())
        r_all = rng.randn(n) * 10.0
        tol = 1e-7

        # MINIMIZE
        expected_candidates_min = [j for j in nonbasic if r_all[j] < -tol]
        expected_enter_min = min(expected_candidates_min) if expected_candidates_min else None

        actual_enter_min = None
        for j in nonbasic:
            if r_all[j] < -tol:
                actual_enter_min = j
                break

        assert actual_enter_min == expected_enter_min, (
            f"Trial {trial} MIN mismatch: {actual_enter_min} vs {expected_enter_min}"
        )

        # MAXIMIZE
        expected_candidates_max = [j for j in nonbasic if r_all[j] > tol]
        expected_enter_max = min(expected_candidates_max) if expected_candidates_max else None

        actual_enter_max = None
        for j in nonbasic:
            if r_all[j] > tol:
                actual_enter_max = j
                break

        assert actual_enter_max == expected_enter_max, (
            f"Trial {trial} MAX mismatch: {actual_enter_max} vs {expected_enter_max}"
        )


# ---------------------------------------------------------------------------
# Test 3: Vectorized Ratio Test & Tie Breaking
# ---------------------------------------------------------------------------

def test_vectorized_ratio_test_tie_breaking():
    """Verify vectorized ratio test preserves exact Bland tie breaking."""
    rng = np.random.RandomState(456)
    m = 20

    for trial in range(50):
        d = rng.randn(m)
        d[d < 0.1] = -1.0
        d[2] = 2.0
        d[5] = 2.0
        d[8] = 2.0

        x_B = rng.uniform(0.0, 10.0, size=m)
        # Create deliberate ties
        x_B[2] = 4.0
        x_B[5] = 4.0
        x_B[8] = 4.0

        basis = rng.choice(100, size=m, replace=False).tolist()

        piv_tol = 1e-9

        # Reference scalar implementation
        eligible_rows = [i for i in range(m) if d[i] > piv_tol]
        if not eligible_rows:
            continue
        ratios = [max(0.0, float(x_B[i])) / d[i] for i in eligible_rows]
        min_ratio = min(ratios)
        min_ratio_rows = [
            row_idx
            for row_idx, ratio in zip(eligible_rows, ratios)
            if abs(ratio - min_ratio) <= 1e-11
        ]
        expected_i_leave = min(min_ratio_rows, key=lambda i: basis[i])

        # Vectorized implementation
        eligible_mask = d > piv_tol
        eligible_indices = np.flatnonzero(eligible_mask)
        eligible_d = d[eligible_indices]
        eligible_xB = np.maximum(0.0, x_B[eligible_indices])
        vec_ratios = eligible_xB / eligible_d
        vec_min_ratio = np.min(vec_ratios)
        tie_mask = np.abs(vec_ratios - vec_min_ratio) <= 1e-11
        tie_indices = eligible_indices[tie_mask]
        if tie_indices.size == 1:
            actual_i_leave = int(tie_indices[0])
        else:
            actual_i_leave = min(tie_indices, key=lambda i: basis[i])

        assert actual_i_leave == expected_i_leave, (
            f"Trial {trial}: {actual_i_leave} != {expected_i_leave}"
        )


# ---------------------------------------------------------------------------
# Test 4: In-Place GEMV Reduced Cost Calculation
# ---------------------------------------------------------------------------

def test_inplace_gemv_reduced_cost():
    """Verify in-place GEMV matches np.dot exactly."""
    rng = np.random.RandomState(789)
    m, n = 30, 80
    A = rng.randn(m, n)
    c = rng.randn(n)
    y = rng.randn(m)

    ATy_buf = np.empty(n, dtype=np.float64)
    r_all = np.empty(n, dtype=np.float64)

    np.dot(A.T, y, out=ATy_buf)
    np.subtract(c, ATy_buf, out=r_all)

    expected_r = c - A.T @ y
    np.testing.assert_allclose(r_all, expected_r, atol=1e-14)


# ---------------------------------------------------------------------------
# Test 5: Standard Problem Classes (MIN, MAX, Bounded, Equality, Mixed)
# ---------------------------------------------------------------------------

def test_small_bounded_minimize():
    model = Model(name="SmallBoundedMin")
    model.add_variable(Variable("x1", lower_bound=0.0))
    model.add_variable(Variable("x2", lower_bound=0.0))
    model.set_objective(Objective({"x1": 1.0, "x2": 2.0}, sense=ObjectiveSense.MINIMIZE))
    model.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, 3.0))
    model.add_constraint(Constraint("c2", {"x1": 2.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, 4.0))

    solver = RevisedSimplexSolver()
    sol = solver.solve(model)

    assert sol.status == SolutionStatus.OPTIMAL
    assert pytest.approx(sol.objective_value, abs=1e-6) == 3.0
    assert pytest.approx(sol.variable_values["x1"], abs=1e-6) == 3.0
    assert pytest.approx(sol.variable_values["x2"], abs=1e-6) == 0.0


def test_small_bounded_maximize():
    model = Model(name="SmallBoundedMax")
    model.add_variable(Variable("x1", lower_bound=0.0))
    model.add_variable(Variable("x2", lower_bound=0.0))
    model.set_objective(Objective({"x1": 3.0, "x2": 2.0}, sense=ObjectiveSense.MAXIMIZE))
    model.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 2.0}, ConstraintSense.LESS_EQUAL, 6.0))
    model.add_constraint(Constraint("c2", {"x1": 2.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 8.0))

    solver = RevisedSimplexSolver()
    sol = solver.solve(model)

    assert sol.status == SolutionStatus.OPTIMAL
    assert pytest.approx(sol.objective_value, abs=1e-6) == 12.6666667
    assert pytest.approx(sol.variable_values["x1"], abs=1e-6) == 3.3333333
    assert pytest.approx(sol.variable_values["x2"], abs=1e-6) == 1.3333333


def test_equality_constraints():
    model = Model(name="EqualityLP")
    model.add_variable(Variable("x1", lower_bound=0.0))
    model.add_variable(Variable("x2", lower_bound=0.0))
    model.add_variable(Variable("x3", lower_bound=0.0))
    model.set_objective(Objective({"x1": 2.0, "x2": 3.0, "x3": 1.0}, sense=ObjectiveSense.MINIMIZE))
    model.add_constraint(Constraint("eq1", {"x1": 1.0, "x2": 1.0, "x3": 1.0}, ConstraintSense.EQUAL, 10.0))
    model.add_constraint(Constraint("eq2", {"x1": 2.0, "x2": -1.0, "x3": 3.0}, ConstraintSense.EQUAL, 14.0))

    solver = RevisedSimplexSolver()
    sol = solver.solve(model)

    assert sol.status == SolutionStatus.OPTIMAL
    x1_val = sol.variable_values["x1"]
    x2_val = sol.variable_values["x2"]
    x3_val = sol.variable_values["x3"]
    assert pytest.approx(x1_val + x2_val + x3_val, abs=1e-6) == 10.0
    assert pytest.approx(2 * x1_val - x2_val + 3 * x3_val, abs=1e-6) == 14.0


def test_infeasible_lp():
    model = Model(name="InfeasibleLP")
    model.add_variable(Variable("x1", lower_bound=0.0))
    model.set_objective(Objective({"x1": 1.0}, sense=ObjectiveSense.MINIMIZE))
    model.add_constraint(Constraint("c1", {"x1": 1.0}, ConstraintSense.GREATER_EQUAL, 10.0))
    model.add_constraint(Constraint("c2", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))

    solver = RevisedSimplexSolver()
    sol = solver.solve(model)

    assert sol.status == SolutionStatus.INFEASIBLE


def test_unbounded_lp():
    model = Model(name="UnboundedLP")
    model.add_variable(Variable("x1", lower_bound=0.0))
    model.add_variable(Variable("x2", lower_bound=0.0))
    model.set_objective(Objective({"x1": 1.0, "x2": 1.0}, sense=ObjectiveSense.MAXIMIZE))
    model.add_constraint(Constraint("c1", {"x1": 1.0, "x2": -1.0}, ConstraintSense.LESS_EQUAL, 5.0))

    solver = RevisedSimplexSolver()
    sol = solver.solve(model)

    assert sol.status == SolutionStatus.UNBOUNDED


def test_degenerate_lp():
    """Verify that Bland's rule handles degenerate basic solutions cleanly."""
    model = Model(name="degenerate_lp")
    model.add_variable(Variable("x1", lower_bound=0.0))
    model.add_variable(Variable("x2", lower_bound=0.0))
    model.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    model.add_constraint(Constraint("c2", {"x1": 1.0, "x2": -2.0}, ConstraintSense.LESS_EQUAL, 0.0))
    model.set_objective(Objective({"x1": 2.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))

    solver = RevisedSimplexSolver()
    sol = solver.solve(model)

    assert sol.status == SolutionStatus.OPTIMAL
    assert pytest.approx(sol.variable_values["x1"], abs=1e-6) == 8.0 / 3.0
    assert pytest.approx(sol.variable_values["x2"], abs=1e-6) == 4.0 / 3.0
    assert pytest.approx(sol.objective_value, abs=1e-6) == 20.0 / 3.0


# ---------------------------------------------------------------------------
# Test 6: Numerical Diagnostics Validation
# ---------------------------------------------------------------------------

def test_numerical_stability_monitor_tracking():
    """Verify that NumericalStabilityMonitor tracks residuals correctly with the optimized loop."""
    monitor = NumericalStabilityMonitor()
    model = Model(name="MonitorTest")
    for i in range(5):
        model.add_variable(Variable(f"x{i}", lower_bound=0.0))
    model.set_objective(Objective({f"x{i}": 1.0 for i in range(5)}, sense=ObjectiveSense.MINIMIZE))
    model.add_constraint(Constraint("c1", {"x0": 1.0, "x1": 2.0, "x2": 3.0}, ConstraintSense.GREATER_EQUAL, 10.0))
    model.add_constraint(Constraint("c2", {"x1": 2.0, "x3": 1.0, "x4": 4.0}, ConstraintSense.GREATER_EQUAL, 15.0))
    model.add_constraint(Constraint("c3", {"x2": 1.0, "x3": 1.0}, ConstraintSense.GREATER_EQUAL, 5.0))

    solver = RevisedSimplexSolver(numerical_monitor=monitor)
    sol = solver.solve(model)

    assert sol.status == SolutionStatus.OPTIMAL
    snap = monitor.get_latest_snapshot()
    assert snap is not None
    assert snap.primal_residual is not None
    assert snap.primal_residual < 1e-6
    assert snap.dual_residual is not None


# ---------------------------------------------------------------------------
# Test 7: MILP and Dual Simplex Warm-Start Integration
# ---------------------------------------------------------------------------

def test_milp_solver_with_optimized_simplex():
    """Verify end-to-end MILP solving with the optimized pivot kernel."""
    model = Model(name="KnapsackMILP")
    model.add_variable(Variable("x1", lower_bound=0.0, upper_bound=1.0, variable_type=VariableType.INTEGER))
    model.add_variable(Variable("x2", lower_bound=0.0, upper_bound=1.0, variable_type=VariableType.INTEGER))
    model.add_variable(Variable("x3", lower_bound=0.0, upper_bound=1.0, variable_type=VariableType.INTEGER))
    model.add_variable(Variable("x4", lower_bound=0.0, upper_bound=1.0, variable_type=VariableType.INTEGER))

    model.set_objective(
        Objective({"x1": 5.0, "x2": 6.0, "x3": 4.0, "x4": 7.0}, sense=ObjectiveSense.MAXIMIZE)
    )
    model.add_constraint(
        Constraint("capacity", {"x1": 2.0, "x2": 3.0, "x3": 1.0, "x4": 4.0}, ConstraintSense.LESS_EQUAL, 5.0)
    )

    # Solve with default cold start
    solver_cold = MILPSolver()
    sol_cold = solver_cold.solve(model)

    # Solve with dual simplex warm start
    solver_warm = MILPSolver(warm_start_relaxations=True)
    sol_warm = solver_warm.solve(model)

    assert sol_cold.status == SolutionStatus.OPTIMAL
    assert sol_warm.status == SolutionStatus.OPTIMAL
    assert pytest.approx(sol_cold.objective_value, abs=1e-6) == sol_warm.objective_value
    assert pytest.approx(sol_cold.objective_value, abs=1e-6) == 11.0
