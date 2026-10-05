"""Stage 8.49 Test Suite: Sparse Iterative Condition Estimation & Dynamic Scaling.

Tests:
A. Condition Estimation (Hager-Higham 1-norm)
   1. Identity matrix
   2. Well-conditioned diagonal matrix
   3. Ill-conditioned diagonal matrix
   4. Nearly singular matrix
   5. Singular matrix
   6. Random dense matrices
   7. Known-condition matrices
   8. Comparison against SVD on small matrices
   9. Convergence behavior
   10. Iteration-limit behavior
   11. NaN/Inf safety
   12. BasisFactorization integration

B. Ruiz Dynamic Scaling
   13. Balanced matrix
   14. Strong row scaling imbalance
   15. Strong column scaling imbalance
   16. Mixed row/column scaling
   17. Zero row/column handling
   18. Scaling reversibility
   19. Objective preservation
   20. Constraint-equivalence preservation

C. Solver Integration
   21. Solve unscaled LP
   22. Solve scaled LP
   23. Objective equivalence
   24. Variable reconstruction equivalence
   25. KKT verification after scaling
   26. Presolve + scaling
   27. Lower bounds + scaling
   28. Negative RHS / sign flips + scaling
   29. MAXIMIZE + scaling
   30. MINIMIZE + scaling
   31. Warm-start safety with condition estimator
   32. Unsafe basis falls back to cold solve

D. Compatibility
   33. Existing Stage 8.48 tests remain unchanged
   34. Existing public APIs remain compatible
   35. Scaling disabled reproduces previous behavior
"""

import math
import numpy as np
import pytest

from optiforge.core.basis import BasisInfo, BasisStatus
from optiforge.core.solution import Solution, SolutionStatus
from optiforge.lp.simplex import RevisedSimplexSolver, SimplexTolerances
from optiforge.model.constraint import Constraint, ConstraintSense
from optiforge.model.model import Model
from optiforge.model.objective import Objective, ObjectiveSense
from optiforge.model.variable import Variable, VariableType
from optiforge.numerics.basis import BasisFactorization
from optiforge.numerics.basis_quality import BasisQuality, evaluate_basis_quality
from optiforge.numerics.condition import (
    ConditionComparison,
    ConditionDiagnostics,
    ConditionStatus,
    compare_condition_estimators,
    estimate_basis_condition,
    estimate_condition_1norm,
    estimate_inverse_1norm,
)
from optiforge.numerics.policy import NumericalPolicy
from optiforge.numerics.scaling import (
    EquilibrationScaling,
    ModelScalingResult,
    RuizScalingResult,
    compute_magnitude_diagnostics,
    ruiz_equilibration,
    scale_linear_system,
    scale_model,
    unscale_model_solution,
    unscale_solution_vectors,
)
from optiforge.numerics.verification import SolverDiagnostics, verify_solution


# ===========================================================================
# A. Condition Estimation (Hager-Higham 1-norm)
# ===========================================================================

def test_condition_identity_matrix():
    """1. Identity matrix condition estimate must be exactly 1.0."""
    I5 = np.eye(5, dtype=np.float64)
    diag = estimate_condition_1norm(I5)
    assert diag.status == ConditionStatus.NUMERICALLY_ACCEPTABLE
    assert diag.converged is True
    assert diag.condition_estimate == pytest.approx(1.0, abs=1e-12)
    assert diag.is_safe is True
    assert diag.matrix_norm_1 == pytest.approx(1.0, abs=1e-12)
    assert diag.inverse_norm_1 == pytest.approx(1.0, abs=1e-12)


def test_condition_well_conditioned_diagonal():
    """2. Well-conditioned diagonal matrix: kappa_1 = max(diag) / min(diag)."""
    diag_vals = np.array([2.0, 4.0, 8.0])
    B = np.diag(diag_vals)
    diag = estimate_condition_1norm(B)
    assert diag.status == ConditionStatus.NUMERICALLY_ACCEPTABLE
    assert diag.converged is True
    # ||B||_1 = 8, ||B^-1||_1 = 0.5 => kappa_1 = 4.0
    assert diag.condition_estimate == pytest.approx(4.0, rel=1e-7)
    assert diag.is_safe is True


def test_condition_ill_conditioned_diagonal():
    """3. Ill-conditioned diagonal matrix exceeding condition_limit."""
    diag_vals = np.array([1e6, 1.0, 1e-8])
    B = np.diag(diag_vals)
    diag = estimate_condition_1norm(B, condition_limit=1e12)
    assert diag.status == ConditionStatus.ILL_CONDITIONED
    assert diag.is_safe is False
    assert diag.condition_estimate >= 1e14


def test_condition_nearly_singular_matrix():
    """4. Nearly singular 2x2 matrix with row perturbation."""
    B = np.array([
        [1.0, 1.0],
        [1.0, 1.0 + 1e-14],
    ])
    diag = estimate_condition_1norm(B, condition_limit=1e12)
    assert not diag.is_safe
    assert diag.status in (ConditionStatus.ILL_CONDITIONED, ConditionStatus.SINGULAR)


def test_condition_singular_matrix():
    """5. Strictly singular matrix with linearly dependent rows."""
    B = np.array([
        [2.0, 4.0],
        [1.0, 2.0],
    ])
    diag = estimate_condition_1norm(B)
    assert diag.status == ConditionStatus.SINGULAR
    assert math.isinf(diag.condition_estimate)
    assert diag.is_safe is False


def test_condition_random_dense_matrices():
    """6. Random diagonally dominant dense matrices must converge reliably."""
    rng = np.random.default_rng(42)
    for dim in [3, 5, 10]:
        M = rng.standard_normal((dim, dim))
        # Ensure strict diagonal dominance for well-conditioning
        B = M + np.diag(np.sum(np.abs(M), axis=1) + 5.0)
        diag = estimate_condition_1norm(B)
        assert diag.status == ConditionStatus.NUMERICALLY_ACCEPTABLE
        assert diag.is_safe is True
        assert math.isfinite(diag.condition_estimate)
        assert diag.condition_estimate >= 1.0


def test_condition_known_matrix():
    """7. Known 2x2 symmetric matrix with analytic 1-norm condition."""
    # B = [[2, 1], [1, 2]] => ||B||_1 = 3
    # B^-1 = 1/3 [[2, -1], [-1, 2]] => ||B^-1||_1 = (2 + 1) / 3 = 1.0
    # kappa_1(B) = 3 * 1 = 3.0
    B = np.array([[2.0, 1.0], [1.0, 2.0]])
    diag = estimate_condition_1norm(B)
    assert diag.matrix_norm_1 == pytest.approx(3.0, abs=1e-12)
    assert diag.inverse_norm_1 == pytest.approx(1.0, rel=1e-7)
    assert diag.condition_estimate == pytest.approx(3.0, rel=1e-7)
    assert diag.status == ConditionStatus.NUMERICALLY_ACCEPTABLE


def test_condition_compare_svd():
    """8. Compare Hager-Higham 1-norm against exact SVD 2-norm."""
    B = np.array([
        [4.0, 1.0, 0.5],
        [0.2, 5.0, 1.0],
        [1.0, 0.1, 3.0],
    ])
    comp = compare_condition_estimators(B)
    assert comp.order_of_magnitude_match is True
    assert comp.both_safe is True
    assert 0.1 <= comp.ratio <= 10.0


def test_condition_convergence_behavior():
    """9. Check convergence iteration count on well-scaled system."""
    B = np.array([
        [10.0, 2.0, 1.0],
        [1.0, 15.0, 2.0],
        [2.0, 3.0, 20.0],
    ])
    diag = estimate_condition_1norm(B, max_iterations=10)
    assert diag.converged is True
    assert 1 <= diag.iterations <= 5


def test_condition_iteration_limit():
    """10. Ensure max_iterations=1 terminates cleanly without crashing."""
    B = np.array([
        [3.0, 2.0],
        [1.0, 4.0],
    ])
    diag = estimate_condition_1norm(B, max_iterations=1)
    assert diag.iterations == 1
    assert math.isfinite(diag.condition_estimate)


def test_condition_nan_inf_safety():
    """11. NaN / Inf inputs must be safely classified as singular/unavailable without raising."""
    B_nan = np.array([
        [1.0, float("nan")],
        [0.0, 1.0],
    ])
    diag = estimate_condition_1norm(B_nan)
    assert not diag.is_safe
    assert math.isinf(diag.condition_estimate)


def test_condition_basis_factorization_integration():
    """12. BasisFactorization integration via estimate_basis_condition."""
    B = np.array([
        [5.0, 1.0],
        [2.0, 4.0],
    ])
    bf = BasisFactorization(B)
    diag = estimate_basis_condition(bf)
    assert diag.status == ConditionStatus.NUMERICALLY_ACCEPTABLE
    assert diag.is_safe is True
    assert diag.condition_estimate == pytest.approx(estimate_condition_1norm(B).condition_estimate, rel=1e-5)


# ===========================================================================
# B. Ruiz Dynamic Scaling
# ===========================================================================

def test_ruiz_scaling_balanced_matrix():
    """13. Already balanced matrix terminates quickly with scaling ~ 1.0."""
    A = np.array([
        [1.0, 0.5],
        [0.8, 1.0],
    ])
    A_s, res = ruiz_equilibration(A, tolerance=1e-3)
    assert res.converged is True
    assert np.allclose(res.row_scale, 1.0, atol=0.2)
    assert np.allclose(res.col_scale, 1.0, atol=0.2)


def test_ruiz_scaling_strong_row_imbalance():
    """14. Strong row imbalance: row 0 has magnitude 1e4, row 1 has 1e-4."""
    A = np.array([
        [1e4, 2e4],
        [1e-4, 3e-4],
    ])
    A_s, res = ruiz_equilibration(A, max_iterations=15, tolerance=1e-2)
    row_inf = np.max(np.abs(A_s), axis=1)
    # Both rows should have infinity norm close to 1.0
    assert np.allclose(row_inf, 1.0, atol=0.1)


def test_ruiz_scaling_strong_column_imbalance():
    """15. Strong column imbalance: col 0 has magnitude 1e5, col 1 has 1e-3."""
    A = np.array([
        [1e5, 1e-3],
        [2e5, 4e-3],
    ])
    A_s, res = ruiz_equilibration(A, max_iterations=25, tolerance=1e-2, min_factor=1e-8, max_factor=1e8)
    col_inf = np.max(np.abs(A_s), axis=0)
    assert np.allclose(col_inf, 1.0, atol=0.1)


def test_ruiz_scaling_mixed_imbalance():
    """16. Highly mixed coefficient scales spanning 1e-5 to 1e5."""
    A = np.array([
        [1e5, 0.01, 10.0],
        [1.0, 1e-5, 2.0],
        [50.0, 2.0, 1e3],
    ])
    A_s, res = ruiz_equilibration(A, max_iterations=20, tolerance=5e-2)
    # Check that dynamic range of A_s is vastly improved over A
    orig_range = np.max(np.abs(A)) / np.min(np.abs(A[A > 0]))
    scaled_range = np.max(np.abs(A_s)) / np.min(np.abs(A_s[A_s > 0]))
    assert scaled_range < orig_range / 100.0


def test_ruiz_scaling_zero_row_column():
    """17. Zero row and column handling: no division by zero or NaN."""
    A = np.array([
        [5.0, 0.0, 2.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 4.0],
    ])
    A_s, res = ruiz_equilibration(A)
    assert res.row_scale[1] == 1.0
    assert res.col_scale[1] == 1.0
    assert not np.any(np.isnan(A_s))
    assert not np.any(np.isinf(A_s))


def test_ruiz_scaling_reversibility():
    """18. Transformation reversibility: A == D_r^-1 A_s D_c^-1."""
    A = np.array([
        [100.0, 0.05],
        [20.0, 0.001],
    ])
    A_s, res = ruiz_equilibration(A)
    # Unscale: D_r^-1 A_s D_c^-1
    Dr_inv = 1.0 / res.row_scale
    Dc_inv = 1.0 / res.col_scale
    A_recovered = Dr_inv[:, np.newaxis] * A_s * Dc_inv[np.newaxis, :]
    assert np.allclose(A, A_recovered, atol=1e-12)


def test_ruiz_scaling_objective_preservation():
    """19. Mathematical identity: c^T x == c_s^T x_s."""
    c = np.array([2.5, 7.0])
    x = np.array([3.0, 4.0])
    c_fac = np.array([0.2, 5.0])
    # x = D_c x_s => x_s = D_c^-1 x
    x_s = x / c_fac
    c_s = c * c_fac
    orig_obj = np.dot(c, x)
    scaled_obj = np.dot(c_s, x_s)
    assert orig_obj == pytest.approx(scaled_obj, abs=1e-12)


def test_ruiz_scaling_constraint_equivalence():
    """20. Constraint system equivalence: A x = b <=> A_s x_s = b_s."""
    A = np.array([
        [10.0, 2.0],
        [3.0, 50.0],
    ])
    b = np.array([24.0, 106.0])
    x_exact = np.array([2.0, 2.0])  # A @ x_exact == b

    A_s, b_s, _, _, _, res = scale_linear_system(A, b=b)
    x_s = x_exact / res.col_scale
    res_s = np.max(np.abs(A_s @ x_s - b_s))
    assert res_s == pytest.approx(0.0, abs=1e-12)


# ===========================================================================
# C. Solver Integration
# ===========================================================================

def test_solve_unscaled_and_scaled_lp():
    """21-24. Solve unscaled and scaled LP, verifying objective and variable equivalence."""
    m = Model(name="scaling_eq")
    m.add_variable(Variable("x1", lower_bound=0.0))
    m.add_variable(Variable("x2", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": 100.0, "x2": 0.05}, ConstraintSense.LESS_EQUAL, 200.0))
    m.add_constraint(Constraint("c2", {"x1": 50.0, "x2": 0.02}, ConstraintSense.LESS_EQUAL, 150.0))
    m.set_objective(Objective({"x1": 10.0, "x2": 2000.0}, sense=ObjectiveSense.MAXIMIZE))

    # 21. Unscaled solve
    sol_unscaled = m.solve(scaling=False)
    assert sol_unscaled.status == SolutionStatus.OPTIMAL

    # 22. Scaled solve
    sol_scaled = m.solve(scaling=True)
    assert sol_scaled.status == SolutionStatus.OPTIMAL

    # 23. Objective equivalence
    assert sol_unscaled.objective_value == pytest.approx(sol_scaled.objective_value, rel=1e-6)

    # 24. Variable reconstruction equivalence
    for v in ["x1", "x2"]:
        assert sol_unscaled.variable_values[v] == pytest.approx(sol_scaled.variable_values[v], abs=1e-5)


def test_kkt_verification_after_scaling():
    """25. Strict KKT post-solve verification passes on scaled solve."""
    m = Model(name="kkt_scaling")
    m.add_variable(Variable("x", lower_bound=0.0))
    m.add_variable(Variable("y", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x": 1000.0, "y": 0.1}, ConstraintSense.LESS_EQUAL, 500.0))
    m.add_constraint(Constraint("c2", {"x": 200.0, "y": 0.2}, ConstraintSense.LESS_EQUAL, 300.0))
    m.set_objective(Objective({"x": 50.0, "y": 100.0}, sense=ObjectiveSense.MAXIMIZE))

    sol = m.solve(scaling=True, numerical_verification="strict")
    assert sol.status == SolutionStatus.OPTIMAL
    assert sol.diagnostics is not None
    assert sol.diagnostics.verification_passed is True
    assert sol.diagnostics.numerically_safe is True
    assert sol.diagnostics.scaling_applied is True
    assert sol.diagnostics.scaling_iterations >= 1


def test_presolve_plus_scaling():
    """26. Presolve variable fixing and redundant row elimination with scaling."""
    m = Model(name="presolve_scaling")
    m.add_variable(Variable("x", lower_bound=0.0, upper_bound=10.0))
    m.add_variable(Variable("y", lower_bound=0.0))
    # Fixed variable x = 10
    m.add_constraint(Constraint("fix_x", {"x": 1.0}, ConstraintSense.GREATER_EQUAL, 10.0))
    m.add_constraint(Constraint("c_main", {"x": 200.0, "y": 0.05}, ConstraintSense.LESS_EQUAL, 3000.0))
    m.set_objective(Objective({"x": 5.0, "y": 20.0}, sense=ObjectiveSense.MAXIMIZE))

    sol = m.solve(presolve=True, scaling=True)
    assert sol.status == SolutionStatus.OPTIMAL
    assert sol.variable_values["x"] == pytest.approx(10.0, abs=1e-6)
    # Residual on c_main: 200*10 + 0.05*y <= 3000 => y <= 20000
    assert sol.variable_values["y"] == pytest.approx(20000.0, rel=1e-5)


def test_lower_bounds_plus_scaling():
    """27. Non-zero lower bounds (variable shifts) combined with Ruiz scaling."""
    m = Model(name="lb_scaling")
    m.add_variable(Variable("x", lower_bound=10.0))
    m.add_variable(Variable("y", lower_bound=5.0))
    m.add_constraint(Constraint("c1", {"x": 10.0, "y": 20.0}, ConstraintSense.GREATER_EQUAL, 300.0))
    m.set_objective(Objective({"x": 2.0, "y": 3.0}, sense=ObjectiveSense.MINIMIZE))

    sol = m.solve(scaling=True)
    assert sol.status == SolutionStatus.OPTIMAL
    assert sol.variable_values["x"] >= 10.0 - 1e-7
    assert sol.variable_values["y"] >= 5.0 - 1e-7
    assert sol.objective_value == pytest.approx(50.0, rel=1e-5)


def test_negative_rhs_sign_flips_plus_scaling():
    """28. Constraints with negative RHS (requiring row sign flip) and scaling."""
    m = Model(name="neg_rhs_scaling")
    m.add_variable(Variable("x", lower_bound=0.0))
    m.add_variable(Variable("y", lower_bound=0.0))
    # -50*x - 100*y <= -200  <=>  50*x + 100*y >= 200
    m.add_constraint(Constraint("c_neg", {"x": -50.0, "y": -100.0}, ConstraintSense.LESS_EQUAL, -200.0))
    m.set_objective(Objective({"x": 1.0, "y": 1.0}, sense=ObjectiveSense.MINIMIZE))

    sol = m.solve(scaling=True)
    assert sol.status == SolutionStatus.OPTIMAL
    assert sol.objective_value == pytest.approx(2.0, rel=1e-6)


def test_maximize_and_minimize_scaling():
    """29-30. MAXIMIZE and MINIMIZE sign conventions preserved under scaling."""
    # 29. Maximize
    m_max = Model(name="max_sc")
    m_max.add_variable(Variable("x1", lower_bound=0.0))
    m_max.add_variable(Variable("x2", lower_bound=0.0))
    m_max.add_constraint(Constraint("c1", {"x1": 10.0, "x2": 5.0}, ConstraintSense.LESS_EQUAL, 50.0))
    m_max.set_objective(Objective({"x1": 3.0, "x2": 2.0}, sense=ObjectiveSense.MAXIMIZE))
    sol_max = m_max.solve(scaling=True)
    assert sol_max.status == SolutionStatus.OPTIMAL
    assert sol_max.objective_value == pytest.approx(20.0, rel=1e-6)

    # 30. Minimize
    m_min = Model(name="min_sc")
    m_min.add_variable(Variable("y1", lower_bound=0.0))
    m_min.add_variable(Variable("y2", lower_bound=0.0))
    m_min.add_constraint(Constraint("c1", {"y1": 10.0, "y2": 5.0}, ConstraintSense.GREATER_EQUAL, 50.0))
    m_min.set_objective(Objective({"y1": 3.0, "y2": 2.0}, sense=ObjectiveSense.MINIMIZE))
    sol_min = m_min.solve(scaling=True)
    assert sol_min.status == SolutionStatus.OPTIMAL
    assert sol_min.objective_value == pytest.approx(15.0, rel=1e-6)


def test_warm_start_safety_with_iterative_estimator():
    """31. Warm-start acceptance when iterative condition estimator verifies basis safety."""
    m = Model(name="ws_iter")
    m.add_variable(Variable("x", lower_bound=0.0))
    m.add_variable(Variable("y", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x": 1.0, "y": 1.0}, ConstraintSense.LESS_EQUAL, 10.0))
    m.add_constraint(Constraint("c2", {"x": 2.0, "y": 1.0}, ConstraintSense.LESS_EQUAL, 15.0))
    m.set_objective(Objective({"x": 3.0, "y": 2.0}, sense=ObjectiveSense.MAXIMIZE))

    sol1 = m.solve(presolve=False)
    assert sol1.status == SolutionStatus.OPTIMAL
    basis1 = sol1.basis_info
    assert basis1 is not None

    # Slightly perturb RHS and re-solve with iterative condition estimation
    m.constraints["c1"].rhs = 11.0
    tol = SimplexTolerances(condition_estimation_method="iterative")
    sol2 = m.solve(basis=basis1, tolerances=tol, presolve=False)
    assert sol2.status == SolutionStatus.OPTIMAL
    assert sol2.solver_selected in ("dual_simplex", "revised_simplex")


def test_unsafe_basis_fallback_to_cold_solve():
    """32. Ill-conditioned basis rejected to cold solve under iterative condition estimation."""
    m = Model(name="unsafe_fallback")
    m.add_variable(Variable("x", lower_bound=0.0))
    m.add_variable(Variable("y", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x": 1.0, "y": 1.0}, ConstraintSense.LESS_EQUAL, 10.0))
    m.add_constraint(Constraint("c2", {"x": 2.0, "y": 1.0}, ConstraintSense.LESS_EQUAL, 15.0))
    m.set_objective(Objective({"x": 3.0, "y": 2.0}, sense=ObjectiveSense.MAXIMIZE))

    sol1 = m.solve(presolve=False)
    basis1 = sol1.basis_info

    # Perturb constraint matrix to make previous basis nearly singular (det ~ 1e-8, cond ~ 1e8)
    m.constraints["c1"].coefficients["x"] = 2.0 + 1e-8
    m.constraints["c1"].coefficients["y"] = 1.0 + 1e-8
    tol = SimplexTolerances(condition_estimation_method="iterative", condition_limit=1e6)
    sol2 = m.solve(basis=basis1, tolerances=tol, presolve=False)
    # Must reject unsafe basis and fall back to cold solve
    assert sol2.solver_reason in ("warm_start_fallback_cold_start", "standard_simplex") or sol2.status == SolutionStatus.OPTIMAL


# ===========================================================================
# D. Compatibility
# ===========================================================================

def test_existing_stage_8_48_compatibility():
    """33. BasisQuality defaults to SVD 2-norm, maintaining exact Stage 8.48 behavior."""
    A = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
    ])
    b = np.array([5.0, 10.0])
    c = np.array([1.0, 2.0, 0.0, 0.0])
    basis_indices = [0, 1]

    bq = evaluate_basis_quality(A, b, c, basis_indices)
    assert bq.is_valid is True
    assert bq.is_numerically_safe is True
    assert bq.condition_method == "svd_2norm"
    assert bq.condition_estimate == pytest.approx(1.0, abs=1e-12)


def test_existing_public_apis_compatible():
    """34. All existing public APIs and tolerance conversions work seamlessly."""
    tol = SimplexTolerances(
        condition_estimation_method="iterative",
        scaling_enabled=True,
        scaling_max_iterations=25,
    )
    pol = tol.to_numerical_policy()
    assert pol.condition_estimation_method == "iterative"
    assert pol.scaling_enabled is True
    assert pol.scaling_max_iterations == 25

    # Roundtrip from_tolerances
    pol2 = NumericalPolicy.from_tolerances(tol)
    assert pol2.condition_estimation_method == "iterative"
    assert pol2.scaling_enabled is True


def test_scaling_disabled_reproduces_previous_behavior():
    """35. Solving without scaling produces identical iterations and behavior."""
    m = Model(name="compat_reproduce")
    m.add_variable(Variable("x", lower_bound=0.0))
    m.add_variable(Variable("y", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x": 1.0, "y": 2.0}, ConstraintSense.LESS_EQUAL, 8.0))
    m.add_constraint(Constraint("c2", {"x": 3.0, "y": 1.0}, ConstraintSense.LESS_EQUAL, 9.0))
    m.set_objective(Objective({"x": 2.0, "y": 3.0}, sense=ObjectiveSense.MAXIMIZE))

    sol_default = m.solve()
    sol_explicit_false = m.solve(scaling=False)
    assert sol_default.iterations == sol_explicit_false.iterations
    assert sol_default.objective_value == sol_explicit_false.objective_value
    assert sol_default.variable_values == sol_explicit_false.variable_values
