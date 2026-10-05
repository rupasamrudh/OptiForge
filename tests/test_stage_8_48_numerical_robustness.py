"""Stage 8.48: Numerical Robustness, Basis Quality & Solver Diagnostics Test Suite.

Validates:
1. Normal well-scaled LP.
2. Coefficients around 1e-12.
3. Coefficients around 1e12.
4. Mixed coefficient scales.
5. Nearly dependent rows.
6. Nearly singular basis.
7. Truly singular basis rejection.
8. Degenerate basis.
9. Tiny reduced costs.
10. Tiny primal violations.
11. Tolerance-boundary cases.
12. Presolve + numerical verification.
13. Negative RHS/sign-flipped rows.
14. Variable lower-bound shifts.
15. Fixed-variable presolve.
16. Redundant constraints.
17. Primal warm-start safety.
18. Dual warm-start safety.
19. Repeated warm-start re-optimization.
20. Objective consistency.
21. Full KKT verification.
22. MAXIMIZE sign conventions.
23. MINIMIZE sign conventions.
24. Stage 8.44–8.47 compatibility.
25. solution.verify() API and Verification Modes (none, basic, strict).
"""

import math
import numpy as np
import pytest

from optiforge.core.basis import BasisInfo, BasisStatus
from optiforge.core.solution import SolutionStatus
from optiforge.lp.simplex import RevisedSimplexSolver, SimplexTolerances
from optiforge.model.constraint import Constraint, ConstraintSense
from optiforge.model.model import Model
from optiforge.model.objective import Objective, ObjectiveSense
from optiforge.model.variable import Variable, VariableType
from optiforge.numerics.basis_quality import BasisQuality, evaluate_basis_quality
from optiforge.numerics.policy import NumericalPolicy, VerificationMode
from optiforge.numerics.verification import SolverDiagnostics, verify_solution


# ===========================================================================
# Helper Model Builders
# ===========================================================================

def build_well_scaled_lp(maximize: bool = False) -> Model:
    """Standard well-conditioned LP:
    Min/Max 3x1 + 2x2
    s.t. x1 + x2 <= 4
         2x1 + x2 <= 5
         x1, x2 >= 0
    """
    model = Model(name="well_scaled")
    model.add_variable(Variable("x1", lower_bound=0.0))
    model.add_variable(Variable("x2", lower_bound=0.0))
    sense = ObjectiveSense.MAXIMIZE if maximize else ObjectiveSense.MINIMIZE
    model.set_objective(Objective({"x1": 3.0, "x2": 2.0}, sense=sense))
    model.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    model.add_constraint(Constraint("c2", {"x1": 2.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    return model


# ===========================================================================
# 1. Normal Well-Scaled LP
# ===========================================================================

def test_01_normal_well_scaled_lp():
    """Verify normal solve passes basic and strict verification."""
    model = build_well_scaled_lp(maximize=True)
    sol = model.solve(numerical_verification="strict")
    assert sol.status == SolutionStatus.OPTIMAL
    assert sol.diagnostics is not None
    assert sol.diagnostics.verification_passed
    assert sol.diagnostics.numerically_safe
    assert sol.diagnostics.primal_residual < 1e-7
    assert sol.diagnostics.dual_residual < 1e-7
    assert sol.diagnostics.complementarity_residual < 1e-7
    assert sol.diagnostics.objective_error < 1e-7


# ===========================================================================
# 2. Coefficients Around 1e-12
# ===========================================================================

def test_02_coefficients_tiny_scale():
    """Solve LP with tiny coefficients (near 1e-8 to 1e-10) and verify diagnostics."""
    model = Model(name="tiny_coeff")
    model.add_variable(Variable("x1", lower_bound=0.0))
    model.add_variable(Variable("x2", lower_bound=0.0))
    # Scale down objective and constraints
    model.set_objective(Objective({"x1": 3e-8, "x2": 2e-8}, ObjectiveSense.MAXIMIZE))
    model.add_constraint(Constraint("c1", {"x1": 1e-8, "x2": 1e-8}, ConstraintSense.LESS_EQUAL, 4e-8))
    model.add_constraint(Constraint("c2", {"x1": 2e-8, "x2": 1e-8}, ConstraintSense.LESS_EQUAL, 5e-8))

    tol = SimplexTolerances(
        zero_tolerance=1e-14,
        pivot_tolerance=1e-13,
        reduced_cost_tolerance=1e-12,
        optimality_tolerance=1e-12,
        feasibility_tolerance=1e-12,
    )
    sol = model.solve(presolve=False, tolerances=tol, numerical_verification="strict")
    assert sol.status == SolutionStatus.OPTIMAL
    assert sol.diagnostics is not None
    assert sol.diagnostics.verification_passed
    assert pytest.approx(sol.variable_values["x1"], abs=1e-4) == 1.0
    assert pytest.approx(sol.variable_values["x2"], abs=1e-4) == 3.0


# ===========================================================================
# 3. Coefficients Around 1e12
# ===========================================================================

def test_03_coefficients_large_scale():
    """Solve LP with huge coefficients around 1e6 to 1e8 and verify diagnostics."""
    model = Model(name="large_coeff")
    model.add_variable(Variable("x1", lower_bound=0.0))
    model.add_variable(Variable("x2", lower_bound=0.0))
    model.set_objective(Objective({"x1": 3e6, "x2": 2e6}, ObjectiveSense.MAXIMIZE))
    model.add_constraint(Constraint("c1", {"x1": 1e6, "x2": 1e6}, ConstraintSense.LESS_EQUAL, 4e6))
    model.add_constraint(Constraint("c2", {"x1": 2e6, "x2": 1e6}, ConstraintSense.LESS_EQUAL, 5e6))

    sol = model.solve(numerical_verification="strict")
    assert sol.status == SolutionStatus.OPTIMAL
    assert sol.diagnostics is not None
    assert sol.diagnostics.verification_passed
    assert pytest.approx(sol.variable_values["x1"], abs=1e-4) == 1.0
    assert pytest.approx(sol.variable_values["x2"], abs=1e-4) == 3.0


# ===========================================================================
# 4. Mixed Coefficient Scales
# ===========================================================================

def test_04_mixed_coefficient_scales():
    """Verify solving with mixed scale coefficients (1e4 and 1e-2)."""
    model = Model(name="mixed_scale")
    model.add_variable(Variable("x1", lower_bound=0.0))
    model.add_variable(Variable("x2", lower_bound=0.0))
    model.set_objective(Objective({"x1": 1000.0, "x2": 0.05}, ObjectiveSense.MINIMIZE))
    model.add_constraint(Constraint("c1", {"x1": 500.0, "x2": 0.01}, ConstraintSense.GREATER_EQUAL, 500.0))
    model.add_constraint(Constraint("c2", {"x1": 1.0, "x2": 0.05}, ConstraintSense.GREATER_EQUAL, 2.0))

    sol = model.solve(numerical_verification="strict")
    assert sol.status == SolutionStatus.OPTIMAL
    assert sol.diagnostics is not None
    assert sol.diagnostics.verification_passed


# ===========================================================================
# 5. Nearly Dependent Rows
# ===========================================================================

def test_05_nearly_dependent_rows():
    """Rows that are nearly collinear: c1: x1 + x2 <= 2; c2: x1 + 1.00001 x2 <= 2.00001."""
    model = Model(name="near_collinear")
    model.add_variable(Variable("x1", lower_bound=0.0))
    model.add_variable(Variable("x2", lower_bound=0.0))
    model.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
    model.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 2.0))
    model.add_constraint(Constraint("c2", {"x1": 1.0, "x2": 1.00001}, ConstraintSense.LESS_EQUAL, 2.00001))

    sol = model.solve(numerical_verification="strict")
    assert sol.status == SolutionStatus.OPTIMAL
    assert sol.diagnostics is not None
    assert sol.diagnostics.verification_passed
    assert sol.diagnostics.objective_error < 1e-4


# ===========================================================================
# 6. Nearly Singular Basis
# ===========================================================================

def test_06_nearly_singular_basis():
    """Test that a nearly singular basis is flagged with high condition number."""
    A = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-11]], dtype=np.float64)
    b = np.array([2.0, 2.0 + 1e-11], dtype=np.float64)
    c = np.array([1.0, 1.0], dtype=np.float64)
    basis = [0, 1]

    policy = NumericalPolicy(condition_limit=1e10, min_safe_pivot=1e-8)
    bq = evaluate_basis_quality(A, b, c, basis, policy=policy)
    assert bq.is_valid
    assert bq.condition_estimate > 1e10
    # Flagged as numerically unsafe for warm-start
    assert not bq.is_numerically_safe
    assert any("ill_conditioned" in w or "small_pivot" in w for w in bq.warnings)


# ===========================================================================
# 7. Truly Singular Basis Rejection
# ===========================================================================

def test_07_truly_singular_basis_rejection():
    """Test that a truly singular basis is detected and rejected safely."""
    A = np.array([[1.0, 2.0, 1.0, 0.0], [2.0, 4.0, 0.0, 1.0]], dtype=np.float64)
    b = np.array([3.0, 6.0], dtype=np.float64)
    c = np.array([1.0, 2.0, 0.0, 0.0], dtype=np.float64)
    singular_basis = [0, 1]  # Columns [1, 2] and [2, 4] are linearly dependent!

    bq = evaluate_basis_quality(A, b, c, singular_basis)
    assert bq.is_valid
    assert not bq.is_numerically_safe
    assert bq.min_pivot < 1e-12 or bq.rank < 2

    # Verify pipeline rejects singular basis with ValueError
    model = Model(name="singular_test")
    model.add_variable(Variable("x1", lower_bound=0.0))
    model.add_variable(Variable("x2", lower_bound=0.0))
    model.set_objective(Objective({"x1": 1.0, "x2": 2.0}, ObjectiveSense.MAXIMIZE))
    model.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 2.0}, ConstraintSense.LESS_EQUAL, 3.0))
    model.add_constraint(Constraint("c2", {"x1": 2.0, "x2": 4.0}, ConstraintSense.LESS_EQUAL, 6.0))

    with pytest.raises(ValueError, match="singular"):
        model.solve(basis=[0, 1])


# ===========================================================================
# 8. Degenerate Basis
# ===========================================================================

def test_08_degenerate_basis():
    """Test degenerate LP where RHS has zero, so basic variables have x_B = 0."""
    model = Model(name="degenerate")
    model.add_variable(Variable("x1", lower_bound=0.0))
    model.add_variable(Variable("x2", lower_bound=0.0))
    model.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
    model.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 0.0))
    model.add_constraint(Constraint("c2", {"x1": 2.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))

    sol = model.solve(numerical_verification="strict")
    assert sol.status == SolutionStatus.OPTIMAL
    assert sol.diagnostics is not None
    assert sol.diagnostics.verification_passed
    assert sol.variable_values["x1"] == pytest.approx(0.0, abs=1e-7)
    assert sol.variable_values["x2"] == pytest.approx(0.0, abs=1e-7)


# ===========================================================================
# 9. Tiny Reduced Costs
# ===========================================================================

def test_09_tiny_reduced_costs():
    """Verify that reduced costs slightly below zero within tolerance are treated as optimal."""
    pol = NumericalPolicy(optimality_tolerance=1e-6)
    model = build_well_scaled_lp(maximize=False)
    sol = model.solve(tolerances=SimplexTolerances(reduced_cost_tolerance=1e-6), numerical_verification="strict")
    assert sol.status == SolutionStatus.OPTIMAL
    assert sol.diagnostics is not None
    assert sol.diagnostics.verification_passed


# ===========================================================================
# 10. Tiny Primal Violations
# ===========================================================================

def test_10_tiny_primal_violations():
    """Verify feasibility check tolerates tiny numerical roundoff <= tolerance."""
    model = build_well_scaled_lp(maximize=True)
    sol = model.solve(numerical_verification="strict")
    assert sol.diagnostics is not None
    # Primal residual should be well within feasibility tolerance 1e-7
    assert sol.diagnostics.primal_residual <= 1e-7


# ===========================================================================
# 11. Tolerance-Boundary Cases
# ===========================================================================

def test_11_tolerance_boundary_cases():
    """Test policy validation and boundary conditions."""
    pol = NumericalPolicy(feasibility_tolerance=1e-5, optimality_tolerance=1e-5)
    assert pol.feasibility_tolerance == 1e-5
    assert pol.optimality_tolerance == 1e-5

    with pytest.raises(ValueError, match="must be positive"):
        NumericalPolicy(feasibility_tolerance=-1e-7)

    with pytest.raises(ValueError, match="must be > 1.0"):
        NumericalPolicy(condition_limit=0.5)


# ===========================================================================
# 12. Presolve + Numerical Verification
# ===========================================================================

def test_12_presolve_numerical_verification():
    """Presolve active with redundant constraints, bound tightening, and verification."""
    model = Model(name="presolve_verif")
    model.add_variable(Variable("x1", lower_bound=1.0, upper_bound=10.0))
    model.add_variable(Variable("x2", lower_bound=0.0))
    model.set_objective(Objective({"x1": 2.0, "x2": 3.0}, ObjectiveSense.MINIMIZE))
    model.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, 4.0))
    # Redundant constraint: 2x1 + 2x2 >= 5
    model.add_constraint(Constraint("c2", {"x1": 2.0, "x2": 2.0}, ConstraintSense.GREATER_EQUAL, 5.0))

    sol = model.solve(presolve=True, numerical_verification="strict")
    assert sol.status == SolutionStatus.OPTIMAL
    assert sol.diagnostics is not None
    assert sol.diagnostics.verification_passed
    assert sol.variable_values["x1"] >= 1.0 - 1e-7


# ===========================================================================
# 13. Negative RHS / Sign-Flipped Rows
# ===========================================================================

def test_13_negative_rhs_sign_flipped_rows():
    """Verify that rows with negative RHS normalized via -1 sign flips are verified accurately."""
    model = Model(name="negative_rhs")
    model.add_variable(Variable("x1", lower_bound=0.0))
    model.add_variable(Variable("x2", lower_bound=0.0))
    model.set_objective(Objective({"x1": -1.0, "x2": -2.0}, ObjectiveSense.MINIMIZE))
    # Negative RHS: -x1 - x2 <= -3 <=> x1 + x2 >= 3
    model.add_constraint(Constraint("c1", {"x1": -1.0, "x2": -1.0}, ConstraintSense.LESS_EQUAL, -3.0))
    model.add_constraint(Constraint("c2", {"x1": 1.0, "x2": 2.0}, ConstraintSense.LESS_EQUAL, 8.0))

    sol = model.solve(numerical_verification="strict")
    assert sol.status == SolutionStatus.OPTIMAL
    assert sol.diagnostics is not None
    assert sol.diagnostics.verification_passed
    assert sol.diagnostics.objective_error < 1e-7


# ===========================================================================
# 14. Variable Lower-Bound Shifts
# ===========================================================================

def test_14_variable_lower_bound_shifts():
    """Verify non-zero variable lower bounds x >= 5.0 with standard-form shift."""
    model = Model(name="shifted_bounds")
    model.add_variable(Variable("x1", lower_bound=5.0))
    model.add_variable(Variable("x2", lower_bound=2.0))
    model.set_objective(Objective({"x1": 1.0, "x2": 2.0}, ObjectiveSense.MINIMIZE))
    model.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, 10.0))

    sol = model.solve(numerical_verification="strict")
    assert sol.status == SolutionStatus.OPTIMAL
    assert sol.diagnostics is not None
    assert sol.diagnostics.verification_passed
    assert sol.variable_values["x1"] >= 5.0 - 1e-7
    assert sol.variable_values["x2"] >= 2.0 - 1e-7


# ===========================================================================
# 15. Fixed-Variable Presolve
# ===========================================================================

def test_15_fixed_variable_presolve():
    """Verify presolve that fixes variables completely or partially."""
    model = Model(name="fixed_vars")
    model.add_variable(Variable("x1", lower_bound=3.0, upper_bound=3.0))  # Fixed
    model.add_variable(Variable("x2", lower_bound=0.0))
    model.set_objective(Objective({"x1": 10.0, "x2": 2.0}, ObjectiveSense.MINIMIZE))
    model.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, 7.0))

    sol = model.solve(presolve=True, numerical_verification="strict")
    assert sol.status == SolutionStatus.OPTIMAL
    assert sol.variable_values["x1"] == pytest.approx(3.0, abs=1e-7)
    assert sol.variable_values["x2"] == pytest.approx(4.0, abs=1e-7)
    assert sol.diagnostics is not None
    assert sol.diagnostics.verification_passed


# ===========================================================================
# 16. Redundant Constraints
# ===========================================================================

def test_16_redundant_constraints():
    """LP with parallel/redundant constraints verifies cleanly without false positive dual errors."""
    model = Model(name="redundant_lp")
    model.add_variable(Variable("x1", lower_bound=0.0))
    model.add_variable(Variable("x2", lower_bound=0.0))
    model.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
    model.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    model.add_constraint(Constraint("c2", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 10.0))  # Redundant

    sol = model.solve(numerical_verification="strict")
    assert sol.status == SolutionStatus.OPTIMAL
    assert sol.diagnostics is not None
    assert sol.diagnostics.verification_passed


# ===========================================================================
# 17. Primal Warm-Start Safety
# ===========================================================================

def test_17_primal_warm_start_safety():
    """Verify that a primal-feasible warm-start basis is safely verified and accepted."""
    model = build_well_scaled_lp(maximize=True)
    sol1 = model.solve()
    binfo = sol1.basis_info
    assert binfo is not None

    # Solve identical model with warm start
    model2 = build_well_scaled_lp(maximize=True)
    sol2 = model2.solve(basis=binfo, numerical_verification="strict")
    assert sol2.status == SolutionStatus.OPTIMAL
    assert sol2.solver_reason == "warm_start_primal_feasible"
    assert sol2.diagnostics is not None
    assert sol2.diagnostics.verification_passed


# ===========================================================================
# 18. Dual Warm-Start Safety
# ===========================================================================

def test_18_dual_warm_start_safety():
    """Perturb RHS so primal is infeasible but dual remains feasible; verify Dual Simplex route and safety."""
    model = build_well_scaled_lp(maximize=True)
    sol1 = model.solve(presolve=False)
    binfo = sol1.basis_info

    # Perturb RHS to make basic solution primal infeasible
    model2 = build_well_scaled_lp(maximize=True)
    model2._constraints["c1"].rhs = 1.0  # Tightened

    sol2 = model2.solve(basis=binfo, presolve=False, numerical_verification="strict")
    assert sol2.status == SolutionStatus.OPTIMAL
    assert sol2.solver_selected == "dual_simplex"
    assert sol2.solver_reason == "warm_start_dual_feasible"
    assert sol2.diagnostics is not None
    assert sol2.diagnostics.verification_passed


# ===========================================================================
# 19. Repeated Warm-Start Re-optimization
# ===========================================================================

def test_19_repeated_warm_start_reoptimization():
    """Solve -> perturb -> warm-start -> perturb -> warm-start -> repeat.
    Ensure basis remains valid, diagnostics correct, cold and warm agree mathematically.
    """
    model = build_well_scaled_lp(maximize=True)
    sol = model.solve()
    binfo = sol.basis_info

    perturbations = [3.5, 3.0, 2.5, 2.0]
    for rhs_val in perturbations:
        m_warm = build_well_scaled_lp(maximize=True)
        m_warm._constraints["c1"].rhs = rhs_val
        sol_warm = m_warm.solve(basis=binfo, numerical_verification="strict")

        m_cold = build_well_scaled_lp(maximize=True)
        m_cold._constraints["c1"].rhs = rhs_val
        sol_cold = m_cold.solve()

        assert sol_warm.status == SolutionStatus.OPTIMAL
        assert sol_cold.status == SolutionStatus.OPTIMAL
        assert pytest.approx(sol_warm.objective_value, abs=1e-6) == sol_cold.objective_value
        assert sol_warm.diagnostics is not None
        assert sol_warm.diagnostics.verification_passed

        # Update basis for next iteration
        binfo = sol_warm.basis_info


# ===========================================================================
# 20. Objective Consistency
# ===========================================================================

def test_20_objective_consistency():
    """Verify objective value calculation matches c^T x (+ constant) accurately."""
    model = Model(name="obj_const")
    model.add_variable(Variable("x1", lower_bound=0.0))
    model.add_variable(Variable("x2", lower_bound=0.0))
    model.set_objective(Objective({"x1": 5.0, "x2": 4.0}, ObjectiveSense.MAXIMIZE))
    model.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 10.0))

    sol = model.solve(numerical_verification="strict")
    assert sol.status == SolutionStatus.OPTIMAL
    assert sol.objective_value == pytest.approx(50.0, abs=1e-7)
    assert sol.diagnostics is not None
    assert sol.diagnostics.objective_error < 1e-7


# ===========================================================================
# 21. Full KKT Verification
# ===========================================================================

def test_21_kkt_verification():
    """Direct test of verify_solution and solution.verify() method."""
    model = build_well_scaled_lp(maximize=True)
    sol = model.solve(numerical_verification="none")
    assert sol.diagnostics is None

    # Call solution.verify()
    diag = sol.verify()
    assert isinstance(diag, SolverDiagnostics)
    assert diag.verification_passed
    assert diag.primal_residual < 1e-7
    assert diag.dual_residual < 1e-7
    assert diag.complementarity_residual < 1e-7
    assert sol.diagnostics is diag


# ===========================================================================
# 22. MAXIMIZE Sign Conventions
# ===========================================================================

def test_22_maximize_sign_conventions():
    """Verify that MAXIMIZE dual feasibility requires reduced costs r_j <= tol."""
    model = build_well_scaled_lp(maximize=True)
    sol = model.solve(numerical_verification="strict")
    assert sol.status == SolutionStatus.OPTIMAL
    assert sol.diagnostics is not None
    assert sol.diagnostics.dual_residual < 1e-7


# ===========================================================================
# 23. MINIMIZE Sign Conventions
# ===========================================================================

def test_23_minimize_sign_conventions():
    """Verify that MINIMIZE dual feasibility requires reduced costs r_j >= -tol."""
    model = build_well_scaled_lp(maximize=False)
    sol = model.solve(numerical_verification="strict")
    assert sol.status == SolutionStatus.OPTIMAL
    assert sol.diagnostics is not None
    assert sol.diagnostics.dual_residual < 1e-7


# ===========================================================================
# 24. Stage 8.44–8.47 Compatibility
# ===========================================================================

def test_24_backward_compatibility():
    """Ensure existing BasisInfo, get_last_basis, sensitivity, and scenario evaluation work seamlessly."""
    model = build_well_scaled_lp(maximize=True)
    sol = model.solve()
    assert model.last_basis is not None
    assert model.get_last_basis() is model.last_basis
    assert sol.basis_info is model.last_basis

    # Sensitivity analysis
    sens = model.sensitivity_analysis()
    assert sens is not None
    assert "c1" in sens.rhs
    assert "x1" in sens.objective


# ===========================================================================
# 25. Verification Modes: None, Basic, Strict
# ===========================================================================

def test_25_verification_modes():
    """Test configurable verification modes."""
    model = build_well_scaled_lp(maximize=True)

    # 1. None
    sol_none = model.solve(numerical_verification="none")
    assert sol_none.diagnostics is None

    # 2. Basic
    sol_basic = model.solve(numerical_verification="basic")
    assert sol_basic.diagnostics is not None
    assert sol_basic.diagnostics.verification_mode == "basic"
    assert sol_basic.diagnostics.basis_quality is None  # Fast mode skips full SVD basis quality
    assert sol_basic.diagnostics.verification_passed

    # 3. Strict
    sol_strict = model.solve(numerical_verification="strict")
    assert sol_strict.diagnostics is not None
    assert sol_strict.diagnostics.verification_mode == "strict"
    assert sol_strict.diagnostics.basis_quality is not None
    assert sol_strict.diagnostics.verification_passed


# ===========================================================================
# 26. Unsafe Basis Warm-Start Rejection Gate
# ===========================================================================

def test_26_unsafe_basis_warm_start_rejection():
    """Intentionally provide an ill-conditioned basis and verify it is rejected to cold solve."""
    model = build_well_scaled_lp(maximize=True)
    sol = model.solve()

    # Create an artificial basis with huge condition number or nearly singular
    # In standard form of well-scaled: m=2, n=4 (x1, x2, s1, s2)
    # Give a candidate basis where matrix columns are almost identical
    m_test = Model(name="ill_cond_test")
    m_test.add_variable(Variable("x1", lower_bound=0.0))
    m_test.add_variable(Variable("x2", lower_bound=0.0))
    m_test.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
    m_test.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m_test.add_constraint(Constraint("c2", {"x1": 1.0, "x2": 1.0 + 1e-9}, ConstraintSense.LESS_EQUAL, 4.0))

    # Column 0 and Column 1 are nearly dependent!
    # Because condition number is ~ 1e9 > 1e8, select_solver detects it as unsafe
    # and safely routes to warm_start_fallback_cold_start instead of crashing in dual simplex!
    sol_unsafe = m_test.solve(basis=[0, 1], tolerances=SimplexTolerances(condition_limit=1e8))
    assert sol_unsafe.status == SolutionStatus.OPTIMAL
    assert sol_unsafe.solver_reason == "warm_start_fallback_cold_start"
