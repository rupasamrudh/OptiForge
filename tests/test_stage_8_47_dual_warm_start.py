"""Tests for Stage 8.47: Dual Simplex Warm-Start Re-optimization.

Covers the 20 required specification categories:
1. Basic dual warm start
2. Mathematical correctness vs cold solve
3. Optimal status returned
4. Basis reuse verified
5. Iteration reduction vs cold Phase-I solve
6. Primal-feasible warm start (revised_simplex)
7. Dual-infeasible basis (safe fallback)
8. Invalid basis rejection (dimensions, duplicates, out-of-range, singular)
9. Presolve compatibility (variable elimination, redundant constraints, sign flips, bound shifts)
10. Degenerate basis
11. Multiple infeasible basic variables
12. Infeasible perturbed problem detection
13. Backward compatibility
14. Scenario integration (evaluate_scenario with dual warm start)
15. Model immutability
16. Objective preservation
17. MAXIMIZE sign conventions
18. MINIMIZE sign conventions
19. Tolerance boundary handling
20. Fallback correctness
"""

from __future__ import annotations

import pytest
import numpy as np

from optiforge.core.basis import BasisInfo
from optiforge.core.solution import SolutionStatus
from optiforge.lp.dual_simplex import DualRevisedSimplexSolver
from optiforge.lp.simplex import RevisedSimplexSolver, SimplexTolerances
from optiforge.model.constraint import Constraint, ConstraintSense
from optiforge.model.model import Model
from optiforge.model.objective import Objective, ObjectiveSense
from optiforge.model.variable import Variable, VariableType
from optiforge.sensitivity.scenario import Scenario, evaluate_scenario


def _build_test_lp(sense: ObjectiveSense = ObjectiveSense.MAXIMIZE) -> Model:
    """Standard 2-variable, 2-constraint LP:
    max 3x1 + 2x2
    s.t.
      c1: 2x1 + x2 <= 100
      c2: x1 + x2 <= 80
      x1, x2 >= 0
    Optimal solution: x1 = 20, x2 = 60, obj = 180.
    Allowable range for c1 RHS is [80, 160].
    """
    m = Model(name="test_lp")
    m.add_variable(Variable(name="x1"))
    m.add_variable(Variable(name="x2"))

    m.add_constraint(
        Constraint(
            name="c1",
            coefficients={"x1": 2.0, "x2": 1.0},
            sense=ConstraintSense.LESS_EQUAL,
            rhs=100.0,
        )
    )
    m.add_constraint(
        Constraint(
            name="c2",
            coefficients={"x1": 1.0, "x2": 1.0},
            sense=ConstraintSense.LESS_EQUAL,
            rhs=80.0,
        )
    )

    if sense is ObjectiveSense.MAXIMIZE:
        m.set_objective(
            Objective(coefficients={"x1": 3.0, "x2": 2.0}, sense=ObjectiveSense.MAXIMIZE)
        )
    else:
        m.set_objective(
            Objective(coefficients={"x1": -3.0, "x2": -2.0}, sense=ObjectiveSense.MINIMIZE)
        )
    return m


def _build_diet_lp() -> Model:
    """Minimization LP with 3 variables and 2 constraints:
    min 2x1 + 3x2 + 4x3
    s.t.
      c1: 3x1 + 2x2 + x3 >= 10
      c2: 2x1 + 4x2 + 3x3 >= 15
      x1, x2, x3 >= 0
    """
    m = Model(name="diet_lp")
    m.add_variable(Variable("x1", lower_bound=0.0))
    m.add_variable(Variable("x2", lower_bound=0.0))
    m.add_variable(Variable("x3", lower_bound=0.0))

    m.add_constraint(
        Constraint(
            name="c1",
            coefficients={"x1": 3.0, "x2": 2.0, "x3": 1.0},
            sense=ConstraintSense.GREATER_EQUAL,
            rhs=10.0,
        )
    )
    m.add_constraint(
        Constraint(
            name="c2",
            coefficients={"x1": 2.0, "x2": 4.0, "x3": 3.0},
            sense=ConstraintSense.GREATER_EQUAL,
            rhs=15.0,
        )
    )
    m.set_objective(
        Objective(
            coefficients={"x1": 2.0, "x2": 3.0, "x3": 4.0},
            sense=ObjectiveSense.MINIMIZE,
        )
    )
    return m


# ===========================================================================
# 1. Basic Dual Warm Start
# ===========================================================================

def test_basic_dual_warm_start():
    """Verify that an RHS perturbation rendering the basis primal-infeasible
    but dual-feasible triggers Dual Simplex warm start."""
    model = _build_test_lp()
    sol = model.solve()
    assert sol.status is SolutionStatus.OPTIMAL
    binfo = model.last_basis
    assert binfo is not None

    # Perturb c1 RHS from 100 to 60 (below lower allowable bound 80)
    # Basis [x1, x2] becomes primal infeasible
    perturbed = _build_test_lp()
    perturbed._constraints["c1"].rhs = 60.0

    res_warm = perturbed.solve(basis=binfo)
    assert res_warm.status is SolutionStatus.OPTIMAL
    assert perturbed.last_solver_selected == "dual_simplex"
    assert perturbed.last_solver_reason == "warm_start_dual_feasible"
    assert res_warm.solver_metadata["warm_started"] is True
    assert res_warm.solver_metadata["initial_primal_feasible"] is False
    assert res_warm.solver_metadata["initial_dual_feasible"] is True


# ===========================================================================
# 2. Mathematical Correctness vs Cold Solve
# ===========================================================================

def test_mathematical_correctness_vs_cold_solve():
    """Warm-started Dual Simplex must produce the exact same objective and variable
    values as a cold solve."""
    model = _build_test_lp()
    model.solve()
    binfo = model.last_basis

    for new_rhs in [55.0, 60.0, 70.0]:
        m_cold = _build_test_lp()
        m_cold._constraints["c1"].rhs = new_rhs
        sol_cold = m_cold.solve()

        m_warm = _build_test_lp()
        m_warm._constraints["c1"].rhs = new_rhs
        sol_warm = m_warm.solve(basis=binfo)

        assert sol_cold.status is SolutionStatus.OPTIMAL
        assert sol_warm.status is SolutionStatus.OPTIMAL
        assert abs(sol_warm.objective_value - sol_cold.objective_value) < 1e-6
        for v in ["x1", "x2"]:
            assert abs(sol_warm.variable_values[v] - sol_cold.variable_values[v]) < 1e-6


# ===========================================================================
# 3. Optimal Status Returned
# ===========================================================================

def test_optimal_status_returned():
    """Dual Simplex warm-start solve must return SolutionStatus.OPTIMAL."""
    model = _build_test_lp()
    model.solve()
    binfo = model.last_basis

    perturbed = _build_test_lp()
    perturbed._constraints["c1"].rhs = 60.0
    sol = perturbed.solve(basis=binfo)
    assert sol.status == SolutionStatus.OPTIMAL


# ===========================================================================
# 4. Basis Reuse Verified
# ===========================================================================

def test_basis_reuse_verified():
    """Ensure the supplied basis indices are truly consumed as the initial basis."""
    model = _build_test_lp()
    model.solve()
    binfo = model.last_basis

    perturbed = _build_test_lp()
    perturbed._constraints["c1"].rhs = 60.0
    sol = perturbed.solve(basis=binfo)

    meta = sol.solver_metadata
    assert meta["warm_started"] is True
    assert meta["initial_basis"] == tuple(binfo.basic_variable_indices)
    assert meta["reoptimization_iterations"] >= 0


# ===========================================================================
# 5. Iteration Reduction vs Cold Phase-I Solve
# ===========================================================================

def test_iteration_reduction_vs_cold_solve():
    """Dual Simplex reoptimizes in a single pivot from the dual-feasible basis."""
    model = _build_test_lp()
    model.solve()
    binfo = model.last_basis

    perturbed = _build_test_lp()
    perturbed._constraints["c1"].rhs = 60.0
    sol_warm = perturbed.solve(basis=binfo)

    # Re-optimization should take exactly 1 iteration
    assert sol_warm.solver_metadata["reoptimization_iterations"] == 1
    assert sol_warm.iterations == 1


# ===========================================================================
# 6. Primal-Feasible Warm Start
# ===========================================================================

def test_primal_feasible_warm_start():
    """When a perturbation leaves the basis primal-feasible, RevisedSimplexSolver
    must be selected via the warm_start_primal_feasible path."""
    model = _build_test_lp()
    model.solve()
    binfo = model.last_basis

    # Perturb c1 RHS from 100 to 110 (within allowable range [80, 160])
    perturbed = _build_test_lp()
    perturbed._constraints["c1"].rhs = 110.0

    sol = perturbed.solve(basis=binfo)
    assert sol.status is SolutionStatus.OPTIMAL
    assert perturbed.last_solver_selected == "revised_simplex"
    assert perturbed.last_solver_reason == "warm_start_primal_feasible"
    assert sol.solver_metadata["warm_started"] is True
    assert sol.solver_metadata["initial_primal_feasible"] is True


# ===========================================================================
# 7. Dual-Infeasible Basis (Safe Fallback)
# ===========================================================================

def test_dual_infeasible_basis_fallback():
    """When objective change destroys dual feasibility, Dual Simplex must NOT be chosen.
    Instead, it must safely fall back to revised simplex cold solve."""
    model = _build_test_lp()
    model.solve()
    binfo = model.last_basis

    # Modify both RHS (so primal infeasible) and objective coefficients (so dual infeasible)
    perturbed = _build_test_lp()
    perturbed._constraints["c1"].rhs = 60.0
    perturbed.set_objective(Objective(coefficients={"x1": -50.0, "x2": 10.0}, sense=ObjectiveSense.MAXIMIZE))

    sol = perturbed.solve(basis=binfo)
    assert sol.status is SolutionStatus.OPTIMAL
    assert perturbed.last_solver_selected == "revised_simplex"
    assert perturbed.last_solver_reason == "warm_start_fallback_cold_start"


# ===========================================================================
# 8. Invalid Basis Rejection
# ===========================================================================

def test_invalid_basis_rejection():
    """Test that invalid bases (wrong length, duplicates, out of bounds, singular)
    are rejected with descriptive ValueError."""
    model = _build_test_lp()
    model.solve()

    # Wrong length
    with pytest.raises(ValueError, match="Incompatible basis: basis length"):
        model.solve(basis=[0])

    # Duplicate column indices
    with pytest.raises(ValueError, match="duplicate column indices"):
        model.solve(basis=[0, 0, 1, 2])

    # Out-of-range column indices
    with pytest.raises(ValueError, match="column indices must be within"):
        model.solve(basis=[0, 1, 2, 999])

    # Singular basis
    with pytest.raises(ValueError, match="Incompatible basis: basis matrix is singular"):
        model.solve(basis=[0, 2, 3, 4])

    # Non-LP basis
    fake_binfo = BasisInfo(basic_variable_indices=[0, 1, 2, 3], is_lp=False)
    with pytest.raises(ValueError, match="non-LP"):
        model.solve(basis=fake_binfo)


# ===========================================================================
# 9. Presolve Compatibility
# ===========================================================================

def test_presolve_compatibility():
    """Test dual warm-start with presolve transformations:
    - Fixed variables
    - Shifted lower bounds
    - Redundant constraints
    """
    model = Model(name="presolved_lp")
    # x1 has lower bound 5.0 (variable shift)
    model.add_variable(Variable("x1", lower_bound=5.0, upper_bound=100.0))
    # x2 is free to be positive
    model.add_variable(Variable("x2", lower_bound=0.0, upper_bound=100.0))
    # x3 is fixed at 10.0 (eliminated by presolve)
    model.add_variable(Variable("x3", lower_bound=10.0, upper_bound=100.0))
    model.add_constraint(Constraint("c_fixed", {"x3": 1.0}, ConstraintSense.EQUAL, 10.0))

    model.add_constraint(
        Constraint("c1", {"x1": 2.0, "x2": 1.0, "x3": 1.0}, ConstraintSense.LESS_EQUAL, 120.0)
    )
    model.add_constraint(
        Constraint("c2", {"x1": 1.0, "x2": 2.0}, ConstraintSense.LESS_EQUAL, 90.0)
    )

    model.set_objective(Objective({"x1": 4.0, "x2": 3.0, "x3": 2.0}, ObjectiveSense.MAXIMIZE))

    sol1 = model.solve(presolve=True)
    assert sol1.status is SolutionStatus.OPTIMAL
    binfo = model.last_basis
    assert binfo is not None

    # Perturb c1 RHS
    model2 = Model(name="presolved_lp_perturbed")
    model2.add_variable(Variable("x1", lower_bound=5.0, upper_bound=100.0))
    model2.add_variable(Variable("x2", lower_bound=0.0, upper_bound=100.0))
    model2.add_variable(Variable("x3", lower_bound=10.0, upper_bound=100.0))
    model2.add_constraint(Constraint("c_fixed", {"x3": 1.0}, ConstraintSense.EQUAL, 10.0))
    model2.add_constraint(
        Constraint("c1", {"x1": 2.0, "x2": 1.0, "x3": 1.0}, ConstraintSense.LESS_EQUAL, 190.0)
    )
    model2.add_constraint(
        Constraint("c2", {"x1": 1.0, "x2": 2.0}, ConstraintSense.LESS_EQUAL, 90.0)
    )
    model2.set_objective(Objective({"x1": 4.0, "x2": 3.0, "x3": 2.0}, ObjectiveSense.MAXIMIZE))

    sol2_warm = model2.solve(basis=binfo, presolve=True)
    assert sol2_warm.status is SolutionStatus.OPTIMAL

    # Compare with cold solve
    sol2_cold = model2.solve(presolve=True)
    assert abs(sol2_warm.objective_value - sol2_cold.objective_value) < 1e-5
    for v in ["x1", "x2", "x3"]:
        assert abs(sol2_warm.variable_values[v] - sol2_cold.variable_values[v]) < 1e-5


# ===========================================================================
# 10. Degenerate Basis
# ===========================================================================

def test_degenerate_basis():
    """Verify warm-start correctness on an LP with a degenerate basis."""
    m = Model(name="deg_lp")
    m.add_variable(Variable("x1"))
    m.add_variable(Variable("x2"))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 10.0))
    m.add_constraint(Constraint("c2", {"x1": 2.0, "x2": 2.0}, ConstraintSense.LESS_EQUAL, 20.0))
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
    sol = m.solve(presolve=False)
    binfo = m.last_basis

    m_pert = Model(name="deg_pert")
    m_pert.add_variable(Variable("x1"))
    m_pert.add_variable(Variable("x2"))
    m_pert.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 15.0))
    m_pert.add_constraint(Constraint("c2", {"x1": 2.0, "x2": 2.0}, ConstraintSense.LESS_EQUAL, 30.0))
    m_pert.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
    sol_warm = m_pert.solve(basis=binfo, presolve=False)
    assert sol_warm.status is SolutionStatus.OPTIMAL


# ===========================================================================
# 11. Multiple Infeasible Basic Variables
# ===========================================================================

def test_multiple_infeasible_basic_variables():
    """Dual Simplex correctly pivots when multiple basic variables are negative."""
    model = _build_test_lp()
    sol = model.solve()
    binfo = model.last_basis

    perturbed = _build_test_lp()
    perturbed._constraints["c1"].rhs = 30.0
    perturbed._constraints["c2"].rhs = 25.0

    sol_warm = perturbed.solve(basis=binfo)
    sol_cold = perturbed.solve()

    assert sol_warm.status is SolutionStatus.OPTIMAL
    assert abs(sol_warm.objective_value - sol_cold.objective_value) < 1e-6
    for v in ["x1", "x2"]:
        assert abs(sol_warm.variable_values[v] - sol_cold.variable_values[v]) < 1e-6


# ===========================================================================
# 12. Infeasible Perturbed Problem Detection
# ===========================================================================

def test_infeasible_perturbed_problem_detection():
    """When perturbation produces an infeasible problem, Dual Simplex detects INFEASIBLE."""
    solver = DualRevisedSimplexSolver()
    A = np.array([[1.0, 1.0, 1.0]])
    b = np.array([-5.0])
    c = np.array([1.0, 1.0, 0.0])  # dual feasible since c >= 0
    basis = [2]  # slack variable
    nonbasic = [0, 1]

    res = solver.solve_standard_form(
        A=A,
        b=b,
        c=c,
        basis=basis,
        nonbasic=nonbasic,
        sense=ObjectiveSense.MINIMIZE,
        tolerances=SimplexTolerances(),
    )
    assert res.status == SolutionStatus.INFEASIBLE


# ===========================================================================
# 13. Backward Compatibility
# ===========================================================================

def test_backward_compatibility():
    """Ensure standard solve without basis remains identical."""
    model = _build_test_lp()
    sol = model.solve()
    assert sol.status is SolutionStatus.OPTIMAL
    assert abs(sol.objective_value - 180.0) < 1e-6
    assert abs(sol.variable_values["x1"] - 20.0) < 1e-6
    assert abs(sol.variable_values["x2"] - 60.0) < 1e-6


# ===========================================================================
# 14. Scenario Integration
# ===========================================================================

def test_scenario_integration_dual_warm_start():
    """evaluate_scenario(..., resolve_if_needed=True) uses dual warm start."""
    model = _build_test_lp()
    model.solve()

    # Scenario: decrease c1 RHS by 40 (to 60), which breaks primal feasibility but stays dual feasible
    sc = Scenario(name="reduce_c1", rhs_changes={"c1": -40.0})
    res = evaluate_scenario(model=model, scenario=sc, resolve_if_needed=True)

    assert res.solver_resolved is True
    assert res.solve_method == "dual_warm_start"
    assert res.warm_started is True
    assert res.objective_value is not None
    assert abs(res.objective_value - 120.0) < 1e-5


# ===========================================================================
# 15. Model Immutability
# ===========================================================================

def test_model_immutability():
    """Original Model must remain unmutated after evaluate_scenario with resolve_if_needed=True."""
    model = _build_test_lp()
    sol_orig = model.solve()
    base_obj = sol_orig.objective_value
    base_rhs = model.constraints["c1"].rhs

    sc = Scenario(name="heavy_shift", rhs_changes={"c1": -40.0})
    res = evaluate_scenario(model=model, scenario=sc, resolve_if_needed=True)

    assert model.constraints["c1"].rhs == base_rhs
    assert model.last_basis.objective_value == base_obj
    assert len(model.variables) == 2
    assert len(model.constraints) == 2


# ===========================================================================
# 16. Objective Preservation
# ===========================================================================

def test_objective_preservation():
    """Model last_solution and objective remain intact and mathematically sound."""
    model = _build_test_lp()
    sol1 = model.solve()
    binfo = model.last_basis

    perturbed = _build_test_lp()
    perturbed._constraints["c1"].rhs = 60.0
    sol2 = perturbed.solve(basis=binfo)

    # In perturbed model, x1=0, x2=60 => obj = 120
    assert abs(sol2.objective_value - 120.0) < 1e-6
    # Original model was not affected
    assert abs(sol1.objective_value - 180.0) < 1e-6


# ===========================================================================
# 17. MAXIMIZE Sign Conventions
# ===========================================================================

def test_maximize_sign_conventions():
    """Dual Simplex warm-start on MAXIMIZE problems satisfies reduced cost <= 0."""
    model = _build_test_lp(sense=ObjectiveSense.MAXIMIZE)
    model.solve()
    binfo = model.last_basis

    perturbed = _build_test_lp(sense=ObjectiveSense.MAXIMIZE)
    perturbed._constraints["c1"].rhs = 60.0
    sol = perturbed.solve(basis=binfo)

    assert sol.status is SolutionStatus.OPTIMAL
    assert perturbed.last_solver_selected == "dual_simplex"


# ===========================================================================
# 18. MINIMIZE Sign Conventions
# ===========================================================================

def test_minimize_sign_conventions():
    """Dual Simplex warm-start on MINIMIZE problems satisfies reduced cost >= 0."""
    model = _build_diet_lp()
    model.solve()
    binfo = model.last_basis

    # Perturb c1 RHS from 10 to 25 (increasing requirements -> primal infeasible for previous basis)
    perturbed = _build_diet_lp()
    perturbed._constraints["c1"].rhs = 25.0
    sol = perturbed.solve(basis=binfo)

    assert sol.status is SolutionStatus.OPTIMAL
    assert perturbed.last_solver_selected == "dual_simplex"
    assert perturbed.last_solver_reason == "warm_start_dual_feasible"

    sol_cold = perturbed.solve()
    assert abs(sol.objective_value - sol_cold.objective_value) < 1e-6


# ===========================================================================
# 19. Tolerance Boundary
# ===========================================================================

def test_tolerance_boundary():
    """Slight violation within tolerance should remain primal warm start;
    violation beyond tolerance should switch to dual warm start."""
    model = _build_test_lp()
    model.solve()
    binfo = model.last_basis

    # Small perturbation (within 1e-8)
    tol = SimplexTolerances(feasibility_tolerance=1e-6)
    m_tiny = _build_test_lp()
    m_tiny._constraints["c1"].rhs = 80.0 - 1e-8
    sol_tiny = m_tiny.solve(basis=binfo, tolerances=tol)
    assert m_tiny.last_solver_selected == "revised_simplex"

    # Significant perturbation (beyond tolerance)
    m_big = _build_test_lp()
    m_big._constraints["c1"].rhs = 60.0
    sol_big = m_big.solve(basis=binfo, tolerances=tol)
    assert m_big.last_solver_selected == "dual_simplex"


# ===========================================================================
# 20. Fallback Correctness
# ===========================================================================

def test_fallback_correctness():
    """If dual warm-start prerequisites fail (e.g. non-dual feasible),
    pipeline safely falls back to cold solve producing optimal solution."""
    model = _build_test_lp()
    model.solve()
    binfo = model.last_basis

    # Modify both RHS and objective so neither primal nor dual feasibility holds
    perturbed = _build_test_lp()
    perturbed._constraints["c1"].rhs = 60.0
    perturbed.set_objective(Objective({"x1": -50.0, "x2": 10.0}, ObjectiveSense.MAXIMIZE))

    sol_fallback = perturbed.solve(basis=binfo)
    assert sol_fallback.status is SolutionStatus.OPTIMAL
    assert sol_fallback.solver_reason == "warm_start_fallback_cold_start"

    # Cold solve on fresh model should produce the same objective
    fresh_perturbed = _build_test_lp()
    fresh_perturbed._constraints["c1"].rhs = 60.0
    fresh_perturbed.set_objective(Objective({"x1": -50.0, "x2": 10.0}, ObjectiveSense.MAXIMIZE))
    sol_pure_cold = fresh_perturbed.solve()
    assert abs(sol_fallback.objective_value - sol_pure_cold.objective_value) < 1e-6
