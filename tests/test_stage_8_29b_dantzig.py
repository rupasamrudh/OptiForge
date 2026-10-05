"""Stage 8.29B: Dantzig Pricing & Anti-Cycling Implementation Tests.

Verifies:
- Beale (1955) cycling problem termination with anti-cycling safeguards
- Demonstrable triggering of Basis Repetition Detection (Safeguard B)
- Demonstrable triggering of Consecutive Degenerate Pivot Counter (Safeguard A)
- Pure Dantzig cycling behavior when safeguards are disabled
- Exact micro-correctness of _select_entering_variable for MIN and MAX
- Boundary tolerances, ties, and basic variable exclusion
- Phase I -> Phase II pricing inheritance
- Status correctness: OPTIMAL, INFEASIBLE, UNBOUNDED, ITERATION_LIMIT
- Numerical consistency across diverse problem classes
- MILP continuous root LP relaxation equivalence
"""

import numpy as np
import pytest

from optiforge.core.solution import SolutionStatus
from optiforge.lp.simplex import RevisedSimplexSolver, SimplexTolerances
from optiforge.model import (
    Constraint,
    ConstraintSense,
    Model,
    Objective,
    ObjectiveSense,
    Variable,
    VariableType,
)


def make_beale_model() -> Model:
    """Construct Beale's (1955) classic 3x7 degenerate cycling problem as a Model.

    min   -0.75 x0 + 20 x1 - 0.5 x2 + 6 x3
    s.t.
          0.25 x0 -  8 x1 -     x2 + 9 x3 <= 0  (slack x4)
           0.5 x0 - 12 x1 - 0.5 x2 + 3 x3 <= 0  (slack x5)
                                x2        <= 1  (slack x6)
          x0..x3 >= 0
    Slacks x4, x5, x6 form the initial BFS at x = (0, 0, 0, 0, 0, 0, 1).
    Theoretical optimum: z* = -1.25.
    """
    m = Model(name="beale_cycling")
    x0 = Variable("x0", lower_bound=0.0)
    x1 = Variable("x1", lower_bound=0.0)
    x2 = Variable("x2", lower_bound=0.0)
    x3 = Variable("x3", lower_bound=0.0)
    for v in [x0, x1, x2, x3]:
        m.add_variable(v)
    m.set_objective(Objective({"x0": -0.75, "x1": 20.0, "x2": -0.5, "x3": 6.0}, ObjectiveSense.MINIMIZE))
    m.add_constraint(Constraint("c1", {"x0": 0.25, "x1": -8.0, "x2": -1.0, "x3": 9.0}, ConstraintSense.LESS_EQUAL, 0.0))
    m.add_constraint(Constraint("c2", {"x0": 0.5, "x1": -12.0, "x2": -0.5, "x3": 3.0}, ConstraintSense.LESS_EQUAL, 0.0))
    m.add_constraint(Constraint("c3", {"x2": 1.0}, ConstraintSense.LESS_EQUAL, 1.0))
    return m


# ============================================================================
# PART 1: BEALE CYCLING & ANTI-CYCLING SAFEGUARD REGRESSION
# ============================================================================

def test_production_default_is_bland():
    """Verify that production default pricing remains Bland's rule."""
    tol = SimplexTolerances()
    assert tol.pricing == "bland"
    solver = RevisedSimplexSolver()
    assert solver.tolerances.pricing == "bland"


def test_beale_cycling_bland_terminates():
    """Verify that reference Bland rule solves Beale's problem in 6 pivots."""
    model = make_beale_model()
    solver = RevisedSimplexSolver(pricing="bland")
    sol = solver.solve(model)

    assert sol.status == SolutionStatus.OPTIMAL
    assert sol.objective_value == pytest.approx(-1.25, abs=1e-7)
    assert sol.iterations == 6
    assert not solver.fallback_triggered


def test_beale_cycling_pure_dantzig_fails_without_safeguards():
    """Verify that pure Dantzig (both safeguards disabled) cycles and hits max_iterations."""
    model = make_beale_model()
    tol = SimplexTolerances(
        pricing="dantzig",
        max_iterations=40,
        enable_basis_repetition_detection=False,
        enable_degenerate_pivot_fallback=False,
    )
    solver = RevisedSimplexSolver(tolerances=tol)
    sol = solver.solve(model)

    # Pure Dantzig without safeguards must fail to terminate (hit iteration limit)
    assert sol.status == SolutionStatus.ITERATION_LIMIT
    assert sol.iterations == 40
    assert not solver.fallback_triggered


def test_beale_cycling_safeguard_basis_repetition_triggers():
    """Verify Dantzig with default safeguards terminates on Beale via basis repetition detection."""
    model = make_beale_model()
    tol = SimplexTolerances(
        pricing="dantzig",
        max_iterations=100,
        enable_basis_repetition_detection=True,
        enable_degenerate_pivot_fallback=True,
    )
    solver = RevisedSimplexSolver(tolerances=tol)
    sol = solver.solve(model)

    assert sol.status == SolutionStatus.OPTIMAL
    assert sol.objective_value == pytest.approx(-1.25, abs=1e-7)
    assert sol.iterations <= 20
    # Safeguard B (Basis repetition) must be demonstrably triggered
    assert solver.fallback_triggered is True
    assert "basis repetition" in solver.fallback_reason.lower()


def test_beale_cycling_safeguard_degenerate_counter_triggers():
    """Verify Dantzig terminates via consecutive degenerate pivot counter when basis repetition is disabled."""
    model = make_beale_model()
    tol = SimplexTolerances(
        pricing="dantzig",
        max_iterations=100,
        max_consecutive_degenerate_pivots=5,  # Trigger early at 5 degenerate pivots
        enable_basis_repetition_detection=False,
        enable_degenerate_pivot_fallback=True,
    )
    solver = RevisedSimplexSolver(tolerances=tol)
    sol = solver.solve(model)

    assert sol.status == SolutionStatus.OPTIMAL
    assert sol.objective_value == pytest.approx(-1.25, abs=1e-7)
    assert sol.iterations <= 20
    # Safeguard A (Degenerate counter) must be demonstrably triggered
    assert solver.fallback_triggered is True
    assert "degenerate pivots" in solver.fallback_reason.lower()


# ============================================================================
# PART 2: PRICING INTERFACE VALIDATION
# ============================================================================

def test_pricing_validation():
    """Unsupported pricing values must be rejected cleanly with ValueError."""
    with pytest.raises(ValueError, match="Unsupported pricing strategy"):
        SimplexTolerances(pricing="invalid_pricing")

    with pytest.raises(ValueError, match="Unsupported pricing strategy"):
        RevisedSimplexSolver(pricing="steepest_edge")

    solver = RevisedSimplexSolver()
    with pytest.raises(ValueError, match="Unsupported pricing strategy"):
        solver.solve(make_beale_model(), pricing="devex")


# ============================================================================
# PART 3: MICRO-CORRECTNESS OF ENTERING VARIABLE SELECTION
# ============================================================================

def test_micro_correctness_minimize():
    """Unit tests for _select_entering_variable under MINIMIZE sense."""
    tol = 1e-7
    nonbasic = [1, 2, 4, 5]

    # One improving candidate
    r1 = np.zeros(6)
    r1[2] = -2.0
    assert RevisedSimplexSolver._select_entering_variable("bland", r1, nonbasic, ObjectiveSense.MINIMIZE, tol) == 2
    assert RevisedSimplexSolver._select_entering_variable("dantzig", r1, nonbasic, ObjectiveSense.MINIMIZE, tol) == 2

    # Multiple improving candidates (Bland chooses first index, Dantzig chooses most negative)
    r2 = np.zeros(6)
    r2[1] = -1.0
    r2[4] = -5.0
    assert RevisedSimplexSolver._select_entering_variable("bland", r2, nonbasic, ObjectiveSense.MINIMIZE, tol) == 1
    assert RevisedSimplexSolver._select_entering_variable("dantzig", r2, nonbasic, ObjectiveSense.MINIMIZE, tol) == 4

    # Exact tolerance boundary (-1e-7 is not strictly < -1e-7 -> None)
    r_bound = np.zeros(6)
    r_bound[1] = -1e-7
    assert RevisedSimplexSolver._select_entering_variable("bland", r_bound, nonbasic, ObjectiveSense.MINIMIZE, tol) is None
    assert RevisedSimplexSolver._select_entering_variable("dantzig", r_bound, nonbasic, ObjectiveSense.MINIMIZE, tol) is None

    # Inside tolerance (-1.001e-7 is strictly < -1e-7 -> selected)
    r_inside = np.zeros(6)
    r_inside[1] = -1.001e-7
    assert RevisedSimplexSolver._select_entering_variable("bland", r_inside, nonbasic, ObjectiveSense.MINIMIZE, tol) == 1
    assert RevisedSimplexSolver._select_entering_variable("dantzig", r_inside, nonbasic, ObjectiveSense.MINIMIZE, tol) == 1

    # Tied minimal candidates: Dantzig tie-breaking chooses lowest index
    r_tie = np.zeros(6)
    r_tie[2] = -3.0
    r_tie[5] = -3.0
    assert RevisedSimplexSolver._select_entering_variable("dantzig", r_tie, nonbasic, ObjectiveSense.MINIMIZE, tol) == 2

    # Basic variable exclusion: variable 0 has large negative reduced cost but is NOT in nonbasic
    r_basic = np.zeros(6)
    r_basic[0] = -100.0
    r_basic[2] = -2.0
    assert RevisedSimplexSolver._select_entering_variable("bland", r_basic, nonbasic, ObjectiveSense.MINIMIZE, tol) == 2
    assert RevisedSimplexSolver._select_entering_variable("dantzig", r_basic, nonbasic, ObjectiveSense.MINIMIZE, tol) == 2

    # All non-negative: optimal
    r_opt = np.array([0.0, 1.0, 2.0, 0.0, 0.5, 3.0])
    assert RevisedSimplexSolver._select_entering_variable("bland", r_opt, nonbasic, ObjectiveSense.MINIMIZE, tol) is None
    assert RevisedSimplexSolver._select_entering_variable("dantzig", r_opt, nonbasic, ObjectiveSense.MINIMIZE, tol) is None


def test_micro_correctness_maximize():
    """Unit tests for _select_entering_variable under MAXIMIZE sense."""
    tol = 1e-7
    nonbasic = [1, 2, 4, 5]

    # One improving candidate
    r1 = np.zeros(6)
    r1[2] = 2.0
    assert RevisedSimplexSolver._select_entering_variable("bland", r1, nonbasic, ObjectiveSense.MAXIMIZE, tol) == 2
    assert RevisedSimplexSolver._select_entering_variable("dantzig", r1, nonbasic, ObjectiveSense.MAXIMIZE, tol) == 2

    # Multiple improving candidates (Bland chooses first index, Dantzig chooses most positive)
    r2 = np.zeros(6)
    r2[1] = 1.0
    r2[4] = 5.0
    assert RevisedSimplexSolver._select_entering_variable("bland", r2, nonbasic, ObjectiveSense.MAXIMIZE, tol) == 1
    assert RevisedSimplexSolver._select_entering_variable("dantzig", r2, nonbasic, ObjectiveSense.MAXIMIZE, tol) == 4

    # Exact tolerance boundary (+1e-7 is not strictly > +1e-7 -> None)
    r_bound = np.zeros(6)
    r_bound[1] = 1e-7
    assert RevisedSimplexSolver._select_entering_variable("bland", r_bound, nonbasic, ObjectiveSense.MAXIMIZE, tol) is None
    assert RevisedSimplexSolver._select_entering_variable("dantzig", r_bound, nonbasic, ObjectiveSense.MAXIMIZE, tol) is None

    # Inside tolerance (+1.001e-7 is strictly > +1e-7 -> selected)
    r_inside = np.zeros(6)
    r_inside[1] = 1.001e-7
    assert RevisedSimplexSolver._select_entering_variable("bland", r_inside, nonbasic, ObjectiveSense.MAXIMIZE, tol) == 1
    assert RevisedSimplexSolver._select_entering_variable("dantzig", r_inside, nonbasic, ObjectiveSense.MAXIMIZE, tol) == 1

    # Tied maximal candidates: Dantzig tie-breaking chooses lowest index
    r_tie = np.zeros(6)
    r_tie[2] = 4.0
    r_tie[5] = 4.0
    assert RevisedSimplexSolver._select_entering_variable("dantzig", r_tie, nonbasic, ObjectiveSense.MAXIMIZE, tol) == 2

    # Basic variable exclusion: variable 0 has large positive reduced cost but is NOT nonbasic
    r_basic = np.zeros(6)
    r_basic[0] = 100.0
    r_basic[2] = 2.0
    assert RevisedSimplexSolver._select_entering_variable("bland", r_basic, nonbasic, ObjectiveSense.MAXIMIZE, tol) == 2
    assert RevisedSimplexSolver._select_entering_variable("dantzig", r_basic, nonbasic, ObjectiveSense.MAXIMIZE, tol) == 2

    # All non-positive: optimal
    r_opt = np.array([0.0, -1.0, -2.0, 0.0, -0.5, -3.0])
    assert RevisedSimplexSolver._select_entering_variable("bland", r_opt, nonbasic, ObjectiveSense.MAXIMIZE, tol) is None
    assert RevisedSimplexSolver._select_entering_variable("dantzig", r_opt, nonbasic, ObjectiveSense.MAXIMIZE, tol) is None


def test_pricing_choice_rules_comparison():
    """Verify exact behavioral differences and tie rules between Bland and Dantzig.

    - Bland and Dantzig choose identical variable when only one candidate exists.
    - Bland chooses lowest-index candidate.
    - Dantzig chooses strongest candidate.
    - Tied Dantzig candidates choose lowest index.
    """
    tol = 1e-7
    nonbasic = [3, 7, 11]

    # 1. Single candidate -> identical choice
    r_single = np.zeros(15)
    r_single[7] = -2.5
    assert (
        RevisedSimplexSolver._select_entering_variable("bland", r_single, nonbasic, ObjectiveSense.MINIMIZE, tol)
        == RevisedSimplexSolver._select_entering_variable("dantzig", r_single, nonbasic, ObjectiveSense.MINIMIZE, tol)
        == 7
    )

    # 2. Bland chooses lowest index, Dantzig chooses strongest
    r_multi = np.zeros(15)
    r_multi[3] = -1.0  # lowest index
    r_multi[7] = -10.0  # strongest
    r_multi[11] = -5.0
    assert RevisedSimplexSolver._select_entering_variable("bland", r_multi, nonbasic, ObjectiveSense.MINIMIZE, tol) == 3
    assert RevisedSimplexSolver._select_entering_variable("dantzig", r_multi, nonbasic, ObjectiveSense.MINIMIZE, tol) == 7

    # 3. Tied Dantzig candidates choose lowest index
    r_tied = np.zeros(15)
    r_tied[7] = -8.0
    r_tied[11] = -8.0
    assert RevisedSimplexSolver._select_entering_variable("dantzig", r_tied, nonbasic, ObjectiveSense.MINIMIZE, tol) == 7



# ============================================================================
# PART 4: PHASE I / PHASE II INTEGRATION
# ============================================================================

def test_phase1_feasible_model_dantzig():
    """Test feasible LP requiring Phase I solved with Dantzig pricing."""
    m = Model(name="phase1_feasible")
    x1 = Variable("x1", lower_bound=0.0)
    x2 = Variable("x2", lower_bound=0.0)
    m.add_variable(x1)
    m.add_variable(x2)
    # Requires artificial variables (>= constraint)
    m.add_constraint(Constraint("c1", {"x1": 2.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, 6.0))
    m.add_constraint(Constraint("c2", {"x1": 1.0, "x2": 2.0}, ConstraintSense.GREATER_EQUAL, 8.0))
    m.set_objective(Objective({"x1": 3.0, "x2": 4.0}, ObjectiveSense.MINIMIZE))

    s_bland = RevisedSimplexSolver(pricing="bland")
    sol_bland = s_bland.solve(m)

    s_dantzig = RevisedSimplexSolver(pricing="dantzig")
    sol_dantzig = s_dantzig.solve(m)

    assert sol_bland.status == SolutionStatus.OPTIMAL
    assert sol_dantzig.status == SolutionStatus.OPTIMAL
    assert sol_dantzig.objective_value == pytest.approx(sol_bland.objective_value, abs=1e-7)
    assert sol_dantzig.variable_values["x1"] == pytest.approx(sol_bland.variable_values["x1"], abs=1e-6)
    assert sol_dantzig.variable_values["x2"] == pytest.approx(sol_bland.variable_values["x2"], abs=1e-6)


def test_phase1_infeasible_model_dantzig():
    """Test infeasible LP requiring Phase I correctly identified as INFEASIBLE under Dantzig."""
    m = Model(name="phase1_infeasible")
    x1 = Variable("x1", lower_bound=0.0)
    m.add_variable(x1)
    m.add_constraint(Constraint("c1", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 2.0))
    m.add_constraint(Constraint("c2", {"x1": 1.0}, ConstraintSense.GREATER_EQUAL, 5.0))
    m.set_objective(Objective({"x1": 1.0}, ObjectiveSense.MINIMIZE))

    s_bland = RevisedSimplexSolver(pricing="bland")
    sol_bland = s_bland.solve(m)

    s_dantzig = RevisedSimplexSolver(pricing="dantzig")
    sol_dantzig = s_dantzig.solve(m)

    assert sol_bland.status == SolutionStatus.INFEASIBLE
    assert sol_dantzig.status == SolutionStatus.INFEASIBLE


def test_phase1_artificial_variable_elimination():
    """Test Phase I cleanup when redundant constraints leave artificial variables basic at zero."""
    m = Model(name="redundant_phase1")
    x1 = Variable("x1", lower_bound=0.0)
    x2 = Variable("x2", lower_bound=0.0)
    m.add_variable(x1)
    m.add_variable(x2)
    # Equality constraints where the second is redundant (c2 = 2 * c1)
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 2.0}, ConstraintSense.EQUAL, 4.0))
    m.add_constraint(Constraint("c2", {"x1": 2.0, "x2": 4.0}, ConstraintSense.EQUAL, 8.0))
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MINIMIZE))

    s_b = RevisedSimplexSolver(pricing="bland")
    sol_b = s_b.solve(m)

    s_d = RevisedSimplexSolver(pricing="dantzig")
    sol_d = s_d.solve(m)

    assert sol_b.status == SolutionStatus.OPTIMAL
    assert sol_d.status == SolutionStatus.OPTIMAL
    assert sol_d.objective_value == pytest.approx(sol_b.objective_value, abs=1e-7)
    assert not s_d.fallback_triggered


def test_direct_phase2_ordinary_optimization():
    """Test ordinary LP optimization not requiring Phase I (all <= with positive RHS)."""
    m = Model(name="ordinary_phase2")
    x1 = Variable("x1", lower_bound=0.0)
    x2 = Variable("x2", lower_bound=0.0)
    m.add_variable(x1)
    m.add_variable(x2)
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 2.0}, ConstraintSense.LESS_EQUAL, 10.0))
    m.add_constraint(Constraint("c2", {"x1": 3.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 15.0))
    m.set_objective(Objective({"x1": 2.0, "x2": 3.0}, ObjectiveSense.MAXIMIZE))

    s_b = RevisedSimplexSolver(pricing="bland")
    sol_b = s_b.solve(m)

    s_d = RevisedSimplexSolver(pricing="dantzig")
    sol_d = s_d.solve(m)

    assert sol_b.status == SolutionStatus.OPTIMAL
    assert sol_d.status == SolutionStatus.OPTIMAL
    assert sol_d.objective_value == pytest.approx(sol_b.objective_value, abs=1e-7)
    assert not s_d.fallback_triggered


def test_phase1_to_phase2_pricing_preserved():
    """Verify that Dantzig pricing is maintained across Phase I -> Phase II transition."""
    m = Model(name="phase1_to_phase2")
    x1 = Variable("x1", lower_bound=0.0)
    x2 = Variable("x2", lower_bound=0.0)
    x3 = Variable("x3", lower_bound=0.0)
    m.add_variable(x1)
    m.add_variable(x2)
    m.add_variable(x3)
    # Requires Phase I
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0, "x3": 1.0}, ConstraintSense.GREATER_EQUAL, 10.0))
    m.add_constraint(Constraint("c2", {"x1": 2.0, "x2": -1.0, "x3": 1.0}, ConstraintSense.LESS_EQUAL, 20.0))
    m.set_objective(Objective({"x1": 5.0, "x2": 4.0, "x3": 6.0}, ObjectiveSense.MINIMIZE))

    s_d = RevisedSimplexSolver(pricing="dantzig")
    sol_d = s_d.solve(m)

    assert sol_d.status == SolutionStatus.OPTIMAL
    assert not s_d.fallback_triggered
    assert s_d.tolerances.pricing == "dantzig"


# ============================================================================
# PART 5: STATUS REGRESSION (OPTIMAL, INFEASIBLE, UNBOUNDED, ITERATION_LIMIT)
# ============================================================================

def test_unbounded_status_dantzig():
    """Test unbounded model correctly identified as UNBOUNDED under Dantzig."""
    m = Model(name="unbounded_model")
    x1 = Variable("x1", lower_bound=0.0)
    x2 = Variable("x2", lower_bound=0.0)
    m.add_variable(x1)
    m.add_variable(x2)
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": -1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.set_objective(Objective({"x1": 2.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))

    s_bland = RevisedSimplexSolver(pricing="bland")
    sol_bland = s_bland.solve(m)

    s_dantzig = RevisedSimplexSolver(pricing="dantzig")
    sol_dantzig = s_dantzig.solve(m)

    assert sol_bland.status == SolutionStatus.UNBOUNDED
    assert sol_dantzig.status == SolutionStatus.UNBOUNDED


def test_iteration_limit_status_bland_and_dantzig():
    """Test iteration limit correctly handled under both Bland and Dantzig."""
    m = Model(name="iter_limit")
    x1 = Variable("x1", lower_bound=0.0)
    x2 = Variable("x2", lower_bound=0.0)
    m.add_variable(x1)
    m.add_variable(x2)
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.add_constraint(Constraint("c2", {"x1": 2.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.set_objective(Objective({"x1": 3.0, "x2": 2.0}, ObjectiveSense.MAXIMIZE))

    tol_b = SimplexTolerances(pricing="bland", max_iterations=1)
    s_b = RevisedSimplexSolver(tolerances=tol_b)
    sol_b = s_b.solve(m)
    assert sol_b.status == SolutionStatus.ITERATION_LIMIT
    assert sol_b.iterations == 1

    tol_d = SimplexTolerances(pricing="dantzig", max_iterations=1)
    s_d = RevisedSimplexSolver(tolerances=tol_d)
    sol_d = s_d.solve(m)
    assert sol_d.status == SolutionStatus.ITERATION_LIMIT
    assert sol_d.iterations == 1


# ============================================================================
# PART 6: NUMERICAL CONSISTENCY (DENSE, SPARSE, DEGENERATE, ILL-CONDITIONED, HIGH DYNAMIC RANGE)
# ============================================================================

def test_dense_lp_numerical_equivalence():
    """Verify objective and primal feasibility match between Bland and Dantzig on dense LP."""
    rng = np.random.default_rng(101)
    m, n = 30, 60
    A = rng.uniform(0.1, 1.0, size=(m, n))
    b = rng.uniform(5.0, 20.0, size=m)
    c = rng.uniform(-10.0, 10.0, size=n)

    model = Model(name="dense_test")
    vars_list = [Variable(f"x_{j}", lower_bound=0.0) for j in range(n)]
    for v in vars_list:
        model.add_variable(v)
    for i in range(m):
        coeffs = {f"x_{j}": float(A[i, j]) for j in range(n)}
        model.add_constraint(Constraint(f"c_{i}", coeffs, ConstraintSense.LESS_EQUAL, float(b[i])))
    model.set_objective(Objective({f"x_{j}": float(c[j]) for j in range(n)}, ObjectiveSense.MINIMIZE))

    sol_bland = RevisedSimplexSolver(tolerances=SimplexTolerances(pricing="bland", use_pivot_factorization=True)).solve(model)
    sol_dantzig = RevisedSimplexSolver(tolerances=SimplexTolerances(pricing="dantzig", use_pivot_factorization=True)).solve(model)

    assert sol_bland.status == SolutionStatus.OPTIMAL
    assert sol_dantzig.status == SolutionStatus.OPTIMAL
    assert sol_dantzig.objective_value == pytest.approx(sol_bland.objective_value, abs=1e-7)
    assert sol_dantzig.iterations < sol_bland.iterations


def test_sparse_lp_numerical_equivalence():
    """Verify Bland vs Dantzig consistency on a sparse LP (25% density)."""
    rng = np.random.default_rng(42)
    m, n = 30, 60
    density = 0.25
    mask = rng.random((m, n)) < density
    A = np.zeros((m, n), dtype=np.float64)
    A[mask] = rng.uniform(0.1, 1.0, size=np.count_nonzero(mask))
    for i in range(m):
        if np.all(A[i, :] == 0):
            A[i, rng.integers(0, n)] = rng.uniform(0.1, 1.0)
    b = rng.uniform(5.0, 20.0, size=m)
    c = rng.uniform(-10.0, 10.0, size=n)

    model = Model(name="sparse_test")
    for j in range(n):
        model.add_variable(Variable(f"x_{j}", lower_bound=0.0))
    for i in range(m):
        coeffs = {f"x_{j}": float(A[i, j]) for j in range(n) if A[i, j] != 0}
        model.add_constraint(Constraint(f"c_{i}", coeffs, ConstraintSense.LESS_EQUAL, float(b[i])))
    model.set_objective(Objective({f"x_{j}": float(c[j]) for j in range(n)}, ObjectiveSense.MINIMIZE))

    s_b = RevisedSimplexSolver(tolerances=SimplexTolerances(pricing="bland", use_pivot_factorization=True))
    sol_b = s_b.solve(model)
    s_d = RevisedSimplexSolver(tolerances=SimplexTolerances(pricing="dantzig", use_pivot_factorization=True))
    sol_d = s_d.solve(model)

    assert sol_b.status == SolutionStatus.OPTIMAL
    assert sol_d.status == SolutionStatus.OPTIMAL
    assert sol_d.objective_value == pytest.approx(sol_b.objective_value, abs=1e-6)
    assert sol_d.iterations < sol_b.iterations


def test_degenerate_lp_numerical_equivalence():
    """Verify Bland vs Dantzig on a degenerate LP where multiple RHS values are zero."""
    rng = np.random.default_rng(103)
    m, n = 20, 40
    A = rng.uniform(0.1, 1.0, size=(m, n))
    b = rng.uniform(5.0, 15.0, size=m)
    b[:7] = 0.0  # 7 degenerate rows
    c = rng.uniform(-8.0, 8.0, size=n)

    model = Model(name="degenerate_test")
    for j in range(n):
        model.add_variable(Variable(f"x_{j}", lower_bound=0.0))
    for i in range(m):
        coeffs = {f"x_{j}": float(A[i, j]) for j in range(n)}
        model.add_constraint(Constraint(f"c_{i}", coeffs, ConstraintSense.LESS_EQUAL, float(b[i])))
    model.set_objective(Objective({f"x_{j}": float(c[j]) for j in range(n)}, ObjectiveSense.MINIMIZE))

    s_b = RevisedSimplexSolver(pricing="bland")
    sol_b = s_b.solve(model)
    s_d = RevisedSimplexSolver(pricing="dantzig")
    sol_d = s_d.solve(model)

    assert sol_b.status == SolutionStatus.OPTIMAL
    assert sol_d.status == SolutionStatus.OPTIMAL
    assert sol_d.objective_value == pytest.approx(sol_b.objective_value, abs=1e-6)


def test_ill_conditioned_lp_numerical_equivalence():
    """Verify Bland vs Dantzig on an ill-conditioned LP system (condition number ~1e5)."""
    rng = np.random.default_rng(2026)
    m_dim, n_dim = 25, 50
    U, _ = np.linalg.qr(rng.standard_normal((m_dim, m_dim)))
    V, _ = np.linalg.qr(rng.standard_normal((n_dim, n_dim)))
    s_vals = np.logspace(0, -5, m_dim)
    S_mat = np.zeros((m_dim, n_dim))
    for i in range(m_dim):
        S_mat[i, i] = s_vals[i]
    A_ill = np.abs(U @ S_mat @ V.T) + 0.01
    b_ill = rng.uniform(10.0, 50.0, size=m_dim)
    c_ill = rng.uniform(-10.0, 10.0, size=n_dim)

    model = Model(name="ill_conditioned_test")
    for j in range(n_dim):
        model.add_variable(Variable(f"x_{j}", lower_bound=0.0))
    for i in range(m_dim):
        coeffs = {f"x_{j}": float(A_ill[i, j]) for j in range(n_dim)}
        model.add_constraint(Constraint(f"c_{i}", coeffs, ConstraintSense.LESS_EQUAL, float(b_ill[i])))
    model.set_objective(Objective({f"x_{j}": float(c_ill[j]) for j in range(n_dim)}, ObjectiveSense.MINIMIZE))

    s_b = RevisedSimplexSolver(tolerances=SimplexTolerances(pricing="bland", use_pivot_factorization=True))
    sol_b = s_b.solve(model)
    s_d = RevisedSimplexSolver(tolerances=SimplexTolerances(pricing="dantzig", use_pivot_factorization=True))
    sol_d = s_d.solve(model)

    assert sol_b.status == SolutionStatus.OPTIMAL
    assert sol_d.status == SolutionStatus.OPTIMAL
    assert sol_d.objective_value == pytest.approx(sol_b.objective_value, rel=1e-4)


def test_high_dynamic_range_lp_numerical_equivalence():
    """Verify Bland vs Dantzig on LP with coefficient magnitudes ranging across 7 orders of magnitude."""
    m = Model(name="hdr_test")
    x1 = Variable("x1", lower_bound=0.0)
    x2 = Variable("x2", lower_bound=0.0)
    x3 = Variable("x3", lower_bound=0.0)
    m.add_variable(x1)
    m.add_variable(x2)
    m.add_variable(x3)

    m.add_constraint(Constraint("c1", {"x1": 1e4, "x2": 1e-2, "x3": 1.0}, ConstraintSense.LESS_EQUAL, 1e4))
    m.add_constraint(Constraint("c2", {"x1": 1e-3, "x2": 1e3, "x3": 2.0}, ConstraintSense.LESS_EQUAL, 2e3))
    m.add_constraint(Constraint("c3", {"x1": 1.0, "x2": 1.0, "x3": 1e2}, ConstraintSense.LESS_EQUAL, 5e2))
    m.set_objective(Objective({"x1": 1e2, "x2": 1e3, "x3": 1e1}, ObjectiveSense.MAXIMIZE))

    s_b = RevisedSimplexSolver(pricing="bland")
    sol_b = s_b.solve(m)
    s_d = RevisedSimplexSolver(pricing="dantzig")
    sol_d = s_d.solve(m)

    assert sol_b.status == SolutionStatus.OPTIMAL
    assert sol_d.status == SolutionStatus.OPTIMAL
    assert sol_d.objective_value == pytest.approx(sol_b.objective_value, rel=1e-5)


# ============================================================================
# PART 7: MILP ROOT LP RELAXATION CONSISTENCY & DISCRETE INTEGRATION
# ============================================================================

def test_milp_root_lp_relaxation_consistency():
    """Verify continuous root LP relaxation produces identical bounds under Dantzig."""
    # Knapsack-style continuous relaxation
    weights = [2, 3, 5, 7, 1, 4, 1]
    values = [10, 15, 20, 25, 8, 12, 6]
    capacity = 12

    m = Model(name="root_knapsack_rel")
    for j in range(len(weights)):
        m.add_variable(Variable(f"x_{j}", lower_bound=0.0, upper_bound=1.0, variable_type=VariableType.CONTINUOUS))
    m.add_constraint(Constraint("cap", {f"x_{j}": float(weights[j]) for j in range(len(weights))}, ConstraintSense.LESS_EQUAL, float(capacity)))
    m.set_objective(Objective({f"x_{j}": float(values[j]) for j in range(len(weights))}, ObjectiveSense.MAXIMIZE))

    s_b = RevisedSimplexSolver(pricing="bland")
    sol_b = s_b.solve(m)

    s_d = RevisedSimplexSolver(pricing="dantzig")
    sol_d = s_d.solve(m)

    assert sol_b.status == SolutionStatus.OPTIMAL
    assert sol_d.status == SolutionStatus.OPTIMAL
    assert sol_d.objective_value == pytest.approx(sol_b.objective_value, abs=1e-8)


def test_milp_discrete_solve_consistency():
    """Verify that MILP solve with integer variables produces identical objective."""
    from optiforge.milp.solver import MILPSolver

    m = Model(name="knapsack_discrete")
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 8]
    capacity = 8

    for j in range(len(weights)):
        m.add_variable(Variable(f"x_{j}", lower_bound=0.0, upper_bound=1.0, variable_type=VariableType.BINARY))
    m.add_constraint(Constraint("cap", {f"x_{j}": float(weights[j]) for j in range(len(weights))}, ConstraintSense.LESS_EQUAL, float(capacity)))
    m.set_objective(Objective({f"x_{j}": float(values[j]) for j in range(len(weights))}, ObjectiveSense.MAXIMIZE))

    # Solve MILP using default solver (which uses root LP relaxation)
    milp_solver = MILPSolver()
    sol = milp_solver.solve(m)

    assert sol.status == SolutionStatus.OPTIMAL
    assert sol.objective_value == pytest.approx(12.0, abs=1e-6)

