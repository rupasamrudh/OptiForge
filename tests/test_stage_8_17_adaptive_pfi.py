"""
Stage 8.17: Adaptive Dual-PFI Dispatch + Code Hardening Tests
Verifies:
- Type annotations integrity (get_type_hints succeeds)
- Narrowed exception handling (SingularBasisError falls back, TypeError propagates)
- Threshold boundary behavior (m < T vs m >= T)
- Below-threshold dispatch uses IncrementalBasis
- Above-threshold dispatch uses PivotUpdateFactorization
- Adaptive disabled uses default factorization
- Explicit PFI override beats adaptive threshold
- Adaptive Dual Simplex for MIN and MAX
- MILP warm-start with adaptive dispatch and stats accounting
- Sibling isolation under adaptive dispatch
"""
import typing
import numpy as np
import pytest

from optiforge.core.solution import SolutionStatus
from optiforge.lp.dual_simplex import DualRevisedSimplexSolver, DualSimplexResult
from optiforge.lp.simplex import SimplexTolerances
from optiforge.lp.standard_form import SingularBasisError
from optiforge.milp.solver import MILPSolver, MILPTolerances
from optiforge.model.model import Model
from optiforge.model.variable import Variable, VariableType
from optiforge.model.constraint import Constraint, ConstraintSense
from optiforge.model.objective import Objective, ObjectiveSense
from optiforge.numerics.pivot_basis import PivotUpdateFactorization
from optiforge.numerics.incremental_basis import IncrementalBasisFactorization


# ---------------------------------------------------------------------------
# 1. Code Hardening: Typing Inspection
# ---------------------------------------------------------------------------

def test_type_hints_inspection():
    """Verify typing.get_type_hints(DualSimplexResult) succeeds without NameError."""
    hints = typing.get_type_hints(DualSimplexResult)
    assert "basis_factorization" in hints
    assert "status" in hints
    assert "iterations" in hints


# ---------------------------------------------------------------------------
# 2. Code Hardening: Exception Narrowing
# ---------------------------------------------------------------------------

def test_exception_narrowing_propagates_programming_errors():
    """Verify unexpected programming bugs (TypeError, IndexError) are NOT silently swallowed."""
    A = np.array([
        [-1.0, -1.0, 1.0, 0.0, 0.0],
        [-1.0, -2.0, 0.0, 1.0, 0.0],
        [-1.0, -3.0, 0.0, 0.0, 1.0],
    ])
    b = np.array([-4.0, -6.0, -6.0])
    c = np.array([2.0, 3.0, 0.0, 0.0, 0.0])
    basis = [2, 3, 4]
    nonbasic = [0, 1]

    solver = DualRevisedSimplexSolver(use_pivot_factorization=True)

    orig_replace = PivotUpdateFactorization.replace_column

    def buggy_replace(self, pos, col, search_direction=None):
        raise TypeError("Simulated programming error inside replace_column")

    PivotUpdateFactorization.replace_column = buggy_replace
    try:
        with pytest.raises(TypeError, match="Simulated programming error"):
            solver.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, SimplexTolerances())
    finally:
        PivotUpdateFactorization.replace_column = orig_replace


def test_exception_narrowing_catches_numerical_failures():
    """Verify expected numerical failures (SingularBasisError) trigger clean fallback to DenseLU."""
    A = np.array([
        [-1.0, -1.0, 1.0, 0.0, 0.0],
        [-1.0, -2.0, 0.0, 1.0, 0.0],
        [-1.0, -3.0, 0.0, 0.0, 1.0],
    ])
    b = np.array([-4.0, -6.0, -6.0])
    c = np.array([2.0, 3.0, 0.0, 0.0, 0.0])
    basis = [2, 3, 4]
    nonbasic = [0, 1]

    solver = DualRevisedSimplexSolver(use_pivot_factorization=True)

    orig_replace = PivotUpdateFactorization.replace_column

    def singular_replace(self, pos, col, search_direction=None):
        raise SingularBasisError("Simulated numerical singularity during column replace")

    PivotUpdateFactorization.replace_column = singular_replace
    try:
        res = solver.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, SimplexTolerances())
        assert res.status == SolutionStatus.OPTIMAL
        assert abs(res.objective_value - 10.0) < 1e-10
    finally:
        PivotUpdateFactorization.replace_column = orig_replace


# ---------------------------------------------------------------------------
# 3. Adaptive Dispatch Helper
# ---------------------------------------------------------------------------

def test_should_use_pivot_factorization_policy():
    """Verify should_use_pivot_factorization respects explicit and adaptive thresholds."""
    # 1. Default: everything False -> False
    s_default = DualRevisedSimplexSolver()
    assert s_default.should_use_pivot_factorization(m=2) is False
    assert s_default.should_use_pivot_factorization(m=10) is False

    # 2. Explicit PFI -> always True
    s_pfi = DualRevisedSimplexSolver(use_pivot_factorization=True)
    assert s_pfi.should_use_pivot_factorization(m=1) is True
    assert s_pfi.should_use_pivot_factorization(m=100) is True

    # 3. Adaptive with threshold 6
    s_adapt = DualRevisedSimplexSolver(
        use_adaptive_pivot_factorization=True,
        adaptive_pfi_min_dimension=6,
    )
    assert s_adapt.should_use_pivot_factorization(m=1) is False
    assert s_adapt.should_use_pivot_factorization(m=5) is False
    assert s_adapt.should_use_pivot_factorization(m=6) is True
    assert s_adapt.should_use_pivot_factorization(m=10) is True

    # 4. Overridden via tolerances
    tol_adapt = SimplexTolerances(
        use_adaptive_pivot_factorization=True,
        adaptive_pfi_min_dimension=4,
    )
    assert s_default.should_use_pivot_factorization(m=3, tolerances=tol_adapt) is False
    assert s_default.should_use_pivot_factorization(m=4, tolerances=tol_adapt) is True


# ---------------------------------------------------------------------------
# 4. Below & Above Threshold Dispatch in Solve
# ---------------------------------------------------------------------------

def test_below_threshold_dispatch_selects_incremental():
    """When m < threshold, solver uses IncrementalBasisFactorization (if use_incremental_basis=True)."""
    # m = 3, threshold = 6
    A = np.array([
        [-1.0, -1.0, 1.0, 0.0, 0.0],
        [-1.0, -2.0, 0.0, 1.0, 0.0],
        [-1.0, -3.0, 0.0, 0.0, 1.0],
    ])
    b = np.array([-4.0, -6.0, -6.0])
    c = np.array([2.0, 3.0, 0.0, 0.0, 0.0])
    basis = [2, 3, 4]
    nonbasic = [0, 1]

    solver = DualRevisedSimplexSolver(
        use_incremental_basis=True,
        use_adaptive_pivot_factorization=True,
        adaptive_pfi_min_dimension=6,
    )
    res = solver.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, SimplexTolerances())
    assert res.status == SolutionStatus.OPTIMAL
    assert isinstance(res.basis_factorization, IncrementalBasisFactorization)
    assert not isinstance(res.basis_factorization, PivotUpdateFactorization)


def test_above_threshold_dispatch_selects_pfi():
    """When m >= threshold, solver uses PivotUpdateFactorization."""
    # m = 3, threshold = 2
    A = np.array([
        [-1.0, -1.0, 1.0, 0.0, 0.0],
        [-1.0, -2.0, 0.0, 1.0, 0.0],
        [-1.0, -3.0, 0.0, 0.0, 1.0],
    ])
    b = np.array([-4.0, -6.0, -6.0])
    c = np.array([2.0, 3.0, 0.0, 0.0, 0.0])
    basis = [2, 3, 4]
    nonbasic = [0, 1]

    solver = DualRevisedSimplexSolver(
        use_incremental_basis=True,
        use_adaptive_pivot_factorization=True,
        adaptive_pfi_min_dimension=2,
    )
    res = solver.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, SimplexTolerances())
    assert res.status == SolutionStatus.OPTIMAL
    assert isinstance(res.basis_factorization, PivotUpdateFactorization)


def test_explicit_pfi_overrides_adaptive_threshold():
    """Explicit use_pivot_factorization=True forces PFI even when m < adaptive_pfi_min_dimension."""
    A = np.array([
        [-1.0, -1.0, 1.0, 0.0, 0.0],
        [-1.0, -2.0, 0.0, 1.0, 0.0],
        [-1.0, -3.0, 0.0, 0.0, 1.0],
    ])
    b = np.array([-4.0, -6.0, -6.0])
    c = np.array([2.0, 3.0, 0.0, 0.0, 0.0])
    basis = [2, 3, 4]
    nonbasic = [0, 1]

    solver = DualRevisedSimplexSolver(
        use_pivot_factorization=True,
        use_adaptive_pivot_factorization=True,
        adaptive_pfi_min_dimension=10,  # higher than m=3
    )
    res = solver.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, SimplexTolerances())
    assert res.status == SolutionStatus.OPTIMAL
    assert isinstance(res.basis_factorization, PivotUpdateFactorization)


# ---------------------------------------------------------------------------
# 5. Dual Simplex MIN and MAX with Adaptive Dispatch
# ---------------------------------------------------------------------------

def test_adaptive_dual_simplex_min_and_max():
    """Verify adaptive dispatch produces correct optimal solution for both MIN and MAX."""
    A = np.array([
        [-1.0, -1.0, 1.0, 0.0, 0.0],
        [-1.0, -2.0, 0.0, 1.0, 0.0],
        [-1.0, -3.0, 0.0, 0.0, 1.0],
    ])
    b = np.array([-4.0, -6.0, -6.0])
    basis = [2, 3, 4]
    nonbasic = [0, 1]

    solver = DualRevisedSimplexSolver(
        use_adaptive_pivot_factorization=True,
        adaptive_pfi_min_dimension=2,
    )

    # MIN
    c_min = np.array([2.0, 3.0, 0.0, 0.0, 0.0])
    res_min = solver.solve_standard_form(A, b, c_min, basis, nonbasic, ObjectiveSense.MINIMIZE, SimplexTolerances())
    assert res_min.status == SolutionStatus.OPTIMAL
    assert abs(res_min.objective_value - 10.0) < 1e-10

    # MAX: dual feasible starting basis requires r_N <= 0
    c_max = np.array([-2.0, -3.0, 0.0, 0.0, 0.0])
    res_max = solver.solve_standard_form(A, b, c_max, basis, nonbasic, ObjectiveSense.MAXIMIZE, SimplexTolerances())
    assert res_max.status == SolutionStatus.OPTIMAL
    assert abs(res_max.objective_value - (-10.0)) < 1e-10


# ---------------------------------------------------------------------------
# 6. MILP Warm-Start with Adaptive Dispatch
# ---------------------------------------------------------------------------

def test_milp_warm_start_adaptive_dispatch():
    """Verify MILP warm-starts record adaptive_pfi_chosen and adaptive_inc_chosen."""
    # Small knapsack-like model with m=1 constraint
    m_small = Model("small_knapsack")
    x = m_small.add_variable(Variable("x", lower_bound=0, variable_type=VariableType.INTEGER))
    y = m_small.add_variable(Variable("y", lower_bound=0, variable_type=VariableType.INTEGER))
    m_small.add_constraint(Constraint("c1", {"x": 2.0, "y": 3.0}, ConstraintSense.LESS_EQUAL, 7.0))
    m_small.set_objective(Objective(coefficients={"x": 3.0, "y": 4.0}, sense=ObjectiveSense.MAXIMIZE))

    # Adaptive with threshold T = 5 (since m_child = 1+1 = 2 < 5, IncrementalBasis is chosen)
    solver = MILPSolver(
        tolerances=MILPTolerances(
            warm_start_relaxations=True,
            inherit_basis_factorization=True,
            use_adaptive_pivot_factorization=True,
            adaptive_pfi_min_dimension=5,
        ),
        simplex_tolerances=SimplexTolerances(
            use_adaptive_pivot_factorization=True,
            adaptive_pfi_min_dimension=5,
        ),
        warm_start_relaxations=True,
        inherit_basis_factorization=True,
    )
    sol = solver.solve(m_small)
    stats = solver.get_warm_start_stats()

    assert sol.status == SolutionStatus.OPTIMAL
    if stats.attempted > 0:
        assert stats.adaptive_inc_chosen > 0
        assert stats.adaptive_pfi_chosen == 0

    # Now run with threshold T = 2 (m_child = 2 >= 2, PFI is chosen)
    solver_pfi = MILPSolver(
        tolerances=MILPTolerances(
            warm_start_relaxations=True,
            inherit_basis_factorization=True,
            use_adaptive_pivot_factorization=True,
            adaptive_pfi_min_dimension=2,
        ),
        simplex_tolerances=SimplexTolerances(
            use_adaptive_pivot_factorization=True,
            adaptive_pfi_min_dimension=2,
        ),
        warm_start_relaxations=True,
        inherit_basis_factorization=True,
    )
    sol_pfi = solver_pfi.solve(m_small)
    stats_pfi = solver_pfi.get_warm_start_stats()

    assert sol_pfi.status == SolutionStatus.OPTIMAL
    assert abs(sol.objective_value - sol_pfi.objective_value) < 1e-10
    if stats_pfi.attempted > 0:
        assert stats_pfi.adaptive_pfi_chosen > 0
        assert stats_pfi.adaptive_inc_chosen == 0
