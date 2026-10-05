"""
Stage 8.17 Remediation Tests: Adaptive-PFI Dispatch Accounting Fix
Verifies:
TEST 1: Adaptive PFI + feasible child (adaptive_pfi_chosen increments correctly)
TEST 2: Adaptive IncrementalBasis + feasible child (adaptive_inc_chosen increments correctly)
TEST 3: Adaptive PFI + infeasible child (adaptive_pfi_chosen records PFI on infeasible child)
TEST 4: Adaptive IncrementalBasis + infeasible child (adaptive_inc_chosen records IncBasis on infeasible child)
TEST 5: Fallback accounting (attempted = accepted + fallbacks)
TEST 6: Multiple infeasible children (infeasible_children and integer_allocation)
TEST 7: Default mode (adaptive counters remain strictly zero)
TEST 8: Explicit PFI (explicit PFI statistics correct, adaptive counters zero)
"""
import pytest
import numpy as np

from optiforge.core.solution import SolutionStatus
from optiforge.lp.simplex import SimplexTolerances
from optiforge.lp.standard_form import SingularBasisError
from optiforge.milp.solver import MILPSolver, MILPTolerances
from optiforge.model.model import Model
from optiforge.model.variable import Variable, VariableType
from optiforge.model.constraint import Constraint, ConstraintSense
from optiforge.model.objective import Objective, ObjectiveSense


def _create_simple_model():
    """Simple model where root is fractional and children are feasible."""
    m = Model("simple_feasible")
    x = m.add_variable(Variable("x", lower_bound=0, variable_type=VariableType.INTEGER))
    y = m.add_variable(Variable("y", lower_bound=0, variable_type=VariableType.INTEGER))
    m.add_constraint(Constraint("c1", {"x": 2.0, "y": 3.0}, ConstraintSense.LESS_EQUAL, 7.0))
    m.set_objective(Objective(coefficients={"x": 3.0, "y": 4.0}, sense=ObjectiveSense.MAXIMIZE))
    return m


def _create_infeasible_branch_model():
    """Model specifically designed to have an infeasible child branch.
    
    Relaxation at root: x in [0, 1.5], y in [0, 1.5], x + y <= 1.5.
    Branch on x:
      Branch 1: x <= 1 -> feasible
      Branch 2: x >= 2 -> infeasible with x <= 1.5 (or x + y <= 1.5 with x >= 2)
    """
    m = Model("infeasible_branch")
    x = m.add_variable(Variable("x", lower_bound=0, upper_bound=2, variable_type=VariableType.INTEGER))
    y = m.add_variable(Variable("y", lower_bound=0, upper_bound=2, variable_type=VariableType.INTEGER))
    # 2*x + 2*y <= 3 -> x + y <= 1.5
    m.add_constraint(Constraint("c1", {"x": 2.0, "y": 2.0}, ConstraintSense.LESS_EQUAL, 3.0))
    m.set_objective(Objective(coefficients={"x": 2.0, "y": 1.0}, sense=ObjectiveSense.MAXIMIZE))
    return m


# ---------------------------------------------------------------------------
# TEST 1: Adaptive PFI + feasible child
# ---------------------------------------------------------------------------
def test_adaptive_pfi_feasible_child():
    """Verify adaptive PFI increments adaptive_pfi_chosen on feasible child relaxations."""
    m = _create_simple_model()
    # Low threshold: child m >= 2 -> PFI chosen
    solver = MILPSolver(
        tolerances=MILPTolerances(
            warm_start_relaxations=True,
            inherit_basis_factorization=True,
            use_adaptive_pivot_factorization=True,
            adaptive_pfi_min_dimension=2,
        ),
        warm_start_relaxations=True,
        inherit_basis_factorization=True,
    )
    sol = solver.solve(m)
    st = solver.get_warm_start_stats()

    assert sol.status == SolutionStatus.OPTIMAL
    assert st.attempted > 0
    assert st.accepted > 0
    assert st.adaptive_pfi_chosen == st.accepted
    assert st.adaptive_inc_chosen == 0
    assert st.attempted == st.accepted + st.fallbacks


# ---------------------------------------------------------------------------
# TEST 2: Adaptive IncrementalBasis + feasible child
# ---------------------------------------------------------------------------
def test_adaptive_inc_feasible_child():
    """Verify adaptive dispatch increments adaptive_inc_chosen when m < threshold."""
    m = _create_simple_model()
    # High threshold: child m < 50 -> IncrementalBasis chosen
    solver = MILPSolver(
        tolerances=MILPTolerances(
            warm_start_relaxations=True,
            inherit_basis_factorization=True,
            use_adaptive_pivot_factorization=True,
            adaptive_pfi_min_dimension=50,
        ),
        warm_start_relaxations=True,
        inherit_basis_factorization=True,
    )
    sol = solver.solve(m)
    st = solver.get_warm_start_stats()

    assert sol.status == SolutionStatus.OPTIMAL
    assert st.attempted > 0
    assert st.accepted > 0
    assert st.adaptive_inc_chosen == st.accepted
    assert st.adaptive_pfi_chosen == 0
    assert st.attempted == st.accepted + st.fallbacks


# ---------------------------------------------------------------------------
# TEST 3: Adaptive PFI + infeasible child
# ---------------------------------------------------------------------------
def test_adaptive_pfi_infeasible_child():
    """Verify dispatch counter records PFI on child relaxations proven infeasible."""
    m = _create_infeasible_branch_model()
    solver = MILPSolver(
        tolerances=MILPTolerances(
            warm_start_relaxations=True,
            inherit_basis_factorization=True,
            use_adaptive_pivot_factorization=True,
            adaptive_pfi_min_dimension=2,
        ),
        warm_start_relaxations=True,
        inherit_basis_factorization=True,
    )
    sol = solver.solve(m)
    st = solver.get_warm_start_stats()

    assert sol.status == SolutionStatus.OPTIMAL
    assert st.infeasible_pruned > 0, "Expected at least one child relaxation to be infeasible and pruned"
    assert st.adaptive_pfi_chosen == st.accepted, (
        f"Under-count detected: adaptive_pfi_chosen ({st.adaptive_pfi_chosen}) != accepted ({st.accepted})"
    )
    assert st.adaptive_inc_chosen == 0
    assert st.adaptive_pfi_chosen + st.adaptive_inc_chosen == st.accepted
    assert st.attempted == st.accepted + st.fallbacks


# ---------------------------------------------------------------------------
# TEST 4: Adaptive IncrementalBasis + infeasible child
# ---------------------------------------------------------------------------
def test_adaptive_inc_infeasible_child():
    """Verify dispatch counter records IncrementalBasis on child relaxations proven infeasible."""
    m = _create_infeasible_branch_model()
    solver = MILPSolver(
        tolerances=MILPTolerances(
            warm_start_relaxations=True,
            inherit_basis_factorization=True,
            use_adaptive_pivot_factorization=True,
            adaptive_pfi_min_dimension=50,
        ),
        warm_start_relaxations=True,
        inherit_basis_factorization=True,
    )
    sol = solver.solve(m)
    st = solver.get_warm_start_stats()

    assert sol.status == SolutionStatus.OPTIMAL
    assert st.infeasible_pruned > 0, "Expected at least one child relaxation to be infeasible and pruned"
    assert st.adaptive_inc_chosen == st.accepted, (
        f"Under-count detected: adaptive_inc_chosen ({st.adaptive_inc_chosen}) != accepted ({st.accepted})"
    )
    assert st.adaptive_pfi_chosen == 0
    assert st.adaptive_pfi_chosen + st.adaptive_inc_chosen == st.accepted
    assert st.attempted == st.accepted + st.fallbacks


# ---------------------------------------------------------------------------
# TEST 5: Fallback accounting invariant
# ---------------------------------------------------------------------------
def test_fallback_accounting_invariant(monkeypatch):
    """Verify attempted = accepted + fallbacks and adaptive counters only count accepted solves."""
    m = _create_simple_model()

    # Artificially force fallback on the second warm-start call
    call_count = [0]
    from optiforge.lp.dual_simplex import DualRevisedSimplexSolver
    orig_solve_sf = DualRevisedSimplexSolver.solve_standard_form

    def failing_solve_sf(self, *args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 2:
            raise SingularBasisError("Simulated singular basis forcing fallback")
        return orig_solve_sf(self, *args, **kwargs)

    monkeypatch.setattr(DualRevisedSimplexSolver, "solve_standard_form", failing_solve_sf)

    solver = MILPSolver(
        tolerances=MILPTolerances(
            warm_start_relaxations=True,
            inherit_basis_factorization=True,
            use_adaptive_pivot_factorization=True,
            adaptive_pfi_min_dimension=2,
        ),
        warm_start_relaxations=True,
        inherit_basis_factorization=True,
    )
    sol = solver.solve(m)
    st = solver.get_warm_start_stats()

    assert sol.status == SolutionStatus.OPTIMAL
    assert st.fallbacks > 0, "Expected at least one fallback"
    assert st.attempted == st.accepted + st.fallbacks
    assert st.adaptive_pfi_chosen + st.adaptive_inc_chosen == st.accepted


# ---------------------------------------------------------------------------
# TEST 6: Multiple infeasible children across several nodes
# ---------------------------------------------------------------------------
def test_multiple_infeasible_children():
    """Verify accounting accuracy on workloads with multiple infeasible child nodes."""
    import sys
    sys.path.insert(0, 'benchmarks/stage_8_15')
    from generate_workloads import generate_all_workloads

    workloads = generate_all_workloads()
    for wname in ["infeasible_children", "integer_allocation"]:
        w = next(x for x in workloads if x.name == wname)

        # 1. PFI adaptive (T=6)
        s_pfi = MILPSolver(
            tolerances=MILPTolerances(
                warm_start_relaxations=True,
                inherit_basis_factorization=True,
                use_adaptive_pivot_factorization=True,
                adaptive_pfi_min_dimension=6,
            ),
            warm_start_relaxations=True,
            inherit_basis_factorization=True,
        )
        sol_pfi = s_pfi.solve(w.model)
        st_pfi = s_pfi.get_warm_start_stats()

        assert sol_pfi.status == SolutionStatus.OPTIMAL
        assert st_pfi.infeasible_pruned > 0
        assert st_pfi.adaptive_pfi_chosen == st_pfi.accepted
        assert st_pfi.adaptive_inc_chosen == 0
        assert st_pfi.attempted == st_pfi.accepted + st_pfi.fallbacks

        # 2. IncrementalBasis adaptive (T=50)
        s_inc = MILPSolver(
            tolerances=MILPTolerances(
                warm_start_relaxations=True,
                inherit_basis_factorization=True,
                use_adaptive_pivot_factorization=True,
                adaptive_pfi_min_dimension=50,
            ),
            warm_start_relaxations=True,
            inherit_basis_factorization=True,
        )
        sol_inc = s_inc.solve(w.model)
        st_inc = s_inc.get_warm_start_stats()

        assert sol_inc.status == SolutionStatus.OPTIMAL
        assert st_inc.infeasible_pruned > 0
        assert st_inc.adaptive_inc_chosen == st_inc.accepted
        assert st_inc.adaptive_pfi_chosen == 0
        assert st_inc.attempted == st_inc.accepted + st_inc.fallbacks


# ---------------------------------------------------------------------------
# TEST 7: Default mode counters remain zero
# ---------------------------------------------------------------------------
def test_default_mode_counters_remain_zero():
    """Verify adaptive counters remain zero in default solver mode."""
    m = _create_infeasible_branch_model()
    solver = MILPSolver(
        tolerances=MILPTolerances(warm_start_relaxations=True, inherit_basis_factorization=True),
        warm_start_relaxations=True,
        inherit_basis_factorization=True,
    )
    sol = solver.solve(m)
    st = solver.get_warm_start_stats()

    assert sol.status == SolutionStatus.OPTIMAL
    assert st.accepted > 0
    assert st.adaptive_pfi_chosen == 0
    assert st.adaptive_inc_chosen == 0
    assert st.attempted == st.accepted + st.fallbacks


# ---------------------------------------------------------------------------
# TEST 8: Explicit PFI statistics correct
# ---------------------------------------------------------------------------
def test_explicit_pfi_statistics():
    """Verify explicit PFI records PFI telemetry while adaptive counters remain zero."""
    m = _create_infeasible_branch_model()
    solver = MILPSolver(
        tolerances=MILPTolerances(warm_start_relaxations=True, inherit_basis_factorization=True),
        simplex_tolerances=SimplexTolerances(use_pivot_factorization=True, use_adaptive_pivot_factorization=False),
        warm_start_relaxations=True,
        inherit_basis_factorization=True,
    )
    sol = solver.solve(m)
    st = solver.get_warm_start_stats()

    assert sol.status == SolutionStatus.OPTIMAL
    assert st.accepted > 0
    assert st.pfi_updates >= 0
    # Since use_adaptive_pivot_factorization is False, adaptive counters must be 0
    assert st.adaptive_pfi_chosen == 0
    assert st.adaptive_inc_chosen == 0
    assert st.attempted == st.accepted + st.fallbacks
