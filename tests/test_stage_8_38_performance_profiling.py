"""Tests for Stage 8.38: Post-Optimization Bottleneck Profiling.

Verifies:
1. SimplexProfiler and SimplexProfileStats calculations (shares, Amdahl maximums, bottleneck ranking).
2. Zero-overhead contract when enable_profiling=False (profile_stats is None).
3. Complete instrumentation coverage in DualRevisedSimplexSolver.
4. Numerical invariance (profiling enabled produces identical solution, objective, basis, iterations).
5. Both Python and SciPy triangular solve providers operate correctly with profiling.
6. MILP and DenseLU / PFI compatibility with profiling enabled.
"""

from __future__ import annotations

import numpy as np
import pytest

from optiforge.core.solution import SolutionStatus
from optiforge.lp.dual_simplex import DualRevisedSimplexSolver
from optiforge.lp.profiling import SimplexProfileStats, SimplexProfiler
from optiforge.lp.simplex import SimplexTolerances
from optiforge.milp.solver import MILPSolver, MILPTolerances
from optiforge.model import Constraint, ConstraintSense, Model, Objective, ObjectiveSense, Variable, VariableType
from optiforge.numerics.triangular_provider import is_scipy_available, reset_provider_stats


# ---------------------------------------------------------------------------
# Helpers to build test problems
# ---------------------------------------------------------------------------

def _build_test_problem_1():
    """Feasible 2x4 standard-form LP that requires dual pivots to reach optimal.
    min x1 + x2
    s.t. -x1 - 2x2 + s1 = -4
         -2x1 - x2 + s2 = -5
    x1, x2, s1, s2 >= 0
    Optimal solution: x1 = 2, x2 = 1, obj = 3.0
    """
    A = np.array([
        [-1.0, -2.0, 1.0, 0.0],
        [-2.0, -1.0, 0.0, 1.0],
    ], dtype=np.float64)
    b = np.array([-4.0, -5.0], dtype=np.float64)
    c = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64)
    basis = [2, 3]
    nonbasic = [0, 1]
    return A, b, c, basis, nonbasic


def _build_simple_milp() -> Model:
    """Build a simple integer LP:
    max x1 + x2
    s.t. x1 + x2 <= 3
         x1, x2 in Z+
    """
    m = Model("simple_int")
    m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 3.0))
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
    return m


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

class TestSimplexProfilerUnit:
    """Tests for the SimplexProfiler and SimplexProfileStats classes."""

    def test_disabled_profiler_returns_none(self):
        profiler = SimplexProfiler(enabled=False)
        profiler.record("phase_a", 0.05)
        profiler.record("phase_b", 0.10)
        assert profiler.finish(iterations=5) is None

    def test_profile_stats_calculations(self):
        stats = SimplexProfileStats(
            total_time=1.0,
            iterations=10,
            component_times={
                "phase_a": 0.50,
                "phase_b": 0.30,
                "phase_c": 0.20,
            },
            component_counts={
                "phase_a": 5,
                "phase_b": 10,
                "phase_c": 20,
            },
        )
        shares = stats.runtime_shares()
        assert pytest.approx(shares["phase_a"], rel=1e-5) == 0.50
        assert pytest.approx(shares["phase_b"], rel=1e-5) == 0.30
        assert pytest.approx(shares["phase_c"], rel=1e-5) == 0.20

        # Amdahl maximums: S_max = 1 / (1 - f_i)
        amdahl = stats.amdahl_maximums()
        assert pytest.approx(amdahl["phase_a"], rel=1e-5) == 1.0 / (1.0 - 0.50)  # 2.0x
        assert pytest.approx(amdahl["phase_b"], rel=1e-5) == 1.0 / (1.0 - 0.30)  # ~1.43x
        assert pytest.approx(amdahl["phase_c"], rel=1e-5) == 1.0 / (1.0 - 0.20)  # 1.25x

        # Bottleneck ranking
        ranking = stats.bottleneck_ranking()
        assert [r[0] for r in ranking] == ["phase_a", "phase_b", "phase_c"]

        # Summary string generation
        summary = stats.summary()
        assert "Simplex Profile Summary" in summary
        assert "phase_a" in summary
        assert "2.00x" in summary


class TestDualSimplexProfilingInstrumentation:
    """Tests that DualRevisedSimplexSolver records expected phases and statistics."""

    def test_profiling_disabled_by_default(self):
        A, b, c, basis, nonbasic = _build_test_problem_1()
        solver = DualRevisedSimplexSolver(enable_profiling=False)
        res = solver.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, SimplexTolerances())
        assert res.profile_stats is None
        assert solver.last_profile_stats is None

    def test_profiling_enabled_records_phases(self):
        A, b, c, basis, nonbasic = _build_test_problem_1()
        solver = DualRevisedSimplexSolver(enable_profiling=True, use_pivot_factorization=True)
        res = solver.solve_standard_form(
            A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE,
            SimplexTolerances(enable_profiling=True)
        )
        assert res.status == SolutionStatus.OPTIMAL
        assert res.iterations > 0
        assert pytest.approx(res.objective_value, rel=1e-6) == 3.0
        assert res.profile_stats is not None
        stats = res.profile_stats

        assert stats.iterations == res.iterations
        assert stats.total_time > 0.0

        # Check required component phases are recorded
        expected_phases = [
            "initial_factorization", "primal_solve", "pricing",
            "btran_v", "gemv_atv", "ratio_test", "basis_update", "pfi_update"
        ]
        for phase in expected_phases:
            assert phase in stats.component_times, f"Expected phase '{phase}' in component_times"
            assert stats.component_counts[phase] >= 1

        # Check provider stats attached
        assert "total_solves" in stats.provider_stats

    def test_numerical_invariance_contract(self):
        """Verify profiling enabled produces 100% numerically identical output."""
        A, b, c, basis, nonbasic = _build_test_problem_1()

        # Run with profiling OFF
        solver_off = DualRevisedSimplexSolver(enable_profiling=False, use_pivot_factorization=True)
        res_off = solver_off.solve_standard_form(
            A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE,
            SimplexTolerances(enable_profiling=False)
        )

        # Run with profiling ON
        solver_on = DualRevisedSimplexSolver(enable_profiling=True, use_pivot_factorization=True)
        res_on = solver_on.solve_standard_form(
            A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE,
            SimplexTolerances(enable_profiling=True)
        )

        assert res_off.status == res_on.status
        assert res_off.iterations == res_on.iterations
        assert np.isclose(res_off.objective_value, res_on.objective_value, atol=1e-12)
        assert np.allclose(res_off.x, res_on.x, atol=1e-12)
        assert res_off.basis_indices == res_on.basis_indices
        assert res_off.nonbasic_indices == res_on.nonbasic_indices

    def test_denselu_vs_pfi_profiling(self):
        """Verify profiling works for both DenseLU and PFI."""
        A, b, c, basis, nonbasic = _build_test_problem_1()

        # DenseLU
        solver_lu = DualRevisedSimplexSolver(enable_profiling=True, use_pivot_factorization=False)
        res_lu = solver_lu.solve_standard_form(
            A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE,
            SimplexTolerances(enable_profiling=True, use_pivot_factorization=False)
        )
        assert res_lu.profile_stats is not None

        # PFI
        solver_pfi = DualRevisedSimplexSolver(enable_profiling=True, use_pivot_factorization=True)
        res_pfi = solver_pfi.solve_standard_form(
            A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE,
            SimplexTolerances(enable_profiling=True, use_pivot_factorization=True)
        )
        assert res_pfi.profile_stats is not None
        assert np.isclose(res_lu.objective_value, res_pfi.objective_value, atol=1e-12)

    def test_python_vs_scipy_provider_profiling(self):
        """Verify profiling works under both python and scipy providers."""
        A, b, c, basis, nonbasic = _build_test_problem_1()

        # Python provider
        solver_py = DualRevisedSimplexSolver(
            enable_profiling=True,
            use_pivot_factorization=True,
            triangular_solve_provider="python",
        )
        res_py = solver_py.solve_standard_form(
            A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE,
            SimplexTolerances(enable_profiling=True)
        )
        assert res_py.profile_stats is not None

        if is_scipy_available():
            reset_provider_stats()
            solver_sp = DualRevisedSimplexSolver(
                enable_profiling=True,
                use_pivot_factorization=True,
                triangular_solve_provider="scipy",
                scipy_triangular_min_dimension=1,  # Force scipy on small 2x4
            )
            res_sp = solver_sp.solve_standard_form(
                A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE,
                SimplexTolerances(enable_profiling=True, scipy_triangular_min_dimension=1)
            )
            assert res_sp.profile_stats is not None
            assert np.isclose(res_py.objective_value, res_sp.objective_value, atol=1e-12)
            assert res_sp.profile_stats.provider_stats["scipy_solves"] > 0


class TestMILPProfilingIntegration:
    """Verify MILP solver passes through profiling without regression."""

    def test_milp_profiling_enabled(self):
        model = _build_simple_milp()
        solver = MILPSolver(
            tolerances=MILPTolerances(enable_profiling=True, warm_start_relaxations=True)
        )
        sol = solver.solve(model)
        assert sol.status == SolutionStatus.OPTIMAL
        assert pytest.approx(sol.objective_value, rel=1e-5) == 3.0
        # Verification that integer constraints are satisfied
        for vname, val in sol.variable_values.items():
            assert abs(val - round(val)) < 1e-6
