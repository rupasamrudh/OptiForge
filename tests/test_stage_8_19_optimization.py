"""Stage 8.19: Tests for optimized PFI (fast_pivot_update).

Verifies correctness, numerical invariance, fallback safety, and default preservation.
"""

import numpy as np
import pytest
from optiforge.lp.simplex import RevisedSimplexSolver, SimplexTolerances
from optiforge.lp.dual_simplex import DualRevisedSimplexSolver
from optiforge.milp.solver import MILPSolver, MILPTolerances
from optiforge.numerics.pivot_basis import PivotUpdateFactorization, RefactorizationReason
from optiforge.numerics.exceptions import SingularBasisError
from optiforge.model import Model, Variable, Constraint, Objective, ConstraintSense, ObjectiveSense, VariableType
from benchmarks.run_stage_8_18 import generate_scale_sweep_lp
from benchmarks.stage_8_15.generate_workloads import generate_all_workloads


def test_default_flag_is_false():
    """Verify that use_fast_pivot_update defaults to False everywhere."""
    s_tol = SimplexTolerances()
    assert s_tol.use_fast_pivot_update is False
    assert s_tol.use_adaptive_pivot_factorization is False

    m_tol = MILPTolerances()
    assert m_tol.use_fast_pivot_update is False
    assert m_tol.use_adaptive_pivot_factorization is False

    B = np.eye(3)
    pfi = PivotUpdateFactorization(B)
    assert pfi.fast_pivot_update is False


def test_pfi_fast_update_numerical_equivalence():
    """Verify solutions are identical between baseline PFI and fast PFI."""
    for m, n in [(6, 15), (10, 20), (20, 30)]:
        model = generate_scale_sweep_lp(m, n, seed=123)

        # Baseline PFI
        tol_base = SimplexTolerances(use_pivot_factorization=True, use_fast_pivot_update=False)
        sol_base = RevisedSimplexSolver().solve(model, tolerances=tol_base)

        # Fast PFI
        tol_fast = SimplexTolerances(use_pivot_factorization=True, use_fast_pivot_update=True)
        sol_fast = RevisedSimplexSolver().solve(model, tolerances=tol_fast)

        assert sol_base.status == sol_fast.status
        assert np.isclose(sol_base.objective_value, sol_fast.objective_value, atol=1e-9)
        for k in sol_base.variable_values:
            assert np.isclose(sol_base.variable_values[k], sol_fast.variable_values[k], atol=1e-8)


def test_dual_simplex_fast_pfi_equivalence():
    """Verify dual simplex produces identical results with fast PFI."""
    all_wl = {w.name: w.model for w in generate_all_workloads()}
    model = all_wl["phase1_heavy_lp"]

    tol_base = SimplexTolerances(use_pivot_factorization=True, use_fast_pivot_update=False)
    solver_base = DualRevisedSimplexSolver(tolerances=tol_base, use_pivot_factorization=True)
    sol_base = solver_base.solve(model)

    tol_fast = SimplexTolerances(use_pivot_factorization=True, use_fast_pivot_update=True)
    solver_fast = DualRevisedSimplexSolver(tolerances=tol_fast, use_pivot_factorization=True)
    sol_fast = solver_fast.solve(model)

    assert sol_base.status == sol_fast.status
    assert np.isclose(sol_base.objective_value, sol_fast.objective_value, atol=1e-9)


def test_milp_fast_pfi_equivalence():
    """Verify branch-and-bound produces identical results with fast PFI."""
    all_wl = {w.name: w.model for w in generate_all_workloads()}
    model = all_wl["branching_heavy"]

    tol_base = MILPTolerances(
        warm_start_relaxations=True,
        inherit_basis_factorization=True,
        use_adaptive_pivot_factorization=True,
        adaptive_pfi_min_dimension=1,
        use_fast_pivot_update=False,
    )
    sol_base = MILPSolver(tolerances=tol_base).solve(model)

    tol_fast = MILPTolerances(
        warm_start_relaxations=True,
        inherit_basis_factorization=True,
        use_adaptive_pivot_factorization=True,
        adaptive_pfi_min_dimension=1,
        use_fast_pivot_update=True,
    )
    sol_fast = MILPSolver(tolerances=tol_fast).solve(model)

    assert sol_base.status == sol_fast.status
    assert np.isclose(sol_base.objective_value, sol_fast.objective_value, atol=1e-9)
    for k in sol_base.variable_values:
        assert np.isclose(sol_base.variable_values[k], sol_fast.variable_values[k], atol=1e-8)


def test_fast_pfi_singularity_handling():
    """Verify that singular column replacement raises SingularBasisError cleanly."""
    B = np.eye(3)
    pfi = PivotUpdateFactorization(B, fast_pivot_update=True)

    # Replace column 0 with a zero vector (singular)
    zero_col = np.zeros(3)
    # The search direction d = B^{-1} * 0 = 0, so pivot_val = 0 < pivot_tolerance.
    # It will trigger refactorization of singular B, raising SingularBasisError.
    with pytest.raises(SingularBasisError):
        pfi.replace_column(0, zero_col)

    # Verify B column 0 was restored
    assert np.allclose(pfi.B[:, 0], [1.0, 0.0, 0.0])


def test_fast_pfi_nonfinite_column_handling():
    """Verify non-finite column raises SingularBasisError."""
    B = np.eye(3)
    pfi = PivotUpdateFactorization(B, fast_pivot_update=True)

    nan_col = np.array([1.0, np.nan, 0.0])
    with pytest.raises(SingularBasisError):
        pfi.replace_column(0, nan_col)


def test_fast_pfi_adversarial_large_finite_overflow():
    """Verify very large finite column whose sum overflows is NOT rejected as non-finite."""
    # B = 2x2 identity matrix
    B = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    pfi = PivotUpdateFactorization(B, fast_pivot_update=True)

    # Column has finite entries [1e308, 1e308], but 1e308 + 1e308 = +inf in float64
    large_col = np.array([1e308, 1e308], dtype=np.float64)
    # This must NOT fail at the finiteness check.
    # It replaces col 0: B becomes [[1e308, 0], [1e308, 1]] which has det = 1e308 (non-singular!).
    # Search direction d = B_init^{-1} * large_col = [1e308, 1e308].
    # Pivot element d[0] = 1e308 >= pivot_tolerance.
    res = pfi.replace_column(0, large_col)
    assert res is True
    assert pfi.num_etas == 1
    # Verify solve works with this large basis
    rhs = np.array([1.0, 1.0], dtype=np.float64)
    sol = pfi.solve(rhs)
    assert np.all(np.isfinite(sol))
    expected = np.linalg.solve(pfi.B, rhs)
    assert np.allclose(sol, expected)


def test_fast_pfi_adversarial_infinities_and_nans():
    """Verify NaN, +Inf, -Inf, and mixed non-finite values are cleanly rejected with state restoration."""
    adversarial_cols = [
        ("NaN", np.array([1.0, np.nan])),
        ("+Inf", np.array([1.0, np.inf])),
        ("-Inf", np.array([1.0, -np.inf])),
        ("Mixed Inf", np.array([np.inf, -np.inf])),
        ("NaN and Inf", np.array([np.nan, np.inf])),
    ]

    for name, bad_col in adversarial_cols:
        B = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float64)
        pfi = PivotUpdateFactorization(B, fast_pivot_update=True)
        orig_col0 = pfi.B[:, 0].copy()
        orig_etas = pfi.num_etas

        with pytest.raises(SingularBasisError):
            pfi.replace_column(0, bad_col)

        # Verification H: State restoration
        assert np.allclose(pfi.B[:, 0], orig_col0), f"Basis column 0 not restored for {name}"
        assert pfi.num_etas == orig_etas, f"Eta chain corrupted for {name}"

        # Subsequent solve succeeds
        test_rhs = np.array([3.0, 4.0], dtype=np.float64)
        expected_sol = np.linalg.solve(B, test_rhs)
        actual_sol = pfi.solve(test_rhs)
        assert np.allclose(actual_sol, expected_sol), f"Subsequent solve failed after {name} rejection"


def test_fast_pfi_adversarial_singular_rollback():
    """Verify singular column rollback: column restored, etas intact, subsequent solve succeeds."""
    B = np.array([[3.0, 2.0], [1.0, 4.0]], dtype=np.float64)
    pfi = PivotUpdateFactorization(B, fast_pivot_update=True)
    orig_col0 = pfi.B[:, 0].copy()

    # Column 0 replaced by multiple of column 1: [4.0, 8.0] = 2 * [2.0, 4.0]
    singular_col = np.array([4.0, 8.0], dtype=np.float64)
    with pytest.raises(SingularBasisError):
        pfi.replace_column(0, singular_col)

    assert np.allclose(pfi.B[:, 0], orig_col0)
    assert pfi.num_etas == 0

    # Subsequent solve succeeds
    test_rhs = np.array([5.0, 5.0], dtype=np.float64)
    expected_sol = np.linalg.solve(B, test_rhs)
    assert np.allclose(pfi.solve(test_rhs), expected_sol)

