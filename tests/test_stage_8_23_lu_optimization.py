"""Tests for Stage 8.23 Base LU Factorization Optimization."""
import numpy as np
import pytest

from optiforge.numerics.lu import (
    DenseLU,
    dense_lu_factor,
    dense_lu_factor_reference,
    lu_solve,
    lu_solve_transpose,
)
from optiforge.numerics.exceptions import NumericalSingularityError
from optiforge.numerics.triangular_provider import (
    is_scipy_available,
    triangular_lu_factor,
    get_provider_stats,
    reset_provider_stats,
)
from optiforge.numerics.pivot_basis import PivotUpdateFactorization, RefactorizationReason


def test_equivalence_with_reference():
    """Verify exact equivalence between optimized dense_lu_factor and reference."""
    rng = np.random.RandomState(42)
    for m in [1, 2, 4, 8, 15, 30]:
        B = rng.randn(m, m) + 5.0 * np.eye(m)
        lu_opt = dense_lu_factor(B)
        lu_ref = dense_lu_factor_reference(B)

        # Permutations must match
        assert np.array_equal(lu_opt.perm, lu_ref.perm)
        assert lu_opt.det_sign == lu_ref.det_sign

        # Factors must match to machine precision
        assert np.allclose(lu_opt.L, lu_ref.L, atol=1e-14)
        assert np.allclose(lu_opt.U, lu_ref.U, atol=1e-14)

        # Residuals P B - L U must be identical
        res_opt = np.linalg.norm(lu_opt.P @ B - lu_opt.L @ lu_opt.U, ord=np.inf)
        res_ref = np.linalg.norm(lu_ref.P @ B - lu_ref.L @ lu_ref.U, ord=np.inf)
        assert abs(res_opt - res_ref) < 1e-14


def test_singularity_detection_matches_reference():
    """Verify singular matrices trigger identical NumericalSingularityError."""
    rng = np.random.RandomState(42)
    B = rng.randn(6, 6)
    B[:, -1] = B[:, 0] + B[:, 1]  # rank deficient

    with pytest.raises(NumericalSingularityError) as exc_opt:
        dense_lu_factor(B)

    with pytest.raises(NumericalSingularityError) as exc_ref:
        dense_lu_factor_reference(B)

    assert exc_opt.value.step == exc_ref.value.step
    assert abs(exc_opt.value.pivot - exc_ref.value.pivot) < 1e-14


def test_hilbert_matrix():
    """Verify accuracy on ill-conditioned Hilbert matrix."""
    H = 1.0 / (np.arange(1, 8)[:, None] + np.arange(1, 8)[None, :] - 1.0)
    lu = dense_lu_factor(H)
    res = np.linalg.norm(lu.P @ H - lu.L @ lu.U, ord=np.inf)
    assert res < 1e-14


def test_input_non_mutation():
    """Verify input matrix B is not mutated by dense_lu_factor."""
    rng = np.random.RandomState(42)
    B = rng.randn(10, 10)
    B_orig = B.copy()
    _ = dense_lu_factor(B)
    assert np.array_equal(B, B_orig)


def test_read_only_and_non_contiguous():
    """Verify factorization works cleanly on read-only and Fortran-order matrices."""
    rng = np.random.RandomState(42)
    B = rng.randn(8, 8) + 4.0 * np.eye(8)

    # Read-only
    B_ro = B.copy()
    B_ro.flags.writeable = False
    lu_ro = dense_lu_factor(B_ro)
    assert np.allclose(lu_ro.P @ B, lu_ro.L @ lu_ro.U)

    # Non-contiguous (Fortran order)
    B_f = np.asfortranarray(B)
    assert not B_f.flags.c_contiguous
    lu_f = dense_lu_factor(B_f)
    assert np.allclose(lu_f.P @ B, lu_f.L @ lu_f.U)


def test_determinism():
    """Verify repeated factorizations yield bit-identical factors."""
    rng = np.random.RandomState(42)
    B = rng.randn(12, 12) + 3.0 * np.eye(12)
    lu1 = dense_lu_factor(B)
    lu2 = dense_lu_factor(B)
    assert np.array_equal(lu1.L, lu2.L)
    assert np.array_equal(lu1.U, lu2.U)
    assert np.array_equal(lu1.perm, lu2.perm)


def test_pfi_integration_with_optimized_factor():
    """Verify PFI basis updates and refactorizations work cleanly with optimized LU."""
    rng = np.random.RandomState(42)
    m = 8
    B0 = rng.randn(m, m) + 5.0 * np.eye(m)
    pfi = PivotUpdateFactorization(B0, max_updates=4)

    for step in range(5):
        col = rng.randn(m)
        pfi.replace_column(step % m, col)

    # Reached update limit -> refactored
    assert pfi.total_refactorizations == 2
    assert pfi._last_refactor_reason == RefactorizationReason.UPDATE_LIMIT

    # Solve accuracy
    rhs = rng.randn(m)
    x = pfi.solve(rhs)
    assert np.linalg.norm(pfi.B @ x - rhs, ord=np.inf) < 1e-10


@pytest.mark.skipif(not is_scipy_available(), reason="SciPy not installed in this environment")
def test_scipy_lu_provider():
    """Verify optional SciPy LU factorization provider when available."""
    reset_provider_stats()
    rng = np.random.RandomState(42)
    m = 20
    B = rng.randn(m, m) + 5.0 * np.eye(m)

    lu_sp = dense_lu_factor(B, use_scipy_lu_factor=True, scipy_lu_min_dimension=10)
    assert lu_sp.use_scipy_lu_factor is True

    # Check reconstruction
    res = np.linalg.norm(lu_sp.P @ B - lu_sp.L @ lu_sp.U, ord=np.inf)
    assert res < 1e-12

    # Check telemetry
    stats = get_provider_stats()
    assert stats["scipy_factors"] >= 1


@pytest.mark.skipif(not is_scipy_available(), reason="SciPy not installed in this environment")
def test_scipy_lu_provider_singular():
    """Verify optional SciPy provider correctly detects and raises singularity."""
    rng = np.random.RandomState(42)
    m = 15
    B = rng.randn(m, m)
    B[5, :] = 0.0  # Zero row

    with pytest.raises(NumericalSingularityError):
        dense_lu_factor(B, use_scipy_lu_factor=True, scipy_lu_min_dimension=10)
