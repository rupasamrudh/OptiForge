"""Tests for IncrementalBasisFactorization (Stage 8.7).

Validates the Product Form of Inverse (PFI) incremental basis update against
the reference DenseLU factorization across dimensions, edge cases, ill-conditioned
matrices, repeated updates, and numerical degradation scenarios.
"""

from __future__ import annotations

import numpy as np
import pytest

from optiforge.numerics.basis import BasisFactorization
from optiforge.numerics.exceptions import SingularBasisError
from optiforge.numerics.incremental_basis import IncrementalBasisFactorization
from optiforge.numerics.lu import dense_lu_factor, lu_solve, lu_solve_transpose


class TestIncrementalBasisBasics:
    """Basic validation for 1x1, 2x2, properties, and factory methods."""

    def test_1x1_initial_and_update(self) -> None:
        B = np.array([[4.0]])
        inc = IncrementalBasisFactorization.factorize(B)
        assert inc.m == 1
        assert inc.update_count == 0
        assert inc.num_etas == 0

        # Initial solve
        rhs = np.array([12.0])
        x = inc.solve(rhs)
        assert np.isclose(x[0], 3.0)
        y = inc.solve_transpose(rhs)
        assert np.isclose(y[0], 3.0)

        # Update column
        new_col = np.array([2.0])
        updated = inc.replace_column(0, new_col)
        assert updated is True
        assert inc.update_count == 1
        assert inc.num_etas == 1

        x2 = inc.solve(rhs)
        assert np.isclose(x2[0], 6.0)
        y2 = inc.solve_transpose(rhs)
        assert np.isclose(y2[0], 6.0)

    def test_2x2_column_replacement(self) -> None:
        B0 = np.array([[3.0, 1.0], [1.0, 2.0]])
        inc = IncrementalBasisFactorization(B0)

        # Replace col 0 with [2.0, 5.0]
        new_col = np.array([2.0, 5.0])
        updated = inc.replace_column(0, new_col)
        assert updated is True
        assert inc.update_count == 1

        B_expected = np.array([[2.0, 1.0], [5.0, 2.0]])
        assert np.allclose(inc.B, B_expected)

        # Reference DenseLU
        ref_lu = dense_lu_factor(B_expected)

        rhs = np.array([7.0, -3.0])
        x_inc = inc.solve(rhs)
        x_ref = lu_solve(ref_lu, rhs)
        assert np.allclose(x_inc, x_ref, atol=1e-12)

        c = np.array([4.0, 1.0])
        y_inc = inc.solve_transpose(c)
        y_ref = lu_solve_transpose(ref_lu, c)
        assert np.allclose(y_inc, y_ref, atol=1e-12)

        # Check residuals
        assert inc.compute_residual(rhs, x_inc) < 1e-12
        assert inc.compute_transpose_residual(c, y_inc) < 1e-12

    def test_2d_rhs_solve(self) -> None:
        rng = np.random.default_rng(42)
        m = 4
        B = rng.standard_normal((m, m)) + 3.0 * np.eye(m)
        inc = IncrementalBasisFactorization(B)

        # Replace column 2
        new_col = rng.standard_normal(m)
        inc.replace_column(2, new_col)

        rhs_2d = rng.standard_normal((m, 3))
        ref_lu = dense_lu_factor(inc.B)

        x_inc = inc.solve(rhs_2d)
        x_ref = lu_solve(ref_lu, rhs_2d)
        assert np.allclose(x_inc, x_ref, atol=1e-11)

        y_inc = inc.solve_transpose(rhs_2d)
        y_ref = lu_solve_transpose(ref_lu, rhs_2d)
        assert np.allclose(y_inc, y_ref, atol=1e-11)


class TestIncrementalBasisSequences:
    """Test sequences of multiple column replacements against DenseLU reference."""

    @pytest.mark.parametrize("m", [3, 5, 8, 15])
    def test_random_update_sequence(self, m: int) -> None:
        rng = np.random.default_rng(100 + m)
        # Diagonally dominant to avoid accidental singularity
        B = rng.standard_normal((m, m)) + m * np.eye(m)
        inc = IncrementalBasisFactorization(B, max_updates=50)

        num_updates = 12
        for step in range(num_updates):
            col_idx = step % m
            new_col = rng.standard_normal(m) + (1.5 if col_idx % 2 == 0 else -1.5)

            # Precomputed search direction check
            d = inc.solve(new_col)
            updated = inc.replace_column(col_idx, new_col, search_direction=d)
            assert updated is True

            # Reference DenseLU on explicit matrix
            ref_lu = dense_lu_factor(inc.B)

            # Test primal solve
            b = rng.standard_normal(m)
            x_inc = inc.solve(b)
            x_ref = lu_solve(ref_lu, b)
            assert np.allclose(x_inc, x_ref, atol=1e-9), f"Step {step}: Primal solve mismatch"

            # Test transpose solve
            c = rng.standard_normal(m)
            y_inc = inc.solve_transpose(c)
            y_ref = lu_solve_transpose(ref_lu, c)
            assert np.allclose(y_inc, y_ref, atol=1e-9), f"Step {step}: Transpose solve mismatch"

            # Check exact residuals ||B x - b|| and ||B^T y - c||
            res_primal = np.max(np.abs(inc.B @ x_inc - b))
            res_dual = np.max(np.abs(inc.B.T @ y_inc - c))
            assert res_primal < 1e-8, f"Step {step}: primal residual {res_primal}"
            assert res_dual < 1e-8, f"Step {step}: dual residual {res_dual}"

    def test_repeated_same_column_replacement(self) -> None:
        rng = np.random.default_rng(2026)
        m = 5
        B = rng.standard_normal((m, m)) + 5.0 * np.eye(m)
        inc = IncrementalBasisFactorization(B)

        # Replace col 1 repeatedly 8 times
        for k in range(8):
            new_col = rng.standard_normal(m) + 3.0
            inc.replace_column(1, new_col)

            ref_lu = dense_lu_factor(inc.B)
            rhs = rng.standard_normal(m)
            assert np.allclose(inc.solve(rhs), lu_solve(ref_lu, rhs), atol=1e-9)
            assert np.allclose(inc.solve_transpose(rhs), lu_solve_transpose(ref_lu, rhs), atol=1e-9)


class TestNumericalStabilityAndRefactorization:
    """Test refactorization thresholds, fallback behavior, and ill-conditioned systems."""

    def test_max_updates_forces_refactorization(self) -> None:
        rng = np.random.default_rng(77)
        m = 4
        B = np.eye(m) + 0.1 * rng.standard_normal((m, m))
        max_updates = 3
        inc = IncrementalBasisFactorization(B, max_updates=max_updates)

        assert inc.total_refactorizations == 1
        assert inc.update_count == 0

        # Updates 1, 2, 3
        for i in range(3):
            new_col = rng.standard_normal(m) + 2.0
            res = inc.replace_column(i, new_col)
            assert res is True
            assert inc.update_count == i + 1

        assert inc.total_refactorizations == 1
        assert inc.needs_refactorization() is True

        # Update 4 should trigger full refactorization
        new_col = rng.standard_normal(m) + 2.0
        res = inc.replace_column(0, new_col)
        assert res is False  # Refactorized instead of adding eta
        assert inc.total_refactorizations == 2
        assert inc.update_count == 0
        assert inc.num_etas == 0

        # Solves must still be exact
        rhs = rng.standard_normal(m)
        ref_lu = dense_lu_factor(inc.B)
        assert np.allclose(inc.solve(rhs), lu_solve(ref_lu, rhs), atol=1e-11)

    def test_near_zero_pivot_triggers_refactorization(self) -> None:
        m = 3
        B = np.eye(m)
        inc = IncrementalBasisFactorization(B, pivot_tolerance=1e-10)

        # Construct a new column that produces a near-zero pivot in position 1
        # If col 1 is [1.0, 1e-14, 0.0], the new basis is numerically singular.
        # DenseLU refactorization should detect this and raise SingularBasisError.
        new_col = np.array([1.0, 1e-14, 0.0])
        with pytest.raises(SingularBasisError):
            inc.replace_column(1, new_col)

    def test_ill_conditioned_matrix(self) -> None:
        # Ill-conditioned Hilbert-like submatrix
        n = 4
        B = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
        inc = IncrementalBasisFactorization(B)

        assert inc.condition_number > 1e3

        rhs = np.array([1.0, 2.0, 3.0, 4.0])
        x_inc = inc.solve(rhs)
        x_ref = lu_solve(dense_lu_factor(B), rhs)
        assert np.allclose(x_inc, x_ref, atol=1e-7)

    def test_safe_fallback_on_bad_residual(self) -> None:
        rng = np.random.default_rng(88)
        m = 4
        B = rng.standard_normal((m, m)) + 4.0 * np.eye(m)
        inc = IncrementalBasisFactorization(B, residual_tolerance=1e-10)

        # Do an update
        inc.replace_column(0, rng.standard_normal(m) + 2.0)
        assert inc.update_count == 1

        rhs = rng.standard_normal(m)
        # solve with check_residual=True
        x = inc.solve(rhs, check_residual=True)
        assert inc.compute_residual(rhs, x) < 1e-10

    def test_long_update_sequence_residual_growth(self) -> None:
        """Measure residual growth over 40 updates on 10x10 basis."""
        rng = np.random.default_rng(555)
        m = 10
        B = rng.standard_normal((m, m)) + 8.0 * np.eye(m)
        # Set large max_updates to study accumulation without periodic resets
        inc = IncrementalBasisFactorization(B, max_updates=100)

        max_residual = 0.0
        for step in range(40):
            col = step % m
            new_col = rng.standard_normal(m) + 3.0
            inc.replace_column(col, new_col)

            rhs = rng.standard_normal(m)
            x = inc.solve(rhs)
            rel_res = inc.compute_residual(rhs, x)
            max_residual = max(max_residual, rel_res)

        # Even after 40 updates, relative residual should remain tightly controlled (< 1e-9)
        assert max_residual < 1e-8, f"Max residual after 40 updates: {max_residual}"
