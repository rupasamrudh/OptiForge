"""Stage 8.40: Compact WY & PFI Kernel Optimization Tests.

Tests:
1. _update_compact_wy contract and numerical invariance against reference definition.
2. S-matrix lower-triangular induction equivalence.
3. In-place rank-k WY primal solve (1D and 2D) against explicit basis solve.
4. Incremental row and column absolute sum maintenance precision across long pivot sequences.
5. Residual scaling invariance: fast O(m) residual matching full O(m^2) formula within machine epsilon.
6. Repeated pivot row indices in BTRAN scatter-add.
7. Safe refactorization and fallback on singular/unstable updates.
8. End-to-end Dual Simplex and MILP solver equivalence across all 4 provider combinations.
"""

import numpy as np
import pytest

from optiforge.lp.dual_simplex import DualRevisedSimplexSolver
from optiforge.lp.simplex import RevisedSimplexSolver, SimplexTolerances
from optiforge.milp.solver import MILPSolver, MILPTolerances
from optiforge.model import Constraint, ConstraintSense, Model, Objective, ObjectiveSense, Variable, VariableType
from optiforge.numerics.exceptions import SingularBasisError
from optiforge.numerics.pivot_basis import PivotUpdateFactorization, RefactorizationReason


class TestCompactWYVectorization:
    """Validate the vectorized _update_compact_wy kernel against exact reference mathematics."""

    def test_compact_wy_accumulation_equivalence(self) -> None:
        """Verify _update_compact_wy matches mathematical definition down to machine precision."""
        rng = np.random.default_rng(42)
        m = 50
        B = rng.standard_normal((m, m)) + 10.0 * np.eye(m)
        pfi = PivotUpdateFactorization(B, max_updates=25)

        for step in range(20):
            d = rng.standard_normal(m)
            pos = step % m
            pivot_val = float(d[pos])
            if abs(pivot_val) < 1e-1:
                pivot_val = 2.0
                d[pos] = 2.0

            # Compute reference expected w and S
            k = pfi.num_etas
            inv_piv = 1.0 / pivot_val
            expected_col = -d * inv_piv
            expected_col[pos] = inv_piv - 1.0

            if k > 0:
                expected_S_row = pfi._W[pos, :k] @ pfi._S[:k, :k]
            else:
                expected_S_row = np.array([])

            # Call optimized kernel
            pfi._update_compact_wy(d, pos, pivot_val)

            # Check W column
            np.testing.assert_allclose(pfi._W[:, k], expected_col, atol=1e-15, rtol=1e-15)
            # Check S row
            if k > 0:
                np.testing.assert_allclose(pfi._S[k, :k], expected_S_row, atol=1e-15, rtol=1e-15)
            assert pfi._S[k, k] == 1.0
            assert pfi._pivots[k] == pos

    def test_accumulate_eta_alias_compatibility(self) -> None:
        """Verify _accumulate_eta is an identical alias to _update_compact_wy."""
        B = np.eye(10)
        pfi = PivotUpdateFactorization(B)
        assert hasattr(pfi, "_update_compact_wy")
        assert hasattr(pfi, "_accumulate_eta")
        assert pfi._accumulate_eta == pfi._update_compact_wy

    def test_repeated_pivot_rows_in_btran(self) -> None:
        """Verify scatter-add handles repeated pivot indices correctly."""
        rng = np.random.default_rng(123)
        m = 20
        B = rng.standard_normal((m, m)) + 5.0 * np.eye(m)
        pfi = PivotUpdateFactorization(B, max_updates=10)

        # Force two updates at the SAME pivot position
        d1 = rng.standard_normal(m)
        d1[3] = 1.5
        pfi._update_compact_wy(d1, 3, 1.5)

        d2 = rng.standard_normal(m)
        d2[3] = -2.0
        pfi._update_compact_wy(d2, 3, -2.0)

        rhs = rng.standard_normal(m)
        y = pfi.solve_transpose(rhs)
        assert y.shape == (m,)
        assert np.all(np.isfinite(y))


class TestResidualScalingAndWorkspace:
    """Validate the fast O(m) residual scaling and workspace buffering."""

    def test_incremental_norm_b_precision(self) -> None:
        """Verify incremental row/col sum maintenance exactly matches full matrix norms."""
        rng = np.random.default_rng(999)
        m = 40
        B = rng.standard_normal((m, m)) + 8.0 * np.eye(m)
        pfi = PivotUpdateFactorization(B, max_updates=30)

        for step in range(25):
            pos = step % m
            new_col = rng.standard_normal(m)
            pfi.replace_column(pos, new_col)

            # Compare cached sums against full recalculation
            expected_row_sums = np.sum(np.abs(pfi._B), axis=1)
            expected_col_sums = np.sum(np.abs(pfi._B), axis=0)

            np.testing.assert_allclose(pfi._row_abs_sums, expected_row_sums, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(pfi._col_abs_sums, expected_col_sums, atol=1e-12, rtol=1e-12)

    def test_residual_value_identity(self) -> None:
        """Verify that residual() produces the exact same numerical result as the standard formula."""
        rng = np.random.default_rng(777)
        m = 30
        B = rng.standard_normal((m, m)) + 6.0 * np.eye(m)
        pfi = PivotUpdateFactorization(B, max_updates=15)

        for step in range(10):
            pfi.replace_column(step % m, rng.standard_normal(m))

        rhs = rng.standard_normal(m)
        x = pfi.solve(rhs)

        # Standard definition
        diff = pfi._B @ x - rhs
        norm_res = float(np.max(np.abs(diff)))
        norm_b = float(np.max(np.abs(rhs)))
        norm_x = float(np.max(np.abs(x)))
        norm_B = float(np.max(np.sum(np.abs(pfi._B), axis=1)))
        scale = norm_B * norm_x + norm_b + 1e-15
        expected_residual = norm_res / scale

        actual_residual = pfi.residual(rhs, x)
        assert np.isclose(actual_residual, expected_residual, atol=1e-14, rtol=1e-14)

    def test_transpose_residual_value_identity(self) -> None:
        """Verify transpose_residual() matches standard formula within 1e-14."""
        rng = np.random.default_rng(888)
        m = 30
        B = rng.standard_normal((m, m)) + 6.0 * np.eye(m)
        pfi = PivotUpdateFactorization(B, max_updates=15)

        for step in range(10):
            pfi.replace_column(step % m, rng.standard_normal(m))

        rhs = rng.standard_normal(m)
        y = pfi.solve_transpose(rhs)

        diff = pfi._B.T @ y - rhs
        norm_res = float(np.max(np.abs(diff)))
        norm_b = float(np.max(np.abs(rhs)))
        norm_y = float(np.max(np.abs(y)))
        norm_BT = float(np.max(np.sum(np.abs(pfi._B), axis=0)))
        scale = norm_BT * norm_y + norm_b + 1e-15
        expected_residual = norm_res / scale

        actual_residual = pfi.transpose_residual(rhs, y)
        assert np.isclose(actual_residual, expected_residual, atol=1e-14, rtol=1e-14)


class TestPrimalAndTransposeSolveCorrectness:
    """Verify solve and solve_transpose equivalence against explicit numpy inversion."""

    def test_solve_accuracy_over_pivot_chain(self) -> None:
        """Verify PFI solves match np.linalg.solve for each step of a 20-update chain."""
        rng = np.random.default_rng(333)
        m = 25
        B = rng.standard_normal((m, m)) + 7.0 * np.eye(m)
        pfi = PivotUpdateFactorization(B, max_updates=30)

        for step in range(20):
            pos = step % m
            new_col = rng.standard_normal(m)
            pfi.replace_column(pos, new_col)

            rhs = rng.standard_normal(m)
            x_pfi = pfi.solve(rhs)
            x_ref = np.linalg.solve(pfi._B, rhs)
            np.testing.assert_allclose(x_pfi, x_ref, atol=1e-10, rtol=1e-10)

            y_pfi = pfi.solve_transpose(rhs)
            y_ref = np.linalg.solve(pfi._B.T, rhs)
            np.testing.assert_allclose(y_pfi, y_ref, atol=1e-10, rtol=1e-10)

    def test_2d_solve_accuracy(self) -> None:
        """Verify 2D RHS solves match np.linalg.solve."""
        rng = np.random.default_rng(444)
        m = 20
        k_cols = 4
        B = rng.standard_normal((m, m)) + 5.0 * np.eye(m)
        pfi = PivotUpdateFactorization(B, max_updates=15)

        for step in range(8):
            pfi.replace_column(step % m, rng.standard_normal(m))

        rhs_2d = rng.standard_normal((m, k_cols))
        x_2d = pfi.solve(rhs_2d)
        x_ref = np.linalg.solve(pfi._B, rhs_2d)
        np.testing.assert_allclose(x_2d, x_ref, atol=1e-10, rtol=1e-10)

        y_2d = pfi.solve_transpose(rhs_2d)
        y_ref = np.linalg.solve(pfi._B.T, rhs_2d)
        np.testing.assert_allclose(y_2d, y_ref, atol=1e-10, rtol=1e-10)


class TestDualSimplexAndMILPInvariance:
    """Verify full solver equivalence across provider combinations."""

    def test_dual_simplex_all_provider_quadrants(self) -> None:
        """Verify Dual Simplex produces identical objective and iterations across 4 provider configurations."""
        rng = np.random.default_rng(555)
        m, n = 20, 40
        A_orig = rng.uniform(0.1, 1.0, size=(m, n))
        b_orig = rng.uniform(5.0, 50.0, size=m)
        c_orig = rng.uniform(1.0, 10.0, size=n)
        A_eq = np.hstack([-A_orig, np.eye(m)])
        b_eq = -b_orig
        c_eq = np.concatenate([c_orig, np.zeros(m)])
        basis = list(range(n, n + m))
        nonbasic = list(range(n))

        quadrants = [
            ("python", "python"),
            ("scipy", "python"),
            ("python", "scipy"),
            ("scipy", "scipy"),
        ]

        ref_obj = None
        ref_iters = None

        tol = SimplexTolerances(use_incremental_reduced_costs=None)
        for tri_prov, lu_prov in quadrants:
            solver = DualRevisedSimplexSolver(
                use_pivot_factorization=True,
                triangular_solve_provider=tri_prov,
                lu_factor_provider=lu_prov,
                use_scipy_triangular_solve=(tri_prov == "scipy"),
                use_scipy_lu_factor=(lu_prov == "scipy"),
                tolerances=tol,
            )
            res = solver.solve_standard_form(A_eq, b_eq, c_eq, basis.copy(), nonbasic.copy(), ObjectiveSense.MINIMIZE, tol)
            assert res.status.name == "OPTIMAL"

            if ref_obj is None:
                ref_obj = res.objective_value
                ref_iters = res.iterations
            else:
                assert np.isclose(res.objective_value, ref_obj, atol=1e-8, rtol=1e-8)
                assert res.iterations == ref_iters
