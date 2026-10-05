"""Stage 8.40: Dedicated PFI and Compact-WY Optimization Tests.

Required Coverage:
1. Compact-WY update equivalence
2. Multiple consecutive PFI updates
3. FTRAN equivalence
4. BTRAN equivalence
5. Primal revised simplex regression
6. Dual revised simplex regression
7. Python LU + Python triangular provider
8. Python LU + SciPy triangular provider
9. SciPy LU + Python triangular provider
10. SciPy LU + SciPy triangular provider
11. Workspace reuse across multiple iterations
12. Dimension/refactorization changes
13. Numerical residual preservation
"""

import numpy as np
import pytest

from optiforge.lp.dual_simplex import DualRevisedSimplexSolver
from optiforge.lp.simplex import RevisedSimplexSolver, SimplexTolerances
from optiforge.model import Constraint, ConstraintSense, Model, Objective, ObjectiveSense, Variable, VariableType
from optiforge.numerics.pivot_basis import PivotUpdateFactorization, RefactorizationReason


class TestStage840PFIOptimization:
    """Comprehensive validation suite for Stage 8.40 PFI and Compact-WY optimizations."""

    # 1. Compact-WY update equivalence
    def test_01_compact_wy_update_equivalence(self) -> None:
        """Verify _update_compact_wy matches mathematical definition down to machine precision."""
        rng = np.random.default_rng(101)
        m = 40
        B = rng.standard_normal((m, m)) + 8.0 * np.eye(m)
        pfi = PivotUpdateFactorization(B, max_updates=20)

        for step in range(10):
            d = rng.standard_normal(m)
            pos = step % m
            pivot_val = float(d[pos])
            if abs(pivot_val) < 0.2:
                pivot_val = 1.5
                d[pos] = 1.5

            k = pfi.num_etas
            inv_piv = 1.0 / pivot_val
            expected_col = -d * inv_piv
            expected_col[pos] = inv_piv - 1.0

            if k > 0:
                expected_S_row = pfi._W[pos, :k] @ pfi._S[:k, :k]
            else:
                expected_S_row = np.array([])

            pfi._update_compact_wy(d, pos, pivot_val)

            np.testing.assert_allclose(pfi._W[:, k], expected_col, atol=1e-15, rtol=1e-15)
            if k > 0:
                np.testing.assert_allclose(pfi._S[k, :k], expected_S_row, atol=1e-15, rtol=1e-15)
            assert pfi._S[k, k] == 1.0
            assert pfi._pivots[k] == pos

    # 2. Multiple consecutive PFI updates
    def test_02_multiple_consecutive_pfi_updates(self) -> None:
        """Verify 25 consecutive column replacements correctly update basis matrix and Compact WY."""
        rng = np.random.default_rng(102)
        m = 30
        B = rng.standard_normal((m, m)) + 6.0 * np.eye(m)
        pfi = PivotUpdateFactorization(B, max_updates=30)

        for step in range(25):
            pos = step % m
            new_col = rng.standard_normal(m)
            success = pfi.replace_column(pos, new_col)
            assert success is True
            assert pfi.update_count == step + 1
            assert pfi.num_etas == step + 1

    # 3. FTRAN equivalence
    def test_03_ftran_equivalence(self) -> None:
        """Verify primal solve (FTRAN) matches explicit numpy solve across update sequence."""
        rng = np.random.default_rng(103)
        m = 25
        B = rng.standard_normal((m, m)) + 5.0 * np.eye(m)
        pfi = PivotUpdateFactorization(B, max_updates=20)

        for step in range(15):
            pos = step % m
            new_col = rng.standard_normal(m)
            pfi.replace_column(pos, new_col)

            rhs = rng.standard_normal(m)
            x_ftran = pfi.solve(rhs, check_residual=False)
            x_ref = np.linalg.solve(pfi.B, rhs)
            np.testing.assert_allclose(x_ftran, x_ref, atol=1e-10, rtol=1e-10)

    # 4. BTRAN equivalence
    def test_04_btran_equivalence(self) -> None:
        """Verify transpose solve (BTRAN) matches explicit numpy transpose solve."""
        rng = np.random.default_rng(104)
        m = 25
        B = rng.standard_normal((m, m)) + 5.0 * np.eye(m)
        pfi = PivotUpdateFactorization(B, max_updates=20)

        for step in range(15):
            pos = step % m
            new_col = rng.standard_normal(m)
            pfi.replace_column(pos, new_col)

            c = rng.standard_normal(m)
            y_btran = pfi.solve_transpose(c)
            y_ref = np.linalg.solve(pfi.B.T, c)
            np.testing.assert_allclose(y_btran, y_ref, atol=1e-10, rtol=1e-10)

    # 5. Primal revised simplex regression
    def test_05_primal_revised_simplex_regression(self) -> None:
        """Verify Primal Revised Simplex solves a standard LP correctly with optimized PFI."""
        model = Model(name="primal_test")
        x0 = model.add_variable(Variable("x0", lower_bound=0.0))
        x1 = model.add_variable(Variable("x1", lower_bound=0.0))
        model.add_constraint(Constraint("c0", coefficients={"x0": 1.0, "x1": 2.0},
                                        sense=ConstraintSense.LESS_EQUAL, rhs=4.0))
        model.add_constraint(Constraint("c1", coefficients={"x0": 3.0, "x1": 1.0},
                                        sense=ConstraintSense.LESS_EQUAL, rhs=6.0))
        model.set_objective(Objective({"x0": 3.0, "x1": 2.0}, sense=ObjectiveSense.MAXIMIZE))

        tol = SimplexTolerances(use_pivot_factorization=True)
        sol = RevisedSimplexSolver(tolerances=tol).solve(model)
        assert sol.status.name == "OPTIMAL"
        assert np.isclose(sol.objective_value, 7.2, atol=1e-8)
        assert np.isclose(sol.variable_values["x0"], 1.6, atol=1e-8)
        assert np.isclose(sol.variable_values["x1"], 1.2, atol=1e-8)

    # 6. Dual revised simplex regression
    def test_06_dual_revised_simplex_regression(self) -> None:
        """Verify Dual Revised Simplex solves standard LP correctly with optimized PFI."""
        rng = np.random.default_rng(106)
        m, n = 15, 30
        A_orig = rng.uniform(0.1, 1.0, size=(m, n))
        b_orig = rng.uniform(5.0, 50.0, size=m)
        c_orig = rng.uniform(1.0, 10.0, size=n)
        A_eq = np.hstack([-A_orig, np.eye(m)])
        b_eq = -b_orig
        c_eq = np.concatenate([c_orig, np.zeros(m)])
        basis = list(range(n, n + m))
        nonbasic = list(range(n))

        tol = SimplexTolerances(use_pivot_factorization=True, use_incremental_reduced_costs=None)
        solver = DualRevisedSimplexSolver(use_pivot_factorization=True, tolerances=tol)
        res = solver.solve_standard_form(A_eq, b_eq, c_eq, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)
        assert res.status.name == "OPTIMAL"
        assert res.iterations > 0

    # 7. Python LU + Python triangular provider
    def test_07_provider_py_lu_py_tri(self) -> None:
        """Verify quadrant 1: Python LU + Python triangular solve."""
        self._run_quadrant_test("python", "python")

    # 8. Python LU + SciPy triangular provider
    def test_08_provider_py_lu_sp_tri(self) -> None:
        """Verify quadrant 2: Python LU + SciPy triangular solve."""
        self._run_quadrant_test("scipy", "python")

    # 9. SciPy LU + Python triangular provider
    def test_09_provider_sp_lu_py_tri(self) -> None:
        """Verify quadrant 3: SciPy LU + Python triangular solve."""
        self._run_quadrant_test("python", "scipy")

    # 10. SciPy LU + SciPy triangular provider
    def test_10_provider_sp_lu_sp_tri(self) -> None:
        """Verify quadrant 4: SciPy LU + SciPy triangular solve."""
        self._run_quadrant_test("scipy", "scipy")

    def _run_quadrant_test(self, tri_prov: str, lu_prov: str) -> None:
        rng = np.random.default_rng(200)
        m, n = 15, 30
        A_orig = rng.uniform(0.1, 1.0, size=(m, n))
        b_orig = rng.uniform(5.0, 50.0, size=m)
        c_orig = rng.uniform(1.0, 10.0, size=n)
        A_eq = np.hstack([-A_orig, np.eye(m)])
        b_eq = -b_orig
        c_eq = np.concatenate([c_orig, np.zeros(m)])
        basis = list(range(n, n + m))
        nonbasic = list(range(n))

        tol = SimplexTolerances(
            use_pivot_factorization=True,
            use_incremental_reduced_costs=None,
            triangular_solve_provider=tri_prov,
            lu_factor_provider=lu_prov,
            use_scipy_triangular_solve=(tri_prov == "scipy"),
            use_scipy_lu_factor=(lu_prov == "scipy"),
        )
        solver = DualRevisedSimplexSolver(
            use_pivot_factorization=True,
            triangular_solve_provider=tri_prov,
            lu_factor_provider=lu_prov,
            use_scipy_triangular_solve=(tri_prov == "scipy"),
            use_scipy_lu_factor=(lu_prov == "scipy"),
            tolerances=tol,
        )
        res = solver.solve_standard_form(A_eq, b_eq, c_eq, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)
        assert res.status.name == "OPTIMAL"

    # 11. Workspace reuse across multiple iterations
    def test_11_workspace_reuse_across_multiple_iterations(self) -> None:
        """Verify internal buffers (_diff_buf, _row_abs_sums) correctly persist and reuse without leaking."""
        rng = np.random.default_rng(111)
        m = 20
        B = rng.standard_normal((m, m)) + 5.0 * np.eye(m)
        pfi = PivotUpdateFactorization(B, max_updates=25)

        for step in range(20):
            pfi.replace_column(step % m, rng.standard_normal(m))
            rhs = rng.standard_normal(m)
            x = pfi.solve(rhs, check_residual=True)
            assert pfi._diff_buf is not None
            assert pfi._diff_buf.shape == (m,)
            assert pfi._row_abs_sums is not None
            assert pfi._row_abs_sums.shape == (m,)

    # 12. Dimension/refactorization changes
    def test_12_dimension_refactorization_changes(self) -> None:
        """Verify automatic refactorization on UPDATE_LIMIT correctly resets Compact WY state."""
        rng = np.random.default_rng(112)
        m = 15
        B = rng.standard_normal((m, m)) + 5.0 * np.eye(m)
        pfi = PivotUpdateFactorization(B, max_updates=5)

        for step in range(5):
            pfi.replace_column(step % m, rng.standard_normal(m))
        assert pfi.num_etas == 5
        assert pfi.update_count == 5

        # 6th update reaches max_updates -> triggers refactorization
        success = pfi.replace_column(5, rng.standard_normal(m))
        assert success is False
        assert pfi.get_refactorization_reason() == RefactorizationReason.UPDATE_LIMIT
        assert pfi.num_etas == 0
        assert pfi.update_count == 0
        assert pfi.get_total_refactorizations() == 2

    # 13. Numerical residual preservation
    def test_13_numerical_residual_preservation(self) -> None:
        """Verify ||B x - b||_inf / scale matches standard definition within 1e-14."""
        rng = np.random.default_rng(113)
        m = 30
        B = rng.standard_normal((m, m)) + 6.0 * np.eye(m)
        pfi = PivotUpdateFactorization(B, max_updates=20)

        for step in range(12):
            pfi.replace_column(step % m, rng.standard_normal(m))

        rhs = rng.standard_normal(m)
        x = pfi.solve(rhs)

        diff = pfi.B @ x - rhs
        norm_res = float(np.max(np.abs(diff)))
        norm_b = float(np.max(np.abs(rhs)))
        norm_x = float(np.max(np.abs(x)))
        norm_B = float(np.max(np.sum(np.abs(pfi.B), axis=1)))
        scale = norm_B * norm_x + norm_b + 1e-15
        expected_res = norm_res / scale

        actual_res = pfi.residual(rhs, x)
        assert np.isclose(actual_res, expected_res, atol=1e-14, rtol=1e-14)
