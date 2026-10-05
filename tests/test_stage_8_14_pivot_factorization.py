"""Stage 8.14: Incremental Basis Factorization for Revised Simplex Pivot Updates.

Dedicated test suite verifying:
1. One-update primal solve
2. One-update transpose solve
3. Repeated updates
4. 5 updates
5. 10 updates
6. 25 updates
7. 50 updates
8. Explicit refactorization
9. Residual trigger
10. Singular fallback
11. NaN/Inf fallback
12. Near-singular pivot
13. Memory isolation
14. 2D RHS
15. Primal simplex equivalence
16. Phase-I compatibility
17. Feature-flag isolation
"""

from __future__ import annotations

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
)
from optiforge.numerics.exceptions import SingularBasisError
from optiforge.numerics.lu import dense_lu_factor, lu_solve, lu_solve_transpose
from optiforge.numerics.pivot_basis import (
    EtaMatrixUpdate,
    PivotUpdateFactorization,
    RefactorizationReason,
)


def _generate_well_conditioned_basis(m: int, seed: int = 42) -> np.ndarray:
    """Generate a well-conditioned random square basis matrix."""
    rng = np.random.default_rng(seed)
    # Random orthogonal matrix with controlled singular values
    H = rng.standard_normal((m, m))
    Q, _ = np.linalg.qr(H)
    s = np.linspace(1.0, 10.0, m)
    return Q @ np.diag(s) @ Q.T


class TestPivotUpdateFactorizationUnit:
    """Mathematical and numerical verification of PivotUpdateFactorization."""

    def test_01_one_update_primal_solve(self) -> None:
        """Verify primal solve B_new x = b after 1 column replacement."""
        m = 10
        B = _generate_well_conditioned_basis(m, seed=1)
        fact = PivotUpdateFactorization(B, max_updates=30)

        rng = np.random.default_rng(2)
        p = 3
        new_col = rng.standard_normal(m) + 2.0
        success = fact.replace_column(p, new_col)
        assert success is True
        assert fact.get_update_count() == 1

        b = rng.standard_normal(m)
        x_inc = fact.solve(b)

        # Fresh reference LU
        B_ref = B.copy()
        B_ref[:, p] = new_col
        lu_ref = dense_lu_factor(B_ref)
        x_ref = lu_solve(lu_ref, b)

        np.testing.assert_allclose(x_inc, x_ref, rtol=1e-11, atol=1e-11)
        assert fact.residual(b, x_inc) < 1e-11

    def test_02_one_update_transpose_solve(self) -> None:
        """Verify transpose solve B_new^T y = c after 1 column replacement."""
        m = 10
        B = _generate_well_conditioned_basis(m, seed=10)
        fact = PivotUpdateFactorization(B, max_updates=30)

        rng = np.random.default_rng(11)
        p = 5
        new_col = rng.standard_normal(m)
        fact.replace_column(p, new_col)

        c = rng.standard_normal(m)
        y_inc = fact.solve_transpose(c)

        B_ref = B.copy()
        B_ref[:, p] = new_col
        lu_ref = dense_lu_factor(B_ref)
        y_ref = lu_solve_transpose(lu_ref, c)

        np.testing.assert_allclose(y_inc, y_ref, rtol=1e-11, atol=1e-11)
        assert fact.transpose_residual(c, y_inc) < 1e-11

    def test_03_repeated_updates(self) -> None:
        """Verify primal and transpose solves match fresh LU after each update in a chain."""
        m = 8
        B = _generate_well_conditioned_basis(m, seed=20)
        fact = PivotUpdateFactorization(B, max_updates=30)
        B_curr = B.copy()

        rng = np.random.default_rng(21)
        for step in range(1, 15):
            p = step % m
            new_col = rng.standard_normal(m) + 1.0
            fact.replace_column(p, new_col)
            B_curr[:, p] = new_col

            rhs = rng.standard_normal(m)
            x_inc = fact.solve(rhs)
            y_inc = fact.solve_transpose(rhs)

            lu_curr = dense_lu_factor(B_curr)
            x_ref = lu_solve(lu_curr, rhs)
            y_ref = lu_solve_transpose(lu_curr, rhs)

            np.testing.assert_allclose(x_inc, x_ref, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(y_inc, y_ref, rtol=1e-10, atol=1e-10)

    def test_04_5_updates(self) -> None:
        """Test exact accuracy after 5 sequential updates."""
        m = 12
        B = _generate_well_conditioned_basis(m, seed=30)
        fact = PivotUpdateFactorization(B, max_updates=30)
        B_curr = B.copy()
        rng = np.random.default_rng(31)

        for step in range(5):
            p = step * 2 % m
            col = rng.standard_normal(m) + 3.0
            fact.replace_column(p, col)
            B_curr[:, p] = col

        assert fact.get_update_count() == 5
        b = rng.standard_normal(m)
        lu_ref = dense_lu_factor(B_curr)
        np.testing.assert_allclose(fact.solve(b), lu_solve(lu_ref, b), rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(fact.solve_transpose(b), lu_solve_transpose(lu_ref, b), rtol=1e-11, atol=1e-11)

    def test_05_10_updates(self) -> None:
        """Test exact accuracy after 10 sequential updates."""
        m = 15
        B = _generate_well_conditioned_basis(m, seed=40)
        fact = PivotUpdateFactorization(B, max_updates=30)
        B_curr = B.copy()
        rng = np.random.default_rng(41)

        for step in range(10):
            p = step % m
            col = rng.standard_normal(m) + 2.0
            fact.replace_column(p, col)
            B_curr[:, p] = col

        assert fact.get_update_count() == 10
        b = rng.standard_normal(m)
        lu_ref = dense_lu_factor(B_curr)
        np.testing.assert_allclose(fact.solve(b), lu_solve(lu_ref, b), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(fact.solve_transpose(b), lu_solve_transpose(lu_ref, b), rtol=1e-10, atol=1e-10)

    def test_06_25_updates(self) -> None:
        """Test exact accuracy after 25 sequential updates without refactorization."""
        m = 20
        B = _generate_well_conditioned_basis(m, seed=50)
        fact = PivotUpdateFactorization(B, max_updates=30)
        B_curr = B.copy()
        rng = np.random.default_rng(51)

        for step in range(25):
            p = step % m
            col = rng.standard_normal(m) + 1.5
            fact.replace_column(p, col)
            B_curr[:, p] = col

        assert fact.get_update_count() == 25
        assert fact.get_total_refactorizations() == 1
        b = rng.standard_normal(m)
        lu_ref = dense_lu_factor(B_curr)
        np.testing.assert_allclose(fact.solve(b), lu_solve(lu_ref, b), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(fact.solve_transpose(b), lu_solve_transpose(lu_ref, b), rtol=1e-9, atol=1e-9)

    def test_07_50_updates(self) -> None:
        """Test accuracy across 50 updates spanning across periodic refactorization at max_updates=30."""
        m = 25
        B = _generate_well_conditioned_basis(m, seed=60)
        fact = PivotUpdateFactorization(B, max_updates=30)
        B_curr = B.copy()
        rng = np.random.default_rng(61)

        for step in range(50):
            p = step % m
            col = rng.standard_normal(m) + 1.0
            fact.replace_column(p, col)
            B_curr[:, p] = col

        # Refactorization should have triggered at update 30
        assert fact.get_total_refactorizations() >= 2
        assert fact.get_update_count() == 19
        b = rng.standard_normal(m)
        lu_ref = dense_lu_factor(B_curr)
        np.testing.assert_allclose(fact.solve(b), lu_solve(lu_ref, b), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(fact.solve_transpose(b), lu_solve_transpose(lu_ref, b), rtol=1e-9, atol=1e-9)

    def test_08_refactorization(self) -> None:
        """Verify explicit refactorization resets update chain and updates reason code."""
        m = 6
        B = _generate_well_conditioned_basis(m, seed=70)
        fact = PivotUpdateFactorization(B, max_updates=10)

        rng = np.random.default_rng(71)
        for _ in range(3):
            fact.replace_column(0, rng.standard_normal(m) + 1.0)
        assert fact.get_update_count() == 3

        fact.refactorize(reason=RefactorizationReason.EXPLICIT_REQUEST)
        assert fact.get_update_count() == 0
        assert fact.get_refactorization_reason() == RefactorizationReason.EXPLICIT_REQUEST
        assert fact.needs_refactorization() is False

    def test_09_residual_trigger(self) -> None:
        """Verify residual check detects degradation and triggers refactorization."""
        m = 5
        B = _generate_well_conditioned_basis(m, seed=80)
        fact = PivotUpdateFactorization(B, residual_tolerance=1e-8)

        # Corrupt WY state artificially to simulate numerical degradation.
        # Stage 8.25: internal storage is Compact WY (W, S, pivots); zero out
        # the first W column so the accumulated update is suppressed and the
        # residual check detects the degradation.
        fact.replace_column(0, np.array([2.0, 1.0, 0.0, 0.0, 0.0]))
        assert fact.get_update_count() == 1
        fact._W[:, 0] = 0.0  # suppress the eta correction → solution will be wrong

        initial_refactors = fact.get_total_refactorizations()
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x = fact.solve(b, check_residual=True)

        # Should have fallen back and refactored
        assert fact.get_total_refactorizations() > initial_refactors
        assert fact.get_refactorization_reason() == RefactorizationReason.RESIDUAL_LIMIT
        assert fact.residual(b, x) < 1e-12

    def test_10_singular_fallback(self) -> None:
        """Verify singular column replacement triggers SingularBasisError and preserves state."""
        m = 4
        B = np.eye(m)
        fact = PivotUpdateFactorization(B)

        # Replaced column is all zeros -> singular
        zero_col = np.zeros(m)
        with pytest.raises(SingularBasisError):
            fact.replace_column(0, zero_col)
        # Factorizer rolled back column 0 and remains invertible
        assert np.allclose(fact.B, np.eye(m))

    def test_11_nan_inf_fallback(self) -> None:
        """Verify NaN or Inf in entering column triggers SingularBasisError/instability."""
        m = 4
        B = np.eye(m)
        fact = PivotUpdateFactorization(B)

        nan_col = np.array([np.nan, 1.0, 0.0, 0.0])
        with pytest.raises(SingularBasisError):
            fact.replace_column(0, nan_col)

    def test_12_near_singular_pivot(self) -> None:
        """Verify pivot with element < pivot_tolerance triggers PIVOT_INSTABILITY / SingularBasisError."""
        m = 4
        B = np.eye(m)
        fact = PivotUpdateFactorization(B, pivot_tolerance=1e-8)

        tiny_col = np.array([1e-10, 0.0, 0.0, 0.0])
        with pytest.raises(SingularBasisError):
            fact.replace_column(0, tiny_col)

    def test_13_memory_isolation(self) -> None:
        """Verify mutating caller arrays does not corrupt factorizer state."""
        m = 5
        B = np.eye(m)
        fact = PivotUpdateFactorization(B)

        col = np.array([2.0, 0.0, 0.0, 0.0, 0.0])
        fact.replace_column(0, col)
        col[0] = 999.0  # mutate caller array

        assert fact.B[0, 0] == 2.0  # internal B unchanged

    def test_14_2d_rhs(self) -> None:
        """Verify primal and transpose solves with 2D RHS arrays of shape (m, k)."""
        m = 6
        k = 3
        B = _generate_well_conditioned_basis(m, seed=90)
        fact = PivotUpdateFactorization(B)

        rng = np.random.default_rng(91)
        for _ in range(4):
            fact.replace_column(rng.integers(0, m), rng.standard_normal(m) + 1.0)

        rhs_2d = rng.standard_normal((m, k))
        X_inc = fact.solve(rhs_2d)
        Y_inc = fact.solve_transpose(rhs_2d)

        lu_ref = dense_lu_factor(fact.B)
        X_ref = lu_solve(lu_ref, rhs_2d)
        Y_ref = lu_solve_transpose(lu_ref, rhs_2d)

        np.testing.assert_allclose(X_inc, X_ref, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(Y_inc, Y_ref, rtol=1e-10, atol=1e-10)


class TestPivotFactorizationSimplexIntegration:
    """Integration of PivotUpdateFactorization into Revised Simplex."""

    def test_15_primal_simplex_equivalence(self) -> None:
        """Verify Revised Simplex with use_pivot_factorization produces identical solution to reference."""
        m = Model("simplex_pfi_equiv")
        x = m.add_variable(Variable("x", lower_bound=0.0))
        y = m.add_variable(Variable("y", lower_bound=0.0))
        m.set_objective(Objective(coefficients={"x": 3.0, "y": 2.0}, sense=ObjectiveSense.MAXIMIZE))
        m.add_constraint(Constraint("c1", coefficients={"x": 2.0, "y": 1.0}, sense=ConstraintSense.LESS_EQUAL, rhs=10.0))
        m.add_constraint(Constraint("c2", coefficients={"x": 1.0, "y": 2.0}, sense=ConstraintSense.LESS_EQUAL, rhs=8.0))

        solver_ref = RevisedSimplexSolver(tolerances=SimplexTolerances(use_pivot_factorization=False))
        sol_ref = solver_ref.solve(m)

        solver_pfi = RevisedSimplexSolver(tolerances=SimplexTolerances(use_pivot_factorization=True))
        sol_pfi = solver_pfi.solve(m)

        assert sol_ref.status == sol_pfi.status == SolutionStatus.OPTIMAL
        assert np.isclose(sol_ref.objective_value, sol_pfi.objective_value, atol=1e-8)
        assert np.isclose(sol_ref.variable_values["x"], sol_pfi.variable_values["x"], atol=1e-8)
        assert np.isclose(sol_ref.variable_values["y"], sol_pfi.variable_values["y"], atol=1e-8)
        assert sol_ref.iterations == sol_pfi.iterations

    def test_16_phase1_compatibility(self) -> None:
        """Verify Phase-I with artificial variables and Phase-I -> Phase-II inheritance works with PFI."""
        m = Model("min_phase1_pfi")
        x1 = m.add_variable(Variable("x1", lower_bound=0.0))
        x2 = m.add_variable(Variable("x2", lower_bound=0.0))
        m.set_objective(Objective(coefficients={"x1": 4.0, "x2": 1.0}, sense=ObjectiveSense.MINIMIZE))
        m.add_constraint(Constraint("c1", coefficients={"x1": 3.0, "x2": 1.0}, sense=ConstraintSense.EQUAL, rhs=3.0))
        m.add_constraint(Constraint("c2", coefficients={"x1": 4.0, "x2": 3.0}, sense=ConstraintSense.GREATER_EQUAL, rhs=6.0))
        m.add_constraint(Constraint("c3", coefficients={"x1": 1.0, "x2": 2.0}, sense=ConstraintSense.LESS_EQUAL, rhs=4.0))

        solver_ref = RevisedSimplexSolver(tolerances=SimplexTolerances(use_pivot_factorization=False))
        sol_ref = solver_ref.solve(m)

        solver_pfi = RevisedSimplexSolver(tolerances=SimplexTolerances(use_pivot_factorization=True))
        sol_pfi = solver_pfi.solve(m)

        assert sol_ref.status == sol_pfi.status == SolutionStatus.OPTIMAL
        assert np.isclose(sol_ref.objective_value, sol_pfi.objective_value, atol=1e-8)
        assert np.isclose(sol_ref.variable_values["x1"], sol_pfi.variable_values["x1"], atol=1e-8)
        assert np.isclose(sol_ref.variable_values["x2"], sol_pfi.variable_values["x2"], atol=1e-8)
        assert sol_ref.iterations == sol_pfi.iterations

    def test_17_feature_flag_isolation(self) -> None:
        """Verify that default use_pivot_factorization is False and uses reference factorization."""
        tol_default = SimplexTolerances()
        assert tol_default.use_pivot_factorization is False

        # Solve standard model with default tolerances
        m = Model("default_tol")
        x = m.add_variable(Variable("x", lower_bound=0.0))
        m.set_objective(Objective(coefficients={"x": 1.0}, sense=ObjectiveSense.MINIMIZE))
        m.add_constraint(Constraint("c1", coefficients={"x": 1.0}, sense=ConstraintSense.GREATER_EQUAL, rhs=2.0))

        solver = RevisedSimplexSolver(tolerances=tol_default)
        sol = solver.solve(m)
        assert sol.status == SolutionStatus.OPTIMAL
        assert np.isclose(sol.objective_value, 2.0)
