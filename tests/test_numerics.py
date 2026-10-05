"""Tests for OptiForge Stage 4: Numerical Core.

Covers:
1. LU factorization reconstructs the matrix: P B ≈ L U
2. Simple 2x2 solve
3. Larger random nonsingular matrices
4. Solve against np.linalg.solve as independent reference
5. Transpose solve against np.linalg.solve(B.T, rhs)
6. Multiple RHS solves reusing one factorization
7. Identity matrix
8. Diagonal matrix
9. Well-conditioned basis
10. Singular matrix detection
11. Near-singular / tiny-pivot detection
12. Row permutation case requiring partial pivoting
13. Zero RHS
14. Numerical tolerance behavior
15. Integration with Revised Simplex
16. Numerical diagnostics
17. Matrix scaling utilities
18. Absence of forbidden dependencies (AST scan)
"""

from __future__ import annotations

import ast
from pathlib import Path
import numpy as np
import pytest

from optiforge.core.solution import SolutionStatus
from optiforge.lp.simplex import RevisedSimplexSolver, SimplexTolerances
from optiforge.lp.standard_form import SingularBasisError, to_standard_form
from optiforge.model.constraint import Constraint, ConstraintSense
from optiforge.model.model import Model
from optiforge.model.objective import Objective, ObjectiveSense
from optiforge.model.variable import Variable
from optiforge.numerics.basis import BasisFactorization
from optiforge.numerics.diagnostics import compute_diagnostics
from optiforge.numerics.exceptions import NumericalSingularityError
from optiforge.numerics.lu import (
    dense_lu_factor,
    forward_substitution,
    back_substitution,
    lu_solve,
    lu_solve_transpose,
)
from optiforge.numerics.scaling import (
    EquilibrationScaling,
    compute_magnitude_diagnostics,
)


# ===========================================================================
# 1. LU Factorization Reconstruction (P B ≈ L U)
# ===========================================================================

def test_lu_reconstruction_random():
    """Verify P B ≈ L U, unit lower L, upper U, and orthogonal P."""
    np.random.seed(123)
    for m in [2, 3, 5, 10, 20]:
        # Generate well-conditioned random matrix
        M = np.random.randn(m, m)
        B = M + m * np.eye(m)

        lu = dense_lu_factor(B)

        # 1. P B ≈ L U
        P_mat = lu.P
        reconstructed = lu.L @ lu.U
        expected = P_mat @ B
        assert np.allclose(expected, reconstructed, atol=1e-11, rtol=1e-11)

        # 2. L is unit lower triangular
        assert np.allclose(np.diag(lu.L), 1.0)
        assert np.allclose(np.triu(lu.L, k=1), 0.0)

        # 3. U is upper triangular
        assert np.allclose(np.tril(lu.U, k=-1), 0.0)

        # 4. P is orthogonal permutation
        assert np.allclose(P_mat @ P_mat.T, np.eye(m))
        assert np.allclose(P_mat.T @ P_mat, np.eye(m))


# ===========================================================================
# 2. Simple 2x2 Solve
# ===========================================================================

def test_simple_2x2_solve():
    """Verify analytical solution of 2x2 system."""
    # [3 1] [x1] = [9]  => x1 = 2, x2 = 3
    # [1 2] [x2]   [8]
    B = np.array([[3.0, 1.0], [1.0, 2.0]])
    rhs = np.array([9.0, 8.0])

    bf = BasisFactorization(B)
    x = bf.solve(rhs)

    assert np.allclose(x, [2.0, 3.0], atol=1e-12)


# ===========================================================================
# 3. Larger Random Nonsingular Matrices
# ===========================================================================

@pytest.mark.parametrize("dim", [5, 10, 25, 40])
def test_larger_random_matrices(dim: int):
    """Verify solve accuracy on larger dimensions."""
    np.random.seed(dim)
    B = np.random.randn(dim, dim) + dim * np.eye(dim)
    rhs = np.random.randn(dim)

    bf = BasisFactorization(B)
    x = bf.solve(rhs)

    # Check residual norm ||B x - rhs||
    residual = np.linalg.norm(B @ x - rhs)
    assert residual < 1e-10


# ===========================================================================
# 4. Solve Against np.linalg.solve Reference
# ===========================================================================

def test_solve_against_reference():
    """Verify lu_solve matches np.linalg.solve across diverse matrices."""
    np.random.seed(456)
    for _ in range(15):
        m = np.random.randint(3, 15)
        B = np.random.randn(m, m) + 5.0 * np.eye(m)
        rhs = np.random.randn(m)

        bf = BasisFactorization(B)
        x_optiforge = bf.solve(rhs)
        x_reference = np.linalg.solve(B, rhs)

        assert np.allclose(x_optiforge, x_reference, atol=1e-11, rtol=1e-11)


# ===========================================================================
# 5. Transpose Solve Against np.linalg.solve(B.T, rhs) Reference
# ===========================================================================

def test_transpose_solve_against_reference():
    """Verify solve_transpose matches np.linalg.solve(B.T, rhs)."""
    np.random.seed(789)
    for _ in range(15):
        m = np.random.randint(3, 15)
        B = np.random.randn(m, m) + 5.0 * np.eye(m)
        rhs = np.random.randn(m)

        bf = BasisFactorization(B)
        y_optiforge = bf.solve_transpose(rhs)
        y_reference = np.linalg.solve(B.T, rhs)

        assert np.allclose(y_optiforge, y_reference, atol=1e-11, rtol=1e-11)
        # Verify residual ||B^T y - rhs||
        assert np.linalg.norm(B.T @ y_optiforge - rhs) < 1e-10


# ===========================================================================
# 6. Multiple RHS Solves Reusing One Factorization
# ===========================================================================

def test_multiple_rhs_reuse():
    """A single BasisFactorization instance must solve multiple RHS vectors and 2D matrices."""
    np.random.seed(101)
    m = 6
    B = np.random.randn(m, m) + 8.0 * np.eye(m)
    bf = BasisFactorization(B)

    # Multiple 1D vector RHS
    for _ in range(8):
        rhs = np.random.randn(m)
        x = bf.solve(rhs)
        x_ref = np.linalg.solve(B, rhs)
        assert np.allclose(x, x_ref, atol=1e-11)

        y = bf.solve_transpose(rhs)
        y_ref = np.linalg.solve(B.T, rhs)
        assert np.allclose(y, y_ref, atol=1e-11)

    # 2D matrix RHS (m x k)
    k = 4
    RHS_mat = np.random.randn(m, k)
    X_mat = bf.solve(RHS_mat)
    X_ref = np.linalg.solve(B, RHS_mat)
    assert np.allclose(X_mat, X_ref, atol=1e-11)

    Y_mat = bf.solve_transpose(RHS_mat)
    Y_ref = np.linalg.solve(B.T, RHS_mat)
    assert np.allclose(Y_mat, Y_ref, atol=1e-11)


# ===========================================================================
# 7. Identity Matrix
# ===========================================================================

def test_identity_matrix():
    """Factorization of identity matrix must be exact and preserve vectors."""
    for m in [1, 2, 5]:
        I = np.eye(m)
        bf = BasisFactorization(I)

        assert np.allclose(bf.L, I)
        assert np.allclose(bf.U, I)
        assert np.allclose(bf.P, I)

        rhs = np.arange(1.0, m + 1.0)
        assert np.allclose(bf.solve(rhs), rhs)
        assert np.allclose(bf.solve_transpose(rhs), rhs)


# ===========================================================================
# 8. Diagonal Matrix
# ===========================================================================

def test_diagonal_matrix():
    """Diagonal matrix with varied scales."""
    diag_vals = np.array([2.0, -5.0, 0.5, 100.0])
    B = np.diag(diag_vals)
    rhs = np.array([4.0, 10.0, 1.0, 300.0])

    bf = BasisFactorization(B)
    x = bf.solve(rhs)
    expected = rhs / diag_vals
    assert np.allclose(x, expected, atol=1e-12)

    y = bf.solve_transpose(rhs)
    assert np.allclose(y, expected, atol=1e-12)


# ===========================================================================
# 9. Well-Conditioned Basis (Orthonormal)
# ===========================================================================

def test_well_conditioned_basis():
    """Orthonormal matrix has condition number ~ 1.0."""
    np.random.seed(202)
    A = np.random.randn(6, 6)
    # QR factorization to get an orthonormal matrix Q
    Q, _ = np.linalg.qr(A)

    bf = BasisFactorization(Q)
    diag = bf.diagnostics()

    assert abs(diag.condition_number - 1.0) < 1e-8
    assert diag.success is True
    assert diag.is_singular is False

    rhs = np.random.randn(6)
    x = bf.solve(rhs)
    assert np.allclose(Q @ x, rhs, atol=1e-12)


# ===========================================================================
# 10. Singular Matrix Detection
# ===========================================================================

def test_singular_matrix_detection():
    """Singular and rank-deficient matrices must raise NumericalSingularityError."""
    # 1. Zero column
    B_zero_col = np.array([[1.0, 0.0], [2.0, 0.0]])
    with pytest.raises(NumericalSingularityError) as exc_info:
        BasisFactorization(B_zero_col)
    assert exc_info.value.step is not None

    # 2. Linearly dependent rows (row 2 = 2 * row 1)
    B_dep = np.array([[1.0, 2.0], [2.0, 4.0]])
    with pytest.raises(NumericalSingularityError):
        BasisFactorization(B_dep)

    # 3. 3x3 with row 3 = row 1 + row 2
    B_3x3 = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [5.0, 7.0, 9.0],
    ])
    with pytest.raises(NumericalSingularityError):
        BasisFactorization(B_3x3)

    # 4. Backward compatibility: SingularBasisError is also caught
    with pytest.raises(SingularBasisError):
        BasisFactorization(B_dep)


# ===========================================================================
# 11. Near-Singular / Tiny-Pivot Detection
# ===========================================================================

def test_near_singular_tiny_pivot_detection():
    """Matrix with pivot below pivot_tolerance must be rejected."""
    # Diagonal element 1e-10 with pivot_tolerance 1e-8
    B = np.array([
        [1.0, 0.0],
        [0.0, 1e-10],
    ])
    with pytest.raises(NumericalSingularityError) as exc_info:
        BasisFactorization(B, pivot_tolerance=1e-8)
    assert "below tolerance" in str(exc_info.value)


# ===========================================================================
# 12. Row Permutation Requiring Partial Pivoting
# ===========================================================================

def test_row_permutation_requiring_pivoting():
    """Matrix with zero on the diagonal requires row swap."""
    # [0 2]
    # [3 1]
    B = np.array([[0.0, 2.0], [3.0, 1.0]])
    rhs = np.array([4.0, 7.0])

    bf = BasisFactorization(B)
    # Permutation vector must show row 0 and row 1 swapped
    assert np.array_equal(bf.perm, [1, 0])

    # Reconstruct P B = L U
    assert np.allclose(bf.P @ B, bf.L @ bf.U, atol=1e-12)

    # Solve
    x = bf.solve(rhs)
    assert np.allclose(B @ x, rhs, atol=1e-12)
    # Transpose solve
    y = bf.solve_transpose(rhs)
    assert np.allclose(B.T @ y, rhs, atol=1e-12)


# ===========================================================================
# 13. Zero RHS
# ===========================================================================

def test_zero_rhs():
    """Homogeneous system B x = 0 has unique solution x = 0."""
    B = np.array([[2.0, 1.0], [5.0, 3.0]])
    bf = BasisFactorization(B)

    zero_rhs = np.zeros(2)
    x = bf.solve(zero_rhs)
    assert np.allclose(x, 0.0)

    y = bf.solve_transpose(zero_rhs)
    assert np.allclose(y, 0.0)


# ===========================================================================
# 14. Numerical Tolerance Behavior
# ===========================================================================

def test_numerical_tolerance_behavior():
    """Configurable pivot tolerance correctly accepts or rejects near-zero pivots."""
    # Pivot is 1e-6
    B = np.array([[1.0, 0.0], [0.0, 1e-6]])

    # Rejection when tolerance is 1e-5
    with pytest.raises(NumericalSingularityError):
        BasisFactorization(B, pivot_tolerance=1e-5)

    # Acceptance when tolerance is 1e-8
    bf = BasisFactorization(B, pivot_tolerance=1e-8)
    assert bf.solve(np.array([1.0, 1e-6]))[1] == pytest.approx(1.0)


# ===========================================================================
# 15. Integration with Revised Simplex
# ===========================================================================

def test_simplex_integration_standard_lp():
    """Revised Simplex using BasisFactorization solves standard LP correctly."""
    m = Model("production")
    x1 = m.add_variable(Variable("x1", lower_bound=0.0))
    x2 = m.add_variable(Variable("x2", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 2.0}, ConstraintSense.LESS_EQUAL, 8.0))
    m.add_constraint(Constraint("c2", {"x1": 3.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 9.0))
    m.set_objective(Objective({"x1": 3.0, "x2": 2.0}, ObjectiveSense.MAXIMIZE))

    solver = RevisedSimplexSolver()
    sol = solver.solve(m)

    assert sol.status == SolutionStatus.OPTIMAL
    # Optimum at x1=2, x2=3: obj = 3*2 + 2*3 = 12.0 (both constraints tight).
    assert sol.objective_value == pytest.approx(12.0, abs=1e-5)
    assert sol.variable_values["x1"] == pytest.approx(2.0, abs=1e-5)
    assert sol.variable_values["x2"] == pytest.approx(3.0, abs=1e-5)


def test_simplex_integration_phase1_two_phase():
    """Revised Simplex using BasisFactorization solves LP requiring Phase-I."""
    m = Model("blending")
    x1 = m.add_variable(Variable("x1", lower_bound=0.0))
    x2 = m.add_variable(Variable("x2", lower_bound=0.0))
    # >= constraint requires Phase-I surplus and artificial variables
    m.add_constraint(Constraint("c1", {"x1": 2.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, 6.0))
    m.add_constraint(Constraint("c2", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.set_objective(Objective({"x1": 2.0, "x2": 3.0}, ObjectiveSense.MINIMIZE))

    solver = RevisedSimplexSolver()
    sol = solver.solve(m)

    assert sol.status == SolutionStatus.OPTIMAL
    # Optimum at x1=3, x2=0: obj = 2*3 + 3*0 = 6.0 (2x1+x2=6 tight, x1+x2=3<=5).
    assert sol.objective_value == pytest.approx(6.0, abs=1e-5)
    assert sol.variable_values["x1"] == pytest.approx(3.0, abs=1e-5)
    assert sol.variable_values["x2"] == pytest.approx(0.0, abs=1e-5)


def test_simplex_integration_unbounded():
    """Revised Simplex correctly detects unbounded problems."""
    m = Model("unbounded")
    m.add_variable(Variable("x1", lower_bound=0.0))
    m.add_variable(Variable("x2", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": -1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.set_objective(Objective({"x1": 2.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))

    solver = RevisedSimplexSolver()
    sol = solver.solve(m)
    assert sol.status == SolutionStatus.UNBOUNDED


def test_simplex_integration_infeasible():
    """Revised Simplex correctly detects infeasible problems."""
    m = Model("infeasible")
    m.add_variable(Variable("x1", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 2.0))
    m.add_constraint(Constraint("c2", {"x1": 1.0}, ConstraintSense.GREATER_EQUAL, 4.0))
    m.set_objective(Objective({"x1": 1.0}, ObjectiveSense.MINIMIZE))

    solver = RevisedSimplexSolver()
    sol = solver.solve(m)
    assert sol.status == SolutionStatus.INFEASIBLE


# ===========================================================================
# 16. Numerical Diagnostics
# ===========================================================================

def test_numerical_diagnostics():
    """Inspect NumericalDiagnostics attributes."""
    B = np.array([
        [4.0, 1.0, -1.0],
        [2.0, 5.0, 1.0],
        [1.0, 1.0, 3.0],
    ])
    bf = BasisFactorization(B)
    diag = bf.diagnostics()

    assert diag.dimension == 3
    assert diag.min_abs_pivot > 0.0
    assert diag.max_abs_pivot >= diag.min_abs_pivot
    assert diag.pivot_growth >= 0.0
    assert diag.condition_number >= 1.0
    assert diag.is_singular is False
    assert diag.success is True


# ===========================================================================
# 17. Scaling Utilities
# ===========================================================================

def test_magnitude_diagnostics():
    """Verify row and column magnitude statistics."""
    A = np.array([
        [10.0, 0.0, 0.1],
        [0.0, 500.0, 2.0],
    ])
    diag = compute_magnitude_diagnostics(A)

    assert diag.max_element == 500.0
    assert diag.min_nonzero_element == 0.1
    assert diag.dynamic_range == pytest.approx(5000.0)
    assert np.allclose(diag.row_max, [10.0, 500.0])
    assert np.allclose(diag.col_max, [10.0, 500.0, 2.0])


def test_equilibration_scaling():
    """Verify matrix equilibration and unscaling."""
    A = np.array([
        [100.0, 0.02],
        [50.0, 0.01],
    ])
    b = np.array([10.0, 5.0])
    c = np.array([1.0, 2.0])

    scaler = EquilibrationScaling.from_max_norm(A)
    A_scaled, b_scaled, c_scaled = scaler.scale_system(A, b, c)

    # Scaled matrix should have max row element near 1.0
    assert np.all(np.max(np.abs(A_scaled), axis=1) <= 1.0 + 1e-12)

    # Primal unscaling check: A x = b <=> A_scaled x_scaled = b_scaled
    x_scaled = np.array([0.5, 1.2])
    x_orig = scaler.unscale_primal(x_scaled)
    assert np.allclose(A @ x_orig, scaler.row_scale**(-1) * (A_scaled @ x_scaled))


# ===========================================================================
# 18. AST Inspection: Zero Prohibited Dependencies
# ===========================================================================

def test_no_prohibited_dependencies():
    """Verify that optiforge has NO imports of scipy or external solvers."""
    prohibited_modules = {
        "scipy",
        "scipy.linalg",
        "scipy.optimize",
        "scipy.sparse",
        "gurobipy",
        "cplex",
        "pulp",
        "pyomo",
        "ortools",
        "highspy",
    }

    optiforge_dir = Path(__file__).resolve().parent.parent / "optiforge"
    for py_file in optiforge_dir.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for prohibited in prohibited_modules:
                        assert not alias.name.startswith(prohibited), (
                            f"Prohibited import '{alias.name}' found in {py_file}"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for prohibited in prohibited_modules:
                        assert not node.module.startswith(prohibited), (
                            f"Prohibited import-from '{node.module}' found in {py_file}"
                        )
