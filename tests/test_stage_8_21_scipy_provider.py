"""Stage 8.21 Tests: Optional SciPy Triangular Solve Provider.

Covers:
A. SciPy unavailable handling
B. Feature disabled behavior (default pure NumPy path)
C. Feature enabled behavior
D. Dimension m below threshold (m < 10: pure NumPy path)
E. Dimension m exactly threshold (m = 10: dispatch permitted)
F. Dimension m above threshold (m > 10: dispatch active)
G. Float32 input precision contract (normalized to float64 output)
H. Non-contiguous RHS vectors
I. C-contiguous matrix layout
J. Fortran-contiguous matrix layout
K. Exact-zero diagonal handling & LinAlgError fallback
L. Near-zero diagonal handling
M. Non-finite values (NaN / Inf) in RHS or matrix
N. Transpose solve correctness and P/L/U ordering
O. Provider exception triggers deterministic fallback to pure NumPy
P. Numerical equivalence between NumPy and SciPy paths
Q. DenseLU integration with solve() and solve_transpose()
R. RevisedSimplexSolver integration
S. DualRevisedSimplexSolver integration
T. MILPSolver warm-start relaxation integration
"""

import math
import numpy as np
import pytest

from optiforge.core.solution import SolutionStatus
from optiforge.lp.simplex import RevisedSimplexSolver, SimplexTolerances
from optiforge.lp.dual_simplex import DualRevisedSimplexSolver
from optiforge.milp.solver import MILPSolver, MILPTolerances
from optiforge.model.constraint import Constraint, ConstraintSense
from optiforge.model.model import Model
from optiforge.model.objective import Objective, ObjectiveSense
from optiforge.model.variable import Variable, VariableType
from optiforge.numerics.basis import BasisFactorization
from optiforge.numerics.exceptions import NumericalSingularityError, SingularBasisError
from optiforge.numerics.lu import (
    DenseLU,
    dense_lu_factor,
    forward_substitution,
    back_substitution,
    lu_solve,
    lu_solve_transpose,
)
from optiforge.numerics.pivot_basis import PivotUpdateFactorization
from optiforge.numerics.triangular_provider import (
    backward_solve,
    forward_solve,
    get_provider_stats,
    is_scipy_available,
    reset_provider_stats,
    triangular_lu_solve,
    triangular_lu_solve_transpose,
)


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def _make_triangular_pair(m: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    L = np.tril(rng.standard_normal((m, m)) * 0.1, -1) + np.eye(m)
    U = np.triu(rng.standard_normal((m, m)) * 0.1, 1) + np.eye(m) * 2.0
    return L, U


# ---------------------------------------------------------------------------
# A & B: Feature Disabled / Unavailable Semantics
# ---------------------------------------------------------------------------

def test_feature_disabled_default_behavior():
    """Default configuration use_scipy=False must use reference NumPy substitution."""
    reset_provider_stats()
    L, U = _make_triangular_pair(20)
    b = np.ones(20)

    y = forward_solve(L, b, use_scipy=False)
    x = backward_solve(U, y, use_scipy=False)

    stats = get_provider_stats()
    assert stats["total_solves"] == 2
    assert stats["scipy_solves"] == 0
    assert stats["fallbacks"] == 0

    # Must match pure NumPy functions exactly
    assert np.allclose(y, forward_substitution(L, b))
    assert np.allclose(x, back_substitution(U, y))


def test_scipy_unavailable_fallback(monkeypatch):
    """When SciPy is reported as unavailable, falls back cleanly to NumPy."""
    reset_provider_stats()
    monkeypatch.setattr("optiforge.numerics.triangular_provider.is_scipy_available", lambda: False)

    L, U = _make_triangular_pair(25)
    b = np.arange(1.0, 26.0)

    y = forward_solve(L, b, use_scipy=True, min_dimension=10)
    stats = get_provider_stats()
    assert stats["scipy_solves"] == 0
    assert np.allclose(y, forward_substitution(L, b))


# ---------------------------------------------------------------------------
# C, D, E, F: Dimension Threshold Policy (m < 10 vs m == 10 vs m > 10)
# ---------------------------------------------------------------------------

def test_dimension_below_threshold():
    """m < threshold (e.g. m=6 with min_dimension=10) must use pure NumPy even if use_scipy=True."""
    reset_provider_stats()
    L, U = _make_triangular_pair(6)
    b = np.ones(6)

    y = forward_solve(L, b, use_scipy=True, min_dimension=10)
    stats = get_provider_stats()
    assert stats["total_solves"] == 1
    assert stats["scipy_solves"] == 0
    assert np.allclose(y, forward_substitution(L, b))


def test_dimension_exactly_threshold():
    """m == threshold (m=10 with min_dimension=10) triggers SciPy dispatch if available."""
    reset_provider_stats()
    L, U = _make_triangular_pair(10)
    b = np.ones(10)

    y = forward_solve(L, b, use_scipy=True, min_dimension=10)
    stats = get_provider_stats()
    if is_scipy_available():
        assert stats["scipy_solves"] == 1
    else:
        assert stats["scipy_solves"] == 0
    assert np.allclose(y, forward_substitution(L, b), atol=1e-12)


def test_dimension_above_threshold():
    """m > threshold (m=50 with min_dimension=10) triggers SciPy dispatch if available."""
    reset_provider_stats()
    L, U = _make_triangular_pair(50)
    b = np.random.default_rng(99).standard_normal(50)

    y = forward_solve(L, b, use_scipy=True, min_dimension=10)
    x = backward_solve(U, y, use_scipy=True, min_dimension=10)
    stats = get_provider_stats()
    if is_scipy_available():
        assert stats["scipy_solves"] == 2
    assert np.allclose(L @ y, b, atol=1e-12)
    assert np.allclose(U @ x, y, atol=1e-12)


# ---------------------------------------------------------------------------
# G: Precision Contract & Float32 Input
# ---------------------------------------------------------------------------

def test_float32_input_precision_contract():
    """Float32 inputs must return float64 outputs and preserve precision."""
    L, U = _make_triangular_pair(15)
    b32 = np.ones(15, dtype=np.float32)

    y = forward_solve(L, b32, use_scipy=True, min_dimension=10)
    x = backward_solve(U, y, use_scipy=True, min_dimension=10)

    assert y.dtype == np.float64
    assert x.dtype == np.float64
    assert np.allclose(L @ y, b32.astype(np.float64), atol=1e-7)


# ---------------------------------------------------------------------------
# H, I, J: Memory Layouts (Strided, C-Contiguous, Fortran-Contiguous)
# ---------------------------------------------------------------------------

def test_non_contiguous_rhs():
    """Strided non-contiguous RHS must solve correctly."""
    L, U = _make_triangular_pair(12)
    b_full = np.arange(24.0)
    b_strided = b_full[::2]
    assert not b_strided.flags.c_contiguous

    y = forward_solve(L, b_strided, use_scipy=True, min_dimension=10)
    assert np.allclose(L @ y, b_strided, atol=1e-12)


def test_c_and_f_contiguous_matrices():
    """Both C-contiguous and Fortran-contiguous matrices solve accurately."""
    m = 20
    L_c, U_c = _make_triangular_pair(m)
    L_f = np.asfortranarray(L_c)
    U_f = np.asfortranarray(U_c)
    b = np.ones(m)

    y_c = forward_solve(L_c, b, use_scipy=True, min_dimension=10)
    y_f = forward_solve(L_f, b, use_scipy=True, min_dimension=10)
    assert np.allclose(y_c, y_f, atol=1e-12)

    x_c = backward_solve(U_c, y_c, use_scipy=True, min_dimension=10)
    x_f = backward_solve(U_f, y_f, use_scipy=True, min_dimension=10)
    assert np.allclose(x_c, x_f, atol=1e-12)


# ---------------------------------------------------------------------------
# K, L, M: Adversarial Numerical Cases & Fallback
# ---------------------------------------------------------------------------

def test_exact_zero_diagonal_fallback():
    """Exact zero on diagonal triggers graceful fallback to pure NumPy without crash."""
    reset_provider_stats()
    m = 12
    _, U = _make_triangular_pair(m)
    U_zero = U.copy()
    U_zero[m - 1, m - 1] = 0.0
    b = np.ones(m)

    # When calling backward_solve, SciPy raises LinAlgError on singular diagonal.
    # The provider must intercept it and fallback to NumPy.
    x = backward_solve(U_zero, b, use_scipy=True, min_dimension=10)
    stats = get_provider_stats()
    if is_scipy_available():
        assert stats["fallbacks"] >= 1
    # Pure NumPy path division by zero yields +/-inf or nan (standard IEEE 754)
    assert np.isinf(x[-1]) or np.isnan(x[-1])


def test_near_zero_diagonal():
    """Near-zero diagonal (1e-16) solves backward stably."""
    m = 15
    _, U = _make_triangular_pair(m)
    U_near = U.copy()
    U_near[m - 1, m - 1] = 1e-16
    b = np.ones(m)

    x_base = backward_solve(U_near, b, use_scipy=False)
    x_scipy = backward_solve(U_near, b, use_scipy=True, min_dimension=10)
    assert np.isclose(x_base[-1], x_scipy[-1], rtol=1e-5)


# ---------------------------------------------------------------------------
# N: Transpose Solves & P/L/U Invariants
# ---------------------------------------------------------------------------

def test_transpose_solve_correctness():
    """Solve B^T x = rhs matches exact transpose problem."""
    m = 20
    rng = np.random.default_rng(101)
    B = rng.standard_normal((m, m)) + np.eye(m) * 5.0
    lu = dense_lu_factor(B, use_scipy_triangular_solve=True, scipy_triangular_min_dimension=10)
    b = rng.standard_normal(m)

    xt_base = lu_solve_transpose(lu, b, use_scipy=False)
    xt_scipy = lu_solve_transpose(lu, b, use_scipy=True, min_dimension=10)

    # Both must satisfy B^T x = b
    assert np.allclose(B.T @ xt_base, b, atol=1e-10)
    assert np.allclose(B.T @ xt_scipy, b, atol=1e-10)
    assert np.allclose(xt_base, xt_scipy, atol=1e-10)


# ---------------------------------------------------------------------------
# O: Simulated Provider Exception Triggers NumPy Fallback
# ---------------------------------------------------------------------------

def test_simulated_exception_fallback(monkeypatch):
    """If SciPy raises LinAlgError, provider falls back to pure NumPy."""
    reset_provider_stats()
    L, U = _make_triangular_pair(15)
    b = np.ones(15)

    def mock_failing_solve(*args, **kwargs):
        raise np.linalg.LinAlgError("Simulated LAPACK singularity")

    monkeypatch.setattr("optiforge.numerics.triangular_provider._SCIPY_SOLVE_TRIANGULAR", mock_failing_solve)
    monkeypatch.setattr("optiforge.numerics.triangular_provider.is_scipy_available", lambda: True)

    y = forward_solve(L, b, use_scipy=True, min_dimension=10)
    stats = get_provider_stats()
    assert stats["fallbacks"] == 1
    assert np.allclose(y, forward_substitution(L, b))


# ---------------------------------------------------------------------------
# P & Q: DenseLU Integration
# ---------------------------------------------------------------------------

def test_denselu_integration():
    """DenseLU.solve() and solve_transpose() respect use_scipy configuration."""
    m = 25
    rng = np.random.default_rng(77)
    B = rng.standard_normal((m, m)) + np.eye(m) * 4.0
    lu = dense_lu_factor(B, use_scipy_triangular_solve=True, scipy_triangular_min_dimension=10)
    b = rng.standard_normal(m)

    x = lu.solve(b)
    xt = lu.solve_transpose(b)

    assert np.allclose(B @ x, b, atol=1e-10)
    assert np.allclose(B.T @ xt, b, atol=1e-10)


# ---------------------------------------------------------------------------
# R: RevisedSimplexSolver Integration
# ---------------------------------------------------------------------------

def test_revised_simplex_integration():
    """RevisedSimplexSolver produces identical optimal solution with SciPy enabled."""
    model = Model("test_simplex_scipy")
    n = 15
    m = 8
    rng = np.random.default_rng(42)
    for j in range(n):
        model.add_variable(Variable(f"x_{j}", lower_bound=0.0))

    A = rng.uniform(0.5, 2.0, (m, n))
    b = rng.uniform(20.0, 50.0, m)
    c = rng.uniform(1.0, 5.0, n)

    for i in range(m):
        coeffs = {f"x_{j}": float(A[i, j]) for j in range(n)}
        model.add_constraint(Constraint(f"c_{i}", coeffs, ConstraintSense.LESS_EQUAL, float(b[i])))
    obj = {f"x_{j}": float(c[j]) for j in range(n)}
    model.set_objective(Objective(obj, ObjectiveSense.MAXIMIZE))

    # 1. Base solve
    tol_base = SimplexTolerances(use_scipy_triangular_solve=False)
    res_base = RevisedSimplexSolver(tolerances=tol_base).solve(model)

    # 2. SciPy solve
    tol_sp = SimplexTolerances(use_scipy_triangular_solve=True, scipy_triangular_min_dimension=5)
    res_sp = RevisedSimplexSolver(tolerances=tol_sp).solve(model)

    assert res_base.status == SolutionStatus.OPTIMAL
    assert res_sp.status == SolutionStatus.OPTIMAL
    assert math.isclose(res_base.objective_value, res_sp.objective_value, rel_tol=1e-6)
    assert res_base.iterations == res_sp.iterations


# ---------------------------------------------------------------------------
# S: DualRevisedSimplexSolver Integration
# ---------------------------------------------------------------------------

def test_dual_simplex_integration():
    """DualRevisedSimplexSolver solves identically with SciPy enabled."""
    model = Model("test_dual_scipy")
    for j in range(6):
        model.add_variable(Variable(f"x_{j}", lower_bound=0.0))

    coeffs1 = {"x_0": 1.0, "x_1": 2.0, "x_2": 1.0}
    model.add_constraint(Constraint("c1", coeffs1, ConstraintSense.LESS_EQUAL, 10.0))
    coeffs2 = {"x_3": 2.0, "x_4": 1.0, "x_5": 3.0}
    model.add_constraint(Constraint("c2", coeffs2, ConstraintSense.LESS_EQUAL, 15.0))
    obj = {"x_0": 2.0, "x_1": 3.0, "x_2": 1.0, "x_3": 4.0, "x_4": 2.0, "x_5": 5.0}
    model.set_objective(Objective(obj, ObjectiveSense.MAXIMIZE))

    tol_base = SimplexTolerances(use_scipy_triangular_solve=False)
    solver_base = DualRevisedSimplexSolver(tolerances=tol_base)
    res_base = solver_base.solve(model)

    tol_sp = SimplexTolerances(use_scipy_triangular_solve=True, scipy_triangular_min_dimension=2)
    solver_sp = DualRevisedSimplexSolver(tolerances=tol_sp, use_scipy_triangular_solve=True, scipy_triangular_min_dimension=2)
    res_sp = solver_sp.solve(model)

    assert res_base.status == SolutionStatus.OPTIMAL
    assert res_sp.status == SolutionStatus.OPTIMAL
    assert math.isclose(res_base.objective_value, res_sp.objective_value, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# T: MILPSolver Warm-Start Integration
# ---------------------------------------------------------------------------

def test_milp_solver_integration():
    """MILP branch-and-bound solves correctly with SciPy provider enabled."""
    model = Model("knapsack_test")
    model.add_variable(Variable("x0", variable_type=VariableType.BINARY))
    model.add_variable(Variable("x1", variable_type=VariableType.BINARY))
    model.add_variable(Variable("x2", variable_type=VariableType.BINARY))

    model.add_constraint(Constraint("w", {"x0": 2.0, "x1": 3.0, "x2": 4.0}, ConstraintSense.LESS_EQUAL, 5.0))
    model.set_objective(Objective({"x0": 3.0, "x1": 4.0, "x2": 5.0}, ObjectiveSense.MAXIMIZE))

    # Base solve
    tol_base = MILPTolerances(use_scipy_triangular_solve=False, warm_start_relaxations=True)
    res_base = MILPSolver(tolerances=tol_base).solve(model)

    # SciPy solve
    tol_sp = MILPTolerances(use_scipy_triangular_solve=True, scipy_triangular_min_dimension=2, warm_start_relaxations=True)
    res_sp = MILPSolver(tolerances=tol_sp).solve(model)

    assert res_base.status == SolutionStatus.OPTIMAL
    assert res_sp.status == SolutionStatus.OPTIMAL
    assert math.isclose(res_base.objective_value, res_sp.objective_value, rel_tol=1e-6)
    assert res_base.node_count == res_sp.node_count
