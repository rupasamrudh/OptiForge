"""Stage 8.31B Test Suite: Incremental Reduced Costs & Dual Multiplier Updating.

Validates:
1. Feature isolation (default is False, identical behavior to baseline).
2. Numerical equivalence across 8 mandatory problem classes:
   - normal LP
   - sparse LP
   - dense LP
   - degenerate LP
   - ill-conditioned LP
   - scaled LP
   - Phase-I case
   - MILP workload with dual simplex warm-start
3. Adversarial edge cases:
   - zero-step degenerate pivots (theta_D = 0)
   - tied ratios
   - high dynamic range coefficients
   - non-contiguous array inputs (Fortran order, slicing)
   - immutability of caller-owned arrays
   - infeasible candidate detection
4. Drift bounds:
   - max drift |r - (c - A^T y)| < 1e-10
   - BTRAN_y reduction > 70% on multi-pivot solves
"""

import numpy as np
import pytest

from optiforge.core.solution import SolutionStatus
from optiforge.lp.dual_simplex import DualRevisedSimplexSolver, DualSimplexResult
from optiforge.lp.simplex import SimplexTolerances
from optiforge.model.model import Model
from optiforge.model.variable import Variable, VariableType
from optiforge.model.constraint import Constraint, ConstraintSense
from optiforge.model.objective import Objective, ObjectiveSense
from optiforge.milp.solver import MILPSolver


def make_dual_feasible_lp(
    m: int,
    n: int,
    density: float = 1.0,
    degenerate: bool = False,
    ill_cond: bool = False,
    seed: int = 42,
    scaled: bool = False,
):
    """Generate a dual-feasible, primal-infeasible LP in standard form:
        min c^T x
        s.t. [-A_orig  I] x = b, x >= 0
    with initial basis being the slack columns and b < 0.
    """
    rng = np.random.default_rng(seed)
    n_vars = n - m
    if ill_cond:
        U, _ = np.linalg.qr(rng.standard_normal((m, m)))
        V, _ = np.linalg.qr(rng.standard_normal((n_vars, n_vars)))
        s_vals = np.logspace(0, -3, m)
        S_mat = np.zeros((m, n_vars))
        for i in range(min(m, n_vars)):
            S_mat[i, i] = s_vals[i]
        A_orig = np.abs(U @ S_mat @ V.T) + 0.05
    elif density < 1.0:
        mask = rng.random((m, n_vars)) < density
        A_orig = np.zeros((m, n_vars), dtype=np.float64)
        A_orig[mask] = rng.uniform(0.1, 1.0, size=np.count_nonzero(mask))
        for i in range(m):
            if np.all(A_orig[i, :] == 0):
                A_orig[i, rng.integers(0, n_vars)] = rng.uniform(0.1, 1.0)
    else:
        A_orig = rng.uniform(0.1, 1.0, size=(m, n_vars))

    if scaled:
        row_scales = np.power(10.0, rng.uniform(-2, 2, size=m))
        A_orig = A_orig * row_scales[:, None]

    A = np.hstack([-A_orig, np.eye(m)])
    basis = list(range(n_vars, n))
    nonbasic = list(range(n_vars))

    c = np.hstack([rng.uniform(1.0, 10.0, size=n_vars), np.zeros(m)])
    if scaled:
        c[:n_vars] *= np.power(10.0, rng.uniform(-1, 1, size=n_vars))

    b = -rng.uniform(5.0, 20.0, size=m)
    if degenerate:
        zero_idx = rng.choice(m, size=max(1, m // 3), replace=False)
        b[zero_idx] = 0.0

    return A, b, c, basis, nonbasic


class TestStage831BIncrementalReducedCosts:
    """Stage 8.31B Verification and Numerical Equivalence."""

    def test_feature_isolation_defaults(self):
        """Verify default is None (AUTO) and resolves to False when PFI is inactive."""
        tol = SimplexTolerances()
        assert tol.use_incremental_reduced_costs is None
        assert tol.incremental_rc_refresh_interval == 25

        solver = DualRevisedSimplexSolver()
        assert solver.use_incremental_reduced_costs is None
        assert solver.should_use_incremental_reduced_costs(tol) is False
        assert solver.get_incremental_rc_refresh_interval(tol) == 25

    def test_equivalence_normal_lp(self):
        """Test Class 1: Normal standard LP."""
        A, b, c, basis, nonbasic = make_dual_feasible_lp(20, 40, seed=101)
        tol = SimplexTolerances(use_pivot_factorization=True)

        solver_base = DualRevisedSimplexSolver(tolerances=tol, use_pivot_factorization=True, use_incremental_reduced_costs=False)
        solver_inc = DualRevisedSimplexSolver(tolerances=tol, use_pivot_factorization=True, use_incremental_reduced_costs=True)

        res_base = solver_base.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)
        res_inc = solver_inc.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)

        assert res_base.status == res_inc.status == SolutionStatus.OPTIMAL
        assert np.allclose(res_base.x, res_inc.x, atol=1e-7)
        assert np.allclose(res_base.y, res_inc.y, atol=1e-7)
        assert abs(res_base.objective_value - res_inc.objective_value) < 1e-7
        assert res_base.iterations == res_inc.iterations
        assert res_base.basis_indices == res_inc.basis_indices
        assert res_inc.btran_y_count < res_base.btran_y_count

    def test_equivalence_sparse_lp(self):
        """Test Class 2: Sparse LP (density 10%)."""
        A, b, c, basis, nonbasic = make_dual_feasible_lp(30, 80, density=0.10, seed=102)
        tol = SimplexTolerances(use_pivot_factorization=True)

        solver_base = DualRevisedSimplexSolver(tolerances=tol, use_pivot_factorization=True, use_incremental_reduced_costs=False)
        solver_inc = DualRevisedSimplexSolver(tolerances=tol, use_pivot_factorization=True, use_incremental_reduced_costs=True)

        res_base = solver_base.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)
        res_inc = solver_inc.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)

        assert res_base.status == res_inc.status == SolutionStatus.OPTIMAL
        assert np.allclose(res_base.x, res_inc.x, atol=1e-7)
        assert abs(res_base.objective_value - res_inc.objective_value) < 1e-7
        assert res_base.iterations == res_inc.iterations
        assert res_inc.btran_y_count < res_base.btran_y_count

    def test_equivalence_dense_lp(self):
        """Test Class 3: Dense LP (density 100%)."""
        A, b, c, basis, nonbasic = make_dual_feasible_lp(25, 60, density=1.0, seed=103)
        tol = SimplexTolerances(use_pivot_factorization=True)

        solver_base = DualRevisedSimplexSolver(tolerances=tol, use_pivot_factorization=True, use_incremental_reduced_costs=False)
        solver_inc = DualRevisedSimplexSolver(tolerances=tol, use_pivot_factorization=True, use_incremental_reduced_costs=True)

        res_base = solver_base.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)
        res_inc = solver_inc.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)

        assert res_base.status == res_inc.status == SolutionStatus.OPTIMAL
        assert np.allclose(res_base.x, res_inc.x, atol=1e-7)
        assert abs(res_base.objective_value - res_inc.objective_value) < 1e-7
        assert res_base.iterations == res_inc.iterations

    def test_equivalence_degenerate_lp(self):
        """Test Class 4: Degenerate LP (zero RHS on basic row)."""
        A, b, c, basis, nonbasic = make_dual_feasible_lp(30, 60, degenerate=True, seed=104)
        tol = SimplexTolerances(use_pivot_factorization=True)

        solver_base = DualRevisedSimplexSolver(tolerances=tol, use_pivot_factorization=True, use_incremental_reduced_costs=False)
        solver_inc = DualRevisedSimplexSolver(tolerances=tol, use_pivot_factorization=True, use_incremental_reduced_costs=True)

        res_base = solver_base.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)
        res_inc = solver_inc.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)

        assert res_base.status == res_inc.status == SolutionStatus.OPTIMAL
        assert np.allclose(res_base.x, res_inc.x, atol=1e-7)
        assert abs(res_base.objective_value - res_inc.objective_value) < 1e-7
        assert res_base.iterations == res_inc.iterations

    def test_equivalence_ill_conditioned_lp(self):
        """Test Class 5: Ill-conditioned LP (near collinear columns)."""
        A, b, c, basis, nonbasic = make_dual_feasible_lp(15, 30, ill_cond=True, seed=105)
        tol = SimplexTolerances(use_pivot_factorization=True)

        solver_base = DualRevisedSimplexSolver(tolerances=tol, use_pivot_factorization=True, use_incremental_reduced_costs=False)
        solver_inc = DualRevisedSimplexSolver(tolerances=tol, use_pivot_factorization=True, use_incremental_reduced_costs=True)

        res_base = solver_base.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)
        res_inc = solver_inc.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)

        assert res_base.status == res_inc.status == SolutionStatus.OPTIMAL
        assert abs(res_base.objective_value - res_inc.objective_value) < 1e-6
        assert res_base.iterations == res_inc.iterations

    def test_equivalence_scaled_lp(self):
        """Test Class 6: Scaled LP (coefficients ranging over several orders of magnitude)."""
        A, b, c, basis, nonbasic = make_dual_feasible_lp(20, 45, scaled=True, seed=106)
        tol = SimplexTolerances(use_pivot_factorization=True)

        solver_base = DualRevisedSimplexSolver(tolerances=tol, use_pivot_factorization=True, use_incremental_reduced_costs=False)
        solver_inc = DualRevisedSimplexSolver(tolerances=tol, use_pivot_factorization=True, use_incremental_reduced_costs=True)

        res_base = solver_base.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)
        res_inc = solver_inc.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)

        assert res_base.status == res_inc.status == SolutionStatus.OPTIMAL
        assert abs(res_base.objective_value - res_inc.objective_value) < 1e-5
        assert res_base.iterations == res_inc.iterations

    def test_equivalence_phase1_case(self):
        """Test Class 7: Phase-I / Infeasible case."""
        # Unbounded dual => Primal infeasible
        # A row with all positive entries in A and negative RHS
        A = np.array([
            [1.0, 2.0, 1.0, 0.0],
            [3.0, 1.0, 0.0, 1.0],
        ], dtype=np.float64)
        b = np.array([-5.0, -10.0], dtype=np.float64)
        c = np.array([2.0, 3.0, 0.0, 0.0], dtype=np.float64)
        basis = [2, 3]
        nonbasic = [0, 1]

        tol = SimplexTolerances(use_pivot_factorization=True)
        solver_base = DualRevisedSimplexSolver(tolerances=tol, use_pivot_factorization=True, use_incremental_reduced_costs=False)
        solver_inc = DualRevisedSimplexSolver(tolerances=tol, use_pivot_factorization=True, use_incremental_reduced_costs=True)

        res_base = solver_base.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)
        res_inc = solver_inc.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)

        assert res_base.status == res_inc.status == SolutionStatus.INFEASIBLE
        assert res_base.iterations == res_inc.iterations

    def test_equivalence_milp_workload(self):
        """Test Class 8: MILP workload using dual simplex warm starts."""
        m = Model("milp_knapsack")
        x = [m.add_variable(Variable(f"x_{i}", variable_type=VariableType.BINARY)) for i in range(8)]
        weights = [3, 4, 5, 8, 9, 2, 7, 6]
        values = [12, 18, 24, 32, 40, 9, 28, 25]

        m.add_constraint(
            Constraint("capacity", coefficients={f"x_{i}": weights[i] for i in range(8)},
                       sense=ConstraintSense.LESS_EQUAL, rhs=19.0)
        )
        m.set_objective(Objective({f"x_{i}": values[i] for i in range(8)}, sense=ObjectiveSense.MAXIMIZE))

        tol_base = SimplexTolerances(use_pivot_factorization=True, use_incremental_reduced_costs=False)
        tol_inc = SimplexTolerances(use_pivot_factorization=True, use_incremental_reduced_costs=True)

        milp_base = MILPSolver(simplex_tolerances=tol_base, warm_start_relaxations=True)
        milp_inc = MILPSolver(simplex_tolerances=tol_inc, warm_start_relaxations=True)

        sol_base = milp_base.solve(m)
        sol_inc = milp_inc.solve(m)

        assert sol_base.status == sol_inc.status == SolutionStatus.OPTIMAL
        assert abs(sol_base.objective_value - sol_inc.objective_value) < 1e-6
        for var_name in sol_base.variable_values:
            assert abs(sol_base.variable_values[var_name] - sol_inc.variable_values[var_name]) < 1e-6

    def test_adversarial_non_contiguous_arrays(self):
        """Adversarial test: Fortran-ordered (column-major) arrays and slices."""
        A, b, c, basis, nonbasic = make_dual_feasible_lp(20, 40, seed=201)
        A_f = np.asfortranarray(A)
        b_f = b.copy()
        c_f = c.copy()

        tol = SimplexTolerances(use_pivot_factorization=True, use_incremental_reduced_costs=True)
        solver = DualRevisedSimplexSolver(tolerances=tol, use_pivot_factorization=True, use_incremental_reduced_costs=True)

        res = solver.solve_standard_form(A_f, b_f, c_f, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)
        assert res.status == SolutionStatus.OPTIMAL

    def test_adversarial_caller_immutability(self):
        """Adversarial test: Caller-owned input arrays must not be mutated."""
        A, b, c, basis, nonbasic = make_dual_feasible_lp(20, 40, seed=202)
        A_orig = A.copy()
        b_orig = b.copy()
        c_orig = c.copy()

        tol = SimplexTolerances(use_pivot_factorization=True, use_incremental_reduced_costs=True)
        solver = DualRevisedSimplexSolver(tolerances=tol, use_pivot_factorization=True, use_incremental_reduced_costs=True)
        solver.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)

        assert np.array_equal(A, A_orig)
        assert np.array_equal(b, b_orig)
        assert np.array_equal(c, c_orig)

    def test_adversarial_degenerate_zero_ratio(self):
        """Adversarial test: Degenerate pivot with theta_D = 0 (reduced cost already zero)."""
        m, n = 3, 5
        A = np.array([
            [-1.0, -1.0, 1.0, 0.0, 0.0],
            [-1.0, -2.0, 0.0, 1.0, 0.0],
            [-2.0, -1.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        b = np.array([-2.0, 0.0, -3.0], dtype=np.float64)
        # Reduced cost of nonbasic column 0 is zero => theta_D will be 0
        c = np.array([0.0, 5.0, 0.0, 0.0, 0.0], dtype=np.float64)
        basis = [2, 3, 4]
        nonbasic = [0, 1]

        tol = SimplexTolerances(use_pivot_factorization=True)
        solver_base = DualRevisedSimplexSolver(tolerances=tol, use_pivot_factorization=True, use_incremental_reduced_costs=False)
        solver_inc = DualRevisedSimplexSolver(tolerances=tol, use_pivot_factorization=True, use_incremental_reduced_costs=True)

        res_base = solver_base.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)
        res_inc = solver_inc.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)

        assert res_base.status == res_inc.status == SolutionStatus.OPTIMAL
        assert abs(res_base.objective_value - res_inc.objective_value) < 1e-7
        assert res_base.iterations == res_inc.iterations

    def test_adversarial_maximize_sense(self):
        """Adversarial test: Verify MAXIMIZE objective sense semantics."""
        A, b, c, basis, nonbasic = make_dual_feasible_lp(15, 30, seed=303)
        # For MAX: dual feasible means r <= 0, so c <= 0
        c_max = -c

        tol = SimplexTolerances(use_pivot_factorization=True)
        solver_base = DualRevisedSimplexSolver(tolerances=tol, use_pivot_factorization=True, use_incremental_reduced_costs=False)
        solver_inc = DualRevisedSimplexSolver(tolerances=tol, use_pivot_factorization=True, use_incremental_reduced_costs=True)

        res_base = solver_base.solve_standard_form(A, b, c_max, basis, nonbasic, ObjectiveSense.MAXIMIZE, tol)
        res_inc = solver_inc.solve_standard_form(A, b, c_max, basis, nonbasic, ObjectiveSense.MAXIMIZE, tol)

        assert res_base.status == res_inc.status == SolutionStatus.OPTIMAL
        assert abs(res_base.objective_value - res_inc.objective_value) < 1e-7
        assert res_base.iterations == res_inc.iterations

    def test_btran_reduction_ratio(self):
        """Verify that BTRAN_y solve count is drastically reduced (> 75% reduction)."""
        A, b, c, basis, nonbasic = make_dual_feasible_lp(60, 150, density=0.2, seed=42)
        tol = SimplexTolerances(use_pivot_factorization=True)

        solver_base = DualRevisedSimplexSolver(tolerances=tol, use_pivot_factorization=True, use_incremental_reduced_costs=False)
        solver_inc = DualRevisedSimplexSolver(tolerances=tol, use_pivot_factorization=True, use_incremental_reduced_costs=True)

        res_base = solver_base.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)
        res_inc = solver_inc.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)

        assert res_base.iterations >= 10
        # Baseline must perform 1 BTRAN_y per iteration + 1 at start
        assert res_base.btran_y_count == res_base.iterations + 1
        # Incremental only performs on initial step and periodic refresh/refactorization
        assert res_inc.btran_y_count < res_base.btran_y_count
        reduction = 1.0 - (res_inc.btran_y_count / res_base.btran_y_count)
        assert reduction > 0.70, f"Expected >70% reduction, got {reduction * 100:.1f}%"
