"""Stage 8.36 Test Suite: Behavioral Compatibility and Contract Hardening.

Verifies:
A. Default DenseLU: SimplexTolerances() -> incremental RC resolves to False (btran_y == iters + 1).
B. Explicit False: PFI enabled + incremental RC=False -> incremental RC remains disabled (btran_y == iters + 1).
C. Explicit True: PFI enabled + incremental RC=True -> incremental RC enabled (btran_y < iters + 1).
D. AUTO: PFI enabled + incremental RC=None -> incremental RC enabled (btran_y < iters + 1).
E. AUTO without PFI: DenseLU + incremental RC=None -> incremental RC disabled (btran_y == iters + 1).
F. Solver-level precedence:
   - solver=False, tolerance=True -> False (solver overrides tolerance)
   - solver=True, tolerance=False -> True (solver overrides tolerance)
G. AUTO inheritance:
   - solver=None, tolerance=False -> False (tolerance overrides AUTO)
   - solver=None, tolerance=True -> True (tolerance overrides AUTO)
H. Primal simplex:
   - Confirm that configuring use_incremental_reduced_costs does not alter the primal simplex pricing path.
"""

import numpy as np
import pytest

from optiforge.core.solution import SolutionStatus
from optiforge.lp.dual_simplex import DualRevisedSimplexSolver
from optiforge.lp.simplex import RevisedSimplexSolver, SimplexTolerances
from optiforge.model.objective import ObjectiveSense


def make_test_dual_lp(m, n, seed=42):
    rng = np.random.default_rng(seed)
    A_orig = rng.uniform(0.1, 1.0, size=(m, n))
    b_orig = rng.uniform(5.0, 50.0, size=m)
    c_orig = rng.uniform(1.0, 10.0, size=n)
    A_eq = np.hstack([-A_orig, np.eye(m)])
    b_eq = -b_orig
    c_eq = np.concatenate([c_orig, np.zeros(m)])
    basis = list(range(n, n + m))
    nonbasic = list(range(n))
    return A_eq, b_eq, c_eq, basis, nonbasic


def make_test_primal_lp(m, n, seed=42):
    rng = np.random.default_rng(seed)
    A_orig = rng.uniform(0.1, 1.0, size=(m, n))
    b_orig = rng.uniform(10.0, 50.0, size=m)
    c_orig = rng.uniform(1.0, 10.0, size=n)
    A_eq = np.hstack([A_orig, np.eye(m)])
    b_eq = b_orig
    c_eq = np.concatenate([c_orig, np.zeros(m)])
    basis = list(range(n, n + m))
    nonbasic = list(range(n))
    return A_eq, b_eq, c_eq, basis, nonbasic


class TestStage836ContractHardening:
    """Stage 8.36 Behavioral and Contract Verification."""

    def test_case_a_default_denselu_rc_disabled(self):
        """A. Default DenseLU: SimplexTolerances() -> incremental RC resolves to False."""
        tol = SimplexTolerances()
        assert tol.use_incremental_reduced_costs is None
        solver = DualRevisedSimplexSolver(tolerances=tol)
        assert solver.should_use_incremental_reduced_costs(20, tol) is False

        A, b, c, basis, nonbasic = make_test_dual_lp(20, 40, seed=1)
        res = solver.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)
        assert res.status == SolutionStatus.OPTIMAL
        # Behavioral check: In full BTRAN mode, btran_y is executed on every iteration + 1
        assert res.btran_y_count == res.iterations + 1

    def test_case_b_explicit_false_pfi_rc_disabled(self):
        """B. Explicit False: PFI enabled + incremental RC=False -> remains disabled."""
        tol = SimplexTolerances(use_pivot_factorization=True, use_incremental_reduced_costs=False)
        solver = DualRevisedSimplexSolver(tolerances=tol)
        assert solver.should_use_incremental_reduced_costs(20, tol) is False

        A, b, c, basis, nonbasic = make_test_dual_lp(20, 40, seed=2)
        res = solver.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)
        assert res.status == SolutionStatus.OPTIMAL
        assert res.btran_y_count == res.iterations + 1

    def test_case_c_explicit_true_pfi_rc_enabled(self):
        """C. Explicit True: PFI enabled + incremental RC=True -> incremental RC enabled."""
        tol = SimplexTolerances(use_pivot_factorization=True, use_incremental_reduced_costs=True)
        solver = DualRevisedSimplexSolver(tolerances=tol)
        assert solver.should_use_incremental_reduced_costs(20, tol) is True

        A, b, c, basis, nonbasic = make_test_dual_lp(20, 40, seed=3)
        res = solver.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)
        assert res.status == SolutionStatus.OPTIMAL
        assert res.btran_y_count < res.iterations + 1

    def test_case_d_auto_pfi_rc_enabled(self):
        """D. AUTO: PFI enabled + incremental RC=None -> incremental RC enabled."""
        tol = SimplexTolerances(use_pivot_factorization=True, use_incremental_reduced_costs=None)
        solver = DualRevisedSimplexSolver(tolerances=tol)
        assert solver.should_use_incremental_reduced_costs(20, tol) is True

        A, b, c, basis, nonbasic = make_test_dual_lp(20, 40, seed=4)
        res = solver.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)
        assert res.status == SolutionStatus.OPTIMAL
        assert res.btran_y_count < res.iterations + 1

    def test_case_e_auto_denselu_rc_disabled(self):
        """E. AUTO without PFI: DenseLU + incremental RC=None -> incremental RC disabled."""
        tol = SimplexTolerances(use_pivot_factorization=False, use_adaptive_pivot_factorization=False, use_incremental_reduced_costs=None)
        solver = DualRevisedSimplexSolver(tolerances=tol)
        assert solver.should_use_incremental_reduced_costs(20, tol) is False

        A, b, c, basis, nonbasic = make_test_dual_lp(20, 40, seed=5)
        res = solver.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)
        assert res.status == SolutionStatus.OPTIMAL
        assert res.btran_y_count == res.iterations + 1

    def test_case_f_solver_level_precedence(self):
        """F. Solver-level precedence:
        - solver=False, tolerance=True -> False
        - solver=True, tolerance=False -> True
        """
        A, b, c, basis, nonbasic = make_test_dual_lp(20, 40, seed=6)

        # 1. solver=False, tolerance=True -> solver wins (OFF)
        tol_true = SimplexTolerances(use_pivot_factorization=True, use_incremental_reduced_costs=True)
        solver_false = DualRevisedSimplexSolver(tolerances=tol_true, use_incremental_reduced_costs=False)
        assert solver_false.should_use_incremental_reduced_costs(20, tol_true) is False
        res_f1 = solver_false.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol_true)
        assert res_f1.btran_y_count == res_f1.iterations + 1

        # 2. solver=True, tolerance=False -> solver wins (ON)
        tol_false = SimplexTolerances(use_pivot_factorization=True, use_incremental_reduced_costs=False)
        solver_true = DualRevisedSimplexSolver(tolerances=tol_false, use_incremental_reduced_costs=True)
        assert solver_true.should_use_incremental_reduced_costs(20, tol_false) is True
        res_f2 = solver_true.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol_false)
        assert res_f2.btran_y_count < res_f2.iterations + 1

    def test_case_g_auto_inheritance_precedence(self):
        """G. AUTO inheritance:
        - solver=None, tolerance=False -> False (tolerance overrides AUTO)
        - solver=None, tolerance=True -> True (tolerance overrides AUTO)
        """
        A, b, c, basis, nonbasic = make_test_dual_lp(20, 40, seed=7)

        # 1. solver=None, tolerance=False -> False
        tol_false = SimplexTolerances(use_pivot_factorization=True, use_incremental_reduced_costs=False)
        solver_none1 = DualRevisedSimplexSolver(tolerances=tol_false, use_incremental_reduced_costs=None)
        assert solver_none1.should_use_incremental_reduced_costs(20, tol_false) is False
        res_g1 = solver_none1.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol_false)
        assert res_g1.btran_y_count == res_g1.iterations + 1

        # 2. solver=None, tolerance=True -> True
        tol_true = SimplexTolerances(use_pivot_factorization=True, use_incremental_reduced_costs=True)
        solver_none2 = DualRevisedSimplexSolver(tolerances=tol_true, use_incremental_reduced_costs=None)
        assert solver_none2.should_use_incremental_reduced_costs(20, tol_true) is True
        res_g2 = solver_none2.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol_true)
        assert res_g2.btran_y_count < res_g2.iterations + 1

    def test_case_h_primal_simplex_pricing_unaffected(self):
        """H. Primal simplex:
        Confirm that configuring use_incremental_reduced_costs does not alter the primal simplex pricing path.
        """
        A, b, c, basis, nonbasic = make_test_primal_lp(25, 50, seed=8)

        tol_default = SimplexTolerances(use_incremental_reduced_costs=None)
        tol_off = SimplexTolerances(use_incremental_reduced_costs=False)
        tol_on = SimplexTolerances(use_incremental_reduced_costs=True)

        res_default = RevisedSimplexSolver(tolerances=tol_default)._simplex_loop(
            A, b, c, list(basis), list(nonbasic), ObjectiveSense.MINIMIZE, tol_default
        )
        res_off = RevisedSimplexSolver(tolerances=tol_off)._simplex_loop(
            A, b, c, list(basis), list(nonbasic), ObjectiveSense.MINIMIZE, tol_off
        )
        res_on = RevisedSimplexSolver(tolerances=tol_on)._simplex_loop(
            A, b, c, list(basis), list(nonbasic), ObjectiveSense.MINIMIZE, tol_on
        )

        assert res_default.status == res_off.status == res_on.status == SolutionStatus.OPTIMAL
        assert res_default.iterations == res_off.iterations == res_on.iterations
        assert abs(res_default.objective_value - res_off.objective_value) < 1e-12
        assert abs(res_default.objective_value - res_on.objective_value) < 1e-12
        assert np.allclose(res_default.x, res_off.x, atol=1e-12)
        assert np.allclose(res_default.x, res_on.x, atol=1e-12)
        assert res_default.basis_indices == res_off.basis_indices == res_on.basis_indices
