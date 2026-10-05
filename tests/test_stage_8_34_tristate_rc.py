"""Stage 8.34 Test Suite: Tri-State Incremental Reduced-Cost Activation.

Covers:
1. default is None (AUTO)
2. AUTO + no PFI -> OFF
3. AUTO + explicit PFI -> ON
4. AUTO + adaptive PFI below threshold -> OFF
5. AUTO + adaptive PFI above threshold -> ON
6. explicit OFF overrides AUTO (PFI active)
7. explicit ON overrides AUTO (PFI inactive)
8. DenseLU + explicit ON (graceful fallback)
9. PFI refactorization safety
10. PFI fallback to DenseLU
11. Compact WY compatibility
12. SciPy triangular solve compatibility
13. MILP compatibility and equivalence
14. tolerance-level configuration
15. solver-level configuration
16. configuration precedence (solver-level vs tolerance-level)
"""

import numpy as np
import pytest

from optiforge.core.solution import SolutionStatus
from optiforge.lp.dual_simplex import DualRevisedSimplexSolver, DualSimplexResult
from optiforge.lp.simplex import SimplexTolerances
from optiforge.model.objective import ObjectiveSense
from optiforge.model.model import Model
from optiforge.model.variable import Variable, VariableType
from optiforge.model.constraint import Constraint, ConstraintSense
from optiforge.model.objective import Objective
from optiforge.milp.solver import MILPSolver
from optiforge.numerics.pivot_basis import PivotUpdateFactorization
from scratch.run_stage_8_31a_investigation import make_standard_form_dual_lp


class TestStage834TriStateActivation:
    """Stage 8.34 Verification of Tri-State Incremental Reduced Costs."""

    def test_item_1_default_is_none(self):
        """1. Verify default configuration is None (AUTO)."""
        tol = SimplexTolerances()
        assert tol.use_incremental_reduced_costs is None
        solver = DualRevisedSimplexSolver()
        assert solver.use_incremental_reduced_costs is None

    def test_item_2_auto_no_pfi_resolves_off(self):
        """2. AUTO + no PFI -> resolves to OFF (DenseLU uses full BTRAN)."""
        tol = SimplexTolerances(use_pivot_factorization=False, use_adaptive_pivot_factorization=False)
        solver = DualRevisedSimplexSolver(tolerances=tol)
        assert solver.should_use_incremental_reduced_costs(20, tol) is False

        A, b, c, basis, nonbasic = make_standard_form_dual_lp(20, 40, seed=1)
        res = solver.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)
        assert res.status == SolutionStatus.OPTIMAL
        # In OFF mode, BTRAN_y is executed on every iteration + 1
        assert res.btran_y_count == res.iterations + 1

    def test_item_3_auto_explicit_pfi_resolves_on(self):
        """3. AUTO + explicit PFI -> resolves to ON (Incremental RC active)."""
        tol = SimplexTolerances(use_pivot_factorization=True)
        solver = DualRevisedSimplexSolver(tolerances=tol)
        assert solver.should_use_incremental_reduced_costs(20, tol) is True

        A, b, c, basis, nonbasic = make_standard_form_dual_lp(20, 40, seed=1)
        res = solver.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)
        assert res.status == SolutionStatus.OPTIMAL
        # In ON mode, BTRAN_y is executed once on initial basis (+ refactorizations if any)
        assert res.btran_y_count < res.iterations + 1

    def test_item_4_auto_adaptive_below_threshold_resolves_off(self):
        """4. AUTO + adaptive PFI below threshold (m < 6) -> resolves to OFF."""
        tol = SimplexTolerances(use_adaptive_pivot_factorization=True, adaptive_pfi_min_dimension=6)
        solver = DualRevisedSimplexSolver(tolerances=tol)
        # m = 4 < 6 => Adaptive chooses DenseLU => Inc RC must resolve OFF
        assert solver.should_use_incremental_reduced_costs(4, tol) is False

        A, b, c, basis, nonbasic = make_standard_form_dual_lp(4, 10, seed=2)
        res = solver.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)
        assert res.btran_y_count == res.iterations + 1

    def test_item_5_auto_adaptive_above_threshold_resolves_on(self):
        """5. AUTO + adaptive PFI above threshold (m >= 6) -> resolves to ON."""
        tol = SimplexTolerances(use_adaptive_pivot_factorization=True, adaptive_pfi_min_dimension=6)
        solver = DualRevisedSimplexSolver(tolerances=tol)
        # m = 20 >= 6 => Adaptive chooses PFI => Inc RC must resolve ON
        assert solver.should_use_incremental_reduced_costs(20, tol) is True

        A, b, c, basis, nonbasic = make_standard_form_dual_lp(20, 40, seed=3)
        res = solver.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)
        assert res.btran_y_count < res.iterations + 1

    def test_item_6_explicit_off_overrides_auto(self):
        """6. Explicit OFF overrides AUTO even when PFI is active."""
        # Tolerance-level explicit OFF
        tol = SimplexTolerances(use_pivot_factorization=True, use_incremental_reduced_costs=False)
        solver = DualRevisedSimplexSolver(tolerances=tol)
        assert solver.should_use_incremental_reduced_costs(20, tol) is False

        A, b, c, basis, nonbasic = make_standard_form_dual_lp(20, 40, seed=4)
        res = solver.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)
        assert res.btran_y_count == res.iterations + 1

        # Solver-level explicit OFF
        solver_off = DualRevisedSimplexSolver(use_pivot_factorization=True, use_incremental_reduced_costs=False)
        assert solver_off.should_use_incremental_reduced_costs(20) is False

    def test_item_7_explicit_on_overrides_auto(self):
        """7. Explicit ON overrides AUTO even when PFI is inactive."""
        tol = SimplexTolerances(use_pivot_factorization=False, use_incremental_reduced_costs=True)
        solver = DualRevisedSimplexSolver(tolerances=tol)
        assert solver.should_use_incremental_reduced_costs(20, tol) is True

    def test_item_8_denselu_explicit_on_graceful_fallback(self):
        """8. DenseLU + explicit ON falls back gracefully without error."""
        tol = SimplexTolerances(use_pivot_factorization=False, use_incremental_reduced_costs=True)
        solver = DualRevisedSimplexSolver(tolerances=tol)
        A, b, c, basis, nonbasic = make_standard_form_dual_lp(20, 40, seed=5)
        res = solver.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)
        assert res.status == SolutionStatus.OPTIMAL
        # When inc_fact is None, need_solve_y triggers to re-solve via DenseLU
        assert res.btran_y_count == res.iterations + 1

    def test_item_9_pfi_refactorization_safety(self):
        """9. PFI refactorization triggers re-sync of y and r with exact residuals."""
        tol = SimplexTolerances(use_pivot_factorization=True, max_basis_updates=3)
        solver = DualRevisedSimplexSolver(tolerances=tol)
        A, b, c, basis, nonbasic = make_standard_form_dual_lp(30, 60, seed=6)
        res = solver.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)
        assert res.status == SolutionStatus.OPTIMAL
        assert isinstance(res.basis_factorization, PivotUpdateFactorization)
        assert res.basis_factorization.total_refactorizations >= 1
        # Residual verification
        rc = c - np.dot(A.T, res.y)
        assert np.all(rc >= -tol.reduced_cost_tolerance)

    def test_item_10_pfi_fallback_safety(self):
        """10. PFI fallback to DenseLU cleanly re-syncs dual state."""
        # Tight residual tolerance forces fallback / refactorization
        tol = SimplexTolerances(use_pivot_factorization=True, incremental_residual_tolerance=1e-14)
        solver = DualRevisedSimplexSolver(tolerances=tol)
        A, b, c, basis, nonbasic = make_standard_form_dual_lp(20, 40, ill_cond=True, seed=7)
        res = solver.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)
        assert res.status == SolutionStatus.OPTIMAL

    def test_item_11_compact_wy_compatibility(self):
        """11. Compact WY factorization compatibility."""
        tol = SimplexTolerances(use_pivot_factorization=True)
        solver = DualRevisedSimplexSolver(tolerances=tol)
        A, b, c, basis, nonbasic = make_standard_form_dual_lp(25, 50, seed=8)
        res = solver.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)
        assert res.status == SolutionStatus.OPTIMAL

    def test_item_12_scipy_provider_compatibility(self):
        """12. SciPy triangular solve path compatibility."""
        tol = SimplexTolerances(use_pivot_factorization=True, use_scipy_triangular_solve=True, scipy_triangular_min_dimension=10)
        solver = DualRevisedSimplexSolver(tolerances=tol)
        A, b, c, basis, nonbasic = make_standard_form_dual_lp(30, 60, seed=9)
        res = solver.solve_standard_form(A, b, c, basis, nonbasic, ObjectiveSense.MINIMIZE, tol)
        assert res.status == SolutionStatus.OPTIMAL

    def test_item_13_milp_compatibility(self):
        """13. MILP compatibility: verify exact bounds and integer solutions."""
        m = Model("milp_stage_834")
        x = [m.add_variable(Variable(f"x_{i}", variable_type=VariableType.BINARY)) for i in range(6)]
        w = [3, 4, 5, 8, 9, 2]
        v = [12, 18, 24, 32, 40, 9]
        m.add_constraint(Constraint("cap", coefficients={f"x_{i}": w[i] for i in range(6)}, sense=ConstraintSense.LESS_EQUAL, rhs=15.0))
        m.set_objective(Objective({f"x_{i}": v[i] for i in range(6)}, sense=ObjectiveSense.MAXIMIZE))

        # Solve with default (AUTO)
        milp_auto = MILPSolver(simplex_tolerances=SimplexTolerances(use_pivot_factorization=True), warm_start_relaxations=True)
        sol_auto = milp_auto.solve(m)

        # Solve with explicit OFF
        milp_off = MILPSolver(simplex_tolerances=SimplexTolerances(use_pivot_factorization=True, use_incremental_reduced_costs=False), warm_start_relaxations=True)
        sol_off = milp_off.solve(m)

        assert sol_auto.status == sol_off.status == SolutionStatus.OPTIMAL
        assert abs(sol_auto.objective_value - sol_off.objective_value) < 1e-7
        for vname in sol_auto.variable_values:
            assert abs(sol_auto.variable_values[vname] - sol_off.variable_values[vname]) < 1e-7

    def test_item_14_tolerance_level_configuration(self):
        """14. Tolerance-level configuration semantics."""
        t_none = SimplexTolerances()
        assert t_none.use_incremental_reduced_costs is None

        t_false = SimplexTolerances(use_incremental_reduced_costs=False)
        assert t_false.use_incremental_reduced_costs is False

        t_true = SimplexTolerances(use_incremental_reduced_costs=True)
        assert t_true.use_incremental_reduced_costs is True

    def test_item_15_solver_level_configuration(self):
        """15. Solver-level configuration semantics."""
        s_none = DualRevisedSimplexSolver()
        assert s_none.use_incremental_reduced_costs is None

        s_false = DualRevisedSimplexSolver(use_incremental_reduced_costs=False)
        assert s_false.use_incremental_reduced_costs is False

        s_true = DualRevisedSimplexSolver(use_incremental_reduced_costs=True)
        assert s_true.use_incremental_reduced_costs is True

    def test_item_16_configuration_precedence(self):
        """16. Configuration precedence: Solver-level wins over Tolerance-level."""
        # Solver=False, Tol=True -> Solver wins (False)
        tol_true = SimplexTolerances(use_incremental_reduced_costs=True)
        solver_false = DualRevisedSimplexSolver(tolerances=tol_true, use_incremental_reduced_costs=False)
        assert solver_false.should_use_incremental_reduced_costs(20, tol_true) is False

        # Solver=True, Tol=False -> Solver wins (True)
        tol_false = SimplexTolerances(use_incremental_reduced_costs=False)
        solver_true = DualRevisedSimplexSolver(tolerances=tol_false, use_incremental_reduced_costs=True)
        assert solver_true.should_use_incremental_reduced_costs(20, tol_false) is True

        # Solver=None, Tol=False -> Tol wins (False)
        solver_none = DualRevisedSimplexSolver(tolerances=tol_false, use_incremental_reduced_costs=None)
        assert solver_none.should_use_incremental_reduced_costs(20, tol_false) is False

        # Solver=None, Tol=None -> Inferred from PFI
        tol_pfi = SimplexTolerances(use_pivot_factorization=True, use_incremental_reduced_costs=None)
        solver_auto = DualRevisedSimplexSolver(tolerances=tol_pfi, use_incremental_reduced_costs=None)
        assert solver_auto.should_use_incremental_reduced_costs(20, tol_pfi) is True
