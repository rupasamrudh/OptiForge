"""Tests for Stage 8.44: Post-Presolve Basis, Dual & Re-Optimization Readiness.

Verifies:
1. Optimal LP exposes BasisInfo.
2. Basic/nonbasic indices are valid.
3. Reduced costs are exposed correctly.
4. Dual multipliers are exposed correctly where supported.
5. MAXIMIZE sign conventions.
6. MINIMIZE sign conventions.
7. <= / >= / equality constraints.
8. Presolve variable reconstruction preserves mapping.
9. Fixed-variable elimination preserves mapping.
10. Infeasible solution does not expose invalid basis metadata.
11. Unbounded solution does not expose invalid basis metadata.
12. MILP does not expose fake sensitivity information.
13. Basis export is immutable/safely isolated from solver internals.
14. Compatible basis can be passed back into a solve.
15. Incompatible basis is rejected or safely ignored according to chosen API.
16. Existing Stage 8.43 solver-selection metadata remains unchanged.
17. Existing custom configurations remain functional.
18. Re-solving after an RHS/objective modification produces a correct result.
19. Warm-start path does not change mathematical correctness.
20. Degenerate optimal solutions.
"""

from __future__ import annotations

import math
import pytest
from dataclasses import FrozenInstanceError

from optiforge.core import BasisInfo, BasisStatus, SolutionStatus
from optiforge.model import (
    Constraint,
    ConstraintSense,
    Model,
    Objective,
    ObjectiveSense,
    Variable,
    VariableType,
)


def build_simple_lp(maximize: bool = False) -> Model:
    """Build a standard 2-variable LP:
    max/min 3x1 + 2x2
    s.t.
      x1 + x2 <= 4
      2x1 + x2 <= 5
      x1, x2 >= 0
    """
    model = Model(name="simple_lp")
    model.add_variable(Variable(name="x1", lower_bound=0.0))
    model.add_variable(Variable(name="x2", lower_bound=0.0))

    sense = ObjectiveSense.MAXIMIZE if maximize else ObjectiveSense.MINIMIZE
    model.set_objective(
        Objective(coefficients={"x1": 3.0, "x2": 2.0}, sense=sense)
    )

    model.add_constraint(
        Constraint(name="c1", coefficients={"x1": 1.0, "x2": 1.0}, sense=ConstraintSense.LESS_EQUAL, rhs=4.0)
    )
    model.add_constraint(
        Constraint(name="c2", coefficients={"x1": 2.0, "x2": 1.0}, sense=ConstraintSense.LESS_EQUAL, rhs=5.0)
    )
    return model


class TestStage844BasisReadiness:
    """Stage 8.44 verification suite."""

    def test_optimal_lp_exposes_basis_info(self):
        """1. Optimal LP exposes BasisInfo and model.get_last_basis()."""
        model = build_simple_lp(maximize=True)
        sol = model.solve()

        assert sol.status == SolutionStatus.OPTIMAL
        assert sol.basis_info is not None
        assert isinstance(sol.basis_info, BasisInfo)
        assert model.get_last_basis() is sol.basis_info
        assert sol.basis_info.is_lp is True
        assert sol.basis_info.is_optimal is True
        assert sol.basis_info.objective_value == pytest.approx(9.0, abs=1e-5)

    def test_basic_nonbasic_indices_valid(self):
        """2. Basic and nonbasic indices partition standard-form columns."""
        model = build_simple_lp(maximize=True)
        sol = model.solve()
        binfo = sol.basis_info
        assert binfo is not None

        m, n = binfo.standard_form_dimension
        assert len(binfo.basic_variable_indices) == m
        assert len(binfo.nonbasic_variable_indices) == n - m

        all_indices = set(binfo.basic_variable_indices) | set(binfo.nonbasic_variable_indices)
        assert all_indices == set(range(n))
        assert len(set(binfo.basic_variable_indices) & set(binfo.nonbasic_variable_indices)) == 0

        # Check get_basis_indices()
        assert binfo.get_basis_indices() == binfo.basic_variable_indices
        assert binfo.is_compatible(m, n) is True
        assert binfo.is_compatible(m + 1, n) is False

    def test_reduced_costs_exposed(self):
        """3. Reduced costs are zero for basic variables and valid for non-basic."""
        model = build_simple_lp(maximize=False)
        sol = model.solve()
        binfo = sol.basis_info
        assert binfo is not None

        # Min 3x1 + 2x2 s.t. constraints, x >= 0 -> optimal at x1=0, x2=0, obj=0
        assert sol.variable_values["x1"] == pytest.approx(0.0, abs=1e-5)
        assert sol.variable_values["x2"] == pytest.approx(0.0, abs=1e-5)
        # Reduced costs should be non-negative for minimization at lower bound
        assert binfo.reduced_costs["x1"] >= -1e-6
        assert binfo.reduced_costs["x2"] >= -1e-6

    def test_dual_multipliers_and_maximize_signs(self):
        """4. Dual multipliers for MAXIMIZE <= constraints are non-negative."""
        # Max 3x1 + 2x2
        # s.t. x1 + x2 <= 4 (c1)
        #      2x1 + x2 <= 5 (c2)
        # Opt: x1=1, x2=3 -> obj = 9.
        # Both c1 and c2 are tight at optimal: lambda1 = 1.0, lambda2 = 1.0
        model = build_simple_lp(maximize=True)
        sol = model.solve()
        binfo = sol.basis_info
        assert binfo is not None

        duals = binfo.dual_multipliers
        assert "c1" in duals
        assert "c2" in duals
        assert duals["c1"] == pytest.approx(1.0, abs=1e-4)
        assert duals["c2"] == pytest.approx(1.0, abs=1e-4)

    def test_minimize_sign_conventions(self):
        """5. Dual multipliers for MINIMIZE >= constraints are non-negative."""
        # Min 2x1 + 3x2
        # s.t. x1 + x2 >= 3 (c1)
        #      2x1 + x2 >= 4 (c2)
        # Opt: x1=1, x2=2, obj=8.
        # Both c1 and c2 tight: lambda1 = 1.0, lambda2 = 1.0 (since 1+2=3, 1+1=2)
        model = Model(name="min_lp")
        model.add_variable(Variable(name="x1", lower_bound=0.0))
        model.add_variable(Variable(name="x2", lower_bound=0.0))
        model.set_objective(
            Objective(coefficients={"x1": 3.0, "x2": 2.0}, sense=ObjectiveSense.MINIMIZE)
        )
        model.add_constraint(
            Constraint(name="c1", coefficients={"x1": 1.0, "x2": 1.0}, sense=ConstraintSense.GREATER_EQUAL, rhs=3.0)
        )
        model.add_constraint(
            Constraint(name="c2", coefficients={"x1": 2.0, "x2": 1.0}, sense=ConstraintSense.GREATER_EQUAL, rhs=4.0)
        )

        sol = model.solve()
        assert sol.status == SolutionStatus.OPTIMAL
        binfo = sol.basis_info
        assert binfo is not None

        duals = binfo.dual_multipliers
        assert duals["c1"] >= -1e-6
        assert duals["c2"] >= -1e-6

    def test_equality_constraints_duals(self):
        """6. Equality constraints duals are supported."""
        # Min x1 + 2x2
        # s.t. x1 + x2 == 3 (c_eq)
        #      x1 + 2x2 >= 4 (c_ge)
        #      x1, x2 >= 0
        # Opt: x2 = 1, x1 = 2, obj = 1(2) + 2(1) = 4.
        model = Model(name="eq_lp")
        model.add_variable(Variable(name="x1", lower_bound=0.0))
        model.add_variable(Variable(name="x2", lower_bound=0.0))
        model.set_objective(
            Objective(coefficients={"x1": 1.0, "x2": 2.0}, sense=ObjectiveSense.MINIMIZE)
        )
        model.add_constraint(
            Constraint(name="c_eq", coefficients={"x1": 1.0, "x2": 1.0}, sense=ConstraintSense.EQUAL, rhs=3.0)
        )
        model.add_constraint(
            Constraint(name="c_ge", coefficients={"x1": 1.0, "x2": 2.0}, sense=ConstraintSense.GREATER_EQUAL, rhs=4.0)
        )

        sol = model.solve()
        assert sol.status == SolutionStatus.OPTIMAL
        assert sol.variable_values["x1"] == pytest.approx(2.0, abs=1e-4)
        assert sol.variable_values["x2"] == pytest.approx(1.0, abs=1e-4)

        binfo = sol.basis_info
        assert binfo is not None
        assert "c_eq" in binfo.dual_multipliers
        assert "c_ge" in binfo.dual_multipliers
        assert binfo.dual_multipliers["c_ge"] == pytest.approx(1.0, abs=1e-4)

    def test_presolve_variable_reconstruction_mapping(self):
        """7. Presolve reconstruction preserves mapping across bounds/transformations."""
        model = Model(name="presolve_bounds_lp")
        model.add_variable(Variable(name="x1", lower_bound=2.0))
        model.add_variable(Variable(name="x2", lower_bound=3.0))
        model.set_objective(
            Objective(coefficients={"x1": 1.0, "x2": 1.0}, sense=ObjectiveSense.MINIMIZE)
        )
        model.add_constraint(
            Constraint(name="c1", coefficients={"x1": 1.0, "x2": 1.0}, sense=ConstraintSense.GREATER_EQUAL, rhs=10.0)
        )

        sol = model.solve(presolve=True)
        assert sol.status == SolutionStatus.OPTIMAL
        assert sol.basis_info is not None

        # Verify all original variables are in variable_status and reduced_costs
        assert "x1" in sol.basis_info.variable_status
        assert "x2" in sol.basis_info.variable_status
        assert "x1" in sol.basis_info.reduced_costs
        assert "x2" in sol.basis_info.reduced_costs
        assert "c1" in sol.basis_info.dual_multipliers

    def test_fixed_variable_elimination_mapping(self):
        """8. Fixed-variable elimination preserves mapping with BasisStatus.FIXED."""
        model = Model(name="fixed_var_lp")
        model.add_variable(Variable(name="x1", lower_bound=5.0, upper_bound=5.0))
        model.add_variable(Variable(name="x2", lower_bound=0.0))
        model.add_variable(Variable(name="x3", lower_bound=0.0))
        model.set_objective(
            Objective(coefficients={"x1": 2.0, "x2": 3.0, "x3": 4.0}, sense=ObjectiveSense.MINIMIZE)
        )
        model.add_constraint(
            Constraint(name="c1", coefficients={"x1": 1.0, "x2": 1.0, "x3": 1.0}, sense=ConstraintSense.GREATER_EQUAL, rhs=8.0)
        )
        model.add_constraint(
            Constraint(name="c2", coefficients={"x2": 1.0, "x3": 2.0}, sense=ConstraintSense.GREATER_EQUAL, rhs=6.0)
        )

        sol = model.solve(presolve=True)
        assert sol.status == SolutionStatus.OPTIMAL
        assert sol.variable_values["x1"] == pytest.approx(5.0, abs=1e-5)

        binfo = sol.basis_info
        assert binfo is not None
        assert binfo.variable_status.get("x1") == BasisStatus.FIXED
        assert "x1" in binfo.reduced_costs
        assert binfo.reduced_costs["x1"] == pytest.approx(2.0, abs=1e-4)

    def test_infeasible_lp_no_basis_metadata(self):
        """9. Infeasible LP does not expose invalid basis metadata."""
        model = Model(name="infeasible_lp")
        model.add_variable(Variable(name="x1", lower_bound=0.0))
        model.set_objective(
            Objective(coefficients={"x1": 1.0}, sense=ObjectiveSense.MINIMIZE)
        )
        model.add_constraint(
            Constraint(name="c1", coefficients={"x1": 1.0}, sense=ConstraintSense.LESS_EQUAL, rhs=2.0)
        )
        model.add_constraint(
            Constraint(name="c2", coefficients={"x1": 1.0}, sense=ConstraintSense.GREATER_EQUAL, rhs=5.0)
        )

        sol = model.solve()
        assert sol.status == SolutionStatus.INFEASIBLE
        assert sol.basis_info is None
        assert model.get_last_basis() is None

    def test_unbounded_lp_no_basis_metadata(self):
        """10. Unbounded LP does not expose invalid basis metadata."""
        model = Model(name="unbounded_lp")
        model.add_variable(Variable(name="x1", lower_bound=0.0))
        model.set_objective(
            Objective(coefficients={"x1": 1.0}, sense=ObjectiveSense.MAXIMIZE)
        )
        # No constraints

        sol = model.solve()
        assert sol.status == SolutionStatus.UNBOUNDED
        assert sol.basis_info is None
        assert model.get_last_basis() is None

    def test_milp_does_not_expose_fake_sensitivity(self):
        """11. MILP final integer solution does not expose fake sensitivity basis."""
        model = Model(name="milp_model")
        model.add_variable(Variable(name="x1", variable_type=VariableType.INTEGER, lower_bound=0.0))
        model.add_variable(Variable(name="x2", variable_type=VariableType.CONTINUOUS, lower_bound=0.0))
        model.set_objective(
            Objective(coefficients={"x1": 5.0, "x2": 4.0}, sense=ObjectiveSense.MAXIMIZE)
        )
        model.add_constraint(
            Constraint(name="c1", coefficients={"x1": 2.0, "x2": 1.0}, sense=ConstraintSense.LESS_EQUAL, rhs=5.0)
        )

        sol = model.solve()
        assert sol.status == SolutionStatus.OPTIMAL
        # Crucial MILP boundary guarantee:
        assert sol.basis_info is None
        assert model.get_last_basis() is None

    def test_basis_info_immutability(self):
        """12. Basis export is frozen/immutable and safely isolated."""
        model = build_simple_lp(maximize=True)
        sol = model.solve()
        binfo = sol.basis_info
        assert binfo is not None

        # Frozen dataclass check
        with pytest.raises(FrozenInstanceError):
            binfo.objective_value = 999.0  # type: ignore

        with pytest.raises(FrozenInstanceError):
            binfo.is_optimal = False  # type: ignore

        # get_basis_indices returns a copy
        indices = binfo.get_basis_indices()
        indices[0] = -99
        assert binfo.basic_variable_indices[0] != -99

    def test_compatible_basis_warm_start(self):
        """13. Compatible basis can be passed back into model.solve(basis=...)."""
        model = build_simple_lp(maximize=True)
        sol1 = model.solve()
        basis = sol1.basis_info
        assert basis is not None

        # Re-solve same model with basis
        sol2 = model.solve(basis=basis)
        assert sol2.status == SolutionStatus.OPTIMAL
        assert sol2.objective_value == pytest.approx(sol1.objective_value, abs=1e-5)
        for v in model.variables:
            assert sol2.variable_values[v] == pytest.approx(sol1.variable_values[v], abs=1e-5)

    def test_incompatible_basis_rejected(self):
        """14. Incompatible basis (wrong dimension / invalid index) raises ValueError."""
        model = build_simple_lp(maximize=True)
        model.solve()

        # Incompatible basis dimension (e.g. 5 basic indices when m=2)
        invalid_basis = BasisInfo(
            basic_variable_indices=[0, 1, 2, 3, 4],
            standard_form_dimension=(5, 10),
            is_lp=True,
        )
        with pytest.raises(ValueError, match="Incompatible basis"):
            model.solve(basis=invalid_basis)

        # Incompatible non-LP basis
        milp_basis = BasisInfo(
            basic_variable_indices=[0, 1],
            standard_form_dimension=(2, 4),
            is_lp=False,
        )
        with pytest.raises(ValueError, match="Cannot warm-start with a non-LP basis"):
            model.solve(basis=milp_basis)

        # Duplicate index
        dup_basis = BasisInfo(
            basic_variable_indices=[0, 0],
            standard_form_dimension=(2, 4),
            is_lp=True,
        )
        with pytest.raises(ValueError, match="duplicate column indices"):
            model.solve(basis=dup_basis)

    def test_solver_selection_metadata_preserved(self):
        """15. Existing Stage 8.43 solver-selection metadata is fully preserved."""
        model = build_simple_lp(maximize=True)
        sol = model.solve()

        assert sol.solver_selected is not None
        assert sol.problem_fingerprint is not None
        assert model.last_solver_selected is not None
        assert model.last_solver_reason is not None

    def test_custom_configurations_functional(self):
        """16. Explicit primal, dual solver preference and pricing options remain functional."""
        # Primal
        m1 = build_simple_lp(maximize=True)
        s1 = m1.solve(solver="primal", pricing="dantzig")
        assert s1.status == SolutionStatus.OPTIMAL
        assert s1.solver_selected == "revised_simplex"
        assert s1.basis_info is not None

        # Dual
        m2 = build_simple_lp(maximize=True)
        s2 = m2.solve(solver="dual")
        assert s2.status == SolutionStatus.OPTIMAL
        assert s2.solver_selected == "dual_simplex"
        assert s2.basis_info is not None

    def test_resolving_after_rhs_modification(self):
        """17. Re-solving after RHS modification produces correct result."""
        # Max 3x1 + 2x2
        # s.t. x1 + x2 <= 4, 2x1 + x2 <= 5
        model = build_simple_lp(maximize=True)
        sol1 = model.solve()
        basis = sol1.basis_info
        assert basis is not None

        # Change RHS of c1 from 4.0 to 4.5
        model.constraints["c1"].rhs = 4.5

        # Solve with previous basis
        sol_warm = model.solve(basis=basis)
        assert sol_warm.status == SolutionStatus.OPTIMAL

        # Solve fresh model without warm-start for baseline
        m_fresh = Model(name="fresh")
        m_fresh.add_variable(Variable(name="x1", lower_bound=0.0))
        m_fresh.add_variable(Variable(name="x2", lower_bound=0.0))
        m_fresh.set_objective(
            Objective(coefficients={"x1": 3.0, "x2": 2.0}, sense=ObjectiveSense.MAXIMIZE)
        )
        m_fresh.add_constraint(
            Constraint(name="c1", coefficients={"x1": 1.0, "x2": 1.0}, sense=ConstraintSense.LESS_EQUAL, rhs=4.5)
        )
        m_fresh.add_constraint(
            Constraint(name="c2", coefficients={"x1": 2.0, "x2": 1.0}, sense=ConstraintSense.LESS_EQUAL, rhs=5.0)
        )
        sol_fresh = m_fresh.solve()

        assert sol_warm.objective_value == pytest.approx(sol_fresh.objective_value, abs=1e-5)
        for v in ["x1", "x2"]:
            assert sol_warm.variable_values[v] == pytest.approx(sol_fresh.variable_values[v], abs=1e-5)

    def test_resolving_after_objective_modification(self):
        """18. Re-solving after objective modification produces correct result."""
        # Max 3x1 + 2x2
        # Modify objective to 2x1 + 4x2
        model = build_simple_lp(maximize=True)
        sol1 = model.solve()
        basis = sol1.basis_info
        assert basis is not None

        # Perturb objective
        model.set_objective(
            Objective(coefficients={"x1": 2.0, "x2": 4.0}, sense=ObjectiveSense.MAXIMIZE)
        )

        # Solve with basis
        sol_warm = model.solve(basis=basis)
        assert sol_warm.status == SolutionStatus.OPTIMAL

        # Fresh solve baseline
        m_fresh = Model(name="fresh_obj")
        m_fresh.add_variable(Variable(name="x1", lower_bound=0.0))
        m_fresh.add_variable(Variable(name="x2", lower_bound=0.0))
        m_fresh.set_objective(
            Objective(coefficients={"x1": 2.0, "x2": 4.0}, sense=ObjectiveSense.MAXIMIZE)
        )
        m_fresh.add_constraint(
            Constraint(name="c1", coefficients={"x1": 1.0, "x2": 1.0}, sense=ConstraintSense.LESS_EQUAL, rhs=4.0)
        )
        m_fresh.add_constraint(
            Constraint(name="c2", coefficients={"x1": 2.0, "x2": 1.0}, sense=ConstraintSense.LESS_EQUAL, rhs=5.0)
        )
        sol_fresh = m_fresh.solve()

        assert sol_warm.objective_value == pytest.approx(sol_fresh.objective_value, abs=1e-5)
        for v in ["x1", "x2"]:
            assert sol_warm.variable_values[v] == pytest.approx(sol_fresh.variable_values[v], abs=1e-5)

    def test_warm_start_does_not_change_mathematical_correctness(self):
        """19. Mathematical correctness is identical between cold start and warm start."""
        # 3-variable, 2-constraint LP
        model = Model(name="test_lp3")
        model.add_variable(Variable(name="x", lower_bound=0.0))
        model.add_variable(Variable(name="y", lower_bound=0.0))
        model.add_variable(Variable(name="z", lower_bound=0.0))
        model.set_objective(
            Objective(coefficients={"x": 5.0, "y": 4.0, "z": 3.0}, sense=ObjectiveSense.MAXIMIZE)
        )
        model.add_constraint(
            Constraint(name="c1", coefficients={"x": 2.0, "y": 3.0, "z": 1.0}, sense=ConstraintSense.LESS_EQUAL, rhs=12.0)
        )
        model.add_constraint(
            Constraint(name="c2", coefficients={"x": 4.0, "y": 1.0, "z": 2.0}, sense=ConstraintSense.LESS_EQUAL, rhs=10.0)
        )

        sol_cold = model.solve()
        assert sol_cold.status == SolutionStatus.OPTIMAL
        basis = sol_cold.basis_info
        assert basis is not None

        sol_warm = model.solve(basis=basis)
        assert sol_warm.status == SolutionStatus.OPTIMAL
        assert sol_warm.objective_value == pytest.approx(sol_cold.objective_value, abs=1e-6)
        for v in ["x", "y", "z"]:
            assert sol_warm.variable_values[v] == pytest.approx(sol_cold.variable_values[v], abs=1e-6)

    def test_degenerate_optimal_solution(self):
        """20. Degenerate LP handles basis extraction and reduced costs cleanly."""
        # LP with redundant tight constraints at origin
        model = Model(name="degenerate_lp")
        model.add_variable(Variable(name="x1", lower_bound=0.0))
        model.add_variable(Variable(name="x2", lower_bound=0.0))
        model.set_objective(
            Objective(coefficients={"x1": 1.0, "x2": 1.0}, sense=ObjectiveSense.MINIMIZE)
        )
        model.add_constraint(
            Constraint(name="c1", coefficients={"x1": 1.0, "x2": 1.0}, sense=ConstraintSense.GREATER_EQUAL, rhs=0.0)
        )
        model.add_constraint(
            Constraint(name="c2", coefficients={"x1": 2.0, "x2": 1.0}, sense=ConstraintSense.GREATER_EQUAL, rhs=0.0)
        )

        sol = model.solve()
        assert sol.status == SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(0.0, abs=1e-6)
        assert sol.basis_info is not None
        assert sol.basis_info.is_optimal is True
