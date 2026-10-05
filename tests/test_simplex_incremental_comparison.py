"""Stage 8.7 Phase 6: Simplex Incremental Integration & Reference Comparison.

Compares solutions obtained with the default DenseLU path vs the Incremental
Basis Factorization prototype path across all problem types:
- Maximization
- Minimization
- Phase I (equality/surplus artificials)
- Phase II (direct slack basis)
- Infeasible problems
- Unbounded problems
- Degenerate problems
- Negative bounds
- Negative objective coefficients
- Multiple optimal solutions
- Beale cycling problem
- Random LPs
"""

from __future__ import annotations

import numpy as np
import pytest

from optiforge.core.solution import SolutionStatus
from optiforge.lp.simplex import RevisedSimplexSolver, SimplexTolerances
from optiforge.model import Constraint, ConstraintSense, Model, Objective, ObjectiveSense, Variable, VariableType
from optiforge.numerics.numerical_monitor import NumericalStabilityMonitor


def solve_both(model: Model) -> tuple[any, any, NumericalStabilityMonitor, NumericalStabilityMonitor]:
    """Solve model using both reference DenseLU path and Incremental path."""
    mon_ref = NumericalStabilityMonitor()
    solver_ref = RevisedSimplexSolver(
        tolerances=SimplexTolerances(use_incremental_basis=False),
        numerical_monitor=mon_ref,
    )
    sol_ref = solver_ref.solve(model)

    mon_inc = NumericalStabilityMonitor()
    solver_inc = RevisedSimplexSolver(
        tolerances=SimplexTolerances(use_incremental_basis=True, max_basis_updates=20),
        numerical_monitor=mon_inc,
    )
    sol_inc = solver_inc.solve(model)

    return sol_ref, sol_inc, mon_ref, mon_inc


class TestSimplexIncrementalComparison:
    """Verifies that enabling use_incremental_basis produces identical results."""

    def test_maximization_standard(self) -> None:
        m = Model("max_standard")
        x = m.add_variable(Variable("x", lower_bound=0.0))
        y = m.add_variable(Variable("y", lower_bound=0.0))
        m.set_objective(Objective(coefficients={"x": 3.0, "y": 2.0}, sense=ObjectiveSense.MAXIMIZE))
        m.add_constraint(Constraint("c1", coefficients={"x": 2.0, "y": 1.0}, sense=ConstraintSense.LESS_EQUAL, rhs=10.0))
        m.add_constraint(Constraint("c2", coefficients={"x": 1.0, "y": 2.0}, sense=ConstraintSense.LESS_EQUAL, rhs=8.0))

        sol_ref, sol_inc, _, _ = solve_both(m)
        assert sol_ref.status == sol_inc.status == SolutionStatus.OPTIMAL
        assert np.isclose(sol_ref.objective_value, sol_inc.objective_value, atol=1e-8)
        assert np.isclose(sol_ref.variable_values["x"], sol_inc.variable_values["x"], atol=1e-8)
        assert np.isclose(sol_ref.variable_values["y"], sol_inc.variable_values["y"], atol=1e-8)
        assert sol_ref.iterations == sol_inc.iterations

    def test_minimization_with_surplus_and_artificial(self) -> None:
        """Phase I and Phase II test with >= constraint."""
        m = Model("min_phase1_phase2")
        x1 = m.add_variable(Variable("x1", lower_bound=0.0))
        x2 = m.add_variable(Variable("x2", lower_bound=0.0))
        m.set_objective(Objective(coefficients={"x1": 4.0, "x2": 1.0}, sense=ObjectiveSense.MINIMIZE))
        m.add_constraint(Constraint("c1", coefficients={"x1": 3.0, "x2": 1.0}, sense=ConstraintSense.EQUAL, rhs=3.0))
        m.add_constraint(Constraint("c2", coefficients={"x1": 4.0, "x2": 3.0}, sense=ConstraintSense.GREATER_EQUAL, rhs=6.0))
        m.add_constraint(Constraint("c3", coefficients={"x1": 1.0, "x2": 2.0}, sense=ConstraintSense.LESS_EQUAL, rhs=4.0))

        sol_ref, sol_inc, _, _ = solve_both(m)
        assert sol_ref.status == sol_inc.status == SolutionStatus.OPTIMAL
        assert np.isclose(sol_ref.objective_value, sol_inc.objective_value, atol=1e-8)
        assert np.isclose(sol_ref.variable_values["x1"], sol_inc.variable_values["x1"], atol=1e-8)
        assert np.isclose(sol_ref.variable_values["x2"], sol_inc.variable_values["x2"], atol=1e-8)
        assert sol_ref.iterations == sol_inc.iterations

    def test_infeasible_lp(self) -> None:
        m = Model("infeasible")
        x = m.add_variable(Variable("x", lower_bound=0.0))
        m.set_objective(Objective(coefficients={"x": 1.0}, sense=ObjectiveSense.MINIMIZE))
        m.add_constraint(Constraint("c1", coefficients={"x": 1.0}, sense=ConstraintSense.GREATER_EQUAL, rhs=5.0))
        m.add_constraint(Constraint("c2", coefficients={"x": 1.0}, sense=ConstraintSense.LESS_EQUAL, rhs=2.0))

        sol_ref, sol_inc, _, _ = solve_both(m)
        assert sol_ref.status == sol_inc.status == SolutionStatus.INFEASIBLE

    def test_unbounded_lp(self) -> None:
        m = Model("unbounded")
        x = m.add_variable(Variable("x", lower_bound=0.0))
        y = m.add_variable(Variable("y", lower_bound=0.0))
        m.set_objective(Objective(coefficients={"x": 1.0, "y": 1.0}, sense=ObjectiveSense.MAXIMIZE))
        m.add_constraint(Constraint("c1", coefficients={"x": -1.0, "y": 1.0}, sense=ConstraintSense.LESS_EQUAL, rhs=2.0))

        sol_ref, sol_inc, _, _ = solve_both(m)
        assert sol_ref.status == sol_inc.status == SolutionStatus.UNBOUNDED

    def test_degenerate_lp(self) -> None:
        m = Model("degenerate")
        x1 = m.add_variable(Variable("x1", lower_bound=0.0))
        x2 = m.add_variable(Variable("x2", lower_bound=0.0))
        m.set_objective(Objective(coefficients={"x1": 1.0, "x2": 1.0}, sense=ObjectiveSense.MAXIMIZE))
        m.add_constraint(Constraint("c1", coefficients={"x1": 1.0, "x2": 1.0}, sense=ConstraintSense.LESS_EQUAL, rhs=2.0))
        m.add_constraint(Constraint("c2", coefficients={"x1": 1.0}, sense=ConstraintSense.LESS_EQUAL, rhs=1.0))
        m.add_constraint(Constraint("c3", coefficients={"x1": 2.0, "x2": 2.0}, sense=ConstraintSense.LESS_EQUAL, rhs=4.0))

        sol_ref, sol_inc, _, _ = solve_both(m)
        assert sol_ref.status == sol_inc.status == SolutionStatus.OPTIMAL
        assert np.isclose(sol_ref.objective_value, sol_inc.objective_value, atol=1e-8)
        assert sol_ref.iterations == sol_inc.iterations

    def test_negative_bounds_and_coefficients(self) -> None:
        m = Model("negative_bounds_coeffs")
        x = m.add_variable(Variable("x", lower_bound=-5.0, upper_bound=5.0))
        y = m.add_variable(Variable("y", lower_bound=-2.0, upper_bound=8.0))
        m.set_objective(Objective(coefficients={"x": -3.0, "y": -4.0}, sense=ObjectiveSense.MINIMIZE))
        m.add_constraint(Constraint("c1", coefficients={"x": 1.0, "y": -1.0}, sense=ConstraintSense.LESS_EQUAL, rhs=3.0))
        m.add_constraint(Constraint("c2", coefficients={"x": -2.0, "y": 1.0}, sense=ConstraintSense.GREATER_EQUAL, rhs=-6.0))

        sol_ref, sol_inc, _, _ = solve_both(m)
        assert sol_ref.status == sol_inc.status == SolutionStatus.OPTIMAL
        assert np.isclose(sol_ref.objective_value, sol_inc.objective_value, atol=1e-8)
        assert np.isclose(sol_ref.variable_values["x"], sol_inc.variable_values["x"], atol=1e-8)
        assert np.isclose(sol_ref.variable_values["y"], sol_inc.variable_values["y"], atol=1e-8)

    def test_beale_cycling_problem(self) -> None:
        """Beale's classic cycling problem tests anti-cycling and numerical pivot logic."""
        m = Model("beale_cycling")
        x1 = m.add_variable(Variable("x1", lower_bound=0.0))
        x2 = m.add_variable(Variable("x2", lower_bound=0.0))
        x3 = m.add_variable(Variable("x3", lower_bound=0.0))
        x4 = m.add_variable(Variable("x4", lower_bound=0.0))

        m.set_objective(Objective(
            coefficients={"x1": -10.0, "x2": 57.0, "x3": -9.0, "x4": 24.0},
            sense=ObjectiveSense.MINIMIZE,
        ))
        m.add_constraint(Constraint("c1", coefficients={"x1": 0.5, "x2": -5.5, "x3": -2.5, "x4": 9.0}, sense=ConstraintSense.LESS_EQUAL, rhs=0.0))
        m.add_constraint(Constraint("c2", coefficients={"x1": 0.5, "x2": -1.5, "x3": -0.5, "x4": 1.0}, sense=ConstraintSense.LESS_EQUAL, rhs=0.0))
        m.add_constraint(Constraint("c3", coefficients={"x1": 1.0}, sense=ConstraintSense.LESS_EQUAL, rhs=1.0))

        sol_ref, sol_inc, _, _ = solve_both(m)
        assert sol_ref.status == sol_inc.status == SolutionStatus.UNBOUNDED
        assert sol_ref.iterations == sol_inc.iterations

    def test_multiple_optimal_solutions(self) -> None:
        m = Model("multiple_optimal")
        x = m.add_variable(Variable("x", lower_bound=0.0))
        y = m.add_variable(Variable("y", lower_bound=0.0))
        m.set_objective(Objective(coefficients={"x": 2.0, "y": 2.0}, sense=ObjectiveSense.MAXIMIZE))
        m.add_constraint(Constraint("c1", coefficients={"x": 1.0, "y": 1.0}, sense=ConstraintSense.LESS_EQUAL, rhs=4.0))

        sol_ref, sol_inc, _, _ = solve_both(m)
        assert sol_ref.status == sol_inc.status == SolutionStatus.OPTIMAL
        assert np.isclose(sol_ref.objective_value, sol_inc.objective_value, atol=1e-8)
        assert np.isclose(sol_ref.objective_value, 8.0)

    @pytest.mark.parametrize("seed", [42, 101, 2026])
    def test_random_lps(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        m = Model(f"random_lp_{seed}")
        n_vars = 6
        n_cons = 4
        vars = [m.add_variable(Variable(f"x{i}", lower_bound=0.0, upper_bound=10.0)) for i in range(n_vars)]

        c_obj = {f"x{i}": float(rng.integers(-5, 10)) for i in range(n_vars)}
        m.set_objective(Objective(coefficients=c_obj, sense=ObjectiveSense.MAXIMIZE))

        for j in range(n_cons):
            coeffs = {f"x{i}": float(rng.integers(1, 6)) for i in range(n_vars)}
            rhs = float(rng.integers(15, 50))
            m.add_constraint(Constraint(f"c{j}", coefficients=coeffs, sense=ConstraintSense.LESS_EQUAL, rhs=rhs))

        sol_ref, sol_inc, _, _ = solve_both(m)
        assert sol_ref.status == sol_inc.status
        if sol_ref.status == SolutionStatus.OPTIMAL:
            assert np.isclose(sol_ref.objective_value, sol_inc.objective_value, atol=1e-7)
            for var_name in sol_ref.variable_values:
                assert np.isclose(sol_ref.variable_values[var_name], sol_inc.variable_values[var_name], atol=1e-7)
            assert sol_ref.iterations == sol_inc.iterations
