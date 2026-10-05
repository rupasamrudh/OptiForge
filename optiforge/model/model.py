"""Optimization model container."""

from __future__ import annotations

from typing import Any

from optiforge.core.solution import Solution, SolutionStatus
from optiforge.model.variable import Variable
from optiforge.model.constraint import Constraint
from optiforge.model.objective import Objective


class Model:
    """A mathematical optimization model.

    Holds variables, constraints, and an objective function with full
    validation of names, bounds, and coefficient references.

    Parameters
    ----------
    name : str
        Human-readable name for the model.
    """

    def __init__(self, name: str = "model") -> None:
        self.name: str = name
        self._variables: dict[str, Variable] = {}
        self._constraints: dict[str, Constraint] = {}
        self._objective: Objective | None = None
        self.last_fingerprint: Any | None = None
        self.last_solver_selected: str | None = None
        self.last_solver_reason: str | None = None
        self.last_basis: Any | None = None
        self.last_sensitivity: Any | None = None


    # -- Variables ------------------------------------------------------------

    def add_variable(self, variable: Variable) -> Variable:
        """Add a variable to the model.

        Raises
        ------
        ValueError
            If a variable with the same name already exists.
        """
        if variable.name in self._variables:
            raise ValueError(
                f"Variable '{variable.name}' already exists in model '{self.name}'."
            )
        self._variables[variable.name] = variable
        return variable

    @property
    def variables(self) -> dict[str, Variable]:
        """Return a read-only view of all variables."""
        return dict(self._variables)

    # -- Constraints ----------------------------------------------------------

    def add_constraint(self, constraint: Constraint) -> Constraint:
        """Add a constraint to the model.

        Raises
        ------
        ValueError
            If a constraint with the same name already exists, or if any
            coefficient references an unknown variable.
        """
        if constraint.name in self._constraints:
            raise ValueError(
                f"Constraint '{constraint.name}' already exists in model '{self.name}'."
            )
        self._validate_coefficients(constraint.coefficients, context=f"constraint '{constraint.name}'")
        self._constraints[constraint.name] = constraint
        return constraint

    @property
    def constraints(self) -> dict[str, Constraint]:
        """Return a read-only view of all constraints."""
        return dict(self._constraints)

    # -- Objective ------------------------------------------------------------

    def set_objective(self, objective: Objective) -> None:
        """Set the model's objective function.

        Raises
        ------
        ValueError
            If any coefficient references an unknown variable.
        """
        self._validate_coefficients(objective.coefficients, context="objective")
        self._objective = objective

    @property
    def objective(self) -> Objective | None:
        """Return the current objective (or None)."""
        return self._objective

    # -- Summaries ------------------------------------------------------------

    @property
    def num_variables(self) -> int:
        return len(self._variables)

    @property
    def num_constraints(self) -> int:
        return len(self._constraints)

    def summary(self) -> str:
        """Return a short textual summary of the model."""
        lines = [
            f"Model: {self.name}",
            f"  Variables   : {self.num_variables}",
            f"  Constraints : {self.num_constraints}",
            f"  Objective   : {self._objective}",
        ]
        return "\n".join(lines)

    # -- Solving --------------------------------------------------------------

    def get_last_basis(self) -> Any | None:
        """Return the optimal BasisInfo from the most recent solve, if available."""
        return self.last_basis

    def get_last_sensitivity(self) -> Any | None:
        """Return the SensitivityReport from the most recent solve, if available."""
        return self.last_sensitivity

    def sensitivity_analysis(self) -> Any:
        """Perform post-optimality sensitivity analysis using the last optimal LP basis.

        Returns
        -------
        SensitivityReport
            Allowable variation intervals for constraint RHS and objective coefficients.

        Raises
        ------
        ValueError
            If the model is not a continuous LP or has no optimal basis.
        """
        from optiforge.sensitivity import compute_sensitivity

        report = compute_sensitivity(model=self)
        self.last_sensitivity = report
        return report

    def evaluate_scenario(
        self,
        scenario: Any,
        resolve_if_needed: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Analytically evaluate a what-if scenario against the last optimal LP basis.

        Parameters
        ----------
        scenario : Scenario
            The what-if scenario specification.
        resolve_if_needed : bool, optional
            If True, re-solve the LP using the full solver pipeline if the current basis
            does not remain optimal. Original model remains completely unchanged.
        **kwargs : Any
            Additional options passed to evaluate_scenario.

        Returns
        -------
        ScenarioResult
            Evaluation result with exact basis preservation status and 100% Rule outcome.
        """
        from optiforge.sensitivity import evaluate_scenario

        return evaluate_scenario(
            model=self,
            scenario=scenario,
            resolve_if_needed=resolve_if_needed,
            **kwargs,
        )

    def compare_scenarios(
        self,
        scenarios: list[Any],
        resolve_if_needed: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Evaluate and compare multiple scenarios against this model's baseline.

        Parameters
        ----------
        scenarios : list[Scenario]
            List of scenarios to evaluate.
        resolve_if_needed : bool, optional
            Whether to re-solve non-preserved scenarios.
        **kwargs : Any
            Additional options passed to evaluate_scenario.

        Returns
        -------
        ScenarioComparison
            Structured comparison container with summary metrics.
        """
        from optiforge.sensitivity import compare_scenarios

        return compare_scenarios(
            model=self,
            scenarios=scenarios,
            resolve_if_needed=resolve_if_needed,
            **kwargs,
        )

    def break_even(
        self,
        parameter: str,
        target_value: float,
        target: str = "objective_value",
        parameter_type: str = "auto",
        direction: str | None = None,
    ) -> Any:
        """Perform 1D break-even analysis for a target objective value.

        Parameters
        ----------
        parameter : str
            Constraint name or variable name.
        target_value : float
            Desired target value.
        target : str, optional
            Target metric (default "objective_value").
        parameter_type : str, optional
            "rhs", "objective", or "auto" (default "auto").
        direction : str | None, optional
            Optional expected direction ("increase", "decrease").

        Returns
        -------
        BreakEvenResult
            Calculated break-even delta, parameter value, and basis reachability.
        """
        from optiforge.sensitivity import compute_break_even

        return compute_break_even(
            model=self,
            parameter=parameter,
            target_value=target_value,
            target=target,
            parameter_type=parameter_type,
            direction=direction,
        )


    def solve(
        self,
        tolerances: Any = None,
        stop_event: Any = None,
        solver: str | None = None,
        presolve: bool = True,
        pricing: str | None = None,
        numerical_monitor: Any = None,
        numerical_thresholds: Any = None,
        basis: Any = None,
        numerical_verification: str | None = None,
        scaling: bool | None = None,
        **kwargs: Any,
    ) -> Solution:
        """Solve this model using the unified OptiForge optimization pipeline.

        Orchestrates: Model -> Presolve -> Problem Fingerprint -> Solver Selection -> Postsolve -> Basis -> Solution

        Parameters
        ----------
        tolerances : SimplexTolerances | MILPTolerances | None, optional
            Numerical tolerances for the LP or MILP solver.
        stop_event : threading.Event | None, optional
            Optional cancellation event.
        solver : str | None, optional
            Explicit solver preference ('auto', 'primal', 'dual', 'milp').
            Default is 'auto' (deterministic selection based on model structure).
        presolve : bool, optional
            Whether to apply Stage 5A presolve (default True for continuous LP).
        pricing : str | None, optional
            Simplex pricing strategy ('bland', 'dantzig') if supported by solver.
        numerical_monitor : Any | None, optional
            Optional numerical stability monitor.
        numerical_thresholds : Any | None, optional
            Optional custom numerical thresholds.
        basis : BasisInfo | list[int] | None, optional
            Optional basis for warm-start re-optimization.
        numerical_verification : str | None, optional
            Post-solve verification mode: 'none', 'basic', or 'strict'.
        scaling : bool | None, optional
            Whether to apply iterative Ruiz dynamic scaling prior to solving.
        **kwargs : Any
            Additional solver-specific options forwarded to the selected solver.

        Returns
        -------
        Solution
            Standard OptiForge Solution container with status, variable values,
            objective value, iterations, problem fingerprint, solver selected, and basis info.
        """
        from optiforge.model.pipeline import execute_pipeline

        return execute_pipeline(
            model=self,
            tolerances=tolerances,
            stop_event=stop_event,
            solver=solver,
            presolve=presolve,
            pricing=pricing,
            numerical_monitor=numerical_monitor,
            numerical_thresholds=numerical_thresholds,
            basis=basis,
            numerical_verification=numerical_verification,
            scaling=scaling,
            **kwargs,
        )

    # -- Internal helpers -----------------------------------------------------

    def _validate_coefficients(
        self, coefficients: dict[str, float], *, context: str
    ) -> None:
        """Ensure every key in *coefficients* names a known variable."""
        unknown = set(coefficients) - set(self._variables)
        if unknown:
            raise ValueError(
                f"Unknown variable(s) in {context}: {sorted(unknown)}"
            )

    def __repr__(self) -> str:
        return (
            f"Model(name='{self.name}', vars={self.num_variables}, "
            f"constrs={self.num_constraints})"
        )
