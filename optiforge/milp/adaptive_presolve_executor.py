"""Adaptive Presolve Executor for OptiForge.

This module provides a configurable presolve executor that delegates to the
Stage 5A presolve implementation, selectively applying operations based on an
AdaptivePresolveStrategy decision. It wraps the existing presolve logic without
changing its mathematical semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from optiforge.core.solution import SolutionStatus
from optiforge.milp.adaptive_presolve import PresolveDecision, PresolveOperation
from optiforge.model.constraint import Constraint, ConstraintSense
from optiforge.model.model import Model
from optiforge.model.objective import Objective, ObjectiveSense
from optiforge.model.variable import Variable
from optiforge.presolve.presolve import PresolveResult as Stage5APresolveResult
from optiforge.presolve.presolve import _presolve_model_impl
from optiforge.presolve.presolve import _PresolveInfeasible


@dataclass
class PresolveResult:
    """Outcome of a presolve execution.

    Attributes
    ----------
    reduced_model : Model | None
        Reduced model after all deterministic reductions. None if the problem is
        infeasible or unbounded before entering the simplex stage.
    status : SolutionStatus
        Presolve status (typically FEASIBLE, INFEASIBLE, or UNBOUNDED).
    original_variable_order : list[str]
        Original variable sequence as stored in the input model.
    reduced_variable_mapping : dict[str, str]
        Maps a reduced variable name to the original variable name.
    fixed_variables : dict[str, float]
        Variables fixed during presolve and the values assigned to them.
    removed_constraints : list[dict[str, Any]]
        Records of constraints removed during reduction.
    objective_constant : float
        Constant shift added to the reduced objective so that the original
        objective value is recovered exactly as:
            original_obj = reduced_obj + objective_constant.
    transformation_metadata : list[str]
        Human-readable log of reductions applied.
    statistics : dict[str, int]
        Summary counters for presolve operations.
    original_to_reduced : dict[str, str]
        Mapping of original variable names to their reduced names if preserved.
    reduced_to_original : dict[str, str]
        Reverse mapping for postsolve reconstruction.
    """

    reduced_model: Model | None
    status: SolutionStatus
    original_variable_order: list[str] = field(default_factory=list)
    reduced_variable_mapping: dict[str, str] = field(default_factory=dict)
    fixed_variables: dict[str, float] = field(default_factory=dict)
    removed_constraints: list[dict[str, Any]] = field(default_factory=list)
    objective_constant: float = 0.0
    transformation_metadata: list[str] = field(default_factory=list)
    statistics: dict[str, int] = field(default_factory=dict)
    original_to_reduced: dict[str, str] = field(default_factory=dict)
    reduced_to_original: dict[str, str] = field(default_factory=dict)


class AdaptivePresolveExecutor:
    """Executes presolve operations based on an AdaptivePresolveStrategy decision.

    This executor delegates to the Stage 5A presolve implementation and allows
    selective application of operations based on the decision from
    AdaptivePresolveStrategy.
    """

    def __init__(self, feasibility_tol: float = 1e-7) -> None:
        self.feasibility_tol = float(feasibility_tol)

    def execute(self, model: Model, decision: PresolveDecision) -> PresolveResult:
        """Execute presolve with the given decision.

        Parameters
        ----------
        model : Model
            The model to presolve.
        decision : PresolveDecision
            The decision specifying which operations to apply.

        Returns
        -------
        PresolveResult
            The result of presolve execution.
        """
        # Convert the enabled operations from the decision to a set
        enabled_operations = set(decision.enabled_operations)

        try:
            # Delegate to Stage 5A presolve implementation
            stage5a_result = _presolve_model_impl(
                model=model,
                feasibility_tol=self.feasibility_tol,
                enabled_operations=enabled_operations,
            )
        except _PresolveInfeasible as exc:
            # Return INFEASIBLE result when infeasibility is detected during presolve
            original_vars = list(model.variables.keys())
            original_constraints = list(model.constraints.keys())
            return PresolveResult(
                reduced_model=None,
                status=SolutionStatus.INFEASIBLE,
                original_variable_order=original_vars,
                reduced_variable_mapping={},
                fixed_variables={},
                removed_constraints=[],
                objective_constant=0.0,
                transformation_metadata=[str(exc)],
                statistics={
                    "original_vars": len(original_vars),
                    "original_constraints": len(original_constraints),
                    "reduced_vars": 0,
                    "reduced_constraints": 0,
                    "fixed_vars": 0,
                    "removed_rows": 0,
                    "tightened_bounds": 0,
                    "passes": 0,
                },
                original_to_reduced={},
                reduced_to_original={},
            )

        # Convert Stage 5A PresolveResult to Adaptive PresolveResult
        # They have the same structure, so we can just copy the fields
        return PresolveResult(
            reduced_model=stage5a_result.reduced_model,
            status=stage5a_result.status,
            original_variable_order=stage5a_result.original_variable_order,
            reduced_variable_mapping=stage5a_result.reduced_variable_mapping,
            fixed_variables=stage5a_result.fixed_variables,
            removed_constraints=stage5a_result.removed_constraints,
            objective_constant=stage5a_result.objective_constant,
            transformation_metadata=stage5a_result.transformation_metadata,
            statistics=stage5a_result.statistics,
            original_to_reduced=stage5a_result.original_to_reduced,
            reduced_to_original=stage5a_result.reduced_to_original,
        )


def adaptive_presolve_model(model: Model, decision: PresolveDecision, feasibility_tol: float = 1e-7) -> PresolveResult:
    """Run adaptive presolve with the given decision.

    This is the main entry point for adaptive presolve. It creates an executor
    and runs the presolve operations specified in the decision.

    Parameters
    ----------
    model : Model
        The model to presolve.
    decision : PresolveDecision
        The decision specifying which operations to apply.
    feasibility_tol : float
        Feasibility tolerance for presolve operations.

    Returns
    -------
    PresolveResult
        The result of presolve execution.
    """
    executor = AdaptivePresolveExecutor(feasibility_tol=feasibility_tol)
    return executor.execute(model, decision)