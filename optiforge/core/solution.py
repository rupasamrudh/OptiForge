"""Solution representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SolutionStatus(Enum):
    """Status of an optimization solve."""

    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    ITERATION_LIMIT = "iteration_limit"
    UNKNOWN = "unknown"
    OPTIMAL_WITH_NUMERICAL_WARNING = "optimal_with_numerical_warning"
    NUMERICAL_INSTABILITY = "numerical_instability"
    SINGULAR_BASIS = "singular_basis"


@dataclass
class Solution:
    """Result container returned by a solver.

    Parameters
    ----------
    status : SolutionStatus
        Solve outcome.
    variable_values : dict[str, float]
        Mapping of variable name → value in the solution.
    objective_value : float | None
        Value of the objective at the solution (None if not available).
    message : str
        Human-readable description of the solve outcome.
    iterations : int
        Number of simplex iterations performed (LP) or nodes processed (MILP).
    node_count : int | None
        Number of branch-and-bound nodes processed (MILP only).
    numerical_snapshots : list | None
        Numerical monitoring snapshots (if monitoring was enabled).
    incumbent_objective : float | None
        Best integer-feasible objective value found (MILP only).
    best_bound : float | None
        Best bound on the optimal value (MILP only).
    optimality_gap : float | None
        Relative optimality gap (MILP only).
    absolute_gap : float | None
        Absolute optimality gap (MILP only).
    nodes_remaining : int | None
        Number of unprocessed nodes in the B&B tree (MILP only).
    max_depth : int | None
        Maximum depth of the B&B tree (MILP only).
    feasible_solutions_found : int | None
        Number of integer-feasible solutions found (MILP only).
    """

    status: SolutionStatus = SolutionStatus.UNKNOWN
    variable_values: dict[str, float] = field(default_factory=dict)
    objective_value: float | None = None
    message: str = ""
    iterations: int = 0
    node_count: int | None = None
    numerical_snapshots: list | None = None
    # MILP-specific metrics
    incumbent_objective: float | None = None
    best_bound: float | None = None
    optimality_gap: float | None = None
    absolute_gap: float | None = None
    nodes_remaining: int | None = None
    max_depth: int | None = None
    feasible_solutions_found: int | None = None
    profile_stats: Any | None = None
    problem_fingerprint: Any | None = None
    solver_selected: str | None = None
    solver_reason: str | None = None
    solver_metadata: dict[str, Any] = field(default_factory=dict)
    basis_info: Any | None = None
    sensitivity: Any | None = None
    diagnostics: Any | None = None
    _model: Any | None = None

    def verify(
        self,
        model: Any | None = None,
        policy: Any | None = None,
        mode: str = "strict",
    ) -> Any:
        """Run post-solve verification and return structured SolverDiagnostics."""
        target_m = model if model is not None else self._model
        if target_m is None:
            raise ValueError("Model must be provided to verify solution if not stored on Solution.")
        from optiforge.numerics.verification import verify_solution

        diag = verify_solution(model=target_m, solution=self, policy=policy, mode=mode)
        self.diagnostics = diag
        return diag

    def get_sensitivity(self) -> Any:
        """Return the sensitivity report for this solution, or raise ValueError."""
        if self.sensitivity is None:
            raise ValueError(
                "Sensitivity report is not available for this solution "
                f"(status={self.status.value})."
            )
        return self.sensitivity

    def __repr__(self) -> str:
        return (
            f"Solution(status={self.status.value}, "
            f"obj={self.objective_value}, iters={self.iterations}, "
            f"nodes={self.node_count}, msg='{self.message}')"
        )

