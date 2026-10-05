"""MILP Branch-and-Bound Solver for OptiForge.

Implements a minimal, deterministic Branch-and-Bound algorithm for Mixed-Integer
Linear Programming using the existing LP solver pipeline (Presolve → Standard Form
→ Revised Simplex → Postsolve) for node relaxations.


Stage 7.1 introduced pluggable branching and node-selection strategies plus a
search-state observation callback. The default branching rule deliberately
changed from first-fractional (pre-Stage-7.1) to most-fractional; see
MostFractionalBranching. Termination requires every open node to be pruned by
bound; unprocessed nodes are pruned using their inherited ``lp_bound`` (a valid
relaxation bound), never merely because they are unprocessed.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

from optiforge.core.solution import Solution, SolutionStatus
from optiforge.lp.dual_simplex import DualRevisedSimplexSolver
from optiforge.lp.simplex import RevisedSimplexSolver, SimplexTolerances
from optiforge.lp.standard_form import SingularBasisError, to_standard_form
from optiforge.milp.fingerprint import ProblemFingerprint, fingerprint_model
from optiforge.milp.heuristics import HeuristicResult, LPRoundingHeuristic, PrimalHeuristic
from optiforge.milp.monitor import SearchStateMonitor, SearchStateSnapshot
from optiforge.milp.search_state import SearchState, SearchStateCallback
from optiforge.milp.adaptive_presolve import AdaptivePresolveStrategy, PresolveDecision
from optiforge.milp.adaptive_presolve_executor import adaptive_presolve_model
from optiforge.milp.strategy_switching import (
    BRANCHING_ADAPTIVE,
    BRANCHING_MOST_FRACTIONAL,
    HEURISTIC_CONFIGURED,
    HEURISTIC_NONE,
    NODE_SELECTION_ADAPTIVE,
    NODE_SELECTION_BEST_BOUND,
    StrategyManager,
)
from optiforge.model import Constraint, ConstraintSense, Model, Objective, ObjectiveSense, Variable, VariableType
from optiforge.presolve import PresolveResult, presolve_model, postsolve_solution
from optiforge.numerics.basis import BasisFactorization
from optiforge.numerics.bordered_basis import BorderedBasisFactorization
from optiforge.numerics.pivot_basis import PivotUpdateFactorization
from optiforge.numerics.incremental_basis import IncrementalBasisFactorization
from optiforge.numerics.numerical_monitor import NumericalStabilityMonitor, NumericalThresholds


# ---------------------------------------------------------------------------
# Tolerances & Warm-Start State
# ---------------------------------------------------------------------------

@dataclass
class MILPTolerances:
    """Numerical tolerances for the MILP Branch-and-Bound algorithm."""

    integrality_tolerance: float = 1e-6
    """Maximum absolute deviation from an integer value for a variable to be
    considered integer-feasible."""

    bound_tolerance: float = 1e-7
    """Tolerance for bound comparisons in pruning."""

    max_nodes: int = 10_000
    """Maximum number of nodes to process before declaring iteration limit."""

    max_iterations_per_node: int = 10_000
    """Maximum simplex iterations per node LP relaxation."""

    warm_start_relaxations: bool = False
    """Enable Dual Revised Simplex warm-starting for child LP relaxations."""

    inherit_basis_factorization: bool = False
    """Enable Bordered / Schur-Complement basis factorization inheritance (Stage 8.10)."""

    use_adaptive_pivot_factorization: bool = False
    """Enable adaptive dimension-based PFI dispatch for child LP relaxations (Stage 8.17)."""

    adaptive_pfi_min_dimension: int = 6
    """Minimum constraint matrix row dimension m to enable PFI under adaptive dispatch."""

    use_fast_pivot_update: bool = False
    """Enable optimized PFI pivot update path (Stage 8.19)."""

    use_scipy_triangular_solve: bool = False
    """Enable optional SciPy BLAS triangular solve provider (Stage 8.21)."""

    scipy_triangular_min_dimension: int = 10
    """Minimum constraint matrix row dimension m for SciPy triangular solve dispatch."""

    triangular_solve_provider: str = "python"
    """Triangular solve provider selection ('python' or 'scipy'). Default 'python'."""

    use_scipy_lu_factor: bool = False
    """Enable optional SciPy LAPACK dgetrf LU factorization provider (Stage 8.39)."""

    scipy_lu_min_dimension: int = 10
    """Minimum constraint matrix row dimension m for SciPy LU factorization dispatch."""

    lu_factor_provider: str = "python"
    """LU factorization provider selection ('python' or 'scipy'). Default 'python'."""

    enable_profiling: bool = False
    """Enable lightweight component profiling for LP relaxations and branch-and-bound phases."""

    def __post_init__(self) -> None:
        if self.use_scipy_triangular_solve:
            self.triangular_solve_provider = "scipy"
        elif self.triangular_solve_provider == "scipy":
            self.use_scipy_triangular_solve = True

        if self.use_scipy_lu_factor:
            self.lu_factor_provider = "scipy"
        elif self.lu_factor_provider == "scipy":
            self.use_scipy_lu_factor = True


@dataclass
class WarmStartStats:
    """Statistics for LP relaxation warm-starting in Branch-and-Bound."""

    attempted: int = 0
    accepted: int = 0
    fallbacks: int = 0
    infeasible_pruned: int = 0
    dual_pivots: int = 0
    cold_pivots: int = 0

    # Stage 8.10 Factorization Inheritance Stats:
    inheritance_attempted: int = 0
    inheritance_accepted: int = 0
    inheritance_fallbacks: int = 0
    inherited_factorization_failures: int = 0
    dense_lu_calls_saved: int = 0

    # Stage 8.16 PFI Dual Simplex Stats:
    pfi_updates: int = 0
    pfi_refactorizations: int = 0
    pfi_fallbacks: int = 0

    # Stage 8.17 Adaptive Dispatch Stats:
    adaptive_pfi_chosen: int = 0
    adaptive_inc_chosen: int = 0


@dataclass
class WarmStartBasisState:
    """Carries parent standard-form state and optimal basis for child warm-starts."""

    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    basis_indices: list[int]
    nonbasic_indices: list[int]
    sense: ObjectiveSense
    var_mapping: dict[str, int]
    variable_shifts: dict[int, float] = field(default_factory=dict)
    objective_constant: float = 0.0
    basis_factorization: Any | None = None


# ---------------------------------------------------------------------------
# Node Representation
# ---------------------------------------------------------------------------

@dataclass
class BnBNode:
    """A node in the Branch-and-Bound tree.

    Attributes
    ----------
    node_id : int
        Unique identifier for this node.
    parent_id : int | None
        Parent node ID (None for root).
    depth : int
        Depth in the tree (root = 0).
    bounds : dict[str, tuple[float, float]]
        Additional variable bounds imposed at this node: {var_name: (lb, ub)}.
    constraints : list[Constraint]
        Additional constraints imposed at this node (branching constraints).
    lp_status : SolutionStatus | None
        Status of the LP relaxation at this node.
    lp_objective : float | None
        Objective value of the LP relaxation.
    lp_solution : dict[str, float] | None
        Variable values from the LP relaxation (original variable space).
    lp_bound : float | None
        Best known LP relaxation bound for this node's subtree. For an
        unprocessed node this is inherited from its parent's relaxation
        (a valid bound because branching only tightens the feasible region);
        once the node is processed it equals ``lp_objective``. Used for
        best-bound node selection and search-state gap reporting.
    is_integral : bool
        Whether the LP relaxation solution is integer-feasible.
    fractional_var : str | None
        Name of the variable chosen for branching (if fractional). This is
        the variable actually branched on (the strategy's decision), so it
        is never dead metadata.
    fractional_value : float | None
        Value of the fractional variable chosen for branching.
    warm_start_state : WarmStartBasisState | None
        Standard-form basis state for warm-starting child LP relaxations.
    branch_info : tuple[str, str, float] | None
        Branch specification for this node: (var_name, 'upper' | 'lower', bound_value).
    """

    node_id: int
    parent_id: int | None
    depth: int
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    constraints: list[Constraint] = field(default_factory=list)
    lp_status: SolutionStatus | None = None
    lp_objective: float | None = None
    lp_solution: dict[str, float] | None = None
    lp_bound: float | None = None
    is_integral: bool = False
    fractional_var: str | None = None
    fractional_value: float | None = None
    warm_start_state: WarmStartBasisState | None = None
    branch_info: tuple[str, str, float] | None = None


# ---------------------------------------------------------------------------
# Branching Decision
# ---------------------------------------------------------------------------

@dataclass
class BranchingDecision:
    """Result of a branching variable selection.

    Attributes
    ----------
    variable_name : str
        Name of the variable to branch on.
    value : float
        Current LP relaxation value of the variable.
    """

    variable_name: str
    value: float


class BranchingStrategy:
    """Abstract base class for branching variable selection strategies.

    Subclasses must implement the `choose_branching_variable` method.
    """

    def choose_branching_variable(
        self,
        node: BnBNode,
        model: Model,
        tolerances: MILPTolerances,
    ) -> BranchingDecision | None:
        """Select a fractional integer variable to branch on.

        Parameters
        ----------
        node : BnBNode
            The current B&B node being processed.
        model : Model
            The original MILP model (for variable type information).
        tolerances : MILPTolerances
            Numerical tolerances for integrality checking.

        Returns
        -------
        BranchingDecision | None
            The branching decision, or None if no fractional integer
            variable exists (should not be called if node is integral).
        """
        raise NotImplementedError


class MostFractionalBranching(BranchingStrategy):
    """Most-fractional branching strategy (default).

    Selects the integer variable whose LP relaxation value is furthest
    from the nearest integer (largest deviation).
    """

    def choose_branching_variable(
        self,
        node: BnBNode,
        model: Model,
        tolerances: MILPTolerances,
    ) -> BranchingDecision | None:
        tol = tolerances.integrality_tolerance
        best_var = None
        best_deviation = 0.0

        for var_name, var in model.variables.items():
            if not var.is_integer():
                continue

            value = node.lp_solution.get(var_name, 0.0)
            nearest_int = round(value)
            deviation = abs(value - nearest_int)

            if deviation > tol and deviation > best_deviation:
                best_deviation = deviation
                best_var = var_name

        if best_var is None:
            return None

        return BranchingDecision(
            variable_name=best_var,
            value=node.lp_solution[best_var],
        )


# ---------------------------------------------------------------------------
# MILP Solver
# ---------------------------------------------------------------------------

class MILPSolver:
    """Branch-and-Bound MILP solver using the existing LP pipeline.

    The solver builds a Branch-and-Bound tree where each node solves an LP
    relaxation using the full OptiForge LP pipeline (Presolve → Standard Form
    → Revised Simplex → Postsolve). Nodes are processed in best-bound order
    (best-first search) with deterministic tie-breaking.
    """

    def __init__(
        self,
        tolerances: MILPTolerances | None = None,
        simplex_tolerances: SimplexTolerances | None = None,
        branching_strategy: BranchingStrategy | None = None,
        node_selection_strategy: NodeSelectionStrategy | None = None,
        search_state_callback: SearchStateCallback | None = None,
        search_state_monitor: SearchStateMonitor | None = None,
        primal_heuristic: PrimalHeuristic | None = None,
        strategy_manager: StrategyManager | None = None,
        adaptive_presolve: AdaptivePresolveStrategy | None = None,
        numerical_monitor: NumericalStabilityMonitor | None = None,
        numerical_thresholds: NumericalThresholds | None = None,
        stop_event: Optional[threading.Event] = None,
        warm_start_relaxations: bool = False,
        inherit_basis_factorization: bool = False,
    ) -> None:
        self.tolerances = tolerances or MILPTolerances()
        self.warm_start_relaxations = warm_start_relaxations or self.tolerances.warm_start_relaxations
        self.inherit_basis_factorization = inherit_basis_factorization or self.tolerances.inherit_basis_factorization
        self.warm_start_stats = WarmStartStats()
        self.simplex_tolerances = simplex_tolerances or SimplexTolerances(
            max_iterations=self.tolerances.max_iterations_per_node
        )
        if self.tolerances.use_adaptive_pivot_factorization and not self.simplex_tolerances.use_adaptive_pivot_factorization:
            self.simplex_tolerances.use_adaptive_pivot_factorization = True
            self.simplex_tolerances.adaptive_pfi_min_dimension = self.tolerances.adaptive_pfi_min_dimension
        if self.tolerances.use_fast_pivot_update and not self.simplex_tolerances.use_fast_pivot_update:
            self.simplex_tolerances.use_fast_pivot_update = True
        if self.tolerances.use_scipy_triangular_solve and not self.simplex_tolerances.use_scipy_triangular_solve:
            self.simplex_tolerances.use_scipy_triangular_solve = True
            self.simplex_tolerances.scipy_triangular_min_dimension = self.tolerances.scipy_triangular_min_dimension
            self.simplex_tolerances.triangular_solve_provider = self.tolerances.triangular_solve_provider
        if self.tolerances.use_scipy_lu_factor and not getattr(self.simplex_tolerances, "use_scipy_lu_factor", False):
            self.simplex_tolerances.use_scipy_lu_factor = True
            self.simplex_tolerances.scipy_lu_min_dimension = self.tolerances.scipy_lu_min_dimension
            self.simplex_tolerances.lu_factor_provider = self.tolerances.lu_factor_provider
        if getattr(self.tolerances, "enable_profiling", False) and not getattr(self.simplex_tolerances, "enable_profiling", False):
            self.simplex_tolerances.enable_profiling = True
        self.last_profile_stats = None
        self.numerical_monitor = numerical_monitor
        self.stop_event: Optional[threading.Event] = stop_event
        self.lp_solver = RevisedSimplexSolver(
            tolerances=self.simplex_tolerances,
            numerical_monitor=numerical_monitor,
            numerical_thresholds=numerical_thresholds,
            stop_event=stop_event,
        )

        # Strategy objects (use defaults if not provided).
        # NOTE: injected strategy objects are user-owned and intentionally
        # stateful — the solver never resets them between solves. If a
        # strategy keeps internal counters, those accumulate across solves
        # on the same MILPSolver instance. This is by design so that
        # adaptive strategies (Stage 7.2) can learn across solves.
        self._branching_strategy = branching_strategy or MostFractionalBranching()
        self._node_selection_strategy = node_selection_strategy or BestBoundNodeSelection()
        self._search_state_callback = search_state_callback

        # Search State Monitor (Stage 7.3)
        self._search_state_monitor = search_state_monitor if search_state_monitor is not None else SearchStateMonitor()

        # Primal Heuristic (Stage 7.6) - optional
        self._primal_heuristic = primal_heuristic

        # Dynamic Strategy Manager (Stage 7.7) - optional
        self._strategy_manager = strategy_manager

        # Adaptive Presolve (Stage 7.8) - optional
        self._adaptive_presolve = adaptive_presolve

        # Store the configured heuristic for switching reference
        self._configured_heuristic = primal_heuristic

        # Current context for adaptive strategies (Stage 7.7)
        self._current_fingerprint: ProblemFingerprint | None = None
        self._current_search_state: SearchState | None = None

        # Tree state
        self._nodes: dict[int, BnBNode] = {}
        self._node_counter = 0
        self._open_nodes: list[int] = []  # node IDs of open (unprocessed) nodes

        # Incumbent tracking
        self._incumbent_objective: float | None = None
        self._incumbent_solution: dict[str, float] | None = None

        # Objective sense (set during solve)
        self._objective_sense: ObjectiveSense | None = None

        # Timing (set during solve)
        self._solve_start_time: float | None = None

        # Statistics
        self._nodes_processed = 0
        self._nodes_pruned = 0
        self._nodes_branched = 0
        self._feasible_solutions_found = 0
        self._max_depth = 0

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def solve(self, model: Model) -> Solution:
        """Solve a MILP model using Branch-and-Bound.

        Parameters
        ----------
        model : Model
            The MILP model to solve. Must have at least one INTEGER or BINARY
            variable to trigger Branch-and-Bound; otherwise, the LP solver is
            used directly.

        Returns
        -------
        Solution
            Result container with status, variable values, objective value,
            iteration count, and MILP-specific metadata.
        """
        if self.stop_event is not None and self.stop_event.is_set():
            return self._build_solution(
                status=SolutionStatus.ITERATION_LIMIT,
                message="Solver cancelled (stop signal received).",
            )

        # Apply adaptive presolve if configured (Stage 7.8) - do this for ALL models
        presolve_result = None
        if self._adaptive_presolve is not None:
            fingerprint = self.fingerprint(model)
            decision = self._adaptive_presolve.decide(fingerprint)
            presolve_result = adaptive_presolve_model(model, decision, feasibility_tol=self.tolerances.bound_tolerance)
            
            if presolve_result.status == SolutionStatus.INFEASIBLE:
                return self._build_solution(
                    status=SolutionStatus.INFEASIBLE,
                    message="Presolve detected infeasibility",
                )
            if presolve_result.status == SolutionStatus.UNBOUNDED:
                return self._build_solution(
                    status=SolutionStatus.UNBOUNDED,
                    message="Presolve detected unboundedness",
                )
            # Use the reduced model for solving
            model = presolve_result.reduced_model
            if model is None:
                return self._build_solution(
                    status=SolutionStatus.INFEASIBLE,
                    message="Presolve produced empty model",
                )

        # Check if model has any integer/binary variables (after presolve)
        has_integer_vars = any(v.is_integer() for v in model.variables.values())

        # Handle fully reduced model (no variables remaining after presolve)
        if model.num_variables == 0:
            # All variables were fixed during presolve - construct solution directly
            if presolve_result is not None:
                # Reconstruct original variable values from fixed variables
                original_values = postsolve_solution({}, presolve_result)
                # The objective is just the objective_constant (no reduced objective)
                original_objective = presolve_result.objective_constant
                return self._build_solution(
                    status=SolutionStatus.OPTIMAL,
                    message="Optimal solution found (fully reduced by presolve).",
                    variable_values=original_values,
                    objective_value=original_objective,
                )
            else:
                # No presolve, but somehow no variables - this shouldn't happen
                return self._build_solution(
                    status=SolutionStatus.INFEASIBLE,
                    message="Model has no variables.",
                )

        if not has_integer_vars:
            # Pure LP - delegate to LP solver
            lp_solution = self.lp_solver.solve(model)
            # If we used adaptive presolve, reconstruct the original objective
            if presolve_result is not None and lp_solution.objective_value is not None:
                lp_solution.objective_value += presolve_result.objective_constant
            return lp_solution

        # Initialize tree with root node
        self._reset_state()

        # Store the model reference for fingerprinting in strategy decisions
        self._current_model = model

        # Store objective sense for use in B&B logic (after reset_state)
        if model.objective is not None:
            self._objective_sense = model.objective.sense
        else:
            self._objective_sense = ObjectiveSense.MINIMIZE  # default

        # Initialize SearchStateMonitor (Stage 7.3)
        objective_sense_str = "maximize" if self._objective_sense == ObjectiveSense.MAXIMIZE else "minimize"
        self._search_state_monitor.start_solve(objective_sense_str)

        # Initialize Dynamic Strategy Manager (Stage 7.7)
        if self._strategy_manager is not None:
            self._strategy_manager.start_solve()

        # Start timing for search-state observation (backward compatibility with callback)
        self._solve_start_time = time.monotonic()

        # Initialize tree with root node
        root_node = self._create_root_node(model)
        self._nodes[root_node.node_id] = root_node
        self._open_nodes = [root_node.node_id]

        # Branch-and-Bound main loop
        while self._open_nodes:
            # Cooperative cancellation check — safe boundary before processing a node
            if self.stop_event is not None and self.stop_event.is_set():
                self._search_state_monitor.end_solve()
                if self._strategy_manager is not None:
                    self._strategy_manager.end_solve()
                return self._build_solution(
                    status=SolutionStatus.ITERATION_LIMIT,
                    message="Solver cancelled (stop signal received).",
                )

            if self._nodes_processed >= self.tolerances.max_nodes:
                self._search_state_monitor.end_solve()
                if self._strategy_manager is not None:
                    self._strategy_manager.end_solve()
                return self._build_solution(
                    status=SolutionStatus.ITERATION_LIMIT,
                    message=f"Node limit ({self.tolerances.max_nodes}) reached.",
                )

            # Dynamic strategy selection (Stage 7.7)
            if self._strategy_manager is not None:
                fingerprint = self.fingerprint(model)
                search_state = self._build_search_state()
                # Store for use in _apply_strategy_decision
                self._current_fingerprint = fingerprint
                self._current_search_state = search_state
                decision = self._strategy_manager.decide(fingerprint, search_state)
                self._apply_strategy_decision(decision)

            # Select next node (best-bound first)
            node_id = self._select_node()
            node = self._nodes[node_id]
            self._open_nodes.remove(node_id)

            # Record node start for monitor
            self._search_state_monitor.record_node_start(node.depth)

            # Solve LP relaxation at this node
            self._process_node(model, node)

            # Record root bound after first node (root) is processed
            if self._nodes_processed == 1:
                self._search_state_monitor.record_root_bound(node.lp_objective)

            # Notify search state observers via monitor
            self._notify_search_state()

            # Check termination
            if self._should_terminate():
                break

        self._search_state_monitor.end_solve()
        if self._strategy_manager is not None:
            self._strategy_manager.end_solve()
        
        # If we used adaptive presolve, reconstruct the solution in original variable space
        if presolve_result is not None:
            return self._build_final_solution_with_presolve(presolve_result)
        
        return self._build_final_solution()

    # -----------------------------------------------------------------------
    # Dynamic Strategy Manager (Stage 7.7)
    # -----------------------------------------------------------------------

    def _apply_strategy_decision(self, decision: StrategyDecision) -> None:
        """Apply a strategy decision by updating the current strategies.

        This method is called at each node to potentially switch strategies
        based on the current search state and problem fingerprint.

        Adaptive strategies receive live context via update_context() to
        ensure they always operate with current fingerprint and search state.
        Strategy instances are reused when possible to preserve internal state.
        """
        # Use the stored current context (computed in the solve loop)
        fingerprint = self._current_fingerprint
        search_state = self._current_search_state

        # Apply branching strategy
        if decision.branching_strategy == BRANCHING_ADAPTIVE:
            from optiforge.milp.adaptive_branching import AdaptiveBranchingStrategy
            # Reuse existing adaptive strategy instance if available, otherwise create new
            if isinstance(self._branching_strategy, AdaptiveBranchingStrategy):
                self._branching_strategy.update_context(
                    fingerprint=fingerprint,
                    search_state=search_state
                )
            else:
                self._branching_strategy = AdaptiveBranchingStrategy(
                    fingerprint=fingerprint,
                    search_state=search_state
                )
        elif decision.branching_strategy == BRANCHING_MOST_FRACTIONAL:
            from optiforge.milp.solver import MostFractionalBranching
            self._branching_strategy = MostFractionalBranching()

        # Apply node selection strategy
        if decision.node_selection_strategy == NODE_SELECTION_ADAPTIVE:
            from optiforge.milp.adaptive_node_selection import AdaptiveNodeSelectionStrategy
            # Reuse existing adaptive strategy instance if available, otherwise create new
            if isinstance(self._node_selection_strategy, AdaptiveNodeSelectionStrategy):
                self._node_selection_strategy.update_context(
                    fingerprint=fingerprint,
                    search_state=search_state
                )
            else:
                self._node_selection_strategy = AdaptiveNodeSelectionStrategy(
                    fingerprint=fingerprint,
                    search_state=search_state
                )
        elif decision.node_selection_strategy == NODE_SELECTION_BEST_BOUND:
            from optiforge.milp.solver import BestBoundNodeSelection
            self._node_selection_strategy = BestBoundNodeSelection()

        # Apply heuristic strategy
        if decision.heuristic == HEURISTIC_CONFIGURED:
            # Use the configured heuristic if available
            if self._configured_heuristic is not None:
                self._primal_heuristic = self._configured_heuristic
            else:
                # No heuristic was configured - fall back to none
                # This is the documented behavior: "configured" is unavailable
                # when no heuristic was supplied at construction
                self._primal_heuristic = None
        elif decision.heuristic == HEURISTIC_NONE:
            self._primal_heuristic = None

    # -----------------------------------------------------------------------
    # Problem Fingerprinting (Stage 7.2)
    # -----------------------------------------------------------------------

    def fingerprint(self, model: Model) -> ProblemFingerprint:
        """Compute a deterministic, read-only structural fingerprint of a model.

        This is a descriptive-only observation mechanism. It never mutates the
        model, variables, constraints, objective, or any solver state, and it
        does not influence solve semantics. The fingerprint is intended for
        future adaptive strategies (e.g. branching/node-selection selection).

        Parameters
        ----------
        model : Model
            The MILP model to fingerprint.

        Returns
        -------
        ProblemFingerprint
            A frozen dataclass describing the model's structure.
        """
        return fingerprint_model(model)

    # -----------------------------------------------------------------------
    # Search State Monitor (Stage 7.3)
    # -----------------------------------------------------------------------

    def get_search_state_monitor(self) -> SearchStateMonitor:
        """Get the SearchStateMonitor for this solver.

        The monitor captures detailed search state during the Branch-and-Bound
        process, including history tracking, gap progression, and bound
        progression. It is purely observational and does not influence
        solver decisions.

        Returns
        -------
        SearchStateMonitor
            The search state monitor instance.
        """
        return self._search_state_monitor

    def get_warm_start_stats(self) -> WarmStartStats:
        """Get statistics for LP relaxation warm-starting."""
        return self.warm_start_stats

    # -----------------------------------------------------------------------
    # Node Processing & Warm-Start
    # -----------------------------------------------------------------------

    def _create_root_node(self, model: Model) -> BnBNode:
        """Create the root node of the B&B tree.

        The root's ``lp_bound`` is ``None`` until it is processed, because
        there is no parent relaxation to inherit a bound from.
        """
        self._node_counter += 1
        return BnBNode(
            node_id=self._node_counter,
            parent_id=None,
            depth=0,
        )

    def _solve_root_or_cold(
        self, model: Model
    ) -> tuple[Solution, WarmStartBasisState | None]:
        """Solve an LP relaxation cold and capture the Phase-II basis if available."""
        # Convert all variables to CONTINUOUS for LP relaxation standard form
        rel_model = Model(name=f"{model.name}_rel")
        for v in model.variables.values():
            rel_model.add_variable(Variable(
                name=v.name,
                lower_bound=v.lower_bound,
                upper_bound=v.upper_bound,
                variable_type=VariableType.CONTINUOUS,
            ))
        for c in model.constraints.values():
            rel_model.add_constraint(Constraint(
                name=c.name,
                coefficients=dict(c.coefficients),
                sense=c.sense,
                rhs=c.rhs,
            ))
        if model.objective is not None:
            rel_model.set_objective(Objective(
                coefficients=dict(model.objective.coefficients),
                sense=model.objective.sense,
            ))

        model_sf = to_standard_form(rel_model)
        tol = self.simplex_tolerances
        total_iterations = 0

        if model_sf.needs_phase1:
            phase1_res = self.lp_solver._simplex_loop(
                A=model_sf.A,
                b=model_sf.b,
                c=model_sf.phase1_c,
                basis_indices=list(model_sf.phase1_basis_indices),
                nonbasic_indices=list(model_sf.phase1_nonbasic_indices),
                sense=ObjectiveSense.MINIMIZE,
                tolerances=tol,
                initial_iterations=total_iterations,
                stop_event=self.stop_event,
            )
            total_iterations = phase1_res.iterations
            self.warm_start_stats.cold_pivots += phase1_res.iterations

            if phase1_res.status != SolutionStatus.OPTIMAL:
                return Solution(
                    status=phase1_res.status,
                    iterations=total_iterations,
                    message=phase1_res.message,
                ), None

            w_star = float(np.sum(phase1_res.x[model_sf.artificial_indices]))
            if w_star > tol.phase1_feasibility_tolerance:
                return Solution(
                    status=SolutionStatus.INFEASIBLE,
                    iterations=total_iterations,
                    message=f"Phase-I infeasible (w* = {w_star:.6e}).",
                ), None

            (
                A_phase2,
                b_phase2,
                c_phase2,
                basis_phase2,
                nonbasic_phase2,
                col_mapping,
            ) = self.lp_solver._cleanup_phase1_basis(
                model_sf=model_sf,
                final_basis=phase1_res.basis_indices,
                final_nonbasic=phase1_res.nonbasic_indices,
                tolerances=tol,
            )

            # Inherit Phase-I basis factorization if no artificial variables were basic
            initial_p2_fact = None
            if phase1_res.basis_factorization is not None:
                artificial_set = set(model_sf.artificial_indices)
                if not any(b in artificial_set for b in phase1_res.basis_indices):
                    initial_p2_fact = phase1_res.basis_factorization

            phase2_res = self.lp_solver._simplex_loop(
                A=A_phase2,
                b=b_phase2,
                c=c_phase2,
                basis_indices=basis_phase2,
                nonbasic_indices=nonbasic_phase2,
                sense=model_sf.objective_sense,
                tolerances=tol,
                initial_iterations=total_iterations,
                stop_event=self.stop_event,
                initial_basis_factorization=initial_p2_fact,
            )
            total_iterations = phase2_res.iterations
            self.warm_start_stats.cold_pivots += (phase2_res.iterations - phase1_res.iterations)

            if phase2_res.status != SolutionStatus.OPTIMAL:
                return Solution(
                    status=phase2_res.status,
                    iterations=total_iterations,
                    message=phase2_res.message,
                ), None

            # Phase-I safety check: ensure no artificial columns remain in basis (mapped back to Phase-I space)
            artificial_set = set(model_sf.artificial_indices)
            rev_col_mapping = {new_idx: old_idx for old_idx, new_idx in col_mapping.items()}
            if any(rev_col_mapping.get(col) in artificial_set for col in phase2_res.basis_indices):
                var_values = self.lp_solver._extract_original_variables(
                    model_sf=model_sf, phase2_x=phase2_res.x, col_mapping=col_mapping, tolerances=tol
                )
                orig_obj = phase2_res.objective_value + model_sf.objective_constant
                return Solution(
                    status=SolutionStatus.OPTIMAL,
                    variable_values=var_values,
                    objective_value=orig_obj,
                    iterations=total_iterations,
                ), None

            var_values = self.lp_solver._extract_original_variables(
                model_sf=model_sf, phase2_x=phase2_res.x, col_mapping=col_mapping, tolerances=tol
            )
            orig_obj = phase2_res.objective_value + model_sf.objective_constant
            sol = Solution(
                status=SolutionStatus.OPTIMAL,
                variable_values=var_values,
                objective_value=orig_obj,
                iterations=total_iterations,
            )

            orig_to_p2 = {
                vname: col_mapping[old_idx]
                for vname, old_idx in model_sf.original_variable_mapping.items()
                if old_idx in col_mapping
            }
            p2_shifts = {
                col_mapping[old_idx]: shift
                for old_idx, shift in model_sf.variable_shifts.items()
                if old_idx in col_mapping
            }

            base_fact1 = (
                phase2_res.basis_factorization
                if phase2_res.basis_factorization is not None
                else BasisFactorization(
                    np.array(A_phase2[:, phase2_res.basis_indices], copy=True),
                    pivot_tolerance=tol.pivot_tolerance,
                    use_scipy_triangular_solve=tol.use_scipy_triangular_solve,
                    scipy_triangular_min_dimension=tol.scipy_triangular_min_dimension,
                    triangular_solve_provider=tol.triangular_solve_provider,
                    use_scipy_lu_factor=tol.use_scipy_lu_factor,
                    scipy_lu_min_dimension=tol.scipy_lu_min_dimension,
                    lu_factor_provider=tol.lu_factor_provider,
                )
            )
            warm_state = WarmStartBasisState(
                A=np.array(A_phase2, copy=True),
                b=np.array(b_phase2, copy=True),
                c=np.array(c_phase2, copy=True),
                basis_indices=list(phase2_res.basis_indices),
                nonbasic_indices=list(phase2_res.nonbasic_indices),
                sense=model_sf.objective_sense,
                var_mapping=orig_to_p2,
                variable_shifts=p2_shifts,
                objective_constant=model_sf.objective_constant,
                basis_factorization=base_fact1,
            )
            return sol, warm_state

        else:
            phase2_res = self.lp_solver._simplex_loop(
                A=model_sf.A,
                b=model_sf.b,
                c=model_sf.c,
                basis_indices=list(model_sf.phase1_basis_indices),
                nonbasic_indices=list(model_sf.phase1_nonbasic_indices),
                sense=model_sf.objective_sense,
                tolerances=tol,
                initial_iterations=total_iterations,
                stop_event=self.stop_event,
            )
            total_iterations = phase2_res.iterations
            self.warm_start_stats.cold_pivots += phase2_res.iterations

            if phase2_res.status != SolutionStatus.OPTIMAL:
                return Solution(
                    status=phase2_res.status,
                    iterations=total_iterations,
                    message=phase2_res.message,
                ), None

            var_values = {
                vname: float(phase2_res.x[idx]) + model_sf.variable_shifts.get(idx, 0.0)
                for vname, idx in model_sf.original_variable_mapping.items()
            }
            orig_obj = phase2_res.objective_value + model_sf.objective_constant
            sol = Solution(
                status=SolutionStatus.OPTIMAL,
                variable_values=var_values,
                objective_value=orig_obj,
                iterations=total_iterations,
            )

            base_fact2 = (
                phase2_res.basis_factorization
                if phase2_res.basis_factorization is not None
                else BasisFactorization(
                    np.array(model_sf.A[:, phase2_res.basis_indices], copy=True),
                    pivot_tolerance=tol.pivot_tolerance,
                    use_scipy_triangular_solve=tol.use_scipy_triangular_solve,
                    scipy_triangular_min_dimension=tol.scipy_triangular_min_dimension,
                    triangular_solve_provider=tol.triangular_solve_provider,
                    use_scipy_lu_factor=tol.use_scipy_lu_factor,
                    scipy_lu_min_dimension=tol.scipy_lu_min_dimension,
                    lu_factor_provider=tol.lu_factor_provider,
                )
            )
            warm_state = WarmStartBasisState(
                A=np.array(model_sf.A, copy=True),
                b=np.array(model_sf.b, copy=True),
                c=np.array(model_sf.c, copy=True),
                basis_indices=list(phase2_res.basis_indices),
                nonbasic_indices=list(phase2_res.nonbasic_indices),
                sense=model_sf.objective_sense,
                var_mapping=dict(model_sf.original_variable_mapping),
                variable_shifts=dict(model_sf.variable_shifts),
                objective_constant=model_sf.objective_constant,
                basis_factorization=base_fact2,
            )
            return sol, warm_state

    def _try_warm_start_relaxation(
        self,
        node: BnBNode,
        parent_state: WarmStartBasisState,
    ) -> tuple[Solution | None, WarmStartBasisState | None]:
        """Attempt to solve the child LP relaxation via Dual Revised Simplex warm-start."""
        self.warm_start_stats.attempted += 1

        if node.branch_info is None:
            self.warm_start_stats.fallbacks += 1
            return None, None

        var_name, branch_type, bound_val = node.branch_info
        if var_name not in parent_state.var_mapping:
            self.warm_start_stats.fallbacks += 1
            return None, None

        col_idx = parent_state.var_mapping[var_name]
        shift = parent_state.variable_shifts.get(col_idx, 0.0)
        sf_bound_val = bound_val - shift

        m, n = parent_state.A.shape

        # Augmented matrix and vectors
        A_child = np.zeros((m + 1, n + 1), dtype=np.float64)
        A_child[:m, :n] = parent_state.A
        b_child = np.zeros(m + 1, dtype=np.float64)
        b_child[:m] = parent_state.b
        c_child = np.zeros(n + 1, dtype=np.float64)
        c_child[:n] = parent_state.c

        if branch_type == "upper":
            # y_k + s = bound_val
            A_child[m, col_idx] = 1.0
            A_child[m, n] = 1.0
            b_child[m] = sf_bound_val
        elif branch_type == "lower":
            # -y_k + s = -bound_val
            A_child[m, col_idx] = -1.0
            A_child[m, n] = 1.0
            b_child[m] = -sf_bound_val
        else:
            self.warm_start_stats.fallbacks += 1
            return None, None

        child_basis = list(parent_state.basis_indices) + [n]
        child_nonbasic = sorted(list(parent_state.nonbasic_indices))

        if len(child_basis) != m + 1 or len(set(child_basis)) != m + 1:
            self.warm_start_stats.fallbacks += 1
            return None, None

        dual_solver = DualRevisedSimplexSolver(
            tolerances=self.simplex_tolerances,
            use_incremental_basis=True,
            use_pivot_factorization=self.simplex_tolerances.use_pivot_factorization,
            use_adaptive_pivot_factorization=self.simplex_tolerances.use_adaptive_pivot_factorization,
            adaptive_pfi_min_dimension=self.simplex_tolerances.adaptive_pfi_min_dimension,
            use_scipy_triangular_solve=self.simplex_tolerances.use_scipy_triangular_solve,
            scipy_triangular_min_dimension=self.simplex_tolerances.scipy_triangular_min_dimension,
            stop_event=self.stop_event,
        )

        child_initial_factorization = None
        if self.inherit_basis_factorization and parent_state.basis_factorization is not None:
            self.warm_start_stats.inheritance_attempted += 1
            try:
                a_B = A_child[m, parent_state.basis_indices]
                B_child = A_child[:, child_basis]
                bordered_fact = BorderedBasisFactorization(
                    parent_factorization=parent_state.basis_factorization,
                    a_B=a_B,
                    B=B_child,
                    d=1.0,
                    pivot_tolerance=self.simplex_tolerances.pivot_tolerance,
                    residual_tolerance=self.simplex_tolerances.incremental_residual_tolerance,
                )
                child_initial_factorization = bordered_fact
                self.warm_start_stats.inheritance_accepted += 1
                self.warm_start_stats.dense_lu_calls_saved += 1
            except (SingularBasisError, ArithmeticError):
                self.warm_start_stats.inheritance_fallbacks += 1
                self.warm_start_stats.inherited_factorization_failures += 1
                child_initial_factorization = None

        try:
            res = dual_solver.solve_standard_form(
                A=A_child,
                b=b_child,
                c=c_child,
                basis=child_basis,
                nonbasic=child_nonbasic,
                sense=parent_state.sense,
                tolerances=self.simplex_tolerances,
                stop_event=self.stop_event,
                initial_basis_factorization=child_initial_factorization,
            )
        except (SingularBasisError, ArithmeticError):
            self.warm_start_stats.fallbacks += 1
            return None, None

        self.warm_start_stats.dual_pivots += res.iterations
        if isinstance(res.basis_factorization, PivotUpdateFactorization):
            self.warm_start_stats.pfi_updates += res.basis_factorization.total_updates
            self.warm_start_stats.pfi_refactorizations += res.basis_factorization.total_refactorizations
            self.warm_start_stats.pfi_fallbacks += res.basis_factorization.total_fallbacks

        if res.status in (SolutionStatus.OPTIMAL, SolutionStatus.INFEASIBLE):
            if self.simplex_tolerances.use_adaptive_pivot_factorization:
                if isinstance(res.basis_factorization, PivotUpdateFactorization):
                    self.warm_start_stats.adaptive_pfi_chosen += 1
                elif isinstance(res.basis_factorization, IncrementalBasisFactorization):
                    self.warm_start_stats.adaptive_inc_chosen += 1
                elif dual_solver.should_use_pivot_factorization(m + 1, self.simplex_tolerances):
                    self.warm_start_stats.adaptive_pfi_chosen += 1
                else:
                    self.warm_start_stats.adaptive_inc_chosen += 1

        if res.status == SolutionStatus.OPTIMAL:
            self.warm_start_stats.accepted += 1
            var_values = {
                vname: float(res.x[idx]) + parent_state.variable_shifts.get(idx, 0.0)
                for vname, idx in parent_state.var_mapping.items()
            }
            orig_obj = float(res.objective_value + parent_state.objective_constant)

            child_state = WarmStartBasisState(
                A=A_child,
                b=b_child,
                c=c_child,
                basis_indices=list(res.basis_indices),
                nonbasic_indices=list(res.nonbasic_indices),
                sense=parent_state.sense,
                var_mapping=dict(parent_state.var_mapping),
                variable_shifts=dict(parent_state.variable_shifts),
                objective_constant=parent_state.objective_constant,
                basis_factorization=res.basis_factorization,
            )
            sol = Solution(
                status=SolutionStatus.OPTIMAL,
                variable_values=var_values,
                objective_value=orig_obj,
                iterations=res.iterations,
                message="Optimal solution found via Dual Simplex warm-start.",
            )
            return sol, child_state

        elif res.status == SolutionStatus.INFEASIBLE:
            self.warm_start_stats.accepted += 1
            self.warm_start_stats.infeasible_pruned += 1
            sol = Solution(
                status=SolutionStatus.INFEASIBLE,
                iterations=res.iterations,
                message="LP relaxation proven infeasible via Dual Simplex.",
            )
            return sol, None

        else:
            # Fallback on ITERATION_LIMIT, NUMERICAL_ERROR, etc.
            self.warm_start_stats.fallbacks += 1
            return None, None

    def _process_node(self, model: Model, node: BnBNode) -> None:
        """Solve the LP relaxation at the given node.

        After solving, ``node.lp_bound`` is set to ``node.lp_objective``
        because a processed node's bound is its own LP objective.
        """
        self._nodes_processed += 1

        lp_solution: Solution | None = None

        # Warm-start path for child nodes (if enabled)
        if self.warm_start_relaxations and node.parent_id is not None and node.branch_info is not None:
            parent_node = self._nodes.get(node.parent_id)
            if parent_node is not None and parent_node.warm_start_state is not None:
                lp_solution, node.warm_start_state = self._try_warm_start_relaxation(
                    node, parent_node.warm_start_state
                )

        # Cold solve fallback or default path
        if lp_solution is None:
            node_model = self._build_node_model(model, node)
            if self.warm_start_relaxations:
                lp_solution, node.warm_start_state = self._solve_root_or_cold(node_model)
            else:
                lp_solution = self.lp_solver.solve(node_model)

        node.lp_status = lp_solution.status
        node.lp_objective = lp_solution.objective_value
        node.lp_solution = lp_solution.variable_values

        # For a processed node the best bound equals the LP objective.
        node.lp_bound = node.lp_objective

        # Classify node result
        if lp_solution.status == SolutionStatus.INFEASIBLE:
            # Prune infeasible node
            self._nodes_pruned += 1
            return

        if lp_solution.status == SolutionStatus.UNBOUNDED:
            # Unbounded LP relaxation - for MILP, this means the problem
            # is unbounded in the integer sense only if the direction
            # contains an integer ray. For now, we treat as unbounded.
            # In practice, MILP with unbounded LP relaxation is rare
            # and usually indicates a modeling issue.
            return

        if lp_solution.status != SolutionStatus.OPTIMAL:
            # Other statuses (ITERATION_LIMIT, etc.) - don't branch
            return

        # Check integer feasibility
        node.is_integral, node.fractional_var, node.fractional_value, max_integrality_violation = \
            self._check_integrality(node.lp_solution, model)

        # Record integrality violation in numerical monitor if available
        if self.numerical_monitor is not None and node.lp_solution is not None:
            # Record a snapshot with the integrality violation for this node
            self.numerical_monitor.record_snapshot(
                iteration=None,
                node_id=node.node_id,
                primal_residual=None,
                dual_residual=None,
                constraint_violation=None,
                bound_violation=None,
                integrality_violation=max_integrality_violation,
                pivot_magnitude=None,
                condition_estimate=None,
            )

        if node.is_integral:
            # Integer-feasible solution found - update incumbent
            self._update_incumbent(node)
            self._nodes_pruned += 1  # Prune this node (no need to branch)
            return

        # Fractional solution - check bound for pruning
        if self._should_prune_by_bound(node):
            self._nodes_pruned += 1
            return

        # Try primal heuristic to find a feasible integer solution
        if self._primal_heuristic is not None:
            search_state = self._build_search_state()
            fingerprint = self.fingerprint(model)
            if self._primal_heuristic.should_run(search_state, fingerprint):
                heuristic_result = self._primal_heuristic.run(
                    lp_solution=node.lp_solution,
                    lp_objective=node.lp_objective,
                    model=model,
                    search_state=search_state,
                    fingerprint=fingerprint,
                    tolerances=self.tolerances,
                )
                if heuristic_result.success:
                    # Validate and update incumbent if better
                    self._try_update_incumbent_from_heuristic(heuristic_result)

        # Branch on the fractional variable
        self._branch(node, model)

    def _build_node_model(self, model: Model, node: BnBNode) -> Model:
        """Build a Model for the node with additional bounds and constraints.

        For LP relaxation, all variables are treated as continuous.
        Integer/binary restrictions are enforced via branching.
        """
        node_model = Model(name=f"{model.name}_node{node.node_id}")

        # Copy variables with tightened bounds, converting to continuous for LP relaxation
        for var_name, var in model.variables.items():
            lb = var.lower_bound
            ub = var.upper_bound

            # Apply node-specific bounds
            if var_name in node.bounds:
                node_lb, node_ub = node.bounds[var_name]
                lb = max(lb, node_lb)
                ub = min(ub, node_ub)

            node_model.add_variable(Variable(
                name=var_name,
                lower_bound=lb,
                upper_bound=ub,
                variable_type=VariableType.CONTINUOUS,  # LP relaxation: all continuous
            ))

        # Copy original constraints
        for con in model.constraints.values():
            node_model.add_constraint(Constraint(
                name=con.name,
                coefficients=dict(con.coefficients),
                sense=con.sense,
                rhs=con.rhs,
            ))

        # Add branching constraints (skipping single-variable bounds already enforced by node.bounds)
        for con in node.constraints:
            if len(con.coefficients) == 1:
                vname, coeff = next(iter(con.coefficients.items()))
                if coeff == 1.0 and vname in node.bounds:
                    continue  # Already enforced via tightened variable bounds
            node_model.add_constraint(Constraint(
                name=con.name,
                coefficients=dict(con.coefficients),
                sense=con.sense,
                rhs=con.rhs,
            ))

        # Copy objective
        if model.objective is not None:
            node_model.set_objective(Objective(
                coefficients=dict(model.objective.coefficients),
                sense=model.objective.sense,
            ))

        return node_model

    def _check_integrality(
        self,
        solution: dict[str, float],
        model: Model,
    ) -> tuple[bool, str | None, float | None, float]:
        """Check if a solution is integer-feasible.

        Returns
        -------
        tuple
            (is_integral, fractional_var_name, fractional_value, max_integrality_violation)
        """
        tol = self.tolerances.integrality_tolerance
        max_violation = 0.0

        for var_name, var in model.variables.items():
            if not var.is_integer():
                continue

            value = solution.get(var_name, 0.0)
            nearest_int = round(value)
            deviation = abs(value - nearest_int)

            if deviation > max_violation:
                max_violation = deviation

            if deviation > tol:
                # Found a fractional integer variable
                return False, var_name, value, max_violation

        return True, None, None, max_violation

    def _node_effective_bound(self, node: BnBNode) -> float | None:
        """Return the best known relaxation bound for a node.

        For a processed node this is the node's own LP objective. For an
        unprocessed node it is the bound inherited from the parent's
        relaxation (``lp_bound``), which is valid because branching only
        tightens constraints. Returns ``None`` when no bound is known
        (e.g. the root node before it is processed).
        """
        if node.lp_objective is not None:
            return node.lp_objective
        return node.lp_bound

    def _should_prune_by_bound(self, node: BnBNode) -> bool:
        """Check if node should be pruned by bound.

        Uses the node's effective bound (own LP objective if processed,
        otherwise the inherited ``lp_bound``), so unprocessed nodes can be
        pruned by bound without being solved first.
        """
        if self._incumbent_objective is None:
            return False

        tol = self.tolerances.bound_tolerance
        node_obj = self._node_effective_bound(node)

        if node_obj is None:
            return False

        if self._is_maximize():
            # For MAX: node bound is upper bound; prune if <= incumbent
            return node_obj <= self._incumbent_objective + tol
        else:
            # For MIN: node bound is lower bound; prune if >= incumbent
            return node_obj >= self._incumbent_objective - tol

    def _is_maximize(self) -> bool:
        """Check if the root problem is a maximization."""
        if self._objective_sense is None:
            return True  # default to maximize if not set
        return self._objective_sense == ObjectiveSense.MAXIMIZE

    def _update_incumbent(self, node: BnBNode) -> None:
        """Update incumbent if node solution is better."""
        if node.lp_objective is None or node.lp_solution is None:
            return

        self._feasible_solutions_found += 1

        if self._incumbent_objective is None:
            self._incumbent_objective = node.lp_objective
            self._incumbent_solution = dict(node.lp_solution)
            return

        tol = self.tolerances.bound_tolerance

        if self._is_maximize():
            if node.lp_objective > self._incumbent_objective + tol:
                self._incumbent_objective = node.lp_objective
                self._incumbent_solution = dict(node.lp_solution)
        else:
            if node.lp_objective < self._incumbent_objective - tol:
                self._incumbent_objective = node.lp_objective
                self._incumbent_solution = dict(node.lp_solution)

    def _try_update_incumbent_from_heuristic(self, heuristic_result: HeuristicResult) -> None:
        """Try to update incumbent from a heuristic result.

        The heuristic result has already been validated by the heuristic itself,
        but we still need to check if it's better than the current incumbent.
        """
        if not heuristic_result.success:
            return

        if heuristic_result.variable_values is None or heuristic_result.objective_value is None:
            return

        self._feasible_solutions_found += 1

        if self._incumbent_objective is None:
            self._incumbent_objective = heuristic_result.objective_value
            self._incumbent_solution = dict(heuristic_result.variable_values)
            return

        tol = self.tolerances.bound_tolerance

        if self._is_maximize():
            if heuristic_result.objective_value > self._incumbent_objective + tol:
                self._incumbent_objective = heuristic_result.objective_value
                self._incumbent_solution = dict(heuristic_result.variable_values)
        else:
            if heuristic_result.objective_value < self._incumbent_objective - tol:
                self._incumbent_objective = heuristic_result.objective_value
                self._incumbent_solution = dict(heuristic_result.variable_values)

    def _branch(self, node: BnBNode, model: Model) -> None:
        """Create child nodes by branching on the fractional variable.

        The branching decision comes from the injected ``BranchingStrategy``.
        We validate that the returned variable exists in the model (M5),
        then update ``node.fractional_var`` / ``node.fractional_value`` to
        reflect the *actual* variable branched on (M2 — no dead metadata).
        """
        if node.fractional_var is None or node.fractional_value is None:
            return

        # Use the branching strategy to choose the branching variable
        decision = self._branching_strategy.choose_branching_variable(
            node=node,
            model=model,
            tolerances=self.tolerances,
        )
        if decision is None:
            return

        var_name = decision.variable_name
        value = decision.value

        # Validate the decision — the variable must exist in the model (M5).
        if var_name not in model.variables:
            raise ValueError(
                f"BranchingStrategy returned variable '{var_name}' which does "
                f"not exist in model '{model.name}'. "
                f"Available variables: {list(model.variables.keys())}"
            )

        var = model.variables[var_name]

        # Repurpose fractional_var / fractional_value to reflect the actual
        # branching decision (M2 — no dead metadata).
        node.fractional_var = var_name
        node.fractional_value = value

        floor_val = math.floor(value)
        ceil_val = math.ceil(value)

        # For binary variables, ensure bounds stay within [0, 1]
        if var.is_binary():
            floor_val = max(0.0, floor_val)
            ceil_val = min(1.0, ceil_val)

        # Left child: x <= floor(value)
        left_bounds = dict(node.bounds)
        left_bounds[var_name] = (left_bounds.get(var_name, (var.lower_bound, var.upper_bound))[0], floor_val)

        left_constraints = list(node.constraints)
        left_constraints.append(Constraint(
            name=f"branch_{node.node_id}_L_{var_name}",
            coefficients={var_name: 1.0},
            sense=ConstraintSense.LESS_EQUAL,
            rhs=floor_val,
        ))

        # Right child: x >= ceil(value)
        right_bounds = dict(node.bounds)
        right_bounds[var_name] = (ceil_val, right_bounds.get(var_name, (var.lower_bound, var.upper_bound))[1])

        right_constraints = list(node.constraints)
        right_constraints.append(Constraint(
            name=f"branch_{node.node_id}_R_{var_name}",
            coefficients={var_name: 1.0},
            sense=ConstraintSense.GREATER_EQUAL,
            rhs=ceil_val,
        ))

        # Create child nodes.
        # Children inherit the parent's LP objective as their bound:
        # branching only tightens constraints, so the parent relaxation
        # is a valid bound for both children.
        parent_bound = node.lp_objective

        self._node_counter += 1
        left_node = BnBNode(
            node_id=self._node_counter,
            parent_id=node.node_id,
            depth=node.depth + 1,
            bounds=left_bounds,
            constraints=left_constraints,
            lp_bound=parent_bound,
            branch_info=(var_name, "upper", float(floor_val)),
        )
        self._nodes[left_node.node_id] = left_node
        self._open_nodes.append(left_node.node_id)

        self._node_counter += 1
        right_node = BnBNode(
            node_id=self._node_counter,
            parent_id=node.node_id,
            depth=node.depth + 1,
            bounds=right_bounds,
            constraints=right_constraints,
            lp_bound=parent_bound,
            branch_info=(var_name, "lower", float(ceil_val)),
        )
        self._nodes[right_node.node_id] = right_node
        self._open_nodes.append(right_node.node_id)

        self._nodes_branched += 1
        self._max_depth = max(self._max_depth, left_node.depth, right_node.depth)

    # -----------------------------------------------------------------------
    # Node Selection
    # -----------------------------------------------------------------------

    def _select_node(self) -> int:
        """Select the next node to process using the node selection strategy."""
        return self._node_selection_strategy.select_node(
            open_nodes=self._open_nodes,
            nodes=self._nodes,
            is_maximize=self._is_maximize(),
        )

    def _should_terminate(self) -> bool:
        """Check if the algorithm should terminate.

        Termination requires that every open node is pruned by bound.  An
        unprocessed open node is prunable only when its *inherited* bound
        (``lp_bound``, from the parent's relaxation) proves it cannot beat
        the incumbent — it is never prunable merely because it is
        unprocessed.  This preserves correctness (no node that could
        contain a better solution is skipped) while still allowing early
        termination once the remaining open nodes are dominated by bound.
        """
        if not self._open_nodes:
            return True

        # If we have an incumbent, check if all open nodes are pruned by bound
        if self._incumbent_objective is not None:
            all_pruned = True
            for node_id in self._open_nodes:
                node = self._nodes[node_id]
                if not self._should_prune_by_bound(node):
                    all_pruned = False
                    break
            if all_pruned:
                return True

        return False

    # -----------------------------------------------------------------------
    # Search State Observation
    # -----------------------------------------------------------------------

    def _build_search_state(self) -> SearchState:
        """Build a SearchState snapshot of the current search.

        ``best_open_bound`` is computed from every open node's ``lp_bound``,
        which is valid even for unprocessed nodes (inherited from their
        parent's LP objective).  The ``gap`` and ``absolute_gap`` fields are
        meaningful once both an incumbent and a best open bound exist.
        """
        best_open_bound = None
        for node_id in self._open_nodes:
            node = self._nodes[node_id]
            if node.lp_bound is None:
                continue
            if best_open_bound is None:
                best_open_bound = node.lp_bound
            elif self._is_maximize():
                best_open_bound = max(best_open_bound, node.lp_bound)
            else:
                best_open_bound = min(best_open_bound, node.lp_bound)

        absolute_gap = None
        gap = None
        if self._incumbent_objective is not None and best_open_bound is not None:
            absolute_gap = abs(self._incumbent_objective - best_open_bound)
            if abs(self._incumbent_objective) > 1e-12:
                gap = absolute_gap / abs(self._incumbent_objective)
            elif absolute_gap > 1e-12:
                gap = float("inf")

        # Elapsed wall-clock time since the current solve started.
        elapsed_time = None
        if self._solve_start_time is not None:
            elapsed_time = time.monotonic() - self._solve_start_time

        return SearchState(
            nodes_processed=self._nodes_processed,
            nodes_remaining=len(self._open_nodes),
            incumbent_objective=self._incumbent_objective,
            best_open_bound=best_open_bound,
            gap=gap,
            absolute_gap=absolute_gap,
            feasible_solutions_found=self._feasible_solutions_found,
            nodes_pruned=self._nodes_pruned,
            max_depth=self._max_depth,
            elapsed_time=elapsed_time,
        )

    def _notify_search_state(self) -> None:
        """Invoke the search state callback and monitor, if registered."""
        # Build the base SearchState for backward compatibility with callbacks
        search_state = self._build_search_state()

        # Invoke the legacy callback if registered
        if self._search_state_callback is not None:
            self._search_state_callback(search_state)

        # Record in the SearchStateMonitor (Stage 7.3)
        self._search_state_monitor.record_node_complete(
            nodes_processed=search_state.nodes_processed,
            nodes_remaining=search_state.nodes_remaining,
            incumbent_objective=search_state.incumbent_objective,
            best_open_bound=search_state.best_open_bound,
            feasible_solutions_found=search_state.feasible_solutions_found,
            nodes_pruned=search_state.nodes_pruned,
            max_depth=search_state.max_depth,
        )

    # -----------------------------------------------------------------------
    # Solution Construction
    # -----------------------------------------------------------------------

    def _build_solution(
        self,
        status: SolutionStatus,
        message: str,
        variable_values: dict[str, float] | None = None,
        objective_value: float | None = None,
        incumbent_objective: float | None = None,
        best_bound: float | None = None,
        optimality_gap: float | None = None,
        absolute_gap: float | None = None,
        nodes_remaining: int | None = None,
        max_depth: int | None = None,
        feasible_solutions_found: int | None = None,
    ) -> Solution:
        """Build a Solution object."""
        # Include numerical monitoring snapshots if monitor is available
        numerical_snapshots = None
        if self.numerical_monitor is not None:
            numerical_snapshots = self.numerical_monitor.get_history()
        
        return Solution(
            status=status,
            variable_values=variable_values or {},
            objective_value=objective_value,
            message=message,
            iterations=self._nodes_processed,
            node_count=self._nodes_processed,
            numerical_snapshots=numerical_snapshots,
            # MILP-specific metrics
            incumbent_objective=incumbent_objective,
            best_bound=best_bound,
            optimality_gap=optimality_gap,
            absolute_gap=absolute_gap,
            nodes_remaining=nodes_remaining,
            max_depth=max_depth,
            feasible_solutions_found=feasible_solutions_found,
            profile_stats=self.last_profile_stats,
        )

    def _build_final_solution(self) -> Solution:
        """Build the final solution after B&B terminates."""
        # Build search state for gap/bound information
        search_state = self._build_search_state()
        
        if self._incumbent_solution is not None:
            # When search is complete, the best bound is the incumbent objective
            best_bound = search_state.best_open_bound
            if best_bound is None and self._incumbent_objective is not None:
                best_bound = self._incumbent_objective
            
            return self._build_solution(
                status=SolutionStatus.OPTIMAL,
                message="Optimal integer solution found.",
                variable_values=self._incumbent_solution,
                objective_value=self._incumbent_objective,
                incumbent_objective=self._incumbent_objective,
                best_bound=best_bound,
                optimality_gap=search_state.gap,
                absolute_gap=search_state.absolute_gap,
                nodes_remaining=search_state.nodes_remaining,
                max_depth=self._max_depth,
                feasible_solutions_found=self._feasible_solutions_found,
            )

        # No integer-feasible solution found
        # Check if root was infeasible
        root = self._nodes.get(1)
        if root and root.lp_status == SolutionStatus.INFEASIBLE:
            return self._build_solution(
                status=SolutionStatus.INFEASIBLE,
                message="MILP is infeasible.",
            )

        # LP relaxation was feasible but no integer solution found
        # (e.g., all nodes pruned or iteration limit)
        return self._build_solution(
            status=SolutionStatus.UNKNOWN,
            message="No integer-feasible solution found within node limit.",
            incumbent_objective=None,
            best_bound=search_state.best_open_bound,
            optimality_gap=search_state.gap,
            absolute_gap=search_state.absolute_gap,
            nodes_remaining=search_state.nodes_remaining,
            max_depth=self._max_depth,
            feasible_solutions_found=self._feasible_solutions_found,
        )

    def _build_final_solution_with_presolve(self, presolve_result: PresolveResult) -> Solution:
        """Build the final solution and reconstruct original variable values using presolve result."""
        # Build search state for gap/bound information
        search_state = self._build_search_state()
        
        # Get the objective constant from presolve
        objective_constant = presolve_result.objective_constant
        
        if self._incumbent_solution is not None:
            # Reconstruct original variable values from reduced solution
            original_values = postsolve_solution(self._incumbent_solution, presolve_result)
            # Reconstruct original objective: original_obj = reduced_obj + objective_constant
            original_objective = self._incumbent_objective + objective_constant if self._incumbent_objective is not None else None
            
            # When search is complete, the best bound is the incumbent objective
            best_bound = search_state.best_open_bound
            if best_bound is not None:
                best_bound = best_bound + objective_constant
            elif original_objective is not None:
                best_bound = original_objective
            
            return self._build_solution(
                status=SolutionStatus.OPTIMAL,
                message="Optimal integer solution found.",
                variable_values=original_values,
                objective_value=original_objective,
                incumbent_objective=original_objective,
                best_bound=best_bound,
                optimality_gap=search_state.gap,
                absolute_gap=search_state.absolute_gap,
                nodes_remaining=search_state.nodes_remaining,
                max_depth=self._max_depth,
                feasible_solutions_found=self._feasible_solutions_found,
            )

        # No integer-feasible solution found
        # Check if root was infeasible
        root = self._nodes.get(1)
        if root and root.lp_status == SolutionStatus.INFEASIBLE:
            return self._build_solution(
                status=SolutionStatus.INFEASIBLE,
                message="MILP is infeasible.",
            )

        # LP relaxation was feasible but no integer solution found
        # (e.g., all nodes pruned or iteration limit)
        best_bound = search_state.best_open_bound
        if best_bound is not None:
            best_bound = best_bound + objective_constant
        
        return self._build_solution(
            status=SolutionStatus.UNKNOWN,
            message="No integer-feasible solution found within node limit.",
            incumbent_objective=None,
            best_bound=best_bound,
            optimality_gap=search_state.gap,
            absolute_gap=search_state.absolute_gap,
            nodes_remaining=search_state.nodes_remaining,
            max_depth=self._max_depth,
            feasible_solutions_found=self._feasible_solutions_found,
        )

    def _reset_state(self) -> None:
        """Reset solver state for a new solve.

        Note: injected strategy objects are intentionally NOT reset — see
        the comment in ``__init__``.  User strategy state accumulates across
        solves by design.
        """
        self._nodes.clear()
        self._open_nodes.clear()
        self._node_counter = 0
        self._incumbent_objective = None
        self._incumbent_solution = None
        self._objective_sense = None
        self._solve_start_time = None
        self._nodes_processed = 0
        self._nodes_pruned = 0
        self._nodes_branched = 0
        self._feasible_solutions_found = 0
        self._max_depth = 0
        self.warm_start_stats = WarmStartStats()


# ---------------------------------------------------------------------------
# Node Selection Strategy
# ---------------------------------------------------------------------------

class NodeSelectionStrategy:
    """Abstract base class for node selection strategies.

    Subclasses must implement the `select_node` method.
    """

    def select_node(
        self,
        open_nodes: list[int],
        nodes: dict[int, BnBNode],
        is_maximize: bool,
    ) -> int:
        """Select the next node to process from the open nodes.

        Parameters
        ----------
        open_nodes : list[int]
            List of node IDs that are available for processing.
        nodes : dict[int, BnBNode]
            Dictionary mapping node IDs to BnBNode objects.
        is_maximize : bool
            True if the problem is a maximization, False for minimization.

        Returns
        -------
        int
            The node ID to process next.
        """
        raise NotImplementedError


class BestBoundNodeSelection(NodeSelectionStrategy):
    """Best-bound-first node selection strategy (default).

    For MAXIMIZE: selects node with highest bound (upper bound).
    For MINIMIZE: selects node with lowest bound (lower bound).
    Uses deterministic tie-breaking by node ID.

    A node's bound is its own LP objective when processed, otherwise the
    inherited ``lp_bound`` from the parent's relaxation (consistent with
    ``MILPSolver._node_effective_bound``).  Nodes with no known bound at
    all rank last.
    """

    def select_node(
        self,
        open_nodes: list[int],
        nodes: dict[int, BnBNode],
        is_maximize: bool,
    ) -> int:
        if not open_nodes:
            raise RuntimeError("No open nodes to select")

        def node_key(node_id: int) -> float:
            node = nodes[node_id]
            bound = node.lp_objective if node.lp_objective is not None else node.lp_bound
            if bound is None:
                # No known bound -> rank last regardless of sense.
                return float('inf')
            return -bound if is_maximize else bound

        # Deterministic tie-breaking: sort by node_id as secondary key
        open_nodes.sort(key=lambda nid: (node_key(nid), nid))
        return open_nodes[0]


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def solve_milp(
    model: Model,
    tolerances: MILPTolerances | None = None,
    simplex_tolerances: SimplexTolerances | None = None,
    branching_strategy: BranchingStrategy | None = None,
    node_selection_strategy: NodeSelectionStrategy | None = None,
    search_state_callback: SearchStateCallback | None = None,
    search_state_monitor: SearchStateMonitor | None = None,
    primal_heuristic: PrimalHeuristic | None = None,
    strategy_manager: StrategyManager | None = None,
    adaptive_presolve: AdaptivePresolveStrategy | None = None,
    numerical_monitor: NumericalStabilityMonitor | None = None,
    numerical_thresholds: NumericalThresholds | None = None,
    stop_event: Optional[threading.Event] = None,
    warm_start_relaxations: bool = False,
    inherit_basis_factorization: bool = False,
) -> Solution:
    """Solve a MILP model using Branch-and-Bound.

    Parameters
    ----------
    model : Model
        The MILP model to solve.
    tolerances : MILPTolerances | None
        MILP-specific tolerances.
    simplex_tolerances : SimplexTolerances | None
        LP solver tolerances for node relaxations.
    branching_strategy : BranchingStrategy | None
        Custom branching variable selection strategy.
    node_selection_strategy : NodeSelectionStrategy | None
        Custom node selection strategy.
    search_state_callback : SearchStateCallback | None
        Callback invoked with a SearchState snapshot after each node.
    search_state_monitor : SearchStateMonitor | None
        SearchStateMonitor instance for detailed search state tracking.
    primal_heuristic : PrimalHeuristic | None
        Optional primal heuristic for finding feasible integer solutions.
    strategy_manager : StrategyManager | None
        Optional dynamic strategy manager for switching between strategies.
    adaptive_presolve : AdaptivePresolveStrategy | None
        Optional adaptive presolve strategy for selecting presolve operations.
    numerical_monitor : NumericalStabilityMonitor | None
        Optional numerical stability monitor for tracking numerical behavior.
    numerical_thresholds : NumericalThresholds | None
        Optional custom thresholds for the numerical monitor.
    stop_event : threading.Event | None
        Optional stop event for cooperative cancellation. When set, the solver
        will terminate at the next safe boundary (node boundary for B&B, or
        iteration boundary for simplex) and return ITERATION_LIMIT status.
    warm_start_relaxations : bool
        Enable Dual Revised Simplex warm-starting for child LP relaxations (default False).
    inherit_basis_factorization : bool
        Enable Bordered / Schur-Complement basis factorization inheritance (Stage 8.10, default False).

    Returns
    -------
    Solution
        Result container with status, variable values, objective value,
        and iteration count.
    """
    solver = MILPSolver(
        tolerances=tolerances,
        simplex_tolerances=simplex_tolerances,
        branching_strategy=branching_strategy,
        node_selection_strategy=node_selection_strategy,
        search_state_callback=search_state_callback,
        search_state_monitor=search_state_monitor,
        primal_heuristic=primal_heuristic,
        strategy_manager=strategy_manager,
        adaptive_presolve=adaptive_presolve,
        numerical_monitor=numerical_monitor,
        numerical_thresholds=numerical_thresholds,
        stop_event=stop_event,
        warm_start_relaxations=warm_start_relaxations,
        inherit_basis_factorization=inherit_basis_factorization,
    )
    return solver.solve(model)