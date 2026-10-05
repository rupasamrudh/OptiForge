"""Adaptive Branching Strategy for MILP Branch-and-Bound.

This module implements a conservative, deterministic, explainable adaptive
branching strategy that uses ProblemFingerprint and SearchState information
to make branching decisions among valid fractional integer/binary variables.

The strategy is designed to be:
- Conservative: only uses well-understood, explainable features
- Deterministic: identical inputs always produce identical decisions
- Explainable: scoring formula is explicit and inspectable
- Safe: falls back to MostFractionalBranching when adaptive logic cannot decide
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from optiforge.milp.fingerprint import ProblemFingerprint
from optiforge.milp.search_state import SearchState
from optiforge.milp.solver import (
    BranchingDecision,
    BranchingStrategy,
    BnBNode,
    MILPTolerances,
    Model,
    MostFractionalBranching,
)


@dataclass(frozen=True)
class CandidateFeatures:
    """Features computed for a branching candidate variable.

    These features are used to compute the adaptive branching score.
    All features are normalized to [0, 1] where possible for fair comparison.
    """

    variable_name: str
    """Name of the candidate variable."""

    fractional_value: float
    """Current LP relaxation value of the variable."""

    fractionality: float
    """Distance from nearest integer, normalized to [0, 0.5].
    0.5 means exactly halfway between integers (maximally fractional)."""

    is_binary: bool
    """True if the variable is binary (0/1)."""

    constraint_participation: float
    """Fraction of constraints that reference this variable with non-zero coefficient.
    Normalized to [0, 1] by dividing by total constraint count."""

    binary_constraint_coverage: float
    """From ProblemFingerprint: fraction of constraints referencing binary variables.
    Indicates how widely binary variables are coupled across constraints."""

    integer_constraint_coverage: float
    """From ProblemFingerprint: fraction of constraints referencing integer/binary variables.
    Indicates how widely integrality is coupled across constraints."""

    constraint_density: float
    """From ProblemFingerprint: structural non-zero density of constraint matrix."""

    is_binary_heavy: bool
    """From ProblemFingerprint: True if binary variables >= 50% of all variables."""

    is_equality_heavy: bool
    """From ProblemFingerprint: True if equality constraints >= 50% of all constraints."""

    is_sparse: bool
    """From ProblemFingerprint: True if constraint density < 0.1."""

    is_dense: bool
    """From ProblemFingerprint: True if constraint density > 0.5."""

    current_depth: int
    """Current search tree depth of the node being processed."""

    max_depth: int
    """Maximum depth reached in the B&B tree so far."""

    nodes_processed: int
    """Total number of nodes processed so far."""

    nodes_remaining: int
    """Number of open (unprocessed) nodes in the queue."""

    gap: float | None
    """Relative gap between incumbent and best open bound, if available."""

    absolute_gap: float | None
    """Absolute gap between incumbent and best open bound, if available."""

    incumbent_available: bool
    """True if an incumbent solution has been found."""

    objective_sense: str | None
    """"maximize" or "minimize" indicating optimization direction."""

    def to_dict(self) -> dict[str, Any]:
        """Convert features to dictionary for inspection/debugging."""
        return {
            "variable_name": self.variable_name,
            "fractional_value": self.fractional_value,
            "fractionality": self.fractionality,
            "is_binary": self.is_binary,
            "constraint_participation": self.constraint_participation,
            "binary_constraint_coverage": self.binary_constraint_coverage,
            "integer_constraint_coverage": self.integer_constraint_coverage,
            "constraint_density": self.constraint_density,
            "is_binary_heavy": self.is_binary_heavy,
            "is_equality_heavy": self.is_equality_heavy,
            "is_sparse": self.is_sparse,
            "is_dense": self.is_dense,
            "current_depth": self.current_depth,
            "max_depth": self.max_depth,
            "nodes_processed": self.nodes_processed,
            "nodes_remaining": self.nodes_remaining,
            "gap": self.gap,
            "absolute_gap": self.absolute_gap,
            "incumbent_available": self.incumbent_available,
            "objective_sense": self.objective_sense,
        }


@dataclass(frozen=True)
class BranchingScore:
    """Result of scoring a branching candidate.

    Contains the computed score and the feature values used for transparency.
    """

    variable_name: str
    """Name of the candidate variable."""

    score: float
    """Final computed score (higher is better)."""

    features: CandidateFeatures
    """Features used to compute the score."""

    component_scores: dict[str, float]
    """Breakdown of score components for transparency."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for inspection/debugging."""
        return {
            "variable_name": self.variable_name,
            "score": self.score,
            "features": self.features.to_dict(),
            "component_scores": self.component_scores,
        }


class AdaptiveBranchingStrategy(BranchingStrategy):
    """Adaptive branching strategy using ProblemFingerprint and SearchState.

    This strategy evaluates fractional integer/binary candidates using a
    deterministic scoring formula that combines:
    - Fractionality (distance from nearest integer)
    - Binary vs general integer preference
    - Constraint participation
    - Problem structure (from fingerprint)
    - Search state (gap, depth, incumbent availability)

    The strategy falls back to MostFractionalBranching when:
    - No valid candidates exist
    - Fingerprint or search state is unavailable
    - Scoring produces invalid results
    """

    def __init__(
        self,
        fingerprint: ProblemFingerprint | None = None,
        search_state: SearchState | None = None,
        fallback_strategy: BranchingStrategy | None = None,
    ) -> None:
        """Initialize the adaptive branching strategy.

        Parameters
        ----------
        fingerprint : ProblemFingerprint | None
            Problem fingerprint for structural features. If None, structural
            features will not be used.
        search_state : SearchState | None
            Current search state for dynamic features. If None, dynamic
            features will not be used.
        fallback_strategy : BranchingStrategy | None
            Fallback strategy when adaptive logic cannot decide.
            Defaults to MostFractionalBranching.
        """
        self._fingerprint = fingerprint
        self._search_state = search_state
        self._fallback = fallback_strategy or MostFractionalBranching()

    def update_context(
        self,
        fingerprint: ProblemFingerprint | None = None,
        search_state: SearchState | None = None,
    ) -> None:
        """Update the context for subsequent branching decisions.

        This allows the strategy to adapt as the search progresses.

        Parameters
        ----------
        fingerprint : ProblemFingerprint | None
            Updated problem fingerprint (typically unchanged during solve).
        search_state : SearchState | None
            Updated search state (changes after each node).
        """
        if fingerprint is not None:
            self._fingerprint = fingerprint
        if search_state is not None:
            self._search_state = search_state

    def _extract_candidates(
        self,
        node: BnBNode,
        model: Model,
        tolerances: MILPTolerances,
    ) -> list[tuple[str, float, bool]]:
        """Extract valid fractional integer/binary candidates from the node.

        Returns
        -------
        list[tuple[str, float, bool]]
            List of (variable_name, fractional_value, is_binary) for each
            valid fractional integer/binary candidate.
        """
        tol = tolerances.integrality_tolerance
        candidates = []

        for var_name, var in model.variables.items():
            if not var.is_integer():
                continue

            value = node.lp_solution.get(var_name, 0.0)
            nearest_int = round(value)
            deviation = abs(value - nearest_int)

            if deviation > tol:
                candidates.append((var_name, value, var.is_binary()))

        return candidates

    def _compute_constraint_participation(
        self,
        var_name: str,
        model: Model,
    ) -> float:
        """Compute the fraction of constraints that reference this variable.

        Returns
        -------
        float
            Fraction of constraints referencing this variable with non-zero
            coefficient. Returns 0.0 if no constraints exist.
        """
        constraint_count = len(model.constraints)
        if constraint_count == 0:
            return 0.0

        participation = 0
        for con in model.constraints.values():
            coeff = con.coefficients.get(var_name, 0.0)
            if abs(coeff) > 0:
                participation += 1

        return participation / constraint_count

    def _compute_features(
        self,
        var_name: str,
        fractional_value: float,
        is_binary: bool,
        model: Model,
        node: BnBNode,
    ) -> CandidateFeatures:
        """Compute all features for a candidate variable."""
        # Fractionality: distance from nearest integer, normalized to [0, 0.5]
        nearest_int = round(fractional_value)
        fractionality = abs(fractional_value - nearest_int)
        # Normalize to [0, 1] where 1.0 = maximally fractional (0.5)
        fractionality_normalized = min(fractionality / 0.5, 1.0)

        # Constraint participation
        constraint_participation = self._compute_constraint_participation(var_name, model)

        # Fingerprint features (with defaults if unavailable)
        fp = self._fingerprint
        binary_constraint_coverage = fp.binary_constraint_coverage if fp else 0.0
        integer_constraint_coverage = fp.integer_constraint_coverage if fp else 0.0
        constraint_density = fp.constraint_density if fp else 0.0
        is_binary_heavy = fp.is_binary_heavy if fp else False
        is_equality_heavy = fp.is_equality_heavy if fp else False
        is_sparse = fp.is_sparse if fp else False
        is_dense = fp.is_dense if fp else False

        # Search state features (with defaults if unavailable)
        ss = self._search_state
        current_depth = node.depth
        max_depth = ss.max_depth if ss else 0
        nodes_processed = ss.nodes_processed if ss else 0
        nodes_remaining = ss.nodes_remaining if ss else 0
        gap = ss.gap if ss else None
        absolute_gap = ss.absolute_gap if ss else None
        incumbent_available = ss.incumbent_objective is not None if ss else False
        # SearchState doesn't have objective_sense; use None as default
        objective_sense = None

        return CandidateFeatures(
            variable_name=var_name,
            fractional_value=fractional_value,
            fractionality=fractionality_normalized,
            is_binary=is_binary,
            constraint_participation=constraint_participation,
            binary_constraint_coverage=binary_constraint_coverage,
            integer_constraint_coverage=integer_constraint_coverage,
            constraint_density=constraint_density,
            is_binary_heavy=is_binary_heavy,
            is_equality_heavy=is_equality_heavy,
            is_sparse=is_sparse,
            is_dense=is_dense,
            current_depth=current_depth,
            max_depth=max_depth,
            nodes_processed=nodes_processed,
            nodes_remaining=nodes_remaining,
            gap=gap,
            absolute_gap=absolute_gap,
            incumbent_available=incumbent_available,
            objective_sense=objective_sense,
        )

    def _compute_score(self, features: CandidateFeatures) -> tuple[float, dict[str, float]]:
        """Compute the adaptive branching score for a candidate.

        The score combines multiple components:
        1. Fractionality (weight: 1.0) - prefer more fractional variables
        2. Binary preference (weight: 0.5) - slight preference for binary vars
        3. Constraint participation (weight: 0.3) - prefer vars in more constraints
        3. Binary constraint coverage (weight: 0.2) - prefer when binary vars are widespread
        4. Integer constraint coverage (weight: 0.2) - prefer when integrality is widespread
        5. Depth factor (weight: 0.1) - slight preference for deeper nodes
        6. Gap factor (weight: 0.1) - prefer when gap is large (more exploration needed)

        All components are normalized to [0, 1] where possible.

        Returns
        -------
        tuple[float, dict[str, float]]
            (total_score, component_scores_dict)
        """
        components = {}

        # 1. Fractionality (primary signal) - weight 1.0
        # Already normalized to [0, 1] where 1.0 = maximally fractional
        components["fractionality"] = features.fractionality

        # 2. Binary preference - weight 0.5
        # Binary variables are often more impactful to branch on
        components["binary_preference"] = 1.0 if features.is_binary else 0.0

        # 3. Constraint participation - weight 0.3
        # Variables in more constraints have more impact
        components["constraint_participation"] = features.constraint_participation

        # 4. Binary constraint coverage - weight 0.2
        # When binary vars are widespread, branching on them is more impactful
        components["binary_coverage"] = features.binary_constraint_coverage

        # 5. Integer constraint coverage - weight 0.2
        # When integrality is widespread, branching is more impactful
        components["integer_coverage"] = features.integer_constraint_coverage

        # 6. Depth factor - weight 0.1
        # Deeper nodes: slight preference for variables that might resolve deep conflicts
        if features.max_depth > 0:
            components["depth_factor"] = min(features.current_depth / features.max_depth, 1.0)
        else:
            components["depth_factor"] = 0.0

        # 7. Gap factor - weight 0.1
        # Large gap: more exploration needed, prefer variables with high fractionality
        if features.gap is not None:
            # Normalize gap: large gap -> higher score (more exploration)
            # Cap at 1.0 for very large gaps
            components["gap_factor"] = min(features.gap * 2.0, 1.0)
        else:
            components["gap_factor"] = 0.5  # neutral when no gap info

        # 7b. Incumbent availability - weight 0.05
        # When incumbent exists, we have a bound to work with
        components["incumbent_factor"] = 1.0 if features.incumbent_available else 0.0

        # Weights for each component
        weights = {
            "fractionality": 1.0,
            "binary_preference": 0.5,
            "constraint_participation": 0.3,
            "binary_coverage": 0.2,
            "integer_coverage": 0.2,
            "depth_factor": 0.1,
            "gap_factor": 0.1,
            "incumbent_factor": 0.05,
        }

        # Compute weighted score
        total_score = sum(
            components[name] * weight
            for name, weight in weights.items()
        )

        return total_score, components

    def choose_branching_variable(
        self,
        node: BnBNode,
        model: Model,
        tolerances: MILPTolerances,
    ) -> BranchingDecision | None:
        """Select a fractional integer variable to branch on using adaptive scoring.

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
        # Extract valid candidates
        candidates = self._extract_candidates(node, model, tolerances)

        if not candidates:
            # No valid candidates - fallback
            return self._fallback.choose_branching_variable(node, model, tolerances)

        # If only one candidate, return it directly
        if len(candidates) == 1:
            var_name, value, _ = candidates[0]
            return BranchingDecision(variable_name=var_name, value=value)

        # Score all candidates
        scored_candidates = []
        for var_name, value, is_binary in candidates:
            try:
                features = self._compute_features(var_name, value, is_binary, model, node)
                score, components = self._compute_score(features)
                scored_candidates.append((var_name, value, score, components))
            except Exception:
                # Any scoring error -> fallback
                return self._fallback.choose_branching_variable(node, model, tolerances)

        if not scored_candidates:
            return self._fallback.choose_branching_variable(node, model, tolerances)

        # Select candidate with highest score
        # Deterministic tie-breaking: sort by (-score, variable_name)
        scored_candidates.sort(key=lambda x: (-x[2], x[0]))

        best_var, best_value, best_score, best_components = scored_candidates[0]

        return BranchingDecision(variable_name=best_var, value=best_value)

    def get_fallback_strategy(self) -> BranchingStrategy:
        """Get the fallback strategy."""
        return self._fallback