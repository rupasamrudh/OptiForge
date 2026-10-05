"""Adaptive Presolve for MILP Branch-and-Bound (Stage 7.8).


This module provides a lightweight, deterministic adaptive presolve component
that selects which EXISTING presolve operations should be enabled based on
the ProblemFingerprint. It does NOT implement new presolve mathematics.

The adaptive layer wraps the existing Stage 5A presolve pipeline and decides
which operations to apply based on structural characteristics of the problem.

Design guarantees:
- Deterministic: identical (fingerprint, config) always yields the same decision.
- Conservative: only enables/disables existing validated operations.
- Explainable: PresolveDecision exposes selected operations and reasons.
- Default behavior preserved: if no adaptive presolve is configured, the
  solver uses existing Stage 5A presolve exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from optiforge.milp.fingerprint import ProblemFingerprint
from optiforge.presolve.presolve import PresolveOperation


@dataclass(frozen=True)
class PresolveDecision:
    """Immutable decision produced by the AdaptivePresolveStrategy.

    Attributes
    ----------
    enabled_operations : frozenset[PresolveOperation]
        The set of presolve operations that should be applied.
    selected_operations : tuple[PresolveOperation, ...]
        The selected operations in the default Stage 5A execution order.
        This reflects which operations are enabled, ordered according to
        Stage 5A's fixed, mathematically safe execution order.
        Stage 5A owns the actual mathematical execution order; this field
        documents the selected subset in that canonical order.
    reason : str
        Human-readable explanation of why this decision was made.
    fingerprint_summary : dict[str, Any]
        Relevant fingerprint information that influenced the decision.
    """

    enabled_operations: frozenset[PresolveOperation]
    selected_operations: tuple[PresolveOperation, ...]
    reason: str = ""
    fingerprint_summary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that all enabled operations have a selected entry."""
        enabled_set = set(self.enabled_operations)
        selected_set = set(self.selected_operations)
        if enabled_set != selected_set:
            raise ValueError(
                f"enabled_operations {enabled_set} must match selected_operations {selected_set}"
            )


class AdaptivePresolveStrategy:
    """Deterministic adaptive presolve strategy using ProblemFingerprint.

    This strategy observes the ProblemFingerprint and selects which existing
    presolve operations should be applied. It is purely a *selector* — it
    never implements presolve logic itself.

    The strategy is designed to be:
    - Conservative: only uses well-understood, explainable features
    - Deterministic: identical inputs always produce identical decisions
    - Explainable: decision includes selected operations and reasons
    - Safe: falls back to all operations enabled when uncertain
    """

    # Default operation order (preserves existing Stage 5A safe order)
    DEFAULT_OPERATION_ORDER = (
        PresolveOperation.BOUND_VALIDATION,
        PresolveOperation.ZERO_ROW_REMOVAL,
        PresolveOperation.ZERO_COLUMN_ELIMINATION,
        PresolveOperation.SINGLETON_CONSTRAINTS,
        PresolveOperation.ACTIVITY_CHECKS,
        PresolveOperation.BOUND_PROPAGATION,
        PresolveOperation.FIXED_POINT_ITERATION,
    )

    def __init__(
        self,
        enable_bound_validation: bool = True,
        enable_zero_row_removal: bool = True,
        enable_zero_column_elimination: bool = True,
        enable_singleton_constraints: bool = True,
        enable_activity_checks: bool = True,
        enable_bound_propagation: bool = True,
        enable_fixed_point_iteration: bool = True,
        min_problem_size_for_adaptive: int = 10,
    ) -> None:
        """Initialize the adaptive presolve strategy.

        Parameters
        ----------
        enable_bound_validation : bool
            Whether bound validation can be disabled (default: always enabled).
        enable_zero_row_removal : bool
            Whether zero row removal can be disabled.
        enable_zero_column_elimination : bool
            Whether zero column elimination can be disabled.
        enable_singleton_constraints : bool
            Whether singleton constraint processing can be disabled.
        enable_activity_checks : bool
            Whether activity-based checks can be disabled.
        enable_bound_propagation : bool
            Whether bound propagation can be disabled.
        enable_fixed_point_iteration : bool
            Whether fixed-point iteration can be disabled (single pass only).
        min_problem_size_for_adaptive : int
            Minimum variable count before adaptive decisions apply.
            Below this threshold, all operations are enabled (avoids overhead).
        """
        self.enable_bound_validation = enable_bound_validation
        self.enable_zero_row_removal = enable_zero_row_removal
        self.enable_zero_column_elimination = enable_zero_column_elimination
        self.enable_singleton_constraints = enable_singleton_constraints
        self.enable_activity_checks = enable_activity_checks
        self.enable_bound_propagation = enable_bound_propagation
        self.enable_fixed_point_iteration = enable_fixed_point_iteration
        self.min_problem_size_for_adaptive = min_problem_size_for_adaptive

        # Validate configuration
        if self.min_problem_size_for_adaptive < 0:
            raise ValueError(
                f"min_problem_size_for_adaptive must be >= 0, got {self.min_problem_size_for_adaptive}"
            )

    def decide(self, fingerprint: ProblemFingerprint) -> PresolveDecision:
        """Make a deterministic presolve decision based on the problem fingerprint.

        Parameters
        ----------
        fingerprint : ProblemFingerprint
            The problem fingerprint.

        Returns
        -------
        PresolveDecision
            The selected presolve operations and their order.
        """
        # DEBUG
        print(f"DEBUG: decide() called with variable_count={fingerprint.variable_count}, constraint_count={fingerprint.constraint_count}")

        # 0. Handle edge cases first - BEFORE min_problem_size check
        # If no variables, nothing to do
        if fingerprint.variable_count == 0:
            reasons = ["no variables: no presolve operations needed"]
            enabled = set()
            config_filtered = self._filter_by_config(enabled)
            selected_operations = tuple(op for op in self.DEFAULT_OPERATION_ORDER if op in config_filtered)
            fingerprint_summary = {
                "variable_count": fingerprint.variable_count,
                "constraint_count": fingerprint.constraint_count,
                "constraint_density": fingerprint.constraint_density,
                "is_binary_heavy": fingerprint.is_binary_heavy,
                "is_equality_heavy": fingerprint.is_equality_heavy,
                "is_sparse": fingerprint.is_sparse,
                "is_dense": fingerprint.is_dense,
                "is_bound_heavy": fingerprint.is_bound_heavy,
                "integer_constraint_coverage": fingerprint.integer_constraint_coverage,
                "binary_constraint_coverage": fingerprint.binary_constraint_coverage,
            }
            reason_str = "; ".join(reasons) if reasons else "default: all operations enabled"
            # DEBUG
            print(f"DEBUG: Zero variables detected, returning early with enabled={config_filtered}")
            return PresolveDecision(
                enabled_operations=frozenset(config_filtered),
                selected_operations=selected_operations,
                reason=reason_str,
                fingerprint_summary=fingerprint_summary,
            )

        # If no constraints, only bound validation and zero-column elimination matter
        if fingerprint.constraint_count == 0:
            reasons = ["no constraints: only bound validation and zero-column elimination apply"]
            enabled = {
                PresolveOperation.BOUND_VALIDATION,
                PresolveOperation.ZERO_COLUMN_ELIMINATION,
            }
            config_filtered = self._filter_by_config(enabled)
            selected_operations = tuple(op for op in self.DEFAULT_OPERATION_ORDER if op in config_filtered)
            fingerprint_summary = {
                "variable_count": fingerprint.variable_count,
                "constraint_count": fingerprint.constraint_count,
                "constraint_density": fingerprint.constraint_density,
                "is_binary_heavy": fingerprint.is_binary_heavy,
                "is_equality_heavy": fingerprint.is_equality_heavy,
                "is_sparse": fingerprint.is_sparse,
                "is_dense": fingerprint.is_dense,
                "is_bound_heavy": fingerprint.is_bound_heavy,
                "integer_constraint_coverage": fingerprint.integer_constraint_coverage,
                "binary_constraint_coverage": fingerprint.binary_constraint_coverage,
            }
            reason_str = "; ".join(reasons) if reasons else "default: all operations enabled"
            return PresolveDecision(
                enabled_operations=frozenset(config_filtered),
                selected_operations=selected_operations,
                reason=reason_str,
                fingerprint_summary=fingerprint_summary,
            )

        # For very small problems, enable all operations (avoid adaptive overhead)
        if fingerprint.variable_count < self.min_problem_size_for_adaptive:
            return self._all_operations_decision(fingerprint, "problem too small for adaptive selection")

        # Start with MINIMAL safe operations - only those needed for correctness
        # This is the key change: start minimal, then add based on fingerprint
        enabled = {
            PresolveOperation.BOUND_VALIDATION,  # Always needed for correctness
            PresolveOperation.ZERO_ROW_REMOVAL,  # Always needed for correctness
        }
        reasons = ["minimal safe operations: bound validation, zero-row removal"]

        # Adaptive decisions based on fingerprint signals
        # Conditionally enable operations based on problem structure

        # 1. Sparse problems: bound propagation is very effective
        if fingerprint.is_sparse:
            enabled.add(PresolveOperation.BOUND_PROPAGATION)
            enabled.add(PresolveOperation.SINGLETON_CONSTRAINTS)
            enabled.add(PresolveOperation.ACTIVITY_CHECKS)
            reasons.append("sparse problem: bound propagation, singleton constraints, activity checks enabled")

        # 2. Dense problems: bound propagation may be less effective, but still useful
        elif fingerprint.is_dense:
            enabled.add(PresolveOperation.BOUND_PROPAGATION)
            reasons.append("dense problem: bound propagation enabled")

        # 3. Binary-heavy problems: singleton constraints and bound propagation are very effective
        if fingerprint.is_binary_heavy:
            enabled.add(PresolveOperation.SINGLETON_CONSTRAINTS)
            enabled.add(PresolveOperation.BOUND_PROPAGATION)
            reasons.append("binary-heavy: singleton constraints and bound propagation prioritized")

        # 4. Equality-heavy problems: singleton constraints (equalities) are very effective
        if fingerprint.is_equality_heavy:
            enabled.add(PresolveOperation.SINGLETON_CONSTRAINTS)
            reasons.append("equality-heavy: singleton constraints prioritized")

        # 5. Bound-heavy problems: bound validation and propagation are very effective
        if fingerprint.is_bound_heavy:
            enabled.add(PresolveOperation.BOUND_PROPAGATION)
            reasons.append("bound-heavy: bound propagation prioritized")

        # 6. Large problems: fixed-point iteration is important for convergence
        if fingerprint.variable_count > 50 or fingerprint.constraint_count > 50:
            enabled.add(PresolveOperation.FIXED_POINT_ITERATION)
            reasons.append("large problem: fixed-point iteration enabled for convergence")

        # 7. Very small constraint count: activity checks may be less useful
        if fingerprint.constraint_count <= 5:
            if self.enable_activity_checks:
                reasons.append("few constraints: activity checks may be less impactful")
                # Still keep them enabled by default as they're cheap

        # 8. High integer/binary coverage: bound propagation helps tighten integer variables
        if fingerprint.integer_constraint_coverage > 0.3:
            enabled.add(PresolveOperation.BOUND_PROPAGATION)
            reasons.append("high integer coverage: bound propagation prioritized")

        # 9. High binary constraint coverage: singleton constraints on binary vars are powerful
        if fingerprint.binary_constraint_coverage > 0.3:
            enabled.add(PresolveOperation.SINGLETON_CONSTRAINTS)
            reasons.append("high binary coverage: singleton constraints prioritized")

        # 10. Zero-column elimination: enable if there might be unconstrained variables
        # Heuristic: if density is low, there might be zero columns
        if fingerprint.constraint_density < 0.5:
            enabled.add(PresolveOperation.ZERO_COLUMN_ELIMINATION)
            reasons.append("low density: zero-column elimination enabled")

        # 11. Activity checks: enable for problems with many constraints
        if fingerprint.constraint_count > 10:
            enabled.add(PresolveOperation.ACTIVITY_CHECKS)
            reasons.append("many constraints: activity checks enabled")

        # Filter by configuration flags
        config_filtered = self._filter_by_config(enabled)
        if config_filtered != enabled:
            reasons.append("some operations disabled by configuration")

        # Build selected operations (preserve default order, only include enabled)
        selected_operations = tuple(op for op in self.DEFAULT_OPERATION_ORDER if op in config_filtered)

        # Build fingerprint summary for observability
        fingerprint_summary = {
            "variable_count": fingerprint.variable_count,
            "constraint_count": fingerprint.constraint_count,
            "constraint_density": fingerprint.constraint_density,
            "is_binary_heavy": fingerprint.is_binary_heavy,
            "is_equality_heavy": fingerprint.is_equality_heavy,
            "is_sparse": fingerprint.is_sparse,
            "is_dense": fingerprint.is_dense,
            "is_bound_heavy": fingerprint.is_bound_heavy,
            "integer_constraint_coverage": fingerprint.integer_constraint_coverage,
            "binary_constraint_coverage": fingerprint.binary_constraint_coverage,
        }

        reason_str = "; ".join(reasons) if reasons else "default: all operations enabled"

        return PresolveDecision(
            enabled_operations=frozenset(config_filtered),
            selected_operations=selected_operations,
            reason=reason_str,
            fingerprint_summary=fingerprint_summary,
        )

    def _filter_by_config(self, enabled: set[PresolveOperation]) -> set[PresolveOperation]:
        """Filter enabled operations by configuration flags."""
        filtered = set(enabled)
        if not self.enable_bound_validation:
            filtered.discard(PresolveOperation.BOUND_VALIDATION)
        if not self.enable_zero_row_removal:
            filtered.discard(PresolveOperation.ZERO_ROW_REMOVAL)
        if not self.enable_zero_column_elimination:
            filtered.discard(PresolveOperation.ZERO_COLUMN_ELIMINATION)
        if not self.enable_singleton_constraints:
            filtered.discard(PresolveOperation.SINGLETON_CONSTRAINTS)
        if not self.enable_activity_checks:
            filtered.discard(PresolveOperation.ACTIVITY_CHECKS)
        if not self.enable_bound_propagation:
            filtered.discard(PresolveOperation.BOUND_PROPAGATION)
        if not self.enable_fixed_point_iteration:
            filtered.discard(PresolveOperation.FIXED_POINT_ITERATION)
        return filtered

    def _all_operations_decision(self, fingerprint: ProblemFingerprint, reason: str) -> PresolveDecision:
        """Return a decision with all operations enabled."""
        filtered = self._filter_by_config(set(self.DEFAULT_OPERATION_ORDER))
        selected_operations = tuple(op for op in self.DEFAULT_OPERATION_ORDER if op in filtered)

        fingerprint_summary = {
            "variable_count": fingerprint.variable_count,
            "constraint_count": fingerprint.constraint_count,
            "constraint_density": fingerprint.constraint_density,
        }

        return PresolveDecision(
            enabled_operations=frozenset(filtered),
            selected_operations=selected_operations,
            reason=reason,
            fingerprint_summary=fingerprint_summary,
        )