"""Centralized numerical tolerance and policy abstraction for OptiForge.

Stage 8.48: Numerical Robustness, Basis Quality & Solver Diagnostics.

This module provides a unified policy defining:
- feasibility tolerance (primal feasibility, constraint violations, bound violations)
- optimality/reduced-cost tolerance (dual feasibility, candidate pricing threshold)
- pivot tolerance (ratio-test positive pivot element threshold)
- zero tolerance (floating-point zero filtering and sparsity detection)
- basis singularity tolerance (minimum diagonal pivot magnitude in LU decomposition)
- complementarity tolerance (maximum allowed product |x_j * r_j|)
- condition limit (maximum allowable condition number for safe basis inversion)
- minimum safe pivot (threshold above singularity tolerance for numerically safe bases)
- maximum safe residual (linear system residual threshold for safe warm-starting)
- verification mode ('none', 'basic', 'strict')
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class VerificationMode(Enum):
    """Execution mode for post-solve verification and diagnostics."""

    NONE = "none"
    """No additional post-solve verification or diagnostics (zero overhead)."""

    BASIC = "basic"
    """Cheap primal feasibility, bound compliance, and objective consistency checks."""

    STRICT = "strict"
    """Full KKT verification (primal, dual, complementarity) plus basis quality diagnostics."""


@dataclass(frozen=True)
class NumericalPolicy:
    """Centralized, immutable numerical tolerance policy for OptiForge.

    Parameters & Decisions
    ----------------------
    feasibility_tolerance : float
        Controls primal feasibility checks:
        - Basic variable non-negativity: x_B >= -feasibility_tolerance.
        - Constraint satisfaction: ||Ax - b||_inf <= feasibility_tolerance.
        - Original variable bound satisfaction: x_j >= lb - tol, x_j <= ub + tol.
        Default: 1e-7.

    optimality_tolerance : float
        Controls dual feasibility checks and optimality termination:
        - For MINIMIZE: reduced costs r_j >= -optimality_tolerance for all nonbasic j.
        - For MAXIMIZE: reduced costs r_j <= optimality_tolerance for all nonbasic j.
        - Pricing candidate selection in Revised Simplex.
        Default: 1e-7.

    pivot_tolerance : float
        Controls ratio test pivot selection:
        - Primal ratio test: direction vector element d_i > pivot_tolerance.
        - Dual ratio test: tableau row element alpha_j < -pivot_tolerance.
        Default: 1e-7.

    zero_tolerance : float
        Controls floating-point zero classification:
        - Values with |val| < zero_tolerance are treated as mathematically zero.
        - Sparsity pattern filtering and degeneracy tie-breaking.
        Default: 1e-12.

    basis_singularity_tolerance : float
        Controls LU factorization singularity detection:
        - If min_k |U[k, k]| < basis_singularity_tolerance, matrix is singular.
        Default: 1e-12.

    complementarity_tolerance : float
        Controls complementary slackness validation:
        - For standard form: |x_j * r_j| <= complementarity_tolerance for all j.
        Default: 1e-7.

    condition_limit : float
        Maximum allowable condition number kappa(B) = ||B|| * ||B^-1|| before
        declaring a basis numerically unsafe for warm-start re-optimization.
        Default: 1e12.

    min_safe_pivot : float
        Minimum absolute pivot magnitude for a candidate basis to be considered
        numerically safe for warm-start re-optimization without refactorization risk.
        Default: 1e-8.

    max_safe_residual : float
        Maximum allowable linear solve residual ||B x_B - b||_inf or ||B^T y - c_B||_inf
        for a basis to be accepted into a warm-start gate.
        Default: 1e-4.

    verification_mode : str
        Verification level: 'none', 'basic', or 'strict'.
        Default: 'none'.
    """

    feasibility_tolerance: float = 1e-7
    optimality_tolerance: float = 1e-7
    pivot_tolerance: float = 1e-7
    zero_tolerance: float = 1e-12
    basis_singularity_tolerance: float = 1e-12
    complementarity_tolerance: float = 1e-7
    condition_limit: float = 1e12
    min_safe_pivot: float = 1e-8
    max_safe_residual: float = 1e-4
    verification_mode: str = "none"

    # Stage 8.49: Condition Estimation & Dynamic Scaling
    condition_estimation_method: str = "auto"
    condition_estimation_max_iterations: int = 10
    condition_estimation_tolerance: float = 1e-7
    scaling_enabled: bool = False
    scaling_max_iterations: int = 15
    scaling_min_factor: float = 1e-5
    scaling_max_factor: float = 1e5
    scaling_balance_tolerance: float = 1e-3

    def __post_init__(self) -> None:
        """Validate tolerances to prevent invalid numerical configurations."""
        if self.feasibility_tolerance <= 0:
            raise ValueError(f"feasibility_tolerance must be positive, got {self.feasibility_tolerance}")
        if self.optimality_tolerance <= 0:
            raise ValueError(f"optimality_tolerance must be positive, got {self.optimality_tolerance}")
        if self.pivot_tolerance <= 0:
            raise ValueError(f"pivot_tolerance must be positive, got {self.pivot_tolerance}")
        if self.zero_tolerance <= 0:
            raise ValueError(f"zero_tolerance must be positive, got {self.zero_tolerance}")
        if self.basis_singularity_tolerance <= 0:
            raise ValueError(f"basis_singularity_tolerance must be positive, got {self.basis_singularity_tolerance}")
        if self.complementarity_tolerance <= 0:
            raise ValueError(f"complementarity_tolerance must be positive, got {self.complementarity_tolerance}")
        if self.condition_limit <= 1.0:
            raise ValueError(f"condition_limit must be > 1.0, got {self.condition_limit}")
        if self.min_safe_pivot <= 0:
            raise ValueError(f"min_safe_pivot must be positive, got {self.min_safe_pivot}")
        if self.max_safe_residual <= 0:
            raise ValueError(f"max_safe_residual must be positive, got {self.max_safe_residual}")

        valid_modes = {"none", "basic", "strict"}
        if self.verification_mode.lower() not in valid_modes:
            raise ValueError(
                f"Invalid verification_mode '{self.verification_mode}'. "
                f"Expected one of {sorted(valid_modes)}."
            )

        valid_cond_methods = {"auto", "svd", "iterative", "hager_higham"}
        if self.condition_estimation_method.lower() not in valid_cond_methods:
            raise ValueError(
                f"Invalid condition_estimation_method '{self.condition_estimation_method}'. "
                f"Expected one of {sorted(valid_cond_methods)}."
            )
        if self.condition_estimation_max_iterations <= 0:
            raise ValueError(
                f"condition_estimation_max_iterations must be positive, got {self.condition_estimation_max_iterations}"
            )
        if self.condition_estimation_tolerance <= 0:
            raise ValueError(
                f"condition_estimation_tolerance must be positive, got {self.condition_estimation_tolerance}"
            )
        if self.scaling_max_iterations <= 0:
            raise ValueError(f"scaling_max_iterations must be positive, got {self.scaling_max_iterations}")
        if self.scaling_min_factor <= 0 or self.scaling_max_factor <= self.scaling_min_factor:
            raise ValueError(
                f"scaling factors must satisfy 0 < min ({self.scaling_min_factor}) < max ({self.scaling_max_factor})"
            )
        if self.scaling_balance_tolerance <= 0:
            raise ValueError(
                f"scaling_balance_tolerance must be positive, got {self.scaling_balance_tolerance}"
            )

    @property
    def mode(self) -> VerificationMode:
        """Return structured VerificationMode enum."""
        return VerificationMode(self.verification_mode.lower())

    @classmethod
    def from_tolerances(cls, tol: Any = None, **overrides: Any) -> NumericalPolicy:
        """Factory creating a NumericalPolicy from SimplexTolerances, dict, or existing policy."""
        if isinstance(tol, NumericalPolicy) and not overrides:
            return tol

        params: dict[str, Any] = {}
        if tol is not None:
            if isinstance(tol, NumericalPolicy):
                params = {
                    "feasibility_tolerance": tol.feasibility_tolerance,
                    "optimality_tolerance": tol.optimality_tolerance,
                    "pivot_tolerance": tol.pivot_tolerance,
                    "zero_tolerance": tol.zero_tolerance,
                    "basis_singularity_tolerance": tol.basis_singularity_tolerance,
                    "complementarity_tolerance": tol.complementarity_tolerance,
                    "condition_limit": tol.condition_limit,
                    "min_safe_pivot": tol.min_safe_pivot,
                    "max_safe_residual": tol.max_safe_residual,
                    "verification_mode": tol.verification_mode,
                    "condition_estimation_method": tol.condition_estimation_method,
                    "condition_estimation_max_iterations": tol.condition_estimation_max_iterations,
                    "condition_estimation_tolerance": tol.condition_estimation_tolerance,
                    "scaling_enabled": tol.scaling_enabled,
                    "scaling_max_iterations": tol.scaling_max_iterations,
                    "scaling_min_factor": tol.scaling_min_factor,
                    "scaling_max_factor": tol.scaling_max_factor,
                    "scaling_balance_tolerance": tol.scaling_balance_tolerance,
                }
            elif isinstance(tol, Mapping):
                params = dict(tol)
            else:
                # SimplexTolerances or similar object
                if hasattr(tol, "feasibility_tolerance"):
                    params["feasibility_tolerance"] = tol.feasibility_tolerance
                if hasattr(tol, "reduced_cost_tolerance"):
                    params["optimality_tolerance"] = tol.reduced_cost_tolerance
                if hasattr(tol, "optimality_tolerance"):
                    params["optimality_tolerance"] = tol.optimality_tolerance
                if hasattr(tol, "pivot_tolerance"):
                    params["pivot_tolerance"] = tol.pivot_tolerance
                if hasattr(tol, "zero_tolerance"):
                    params["zero_tolerance"] = tol.zero_tolerance
                if hasattr(tol, "basis_singularity_tolerance"):
                    params["basis_singularity_tolerance"] = tol.basis_singularity_tolerance
                if hasattr(tol, "complementarity_tolerance"):
                    params["complementarity_tolerance"] = tol.complementarity_tolerance
                if hasattr(tol, "condition_limit"):
                    params["condition_limit"] = tol.condition_limit
                if hasattr(tol, "min_safe_pivot"):
                    params["min_safe_pivot"] = tol.min_safe_pivot
                if hasattr(tol, "max_safe_residual"):
                    params["max_safe_residual"] = tol.max_safe_residual
                if hasattr(tol, "numerical_verification"):
                    params["verification_mode"] = str(tol.numerical_verification)
                if hasattr(tol, "condition_estimation_method"):
                    params["condition_estimation_method"] = str(tol.condition_estimation_method)
                if hasattr(tol, "condition_estimation_max_iterations"):
                    params["condition_estimation_max_iterations"] = int(tol.condition_estimation_max_iterations)
                if hasattr(tol, "condition_estimation_tolerance"):
                    params["condition_estimation_tolerance"] = float(tol.condition_estimation_tolerance)
                if hasattr(tol, "scaling_enabled"):
                    params["scaling_enabled"] = bool(tol.scaling_enabled)
                if hasattr(tol, "scaling_max_iterations"):
                    params["scaling_max_iterations"] = int(tol.scaling_max_iterations)
                if hasattr(tol, "scaling_min_factor"):
                    params["scaling_min_factor"] = float(tol.scaling_min_factor)
                if hasattr(tol, "scaling_max_factor"):
                    params["scaling_max_factor"] = float(tol.scaling_max_factor)
                if hasattr(tol, "scaling_balance_tolerance"):
                    params["scaling_balance_tolerance"] = float(tol.scaling_balance_tolerance)

        params.update(overrides)
        return cls(**params)
