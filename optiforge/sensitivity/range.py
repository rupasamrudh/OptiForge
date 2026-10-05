"""Sensitivity range representation for LP post-optimality analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SensitivityRange:
    """Allowable parameter variation interval preserving the current optimal basis.

    Attributes
    ----------
    name : str
        Name of the model entity (constraint name for RHS, variable name for objective).
    current_value : float
        Current value of the parameter in the solved model.
    allowable_decrease : float
        Maximum allowable decrease while the current basis remains optimal (>= 0.0, or inf).
    allowable_increase : float
        Maximum allowable increase while the current basis remains optimal (>= 0.0, or inf).
    lower_bound : float
        Lower boundary value of the parameter (current_value - allowable_decrease).
    upper_bound : float
        Upper boundary value of the parameter (current_value + allowable_increase).
    is_unbounded_below : bool
        True if the parameter can decrease to -inf without changing the optimal basis.
    is_unbounded_above : bool
        True if the parameter can increase to +inf without changing the optimal basis.
    basis_remains_optimal : bool
        True if this range was computed from a verified optimal basis.
    metadata : dict[str, Any]
        Additional context (e.g., binding status, presolve role, degeneracy indicator).
    """

    name: str
    current_value: float
    allowable_decrease: float
    allowable_increase: float
    lower_bound: float
    upper_bound: float
    is_unbounded_below: bool = False
    is_unbounded_above: bool = False
    basis_remains_optimal: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def contains(self, value: float, tol: float = 1e-7) -> bool:
        """Check if a perturbed parameter value lies within this sensitivity range."""
        if not self.is_unbounded_below and value < self.lower_bound - tol:
            return False
        if not self.is_unbounded_above and value > self.upper_bound + tol:
            return False
        return True

    def __repr__(self) -> str:
        lb_str = "-inf" if self.is_unbounded_below else f"{self.lower_bound:.4g}"
        ub_str = "+inf" if self.is_unbounded_above else f"{self.upper_bound:.4g}"
        return (
            f"SensitivityRange('{self.name}': curr={self.current_value:.4g}, "
            f"range=[{lb_str}, {ub_str}], -Delta={self.allowable_decrease:.4g}, +Delta={self.allowable_increase:.4g})"
        )

