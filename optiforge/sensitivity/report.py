"""Sensitivity report container for LP post-optimality analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from optiforge.core.basis import BasisInfo
from optiforge.sensitivity.range import SensitivityRange


@dataclass
class SensitivityReport:
    """Comprehensive LP sensitivity analysis report for constraints and objective coefficients.

    Attributes
    ----------
    rhs : dict[str, SensitivityRange]
        Mapping from constraint name to allowable RHS sensitivity range.
    objective : dict[str, SensitivityRange]
        Mapping from variable name to allowable objective coefficient sensitivity range.
    basis_info : BasisInfo | None
        Optimal BasisInfo reference from which this report was derived.
    solver_selected : str
        Name of the solver that generated the basis.
    is_optimal : bool
        True if the basis is optimal and feasible.
    metadata : dict[str, Any]
        Additional metadata (e.g. presolved status, degeneracy information).
    """

    rhs: dict[str, SensitivityRange] = field(default_factory=dict)
    objective: dict[str, SensitivityRange] = field(default_factory=dict)
    basis_info: BasisInfo | None = None
    solver_selected: str = ""
    is_optimal: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_rhs_range(self, constraint_name: str) -> SensitivityRange:
        """Retrieve allowable RHS range for a specific constraint."""
        if constraint_name not in self.rhs:
            raise KeyError(
                f"Constraint '{constraint_name}' not found in sensitivity report. "
                f"Available constraints: {list(self.rhs.keys())}"
            )
        return self.rhs[constraint_name]

    def get_objective_range(self, variable_name: str) -> SensitivityRange:
        """Retrieve allowable objective coefficient range for a specific variable."""
        if variable_name not in self.objective:
            raise KeyError(
                f"Variable '{variable_name}' not found in sensitivity report. "
                f"Available variables: {list(self.objective.keys())}"
            )
        return self.objective[variable_name]

    def summary(self) -> str:
        """Generate a human-readable textual summary of sensitivity ranges."""
        lines = ["=== LP SENSITIVITY REPORT ==="]
        lines.append(f"Solver: {self.solver_selected} | Optimal: {self.is_optimal}")
        lines.append("\n--- Right-Hand-Side (RHS) Ranges ---")
        lines.append(f"{'Constraint':<20} {'Current':<10} {'Allowable -Delta':<16} {'Allowable +Delta':<16} {'Range'}")
        lines.append("-" * 75)
        for name, r in self.rhs.items():
            lb_str = "-inf" if r.is_unbounded_below else f"{r.lower_bound:.4f}"
            ub_str = "+inf" if r.is_unbounded_above else f"{r.upper_bound:.4f}"
            dec_str = "inf" if r.is_unbounded_below else f"{r.allowable_decrease:.4f}"
            inc_str = "inf" if r.is_unbounded_above else f"{r.allowable_increase:.4f}"
            lines.append(f"{name:<20} {r.current_value:<10.4f} {dec_str:<16} {inc_str:<16} [{lb_str}, {ub_str}]")

        lines.append("\n--- Objective Coefficient Ranges ---")
        lines.append(f"{'Variable':<20} {'Current':<10} {'Allowable -Delta':<16} {'Allowable +Delta':<16} {'Range'}")
        lines.append("-" * 75)
        for name, r in self.objective.items():
            lb_str = "-inf" if r.is_unbounded_below else f"{r.lower_bound:.4f}"
            ub_str = "+inf" if r.is_unbounded_above else f"{r.upper_bound:.4f}"
            dec_str = "inf" if r.is_unbounded_below else f"{r.allowable_decrease:.4f}"
            inc_str = "inf" if r.is_unbounded_above else f"{r.allowable_increase:.4f}"
            lines.append(f"{name:<20} {r.current_value:<10.4f} {dec_str:<14} {inc_str:<14} [{lb_str}, {ub_str}]")

        return "\n".join(lines)
