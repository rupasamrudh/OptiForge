"""Optimal basis and dual multiplier representation for OptiForge."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BasisStatus(Enum):
    """Status of a variable with respect to the optimal basis."""

    BASIC = "basic"
    NONBASIC_LOWER = "nonbasic_lower"
    NONBASIC_UPPER = "nonbasic_upper"
    NONBASIC_ZERO = "nonbasic_zero"
    FIXED = "fixed"
    FREE = "free"


@dataclass(frozen=True)
class BasisInfo:
    """Structured, immutable representation of an optimal LP basis.

    Attributes
    ----------
    basic_variable_indices : list[int]
        Column indices in standard-form problem that are basic (ordered by row).
    nonbasic_variable_indices : list[int]
        Column indices in standard-form problem that are nonbasic.
    basic_variables : list[str]
        Names of basic variables corresponding to basic_variable_indices.
    nonbasic_variables : list[str]
        Names of nonbasic variables.
    variable_status : dict[str, BasisStatus]
        Mapping from variable name to BasisStatus.
    primal_basic_values : dict[str, float]
        Mapping from basic variable name to primal basic value x_B.
    reduced_costs : dict[str, float]
        Mapping from variable name to reduced cost r_j.
    dual_multipliers : dict[str, float]
        Mapping from constraint name to shadow price / dual multiplier lambda_i.
    objective_value : float | None
        Optimal objective value associated with this basis.
    standard_form_dimension : tuple[int, int]
        Dimensions (m, n) of the standard-form constraint matrix.
    solver_selected : str
        Solver that produced the basis ('revised_simplex', 'dual_simplex', etc.).
    is_lp : bool
        True if this basis represents an optimal continuous LP relaxation/problem.
    is_optimal : bool
        True if the basis is optimal and feasible.
    metadata : dict[str, Any]
        Optional solver-specific or presolve reconstruction metadata.
    """

    basic_variable_indices: list[int] = field(default_factory=list)
    nonbasic_variable_indices: list[int] = field(default_factory=list)
    basic_variables: list[str] = field(default_factory=list)
    nonbasic_variables: list[str] = field(default_factory=list)
    variable_status: dict[str, BasisStatus] = field(default_factory=dict)
    primal_basic_values: dict[str, float] = field(default_factory=dict)
    reduced_costs: dict[str, float] = field(default_factory=dict)
    dual_multipliers: dict[str, float] = field(default_factory=dict)
    objective_value: float | None = None
    standard_form_dimension: tuple[int, int] = (0, 0)
    solver_selected: str = ""
    is_lp: bool = True
    is_optimal: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_basis_indices(self) -> list[int]:
        """Return a copy of the basic variable column indices."""
        return list(self.basic_variable_indices)

    def is_compatible(self, m: int, n: int) -> bool:
        """Check if this basis has matching row count m and valid column indices in [0, n-1]."""
        if len(self.basic_variable_indices) != m:
            return False
        if any(idx < 0 or idx >= n for idx in self.basic_variable_indices):
            return False
        if len(set(self.basic_variable_indices)) != m:
            return False
        return True
