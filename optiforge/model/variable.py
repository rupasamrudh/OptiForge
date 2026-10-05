"""Decision variable representation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class VariableType(Enum):
    """Type of a decision variable."""

    CONTINUOUS = "continuous"
    INTEGER = "integer"
    BINARY = "binary"


@dataclass
class Variable:
    """A single decision variable in an optimization model.

    Parameters
    ----------
    name : str
        Unique identifier for the variable.
    lower_bound : float
        Lower bound (default 0.0).
    upper_bound : float
        Upper bound (default +∞).
    variable_type : VariableType
        CONTINUOUS, INTEGER, or BINARY (default CONTINUOUS).
    """

    name: str
    lower_bound: float = 0.0
    upper_bound: float = math.inf
    variable_type: VariableType = field(default=VariableType.CONTINUOUS)

    def __post_init__(self) -> None:
        if isinstance(self.variable_type, str):
            self.variable_type = VariableType(self.variable_type)

        if not self.name:
            raise ValueError("Variable name must be a non-empty string.")

        # Enforce binary bounds.
        if self.variable_type is VariableType.BINARY:
            self.lower_bound = 0.0
            self.upper_bound = 1.0

        if self.lower_bound > self.upper_bound:
            raise ValueError(
                f"Variable '{self.name}': lower_bound ({self.lower_bound}) "
                f"> upper_bound ({self.upper_bound})."
            )

    # -- convenience helpers --------------------------------------------------

    def is_binary(self) -> bool:
        """Return True if this is a binary variable."""
        return self.variable_type is VariableType.BINARY

    def is_integer(self) -> bool:
        """Return True if this is an integer (or binary) variable."""
        return self.variable_type in (VariableType.INTEGER, VariableType.BINARY)

    def __repr__(self) -> str:
        bounds = f"[{self.lower_bound}, {self.upper_bound}]"
        return f"Variable(name='{self.name}', type={self.variable_type.value}, bounds={bounds})"
