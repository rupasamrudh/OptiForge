"""Linear constraint representation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ConstraintSense(Enum):
    """Sense (direction) of a linear constraint."""

    LESS_EQUAL = "<="
    GREATER_EQUAL = ">="
    EQUAL = "=="


@dataclass
class Constraint:
    """A single linear constraint of the form  Σ(aᵢ·xᵢ)  ⊕  rhs.

    Parameters
    ----------
    name : str
        Unique identifier for the constraint.
    coefficients : dict[str, float]
        Mapping of variable name → coefficient.
    sense : ConstraintSense
        One of ``<=``, ``>=``, ``==``.
    rhs : float
        Right-hand-side constant.
    """

    name: str
    coefficients: dict[str, float]
    sense: ConstraintSense
    rhs: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Constraint name must be a non-empty string.")
        if not self.coefficients:
            raise ValueError(
                f"Constraint '{self.name}': coefficients must not be empty."
            )

    def __repr__(self) -> str:
        lhs = " + ".join(
            f"{coeff}*{var}" for var, coeff in self.coefficients.items()
        )
        return f"Constraint('{self.name}': {lhs} {self.sense.value} {self.rhs})"
