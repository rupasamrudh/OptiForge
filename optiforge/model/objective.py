"""Objective function representation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ObjectiveSense(Enum):
    """Direction of optimization."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass
class Objective:
    """A linear objective function  Σ(cᵢ·xᵢ).

    Parameters
    ----------
    coefficients : dict[str, float]
        Mapping of variable name → coefficient.
    sense : ObjectiveSense
        MINIMIZE or MAXIMIZE.
    """

    coefficients: dict[str, float]
    sense: ObjectiveSense

    def __post_init__(self) -> None:
        if not self.coefficients:
            raise ValueError("Objective coefficients must not be empty.")

    def __repr__(self) -> str:
        terms = " + ".join(
            f"{coeff}*{var}" for var, coeff in self.coefficients.items()
        )
        return f"Objective({self.sense.value}: {terms})"
