"""OptiForge numerical core exceptions."""

from __future__ import annotations


class SingularBasisError(ValueError):
    """Raised when a basis matrix B is singular or rank-deficient."""

    def __init__(
        self,
        message: str,
        pivot: float | None = None,
        step: int | None = None,
    ) -> None:
        super().__init__(message)
        self.pivot = pivot
        self.step = step


class NumericalSingularityError(SingularBasisError):
    """Raised when a matrix is singular or numerically singular during factorization."""
