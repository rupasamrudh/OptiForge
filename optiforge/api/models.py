"""API request/response models for OptiForge web interface."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def _validate_finite(value: float, field_name: str) -> float:
    """Validate that a float value is finite (not NaN or infinity)."""
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number (not NaN or infinity)")
    return value


def _validate_not_extreme(value: float, field_name: str, max_abs: float = 1e12) -> float:
    """Validate that a float value is not extremely large."""
    if abs(value) > max_abs:
        raise ValueError(f"{field_name} must not exceed {max_abs} in absolute value")
    return value


class VariableInput(BaseModel):
    """Variable definition for API input."""

    name: str = Field(..., min_length=1, description="Variable name")
    lower_bound: float = Field(default=0.0, description="Lower bound")
    upper_bound: float | None = Field(default=None, description="Upper bound (None for +inf)")
    variable_type: Literal["continuous", "integer", "binary"] = Field(
        default="continuous", description="Variable type"
    )

    @field_validator("lower_bound", "upper_bound", mode="before")
    @classmethod
    def validate_bounds_finite(cls, v: float | None) -> float | None:
        if v is not None:
            _validate_finite(v, "bound")
            _validate_not_extreme(v, "bound")
        return v

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Strip surrounding whitespace and reject blank names."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("variable name must not be empty or whitespace-only")
        return stripped

    @model_validator(mode="after")
    def validate_bounds_order(self) -> "VariableInput":
        # Enforce lb <= ub for all continuous/integer variables
        if self.upper_bound is not None and self.lower_bound > self.upper_bound:
            raise ValueError(
                f"lower_bound ({self.lower_bound}) must not exceed upper_bound ({self.upper_bound})"
            )

        # Validate binary bounds
        if self.variable_type == "binary":
            if self.lower_bound != 0.0:
                raise ValueError("binary variable must have lower_bound = 0")
            if self.upper_bound is not None and self.upper_bound != 1.0:
                raise ValueError("binary variable must have upper_bound = 1 or None")

        # Validate integer bounds consistency
        if self.variable_type == "integer":
            if self.lower_bound != math.floor(self.lower_bound):
                raise ValueError("integer variable lower_bound must be an integer")
            if self.upper_bound is not None and self.upper_bound != math.floor(self.upper_bound):
                raise ValueError("integer variable upper_bound must be an integer")

        return self


class ConstraintInput(BaseModel):
    """Constraint definition for API input."""

    name: str = Field(..., min_length=1, description="Constraint name")
    coefficients: dict[str, float] = Field(
        ..., min_length=1, description="Variable name -> coefficient mapping"
    )
    sense: Literal["<=", ">=", "=="] = Field(..., description="Constraint sense")
    rhs: float = Field(..., description="Right-hand side value")

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Strip surrounding whitespace and reject blank names."""
        stripped = str(v).strip()
        if not stripped:
            raise ValueError("constraint name cannot be blank or whitespace-only")
        return stripped

    @field_validator("coefficients", mode="before")
    @classmethod
    def validate_coefficients(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("constraint must have at least one coefficient")
        result = {}
        for k, v in v.items():
            _validate_finite(v, f"coefficient for variable '{k}'")
            _validate_not_extreme(v, f"coefficient for variable '{k}'")
            result[k] = v
        return result

    @field_validator("rhs", mode="before")
    @classmethod
    def validate_rhs(cls, v: float) -> float:
        _validate_finite(v, "rhs")
        _validate_not_extreme(v, "rhs")
        return v


class ObjectiveInput(BaseModel):
    """Objective function definition for API input."""

    coefficients: dict[str, float] = Field(
        ..., min_length=1, description="Variable name -> coefficient mapping"
    )
    sense: Literal["minimize", "maximize"] = Field(..., description="Optimization direction")

    @field_validator("coefficients", mode="before")
    @classmethod
    def validate_coefficients(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("objective must have at least one coefficient")
        result = {}
        for k, val in v.items():
            _validate_finite(val, f"coefficient for variable '{k}'")
            _validate_not_extreme(val, f"coefficient for variable '{k}'")
            result[k] = val
        if all(coeff == 0.0 for coeff in result.values()):
            raise ValueError(
                "objective coefficients must not all be zero; "
                "a trivial objective (all zeros) is not meaningful"
            )
        return result


class SolveRequest(BaseModel):
    """Complete LP/MILP problem definition for solving."""

    variables: list[VariableInput] = Field(..., min_length=1, description="List of variables")
    constraints: list[ConstraintInput] = Field(default_factory=list, description="List of constraints")
    objective: ObjectiveInput = Field(..., description="Objective function")

    @model_validator(mode="after")
    def validate_dimensions(self) -> "SolveRequest":
        """Validate that all dimensions are consistent."""
        var_names = {v.name for v in self.variables}

        # Check objective coefficients reference valid variables
        obj_vars = set(self.objective.coefficients.keys())
        unknown_obj = obj_vars - var_names
        if unknown_obj:
            raise ValueError(f"Objective references unknown variables: {sorted(unknown_obj)}")

        # Check constraint coefficients reference valid variables
        for constr in self.constraints:
            constr_vars = set(constr.coefficients.keys())
            unknown_constr = constr_vars - var_names
            if unknown_constr:
                raise ValueError(
                    f"Constraint '{constr.name}' references unknown variables: {sorted(unknown_constr)}"
                )

        # Check for duplicate variable names
        if len(var_names) != len(self.variables):
            raise ValueError("Duplicate variable names found")

        # Check for duplicate constraint names
        constr_names = [c.name for c in self.constraints]
        if len(constr_names) != len(set(constr_names)):
            raise ValueError("Duplicate constraint names found")

        # Check for empty constraint coefficients (should not happen due to min_length=1)
        for constr in self.constraints:
            if not constr.coefficients:
                raise ValueError(f"Constraint '{constr.name}' has no coefficients")

        # Check for zero RHS with equality constraint (potential issue)
        for constr in self.constraints:
            if constr.sense == "==" and constr.rhs == 0.0 and len(constr.coefficients) == 1:
                # This is valid but worth noting - single variable equality to 0
                pass

        return self


class VariableValue(BaseModel):
    """Variable value in solution."""

    name: str
    value: float
    variable_type: Literal["continuous", "integer", "binary"] = "continuous"
    lower_bound: float | None = None
    upper_bound: float | None = None
    integrality_valid: bool | None = None
    integrality_violation: float | None = None


class ConstraintEvaluation(BaseModel):
    """Constraint evaluation in solution."""

    name: str
    lhs_value: float
    sense: Literal["<=", ">=", "=="]
    rhs: float
    slack: float | None = None
    residual: float | None = None
    violation: float
    is_satisfied: bool


class SolveResponse(BaseModel):
    """Solver response with results."""

    status: Literal["optimal", "feasible", "infeasible", "unbounded", "iteration_limit", "unknown"]
    objective_value: float | None = None
    variable_values: list[VariableValue] = Field(default_factory=list)
    constraint_evaluations: list[ConstraintEvaluation] = Field(default_factory=list)
    message: str = ""
    iterations: int = 0
    solve_time_ms: float | None = None
    node_count: int | None = None
    numerical_snapshots: list | None = Field(default=None, description="Numerical monitoring snapshots (if monitoring was enabled)")
    # MILP-specific metrics
    incumbent_objective: float | None = None
    best_bound: float | None = None
    optimality_gap: float | None = None
    absolute_gap: float | None = None
    nodes_remaining: int | None = None
    max_depth: int | None = None
    feasible_solutions_found: int | None = None
    # Solver status explanation
    status_explanation: str | None = None
    feasibility_info: str | None = None


class ErrorResponse(BaseModel):
    """Error response for validation or solver errors."""

    error: str
    detail: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "0.1.0"