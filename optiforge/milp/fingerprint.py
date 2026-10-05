"""Problem Fingerprinting for MILP models.

This module provides a deterministic, read-only fingerprinting component that
describes MILP structure for future adaptive strategies. The fingerprint
captures structural characteristics without mutating the model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from optiforge.model import Constraint, ConstraintSense, Model, ObjectiveSense, Variable, VariableType


@dataclass(frozen=True)
class ProblemFingerprint:
    """Deterministic structural fingerprint of a MILP model.

    This is a read-only descriptive snapshot. It never mutates the model,
    variables, constraints, objective, or solver state. All fields are
    computed deterministically from the model's public API.

    Attributes
    ----------
    variable_count : int
        Total number of variables in the model.
    constraint_count : int
        Total number of constraints in the model.
    integer_variable_count : int
        Number of INTEGER variables (excluding BINARY).
    binary_variable_count : int
        Number of BINARY variables.
    continuous_variable_count : int
        Number of CONTINUOUS variables.
    integer_variable_fraction : float
        Fraction of variables that are integer (INTEGER + BINARY) / total.
    binary_variable_fraction : float
        Fraction of variables that are binary / total.
    constraint_sense_le_count : int
        Number of <= constraints.
    constraint_sense_ge_count : int
        Number of >= constraints.
    constraint_sense_eq_count : int
        Number of == constraints.
    objective_sense : ObjectiveSense
        MINIMIZE or MAXIMIZE.
    variable_bounds_finite_count : int
        Number of variables with both finite lower and upper bounds.
    variable_bounds_infinite_count : int
        Number of variables with at least one infinite bound.
    variable_bounds_fixed_count : int
        Number of variables with lower_bound == upper_bound (fixed).
    constraint_density : float
        Fraction of structurally non-zero coefficients in the constraint
        matrix: (count of coefficients with abs(value) > 0) /
        (constraint_count * variable_count). 0.0 if no constraints or
        variables. Stored 0.0 coefficients do NOT count toward density.
    coefficient_magnitude_min : float
        Minimum absolute non-zero coefficient value, or inf if none.
    coefficient_magnitude_max : float
        Maximum absolute coefficient value, or 0.0 if none.
    coefficient_magnitude_mean : float
        Mean absolute coefficient value, or 0.0 if none.
    coefficient_magnitude_std : float
        Standard deviation of absolute coefficient values, or 0.0 if none.
    rhs_magnitude_min : float
        Minimum absolute RHS value, or inf if no constraints.
    rhs_magnitude_max : float
        Maximum absolute RHS value, or 0.0 if no constraints.
    rhs_magnitude_mean : float
        Mean absolute RHS value, or 0.0 if no constraints.
    rhs_magnitude_std : float
        Standard deviation of absolute RHS values, or 0.0 if no constraints.
    integer_constraint_coverage : float
        Fraction of constraints that reference at least one INTEGER or BINARY
        variable with a structurally non-zero coefficient. Bounded in [0, 1];
        0.0 if no constraints. Measures how widely integrality is coupled
        across the constraint set (distinct from integer_variable_fraction,
        which measures the variable side).
    binary_constraint_coverage : float
        Fraction of constraints that reference at least one BINARY variable
        with a structurally non-zero coefficient. Bounded in [0, 1]; 0.0 if
        no constraints. Measures how widely binary variables are coupled
        across the constraint set.
    is_binary_heavy : bool
        True if binary variables >= 50% of all variables and variable_count > 0.
    is_equality_heavy : bool
        True if equality constraints >= 50% of all constraints and constraint_count > 0.
    is_sparse : bool
        True if constraint_density < 0.1 (less than 10% structural nonzeros).
    is_dense : bool
        True if constraint_density > 0.5 (more than 50% structural nonzeros).
    is_bound_heavy : bool
        True if >= 50% of variables have both finite bounds and variable_count > 0.
    """

    # Basic counts
    variable_count: int
    constraint_count: int
    integer_variable_count: int
    binary_variable_count: int
    continuous_variable_count: int

    # Fractions
    integer_variable_fraction: float
    binary_variable_fraction: float

    # Constraint senses
    constraint_sense_le_count: int
    constraint_sense_ge_count: int
    constraint_sense_eq_count: int

    # Objective
    objective_sense: ObjectiveSense

    # Variable bounds
    variable_bounds_finite_count: int
    variable_bounds_infinite_count: int
    variable_bounds_fixed_count: int

    # Sparsity / density
    constraint_density: float

    # Coefficient statistics
    coefficient_magnitude_min: float
    coefficient_magnitude_max: float
    coefficient_magnitude_mean: float
    coefficient_magnitude_std: float

    # RHS statistics
    rhs_magnitude_min: float
    rhs_magnitude_max: float
    rhs_magnitude_mean: float
    rhs_magnitude_std: float

    # Constraint-side integrality coverage (bounded in [0, 1])
    integer_constraint_coverage: float
    binary_constraint_coverage: float

    # Structural indicators
    is_binary_heavy: bool
    is_equality_heavy: bool
    is_sparse: bool
    is_dense: bool
    is_bound_heavy: bool

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate fingerprint fields."""
        # Ensure fractions are in [0, 1]
        for frac_name in ("integer_variable_fraction", "binary_variable_fraction"):
            frac = getattr(self, frac_name)
            if not (0.0 <= frac <= 1.0):
                raise ValueError(f"{frac_name} must be in [0, 1], got {frac}")

        # Ensure density is in [0, 1]
        if not (0.0 <= self.constraint_density <= 1.0):
            raise ValueError(f"constraint_density must be in [0, 1], got {self.constraint_density}")

        # Ensure constraint coverage is in [0, 1]
        for cov_name in ("integer_constraint_coverage", "binary_constraint_coverage"):
            cov = getattr(self, cov_name)
            if not (0.0 <= cov <= 1.0):
                raise ValueError(f"{cov_name} must be in [0, 1], got {cov}")

        # Ensure structural indicators are boolean
        for bool_name in ("is_binary_heavy", "is_equality_heavy", "is_sparse", "is_dense", "is_bound_heavy"):
            val = getattr(self, bool_name)
            if not isinstance(val, bool):
                raise ValueError(f"{bool_name} must be bool, got {type(val)}")

    def to_dict(self) -> dict[str, Any]:
        """Convert fingerprint to a dictionary for serialization."""
        result = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, (ObjectiveSense, VariableType, ConstraintSense)):
                value = value.value
            result[field_name] = value
        return result

    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            f"ProblemFingerprint(",
            f"  variables={self.variable_count} (int={self.integer_variable_count}, bin={self.binary_variable_count}, cont={self.continuous_variable_count})",
            f"  constraints={self.constraint_count} (<=:{self.constraint_sense_le_count}, >=:{self.constraint_sense_ge_count}, ==:{self.constraint_sense_eq_count})",
            f"  objective={self.objective_sense.value}",
            f"  density={self.constraint_density:.4f}",
            f"  int_coverage={self.integer_constraint_coverage:.4f}, bin_coverage={self.binary_constraint_coverage:.4f}",
            f"  coeff_stats: min={self.coefficient_magnitude_min:.2e}, max={self.coefficient_magnitude_max:.2e}, mean={self.coefficient_magnitude_mean:.2e}",
            f"  rhs_stats: min={self.rhs_magnitude_min:.2e}, max={self.rhs_magnitude_max:.2e}, mean={self.rhs_magnitude_mean:.2e}",
            f"  binary_heavy={self.is_binary_heavy}, equality_heavy={self.is_equality_heavy}",
            f"  sparse={self.is_sparse}, dense={self.is_dense}, bound_heavy={self.is_bound_heavy}",
            f")",
        ]
        return "\n".join(lines)


def _compute_stats(values: list[float]) -> tuple[float, float, float, float]:
    """Compute min, max, mean, std of a list of floats.

    Returns (min, max, mean, std). For empty list, returns (inf, 0.0, 0.0, 0.0).
    """
    if not values:
        return float("inf"), 0.0, 0.0, 0.0

    min_val = min(values)
    max_val = max(values)
    mean_val = sum(values) / len(values)
    if len(values) > 1:
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        std_val = math.sqrt(variance)
    else:
        std_val = 0.0
    return min_val, max_val, mean_val, std_val


def fingerprint_model(model: Model) -> ProblemFingerprint:
    """Compute a deterministic structural fingerprint of a MILP model.

    This function is READ-ONLY. It never mutates the model, variables,
    constraints, objective, or any solver state.

    Parameters
    ----------
    model : Model
        The MILP model to fingerprint.

    Returns
    -------
    ProblemFingerprint
        A frozen dataclass containing the structural fingerprint.

    Raises
    ------
    ValueError
        If the model is None.
    """
    if model is None:
        raise ValueError("Model cannot be None")

    variables = model.variables
    constraints = model.constraints
    objective = model.objective

    # Basic counts
    variable_count = len(variables)
    constraint_count = len(constraints)

    # Variable type counts
    integer_variable_count = 0
    binary_variable_count = 0
    continuous_variable_count = 0

    # Variable bounds analysis
    variable_bounds_finite_count = 0
    variable_bounds_infinite_count = 0
    variable_bounds_fixed_count = 0

    for var in variables.values():
        if var.variable_type == VariableType.INTEGER:
            integer_variable_count += 1
        elif var.variable_type == VariableType.BINARY:
            binary_variable_count += 1
        else:
            continuous_variable_count += 1

        # Bounds analysis
        lb_finite = math.isfinite(var.lower_bound)
        ub_finite = math.isfinite(var.upper_bound)

        if lb_finite and ub_finite:
            variable_bounds_finite_count += 1
            if abs(var.lower_bound - var.upper_bound) <= 1e-12:
                variable_bounds_fixed_count += 1
        else:
            variable_bounds_infinite_count += 1

    # Fractions
    if variable_count > 0:
        integer_variable_fraction = (integer_variable_count + binary_variable_count) / variable_count
        binary_variable_fraction = binary_variable_count / variable_count
    else:
        integer_variable_fraction = 0.0
        binary_variable_fraction = 0.0

    # Constraint senses
    constraint_sense_le_count = 0
    constraint_sense_ge_count = 0
    constraint_sense_eq_count = 0

    for con in constraints.values():
        if con.sense == ConstraintSense.LESS_EQUAL:
            constraint_sense_le_count += 1
        elif con.sense == ConstraintSense.GREATER_EQUAL:
            constraint_sense_ge_count += 1
        elif con.sense == ConstraintSense.EQUAL:
            constraint_sense_eq_count += 1

    # Objective sense
    objective_sense = objective.sense if objective is not None else ObjectiveSense.MINIMIZE

    # Constraint density: structural nonzeros / (constraints * variables).
    # A coefficient counts as structurally non-zero only when its value is
    # non-zero; stored 0.0 coefficients do NOT contribute to density. This
    # keeps density consistent with the coefficient magnitude statistics,
    # which also exclude zero-valued coefficients.
    if constraint_count > 0 and variable_count > 0:
        total_nonzeros = sum(
            1
            for con in constraints.values()
            for coeff in con.coefficients.values()
            if abs(coeff) > 0
        )
        constraint_density = total_nonzeros / (constraint_count * variable_count)
    else:
        constraint_density = 0.0

    # Coefficient magnitude statistics (structural nonzeros only).
    coeff_values = []
    for con in constraints.values():
        for coeff in con.coefficients.values():
            if abs(coeff) > 0:
                coeff_values.append(abs(coeff))

    coeff_min, coeff_max, coeff_mean, coeff_std = _compute_stats(coeff_values)

    # RHS magnitude statistics
    rhs_values = [abs(con.rhs) for con in constraints.values()]
    rhs_min, rhs_max, rhs_mean, rhs_std = _compute_stats(rhs_values)

    # Constraint-side integrality coverage: fraction of constraints that
    # reference at least one integer/binary variable with a structurally
    # non-zero coefficient. Bounded in [0, 1]. This is a genuinely different
    # structural signal from integer_variable_fraction / binary_variable_fraction
    # (which measure the variable side): it captures how widely integrality is
    # coupled across the constraint set, which is informative for adaptive
    # branching/node-selection strategy selection.
    integer_constraint_coverage = 0.0
    binary_constraint_coverage = 0.0
    if constraint_count > 0:
        integer_covered = 0
        binary_covered = 0
        for con in constraints.values():
            has_integer = False
            has_binary = False
            for var_name, coeff in con.coefficients.items():
                if abs(coeff) <= 0:
                    continue
                var = variables.get(var_name)
                if var is None:
                    continue
                if var.variable_type == VariableType.BINARY:
                    has_binary = True
                    has_integer = True
                elif var.variable_type == VariableType.INTEGER:
                    has_integer = True
            if has_integer:
                integer_covered += 1
            if has_binary:
                binary_covered += 1
        integer_constraint_coverage = integer_covered / constraint_count
        binary_constraint_coverage = binary_covered / constraint_count

    # Structural indicators. Sparse/dense require a non-empty constraint
    # matrix: a model with no constraints (or no variables) has no matrix
    # to be sparse or dense, so both flags are False.
    has_matrix = constraint_count > 0 and variable_count > 0
    is_binary_heavy = variable_count > 0 and binary_variable_fraction >= 0.5
    is_equality_heavy = constraint_count > 0 and (constraint_sense_eq_count / constraint_count) >= 0.5
    is_sparse = has_matrix and constraint_density < 0.1
    is_dense = has_matrix and constraint_density > 0.5
    is_bound_heavy = variable_count > 0 and (variable_bounds_finite_count / variable_count) >= 0.5

    return ProblemFingerprint(
        variable_count=variable_count,
        constraint_count=constraint_count,
        integer_variable_count=integer_variable_count,
        binary_variable_count=binary_variable_count,
        continuous_variable_count=continuous_variable_count,
        integer_variable_fraction=integer_variable_fraction,
        binary_variable_fraction=binary_variable_fraction,
        constraint_sense_le_count=constraint_sense_le_count,
        constraint_sense_ge_count=constraint_sense_ge_count,
        constraint_sense_eq_count=constraint_sense_eq_count,
        objective_sense=objective_sense,
        variable_bounds_finite_count=variable_bounds_finite_count,
        variable_bounds_infinite_count=variable_bounds_infinite_count,
        variable_bounds_fixed_count=variable_bounds_fixed_count,
        constraint_density=constraint_density,
        coefficient_magnitude_min=coeff_min,
        coefficient_magnitude_max=coeff_max,
        coefficient_magnitude_mean=coeff_mean,
        coefficient_magnitude_std=coeff_std,
        rhs_magnitude_min=rhs_min,
        rhs_magnitude_max=rhs_max,
        rhs_magnitude_mean=rhs_mean,
        rhs_magnitude_std=rhs_std,
        integer_constraint_coverage=integer_constraint_coverage,
        binary_constraint_coverage=binary_constraint_coverage,
        is_binary_heavy=is_binary_heavy,
        is_equality_heavy=is_equality_heavy,
        is_sparse=is_sparse,
        is_dense=is_dense,
        is_bound_heavy=is_bound_heavy,
        metadata={},
    )