"""Tests for Stage 7.2: Problem Fingerprinting."""


import math

import pytest

from optiforge.core.solution import SolutionStatus
from optiforge.milp import (
    MILPSolver,
    ProblemFingerprint,
    fingerprint_model,
    solve_milp,
)
from optiforge.model import (
    Constraint,
    ConstraintSense,
    Model,
    Objective,
    ObjectiveSense,
    Variable,
    VariableType,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _build_pure_lp() -> Model:
    """Pure LP (all continuous):
    max 3x1 + 2x2
    s.t. x1 + x2 <= 4
         x1, x2 >= 0, continuous
    """
    m = Model("pure_lp")
    m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
    m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.set_objective(Objective({"x1": 3.0, "x2": 2.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_binary_milp() -> Model:
    """Binary MILP:
    max 5x1 + 4x2 + 3x3
    s.t. 3x1 + 2x2 + x3 <= 4
         x1, x2, x3 in {0,1}
    """
    m = Model("binary_milp")
    m.add_variable(Variable("x1", variable_type=VariableType.BINARY))
    m.add_variable(Variable("x2", variable_type=VariableType.BINARY))
    m.add_variable(Variable("x3", variable_type=VariableType.BINARY))
    m.add_constraint(Constraint("c1", {"x1": 3.0, "x2": 2.0, "x3": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.set_objective(Objective({"x1": 5.0, "x2": 4.0, "x3": 3.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_mixed_integer() -> Model:
    """Mixed integer + continuous:
    max 2x1 + 3x2
    s.t. x1 + 2x2 <= 4
         x1 >= 0, continuous
         x2 >= 0, integer
    """
    m = Model("mixed_int")
    m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
    m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 2.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.set_objective(Objective({"x1": 2.0, "x2": 3.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_equality_heavy() -> Model:
    """Equality-heavy model:
    min x1 + x2
    s.t. x1 + x2 == 5
         x1 - x2 == 1
         x1, x2 >= 0, integer
    """
    m = Model("equality_heavy")
    m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.EQUAL, 5.0))
    m.add_constraint(Constraint("c2", {"x1": 1.0, "x2": -1.0}, ConstraintSense.EQUAL, 1.0))
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MINIMIZE))
    return m


def _build_sparse_model() -> Model:
    """Sparse model: many variables, few nonzeros per constraint."""
    m = Model("sparse")
    for i in range(20):
        m.add_variable(Variable(f"x{i}", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
    # Each constraint touches only 1-2 variables out of 20
    m.add_constraint(Constraint("c1", {"x0": 1.0, "x1": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.add_constraint(Constraint("c2", {"x2": 1.0, "x3": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.add_constraint(Constraint("c3", {"x4": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.set_objective(Objective({"x0": 1.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_dense_model() -> Model:
    """Dense model: every constraint touches every variable."""
    m = Model("dense")
    for i in range(5):
        m.add_variable(Variable(f"x{i}", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
    for j in range(5):
        m.add_constraint(Constraint(
            f"c{j}",
            {f"x{i}": 1.0 for i in range(5)},
            ConstraintSense.LESS_EQUAL,
            10.0,
        ))
    m.set_objective(Objective({f"x{i}": 1.0 for i in range(5)}, ObjectiveSense.MAXIMIZE))
    return m


def _build_bound_heavy() -> Model:
    """Bound-heavy model: all variables have finite bounds."""
    m = Model("bound_heavy")
    m.add_variable(Variable("x1", lower_bound=0.0, upper_bound=10.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable("x2", lower_bound=0.0, upper_bound=10.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable("x3", lower_bound=0.0, upper_bound=10.0, variable_type=VariableType.CONTINUOUS))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0, "x3": 1.0}, ConstraintSense.LESS_EQUAL, 15.0))
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0, "x3": 1.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_empty_model() -> Model:
    """Empty model: no variables, no constraints, no objective."""
    return Model("empty")


def _build_degenerate_model() -> Model:
    """Degenerate model: fixed variables, zero coefficients, empty constraint set."""
    m = Model("degenerate")
    m.add_variable(Variable("x1", lower_bound=2.0, upper_bound=2.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
    m.set_objective(Objective({"x1": 1.0, "x2": 0.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_zero_coefficient_model() -> Model:
    """Model with zero coefficients in constraints and objective."""
    m = Model("zero_coeff")
    m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
    m.add_constraint(Constraint("c1", {"x1": 0.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.add_constraint(Constraint("c2", {"x1": 1.0, "x2": 0.0}, ConstraintSense.GREATER_EQUAL, 1.0))
    m.set_objective(Objective({"x1": 0.0, "x2": 1.0}, ObjectiveSense.MINIMIZE))
    return m


def _build_negative_coefficient_model() -> Model:
    """Model with negative coefficients."""
    m = Model("neg_coeff")
    m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
    m.add_constraint(Constraint("c1", {"x1": -1.0, "x2": 2.0}, ConstraintSense.LESS_EQUAL, 3.0))
    m.add_constraint(Constraint("c2", {"x1": 1.0, "x2": -1.0}, ConstraintSense.GREATER_EQUAL, -2.0))
    m.set_objective(Objective({"x1": -1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_infinite_bounds_model() -> Model:
    """Model with infinite bounds (free variables)."""
    m = Model("infinite_bounds")
    m.add_variable(Variable("x1", lower_bound=-math.inf, upper_bound=math.inf, variable_type=VariableType.CONTINUOUS))
    m.add_variable(Variable("x2", lower_bound=0.0, upper_bound=math.inf, variable_type=VariableType.INTEGER))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
    return m


def _build_minimize_model() -> Model:
    """MINIMIZE model."""
    m = Model("minimize")
    m.add_variable(Variable("x", lower_bound=0.0, upper_bound=10.0, variable_type=VariableType.INTEGER))
    m.set_objective(Objective({"x": 2.0}, ObjectiveSense.MINIMIZE))
    return m


def _build_maximize_model() -> Model:
    """MAXIMIZE model."""
    m = Model("maximize")
    m.add_variable(Variable("x", lower_bound=0.0, upper_bound=10.0, variable_type=VariableType.INTEGER))
    m.set_objective(Objective({"x": 2.0}, ObjectiveSense.MAXIMIZE))
    return m


# ── Tests ───────────────────────────────────────────────────────────────────


class TestPureLP:
    """Pure LP fingerprinting."""

    def test_pure_lp_counts(self):
        fp = fingerprint_model(_build_pure_lp())
        assert fp.variable_count == 2
        assert fp.constraint_count == 1
        assert fp.continuous_variable_count == 2
        assert fp.integer_variable_count == 0
        assert fp.binary_variable_count == 0
        assert fp.integer_variable_fraction == 0.0
        assert fp.binary_variable_fraction == 0.0

    def test_pure_lp_not_binary_heavy(self):
        fp = fingerprint_model(_build_pure_lp())
        assert not fp.is_binary_heavy

    def test_pure_lp_sense_counts(self):
        fp = fingerprint_model(_build_pure_lp())
        assert fp.constraint_sense_le_count == 1
        assert fp.constraint_sense_ge_count == 0
        assert fp.constraint_sense_eq_count == 0


class TestBinaryMILP:
    """Binary MILP fingerprinting."""

    def test_binary_counts(self):
        fp = fingerprint_model(_build_binary_milp())
        assert fp.variable_count == 3
        assert fp.binary_variable_count == 3
        assert fp.integer_variable_count == 0
        assert fp.continuous_variable_count == 0
        assert fp.binary_variable_fraction == pytest.approx(1.0)
        assert fp.integer_variable_fraction == pytest.approx(1.0)

    def test_binary_heavy(self):
        fp = fingerprint_model(_build_binary_milp())
        assert fp.is_binary_heavy

    def test_binary_bounds(self):
        fp = fingerprint_model(_build_binary_milp())
        # Binary variables have finite [0,1] bounds
        assert fp.variable_bounds_finite_count == 3
        assert fp.variable_bounds_infinite_count == 0


class TestMixedInteger:
    """Mixed-integer fingerprinting."""

    def test_mixed_counts(self):
        fp = fingerprint_model(_build_mixed_integer())
        assert fp.variable_count == 2
        assert fp.continuous_variable_count == 1
        assert fp.integer_variable_count == 1
        assert fp.binary_variable_count == 0
        assert fp.integer_variable_fraction == pytest.approx(0.5)
        assert fp.binary_variable_fraction == pytest.approx(0.0)

    def test_mixed_not_binary_heavy(self):
        fp = fingerprint_model(_build_mixed_integer())
        assert not fp.is_binary_heavy


class TestEqualityHeavy:
    """Equality-heavy fingerprinting."""

    def test_equality_counts(self):
        fp = fingerprint_model(_build_equality_heavy())
        assert fp.constraint_sense_eq_count == 2
        assert fp.constraint_sense_le_count == 0
        assert fp.constraint_sense_ge_count == 0

    def test_equality_heavy_flag(self):
        fp = fingerprint_model(_build_equality_heavy())
        assert fp.is_equality_heavy

    def test_not_equality_heavy_for_le(self):
        fp = fingerprint_model(_build_pure_lp())
        assert not fp.is_equality_heavy


class TestSparseDense:
    """Sparse vs dense fingerprinting."""

    def test_sparse_density(self):
        fp = fingerprint_model(_build_sparse_model())
        # 20 vars, 3 constraints, 5 nonzeros total
        assert fp.constraint_density == pytest.approx(5 / 60)
        assert fp.is_sparse
        assert not fp.is_dense

    def test_dense_density(self):
        fp = fingerprint_model(_build_dense_model())
        # 5 vars, 5 constraints, 25 nonzeros total
        assert fp.constraint_density == pytest.approx(1.0)
        assert fp.is_dense
        assert not fp.is_sparse

    def test_stored_zero_coefficient_not_counted_in_density(self):
        # Stored 0.0 coefficients should NOT count toward structural density
        m = Model("zero_density")
        m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        m.add_constraint(Constraint("c1", {"x1": 0.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
        fp = fingerprint_model(m)
        # 1 structural nonzero / (1 constraint * 2 vars) = 0.5
        assert fp.constraint_density == pytest.approx(0.5)
        assert not fp.is_dense
        assert not fp.is_sparse

    def test_mixed_zero_nonzero_coefficients(self):
        # Mix of zero and nonzero coefficients
        m = Model("mixed_density")
        m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        m.add_variable(Variable("x3", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 0.0, "x3": 2.0}, ConstraintSense.LESS_EQUAL, 5.0))
        m.add_constraint(Constraint("c2", {"x1": 0.0, "x2": 0.0, "x3": 0.0}, ConstraintSense.LESS_EQUAL, 3.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0, "x3": 1.0}, ObjectiveSense.MAXIMIZE))
        fp = fingerprint_model(m)
        # 2 structural nonzeros / (2 constraints * 3 vars) = 2/6 = 1/3
        assert fp.constraint_density == pytest.approx(1/3)
        assert not fp.is_dense
        assert not fp.is_sparse

    def test_all_zero_coefficient_model(self):
        # All coefficients are zero
        m = Model("all_zero")
        m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        m.add_constraint(Constraint("c1", {"x1": 0.0, "x2": 0.0}, ConstraintSense.LESS_EQUAL, 5.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
        fp = fingerprint_model(m)
        # 0 structural nonzeros / (1 constraint * 2 vars) = 0.0
        assert fp.constraint_density == pytest.approx(0.0)
        assert fp.coefficient_magnitude_min == math.inf
        assert fp.coefficient_magnitude_max == 0.0
        assert fp.coefficient_magnitude_mean == 0.0
        assert fp.coefficient_magnitude_std == 0.0
        assert fp.is_sparse
        assert not fp.is_dense

    def test_exact_sparse_threshold(self):
        # Density exactly 0.1 -> NOT sparse (strict < 0.1)
        m = Model("exact_sparse")
        m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        m.add_variable(Variable("x3", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        m.add_variable(Variable("x4", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        m.add_variable(Variable("x5", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        m.add_variable(Variable("x6", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        m.add_variable(Variable("x7", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        m.add_variable(Variable("x8", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        m.add_variable(Variable("x9", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        m.add_variable(Variable("x10", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        # 1 constraint, 1 nonzero -> density = 1/10 = 0.1
        m.add_constraint(Constraint("c1", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
        m.set_objective(Objective({"x1": 1.0}, ObjectiveSense.MAXIMIZE))
        fp = fingerprint_model(m)
        assert fp.constraint_density == pytest.approx(0.1)
        assert not fp.is_sparse  # strict < 0.1
        assert not fp.is_dense

    def test_exact_dense_threshold(self):
        # Density exactly 0.5 -> NOT dense (strict > 0.5)
        m = Model("exact_dense")
        m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        # 1 constraint, 1 nonzero -> density = 1/2 = 0.5
        m.add_constraint(Constraint("c1", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
        m.set_objective(Objective({"x1": 1.0}, ObjectiveSense.MAXIMIZE))
        fp = fingerprint_model(m)
        assert fp.constraint_density == pytest.approx(0.5)
        assert not fp.is_dense  # strict > 0.5
        assert not fp.is_sparse

    def test_sparse_classification_after_fix(self):
        # Sparse model with structural zeros
        m = Model("sparse_fixed")
        for i in range(20):
            m.add_variable(Variable(f"x{i}", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        # 3 constraints, 5 structural nonzeros
        m.add_constraint(Constraint("c1", {"x0": 1.0, "x1": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
        m.add_constraint(Constraint("c2", {"x2": 1.0, "x3": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
        m.add_constraint(Constraint("c3", {"x4": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
        m.set_objective(Objective({"x0": 1.0}, ObjectiveSense.MAXIMIZE))
        fp = fingerprint_model(m)
        assert fp.constraint_density == pytest.approx(5 / 60)
        assert fp.is_sparse
        assert not fp.is_dense

    def test_dense_classification_after_fix(self):
        # Dense model with some stored zeros
        m = Model("dense_fixed")
        for i in range(5):
            m.add_variable(Variable(f"x{i}", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        # 5 constraints, each with 4 structural nonzeros (1 stored zero each)
        for j in range(5):
            coeffs = {f"x{i}": 1.0 for i in range(5)}
            coeffs[f"x{j}"] = 0.0  # one stored zero per constraint
            m.add_constraint(Constraint(f"c{j}", coeffs, ConstraintSense.LESS_EQUAL, 10.0))
        m.set_objective(Objective({f"x{i}": 1.0 for i in range(5)}, ObjectiveSense.MAXIMIZE))
        fp = fingerprint_model(m)
        # 5 constraints * 4 structural nonzeros = 20 / (5*5) = 0.8
        assert fp.constraint_density == pytest.approx(0.8)
        assert fp.is_dense
        assert not fp.is_sparse


class TestBoundHeavy:
    """Bound-heavy fingerprinting."""

    def test_bound_heavy_flag(self):
        fp = fingerprint_model(_build_bound_heavy())
        assert fp.variable_bounds_finite_count == 3
        assert fp.variable_bounds_infinite_count == 0
        assert fp.is_bound_heavy

    def test_not_bound_heavy(self):
        fp = fingerprint_model(_build_pure_lp())
        # x1, x2 have lb=0, ub=inf -> infinite bounds
        assert fp.variable_bounds_finite_count == 0
        assert fp.variable_bounds_infinite_count == 2
        assert not fp.is_bound_heavy


class TestEmptyDegenerate:
    """Empty and degenerate model fingerprinting."""

    def test_empty_model(self):
        fp = fingerprint_model(_build_empty_model())
        assert fp.variable_count == 0
        assert fp.constraint_count == 0
        assert fp.integer_variable_fraction == 0.0
        assert fp.binary_variable_fraction == 0.0
        assert fp.constraint_density == 0.0
        assert fp.coefficient_magnitude_min == math.inf
        assert fp.coefficient_magnitude_max == 0.0
        assert fp.rhs_magnitude_min == math.inf
        assert fp.rhs_magnitude_max == 0.0
        assert not fp.is_binary_heavy
        assert not fp.is_equality_heavy
        assert not fp.is_sparse
        assert not fp.is_dense
        assert not fp.is_bound_heavy

    def test_degenerate_model(self):
        fp = fingerprint_model(_build_degenerate_model())
        assert fp.variable_count == 2
        assert fp.constraint_count == 0
        # x1 is fixed (lb == ub == 2)
        assert fp.variable_bounds_fixed_count == 1
        assert fp.constraint_density == 0.0
        assert fp.coefficient_magnitude_min == math.inf
        assert fp.coefficient_magnitude_max == 0.0

    def test_empty_constraint_set(self):
        m = Model("no_constraints")
        m.add_variable(Variable("x", lower_bound=0.0, variable_type=VariableType.INTEGER))
        m.set_objective(Objective({"x": 1.0}, ObjectiveSense.MAXIMIZE))
        fp = fingerprint_model(m)
        assert fp.constraint_count == 0
        assert fp.constraint_density == 0.0
        assert fp.rhs_magnitude_min == math.inf
        assert fp.rhs_magnitude_max == 0.0
        assert not fp.is_equality_heavy


class TestZeroNegativeCoefficients:
    """Zero and negative coefficient handling."""

    def test_zero_coefficients(self):
        fp = fingerprint_model(_build_zero_coefficient_model())
        # Zero coefficients are excluded from magnitude stats
        assert fp.coefficient_magnitude_min == pytest.approx(1.0)
        assert fp.coefficient_magnitude_max == pytest.approx(1.0)
        assert fp.coefficient_magnitude_mean == pytest.approx(1.0)

    def test_negative_coefficients(self):
        fp = fingerprint_model(_build_negative_coefficient_model())
        # Magnitudes use absolute values: |c1|={1,2}, |c2|={1,1} -> [1,2,1,1]
        assert fp.coefficient_magnitude_min == pytest.approx(1.0)
        assert fp.coefficient_magnitude_max == pytest.approx(2.0)
        assert fp.coefficient_magnitude_mean == pytest.approx(1.25)


class TestInfiniteBounds:
    """Infinite bound handling."""

    def test_infinite_bounds(self):
        fp = fingerprint_model(_build_infinite_bounds_model())
        assert fp.variable_bounds_infinite_count == 2
        assert fp.variable_bounds_finite_count == 0
        assert not fp.is_bound_heavy


class TestObjectiveSense:
    """MAXIMIZE and MINIMIZE fingerprinting."""

    def test_maximize(self):
        fp = fingerprint_model(_build_maximize_model())
        assert fp.objective_sense is ObjectiveSense.MAXIMIZE

    def test_minimize(self):
        fp = fingerprint_model(_build_minimize_model())
        assert fp.objective_sense is ObjectiveSense.MINIMIZE

    def test_no_objective_defaults_to_minimize(self):
        m = Model("no_obj")
        m.add_variable(Variable("x", lower_bound=0.0, variable_type=VariableType.INTEGER))
        fp = fingerprint_model(m)
        assert fp.objective_sense is ObjectiveSense.MINIMIZE


class TestDeterminism:
    """Deterministic repeated fingerprinting."""

    def test_repeated_fingerprinting_identical(self):
        m = _build_mixed_integer()
        fp1 = fingerprint_model(m)
        fp2 = fingerprint_model(m)
        fp3 = fingerprint_model(m)
        assert fp1 == fp2 == fp3

    def test_fingerprint_is_frozen(self):
        fp = fingerprint_model(_build_mixed_integer())
        with pytest.raises(Exception):
            fp.variable_count = 99  # frozen dataclass should reject mutation

    def test_to_dict_deterministic(self):
        m = _build_mixed_integer()
        d1 = fingerprint_model(m).to_dict()
        d2 = fingerprint_model(m).to_dict()
        assert d1 == d2


class TestModelImmutability:
    """Fingerprinting must never mutate the model."""

    def test_model_not_mutated(self):
        m = _build_mixed_integer()
        # Snapshot state before
        before_vars = {name: (v.lower_bound, v.upper_bound, v.variable_type) for name, v in m.variables.items()}
        before_constrs = {
            name: (dict(c.coefficients), c.sense, c.rhs) for name, c in m.constraints.items()
        }
        before_obj = (
            dict(m.objective.coefficients) if m.objective else None,
            m.objective.sense if m.objective else None,
        )
        before_num_vars = m.num_variables
        before_num_constrs = m.num_constraints

        # Fingerprint multiple times
        fingerprint_model(m)
        fingerprint_model(m)
        fingerprint_model(m)

        # Verify state unchanged
        assert m.num_variables == before_num_vars
        assert m.num_constraints == before_num_constrs
        after_vars = {name: (v.lower_bound, v.upper_bound, v.variable_type) for name, v in m.variables.items()}
        after_constrs = {
            name: (dict(c.coefficients), c.sense, c.rhs) for name, c in m.constraints.items()
        }
        after_obj = (
            dict(m.objective.coefficients) if m.objective else None,
            m.objective.sense if m.objective else None,
        )
        assert after_vars == before_vars
        assert after_constrs == before_constrs
        assert after_obj == before_obj

    def test_model_still_solvable_after_fingerprinting(self):
        m = _build_binary_milp()
        fingerprint_model(m)
        sol = solve_milp(m)
        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(8.0, abs=1e-6)


class TestMILPSolverIntegration:
    """MILPSolver fingerprint integration."""

    def test_solver_fingerprint_method(self):
        solver = MILPSolver()
        m = _build_mixed_integer()
        fp = solver.fingerprint(m)
        assert isinstance(fp, ProblemFingerprint)
        assert fp.variable_count == 2
        assert fp.integer_variable_count == 1

    def test_solver_fingerprint_does_not_affect_solve(self):
        solver = MILPSolver()
        m = _build_binary_milp()

        # Fingerprint first
        fp = solver.fingerprint(m)
        assert fp.is_binary_heavy

        # Solve should be unaffected
        sol = solver.solve(m)
        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(8.0, abs=1e-6)

    def test_solver_fingerprint_before_and_after_solve(self):
        solver = MILPSolver()
        m = _build_mixed_integer()

        fp_before = solver.fingerprint(m)
        sol = solver.solve(m)
        fp_after = solver.fingerprint(m)

        assert fp_before == fp_after
        assert sol.status is SolutionStatus.OPTIMAL


class TestFingerprintAPI:
    """Public API surface."""

    def test_fingerprint_model_is_exported(self):
        from optiforge.milp import fingerprint_model as fm
        assert callable(fm)

    def test_problem_fingerprint_is_exported(self):
        from optiforge.milp import ProblemFingerprint as PF
        assert PF is ProblemFingerprint

    def test_fingerprint_is_dataclass(self):
        import dataclasses
        assert dataclasses.is_dataclass(ProblemFingerprint)

    def test_none_model_raises(self):
        with pytest.raises(ValueError):
            fingerprint_model(None)


class TestStructuralIndicators:
    """Structural indicator flags."""

    def test_binary_heavy_threshold(self):
        # 2 binary + 2 continuous = 50% binary -> binary-heavy
        m = Model("half_binary")
        m.add_variable(Variable("b1", variable_type=VariableType.BINARY))
        m.add_variable(Variable("b2", variable_type=VariableType.BINARY))
        m.add_variable(Variable("c1", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        m.add_variable(Variable("c2", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        m.add_constraint(Constraint("c", {"b1": 1.0, "b2": 1.0, "c1": 1.0, "c2": 1.0}, ConstraintSense.LESS_EQUAL, 2.0))
        m.set_objective(Objective({"b1": 1.0, "b2": 1.0, "c1": 1.0, "c2": 1.0}, ObjectiveSense.MAXIMIZE))
        fp = fingerprint_model(m)
        assert fp.binary_variable_fraction == pytest.approx(0.5)
        assert fp.is_binary_heavy

    def test_equality_heavy_threshold(self):
        # 1 equality + 1 inequality = 50% equality -> equality-heavy
        m = Model("half_eq")
        m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
        m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.EQUAL, 5.0))
        m.add_constraint(Constraint("c2", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MINIMIZE))
        fp = fingerprint_model(m)
        assert fp.is_equality_heavy

    def test_bound_heavy_threshold(self):
        # 1 finite + 1 infinite = 50% finite -> bound-heavy
        m = Model("half_bound")
        m.add_variable(Variable("x1", lower_bound=0.0, upper_bound=5.0, variable_type=VariableType.INTEGER))
        m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
        fp = fingerprint_model(m)
        assert fp.variable_bounds_finite_count == 1
        assert fp.variable_bounds_infinite_count == 1
        assert fp.is_bound_heavy


class TestDensityStats:
    """Coefficient and RHS magnitude statistics."""

    def test_coefficient_stats(self):
        m = Model("coeff_stats")
        m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        m.add_constraint(Constraint("c1", {"x1": 2.0, "x2": 4.0}, ConstraintSense.LESS_EQUAL, 10.0))
        m.add_constraint(Constraint("c2", {"x1": 6.0, "x2": 8.0}, ConstraintSense.LESS_EQUAL, 20.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
        fp = fingerprint_model(m)
        assert fp.coefficient_magnitude_min == pytest.approx(2.0)
        assert fp.coefficient_magnitude_max == pytest.approx(8.0)
        assert fp.coefficient_magnitude_mean == pytest.approx(5.0)
        # std of [2,4,6,8] = sqrt(5) ≈ 2.236
        assert fp.coefficient_magnitude_std == pytest.approx(math.sqrt(5.0))

    def test_rhs_stats(self):
        m = Model("rhs_stats")
        m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        m.add_constraint(Constraint("c1", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 3.0))
        m.add_constraint(Constraint("c2", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 7.0))
        m.set_objective(Objective({"x1": 1.0}, ObjectiveSense.MAXIMIZE))
        fp = fingerprint_model(m)
        assert fp.rhs_magnitude_min == pytest.approx(3.0)
        assert fp.rhs_magnitude_max == pytest.approx(7.0)
        assert fp.rhs_magnitude_mean == pytest.approx(5.0)

    def test_negative_rhs_uses_abs(self):
        m = Model("neg_rhs")
        m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        m.add_constraint(Constraint("c1", {"x1": 1.0}, ConstraintSense.GREATER_EQUAL, -5.0))
        m.set_objective(Objective({"x1": 1.0}, ObjectiveSense.MAXIMIZE))
        fp = fingerprint_model(m)
        assert fp.rhs_magnitude_min == pytest.approx(5.0)
        assert fp.rhs_magnitude_max == pytest.approx(5.0)


class TestConstraintCoverage:
    """Constraint-side integrality coverage fingerprinting."""

    def test_integer_constraint_coverage(self):
        m = _build_mixed_integer()
        fp = fingerprint_model(m)
        # 1 constraint, references x2 (integer) -> coverage = 1.0
        assert fp.integer_constraint_coverage == pytest.approx(1.0)
        # No binary variables -> coverage = 0.0
        assert fp.binary_constraint_coverage == pytest.approx(0.0)

    def test_binary_constraint_coverage(self):
        fp = fingerprint_model(_build_binary_milp())
        # 1 constraint, references all 3 binary variables -> coverage = 1.0
        assert fp.binary_constraint_coverage == pytest.approx(1.0)
        assert fp.integer_constraint_coverage == pytest.approx(1.0)

    def test_partial_coverage(self):
        # 2 constraints, only 1 references integer variable
        m = Model("partial_cov")
        m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
        m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
        m.add_constraint(Constraint("c2", {"x2": 1.0}, ConstraintSense.LESS_EQUAL, 3.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
        fp = fingerprint_model(m)
        assert fp.integer_constraint_coverage == pytest.approx(0.5)
        assert fp.binary_constraint_coverage == pytest.approx(0.0)

    def test_zero_coefficient_no_coverage(self):
        # Constraint has integer variable but with zero coefficient -> no coverage
        m = Model("zero_cov")
        m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
        m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        m.add_constraint(Constraint("c1", {"x1": 0.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
        fp = fingerprint_model(m)
        assert fp.integer_constraint_coverage == pytest.approx(0.0)
        assert fp.binary_constraint_coverage == pytest.approx(0.0)

    def test_no_constraints_coverage(self):
        m = Model("no_constr")
        m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
        m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.BINARY))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
        fp = fingerprint_model(m)
        assert fp.integer_constraint_coverage == 0.0
        assert fp.binary_constraint_coverage == 0.0

    def test_all_integer_model_coverage(self):
        m = Model("all_int")
        m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
        m.add_variable(Variable("x2", lower_bound=0.0, variable_type=VariableType.INTEGER))
        m.add_constraint(Constraint("c1", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
        m.add_constraint(Constraint("c2", {"x2": 1.0}, ConstraintSense.LESS_EQUAL, 3.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
        fp = fingerprint_model(m)
        assert fp.integer_constraint_coverage == pytest.approx(1.0)
        assert fp.binary_constraint_coverage == pytest.approx(0.0)

    def test_all_binary_model_coverage(self):
        m = Model("all_bin")
        m.add_variable(Variable("x1", variable_type=VariableType.BINARY))
        m.add_variable(Variable("x2", variable_type=VariableType.BINARY))
        m.add_constraint(Constraint("c1", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 1.0))
        m.add_constraint(Constraint("c2", {"x2": 1.0}, ConstraintSense.LESS_EQUAL, 1.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))
        fp = fingerprint_model(m)
        assert fp.binary_constraint_coverage == pytest.approx(1.0)
        assert fp.integer_constraint_coverage == pytest.approx(1.0)

    def test_mixed_integer_binary_coverage(self):
        m = Model("mixed_cov")
        m.add_variable(Variable("x1", lower_bound=0.0, variable_type=VariableType.INTEGER))
        m.add_variable(Variable("x2", variable_type=VariableType.BINARY))
        m.add_variable(Variable("x3", lower_bound=0.0, variable_type=VariableType.CONTINUOUS))
        m.add_constraint(Constraint("c1", {"x1": 1.0, "x3": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
        m.add_constraint(Constraint("c2", {"x2": 1.0, "x3": 1.0}, ConstraintSense.LESS_EQUAL, 3.0))
        m.set_objective(Objective({"x1": 1.0, "x2": 1.0, "x3": 1.0}, ObjectiveSense.MAXIMIZE))
        fp = fingerprint_model(m)
        # Both constraints have integer variables (c1 has x1, c2 has x2 which is binary=integer)
        assert fp.integer_constraint_coverage == pytest.approx(1.0)
        # Only c2 has binary variable
        assert fp.binary_constraint_coverage == pytest.approx(0.5)