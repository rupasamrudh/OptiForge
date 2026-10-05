"""Unit tests for OptiForge model layer and JSON I/O."""

import math
import pytest

from optiforge.model import (
    Variable,
    VariableType,
    Constraint,
    ConstraintSense,
    Objective,
    ObjectiveSense,
    Model,
)
from optiforge.core import Solution, SolutionStatus
from optiforge.io import model_to_json, model_from_json


# ── Variable tests ──────────────────────────────────────────────────────────


class TestVariable:
    def test_continuous_variable(self):
        v = Variable(name="x", lower_bound=0.0, upper_bound=10.0)
        assert v.name == "x"
        assert v.lower_bound == 0.0
        assert v.upper_bound == 10.0
        assert v.variable_type is VariableType.CONTINUOUS
        assert not v.is_binary()
        assert not v.is_integer()

    def test_integer_variable(self):
        v = Variable(name="y", lower_bound=0.0, upper_bound=5.0, variable_type=VariableType.INTEGER)
        assert v.variable_type is VariableType.INTEGER
        assert v.is_integer()
        assert not v.is_binary()

    def test_binary_variable(self):
        v = Variable(name="b", variable_type=VariableType.BINARY)
        assert v.variable_type is VariableType.BINARY
        assert v.lower_bound == 0.0
        assert v.upper_bound == 1.0
        assert v.is_binary()
        assert v.is_integer()

    def test_binary_bounds_enforced(self):
        """Binary variables must have bounds [0, 1] regardless of what the user passes."""
        v = Variable(name="b", lower_bound=-5.0, upper_bound=99.0, variable_type=VariableType.BINARY)
        assert v.lower_bound == 0.0
        assert v.upper_bound == 1.0

    def test_invalid_bounds(self):
        with pytest.raises(ValueError, match="lower_bound"):
            Variable(name="bad", lower_bound=10.0, upper_bound=5.0)

    def test_empty_name(self):
        with pytest.raises(ValueError, match="non-empty"):
            Variable(name="")

    def test_default_bounds(self):
        v = Variable(name="x")
        assert v.lower_bound == 0.0
        assert v.upper_bound == math.inf


# ── Constraint tests ────────────────────────────────────────────────────────


class TestConstraint:
    def test_constraint_creation(self):
        c = Constraint(
            name="c1",
            coefficients={"x": 1.0, "y": 2.0},
            sense=ConstraintSense.LESS_EQUAL,
            rhs=10.0,
        )
        assert c.name == "c1"
        assert c.coefficients == {"x": 1.0, "y": 2.0}
        assert c.sense is ConstraintSense.LESS_EQUAL
        assert c.rhs == 10.0

    def test_constraint_senses(self):
        for sense in ConstraintSense:
            c = Constraint(name=f"c_{sense.name}", coefficients={"x": 1.0}, sense=sense, rhs=0.0)
            assert c.sense is sense

    def test_empty_coefficients(self):
        with pytest.raises(ValueError, match="coefficients"):
            Constraint(name="bad", coefficients={}, sense=ConstraintSense.EQUAL, rhs=0.0)

    def test_empty_name(self):
        with pytest.raises(ValueError, match="non-empty"):
            Constraint(name="", coefficients={"x": 1.0}, sense=ConstraintSense.EQUAL, rhs=0.0)


# ── Objective tests ─────────────────────────────────────────────────────────


class TestObjective:
    def test_objective_creation(self):
        obj = Objective(coefficients={"x": 3.0, "y": 2.0}, sense=ObjectiveSense.MAXIMIZE)
        assert obj.coefficients == {"x": 3.0, "y": 2.0}
        assert obj.sense is ObjectiveSense.MAXIMIZE

    def test_minimize(self):
        obj = Objective(coefficients={"x": 1.0}, sense=ObjectiveSense.MINIMIZE)
        assert obj.sense is ObjectiveSense.MINIMIZE

    def test_empty_coefficients(self):
        with pytest.raises(ValueError, match="coefficients"):
            Objective(coefficients={}, sense=ObjectiveSense.MINIMIZE)


# ── Model tests ─────────────────────────────────────────────────────────────


class TestModel:
    def _build_simple_model(self) -> Model:
        """Helper: build a small 2-variable LP."""
        m = Model(name="test")
        m.add_variable(Variable(name="x1", lower_bound=0.0))
        m.add_variable(Variable(name="x2", lower_bound=0.0))
        m.add_constraint(
            Constraint(name="c1", coefficients={"x1": 1.0, "x2": 1.0}, sense=ConstraintSense.LESS_EQUAL, rhs=4.0)
        )
        m.set_objective(Objective(coefficients={"x1": 3.0, "x2": 2.0}, sense=ObjectiveSense.MAXIMIZE))
        return m

    def test_model_creation(self):
        m = Model(name="test_model")
        assert m.name == "test_model"
        assert m.num_variables == 0
        assert m.num_constraints == 0
        assert m.objective is None

    def test_add_variable(self):
        m = Model()
        v = m.add_variable(Variable(name="x"))
        assert m.num_variables == 1
        assert "x" in m.variables
        assert v.name == "x"

    def test_duplicate_variable(self):
        m = Model()
        m.add_variable(Variable(name="x"))
        with pytest.raises(ValueError, match="already exists"):
            m.add_variable(Variable(name="x"))

    def test_add_constraint(self):
        m = Model()
        m.add_variable(Variable(name="x"))
        c = m.add_constraint(
            Constraint(name="c1", coefficients={"x": 1.0}, sense=ConstraintSense.LESS_EQUAL, rhs=5.0)
        )
        assert m.num_constraints == 1
        assert c.name == "c1"

    def test_duplicate_constraint(self):
        m = Model()
        m.add_variable(Variable(name="x"))
        m.add_constraint(Constraint(name="c1", coefficients={"x": 1.0}, sense=ConstraintSense.EQUAL, rhs=0.0))
        with pytest.raises(ValueError, match="already exists"):
            m.add_constraint(Constraint(name="c1", coefficients={"x": 1.0}, sense=ConstraintSense.EQUAL, rhs=0.0))

    def test_constraint_unknown_variable(self):
        m = Model()
        m.add_variable(Variable(name="x"))
        with pytest.raises(ValueError, match="Unknown variable"):
            m.add_constraint(
                Constraint(name="c1", coefficients={"z": 1.0}, sense=ConstraintSense.EQUAL, rhs=0.0)
            )

    def test_set_objective(self):
        m = self._build_simple_model()
        assert m.objective is not None
        assert m.objective.sense is ObjectiveSense.MAXIMIZE

    def test_objective_unknown_variable(self):
        m = Model()
        m.add_variable(Variable(name="x"))
        with pytest.raises(ValueError, match="Unknown variable"):
            m.set_objective(Objective(coefficients={"z": 1.0}, sense=ObjectiveSense.MINIMIZE))

    def test_summary(self):
        m = self._build_simple_model()
        s = m.summary()
        assert "test" in s
        assert "2" in s  # 2 variables


# ── Solution tests ──────────────────────────────────────────────────────────


class TestSolution:
    def test_default_solution(self):
        sol = Solution()
        assert sol.status is SolutionStatus.UNKNOWN
        assert sol.variable_values == {}
        assert sol.objective_value is None

    def test_custom_solution(self):
        sol = Solution(
            status=SolutionStatus.OPTIMAL,
            variable_values={"x": 1.0, "y": 2.0},
            objective_value=7.0,
            message="Optimal solution found.",
        )
        assert sol.status is SolutionStatus.OPTIMAL
        assert sol.objective_value == 7.0


# ── JSON I/O tests ──────────────────────────────────────────────────────────


class TestJsonIO:
    def _build_model(self) -> Model:
        m = Model(name="json_test")
        m.add_variable(Variable(name="x1", lower_bound=0.0, upper_bound=10.0))
        m.add_variable(Variable(name="x2", lower_bound=0.0))
        m.add_variable(Variable(name="b", variable_type=VariableType.BINARY))
        m.add_constraint(
            Constraint(name="c1", coefficients={"x1": 1.0, "x2": 2.0}, sense=ConstraintSense.LESS_EQUAL, rhs=8.0)
        )
        m.set_objective(Objective(coefficients={"x1": 3.0, "x2": 1.0}, sense=ObjectiveSense.MAXIMIZE))
        return m

    def test_roundtrip(self):
        original = self._build_model()
        json_str = model_to_json(original)
        restored = model_from_json(json_str)

        assert restored.name == original.name
        assert restored.num_variables == original.num_variables
        assert restored.num_constraints == original.num_constraints

        # Variables match
        for name, orig_var in original.variables.items():
            rest_var = restored.variables[name]
            assert rest_var.name == orig_var.name
            assert rest_var.variable_type == orig_var.variable_type
            assert rest_var.lower_bound == orig_var.lower_bound
            assert rest_var.upper_bound == orig_var.upper_bound

        # Objective matches
        assert restored.objective is not None
        assert restored.objective.sense == original.objective.sense
        assert restored.objective.coefficients == original.objective.coefficients

    def test_json_contains_expected_keys(self):
        m = self._build_model()
        json_str = model_to_json(m)
        import json
        data = json.loads(json_str)
        assert "name" in data
        assert "variables" in data
        assert "constraints" in data
        assert "objective" in data

    def test_infinity_roundtrip(self):
        """Variables with infinite bounds survive serialization."""
        m = Model(name="inf_test")
        m.add_variable(Variable(name="x", lower_bound=0.0))  # upper = inf
        json_str = model_to_json(m)
        restored = model_from_json(json_str)
        assert restored.variables["x"].upper_bound == math.inf
