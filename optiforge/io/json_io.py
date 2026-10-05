"""JSON serialization / deserialization for OptiForge models."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from optiforge.model.variable import Variable, VariableType
from optiforge.model.constraint import Constraint, ConstraintSense
from optiforge.model.objective import Objective, ObjectiveSense
from optiforge.model.model import Model


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _serialize_variable(var: Variable) -> dict[str, Any]:
    return {
        "name": var.name,
        "lower_bound": var.lower_bound if math.isfinite(var.lower_bound) else str(var.lower_bound),
        "upper_bound": var.upper_bound if math.isfinite(var.upper_bound) else str(var.upper_bound),
        "variable_type": var.variable_type.value,
    }


def _serialize_constraint(con: Constraint) -> dict[str, Any]:
    return {
        "name": con.name,
        "coefficients": con.coefficients,
        "sense": con.sense.value,
        "rhs": con.rhs,
    }


def _serialize_objective(obj: Objective) -> dict[str, Any]:
    return {
        "coefficients": obj.coefficients,
        "sense": obj.sense.value,
    }


def model_to_dict(model: Model) -> dict[str, Any]:
    """Convert a Model to a plain dictionary suitable for JSON encoding."""
    data: dict[str, Any] = {
        "name": model.name,
        "variables": [_serialize_variable(v) for v in model.variables.values()],
        "constraints": [_serialize_constraint(c) for c in model.constraints.values()],
    }
    if model.objective is not None:
        data["objective"] = _serialize_objective(model.objective)
    return data


def model_to_json(model: Model, *, indent: int = 2) -> str:
    """Serialize a Model to a JSON string."""
    return json.dumps(model_to_dict(model), indent=indent)


def model_to_file(model: Model, path: str | Path, *, indent: int = 2) -> None:
    """Write a Model to a JSON file."""
    Path(path).write_text(model_to_json(model, indent=indent), encoding="utf-8")


# ---------------------------------------------------------------------------
# Deserialization
# ---------------------------------------------------------------------------

_BOUND_SPECIALS = {"inf": math.inf, "-inf": -math.inf}


def _parse_bound(value: float | str) -> float:
    if isinstance(value, str):
        return _BOUND_SPECIALS.get(value, float(value))
    return float(value)


def _deserialize_variable(d: dict[str, Any]) -> Variable:
    return Variable(
        name=d["name"],
        lower_bound=_parse_bound(d["lower_bound"]),
        upper_bound=_parse_bound(d["upper_bound"]),
        variable_type=VariableType(d["variable_type"]),
    )


def _deserialize_constraint(d: dict[str, Any]) -> Constraint:
    return Constraint(
        name=d["name"],
        coefficients=d["coefficients"],
        sense=ConstraintSense(d["sense"]),
        rhs=d["rhs"],
    )


def _deserialize_objective(d: dict[str, Any]) -> Objective:
    return Objective(
        coefficients=d["coefficients"],
        sense=ObjectiveSense(d["sense"]),
    )


def model_from_dict(data: dict[str, Any]) -> Model:
    """Reconstruct a Model from a plain dictionary (e.g. parsed JSON)."""
    model = Model(name=data.get("name", "model"))

    for var_data in data.get("variables", []):
        model.add_variable(_deserialize_variable(var_data))

    for con_data in data.get("constraints", []):
        model.add_constraint(_deserialize_constraint(con_data))

    if "objective" in data:
        model.set_objective(_deserialize_objective(data["objective"]))

    return model


def model_from_json(text: str) -> Model:
    """Deserialize a Model from a JSON string."""
    return model_from_dict(json.loads(text))


def model_from_file(path: str | Path) -> Model:
    """Read a Model from a JSON file."""
    return model_from_json(Path(path).read_text(encoding="utf-8"))
