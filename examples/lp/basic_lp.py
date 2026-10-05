"""Basic LP example — model construction only (no solver yet).

maximize  3·x1 + 2·x2
subject to
    x1 +  x2 ≤ 4
   2·x1 + x2 ≤ 5
    x1 ≥ 0,  x2 ≥ 0
"""

from optiforge.model import (
    Model,
    Variable,
    VariableType,
    Constraint,
    ConstraintSense,
    Objective,
    ObjectiveSense,
)
from optiforge.io import model_to_json


def main() -> None:
    # ── Build the model ──────────────────────────────────────────────
    model = Model(name="basic_lp")

    x1 = model.add_variable(
        Variable(name="x1", lower_bound=0.0, variable_type=VariableType.CONTINUOUS)
    )
    x2 = model.add_variable(
        Variable(name="x2", lower_bound=0.0, variable_type=VariableType.CONTINUOUS)
    )

    model.add_constraint(
        Constraint(
            name="c1",
            coefficients={"x1": 1.0, "x2": 1.0},
            sense=ConstraintSense.LESS_EQUAL,
            rhs=4.0,
        )
    )
    model.add_constraint(
        Constraint(
            name="c2",
            coefficients={"x1": 2.0, "x2": 1.0},
            sense=ConstraintSense.LESS_EQUAL,
            rhs=5.0,
        )
    )

    model.set_objective(
        Objective(
            coefficients={"x1": 3.0, "x2": 2.0},
            sense=ObjectiveSense.MAXIMIZE,
        )
    )

    # ── Print everything ─────────────────────────────────────────────
    print("=" * 50)
    print(model.summary())
    print("=" * 50)

    print("\nVariables:")
    for var in model.variables.values():
        print(f"  {var}")

    print("\nConstraints:")
    for con in model.constraints.values():
        print(f"  {con}")

    print(f"\nObjective:\n  {model.objective}")

    print("\nJSON representation:")
    print(model_to_json(model))

    print("\n(Solver not implemented yet — model construction complete.)")


if __name__ == "__main__":
    main()
