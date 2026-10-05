"""Standard-form conversion and Phase-I preparation example.

Demonstrates Stage 2.5 features:
1. Canonical Example 1 (All-<= constraints with identity slack basis)
2. Example 2 (>= constraints with surplus and artificial variables for Phase I)
3. Example 3 (Equality constraint with Phase-I basis)
"""

import numpy as np

from optiforge.model import (
    Model,
    Variable,
    VariableType,
    Constraint,
    ConstraintSense,
    Objective,
    ObjectiveSense,
)
from optiforge.lp import to_standard_form


def demo_example_1() -> None:
    print("=" * 60)
    print("EXAMPLE 1: Canonical LP (All <= constraints)")
    print("maximize 3·x1 + 2·x2 s.t. x1 + x2 <= 4, 2·x1 + x2 <= 5")
    print("=" * 60)

    m = Model(name="example_1")
    m.add_variable(Variable(name="x1", lower_bound=0.0))
    m.add_variable(Variable(name="x2", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.add_constraint(Constraint("c2", {"x1": 2.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.set_objective(Objective({"x1": 3.0, "x2": 2.0}, ObjectiveSense.MAXIMIZE))

    sf = to_standard_form(m)
    print(sf.summary())
    print("\nA matrix:\n", sf.A)
    print("b vector: ", sf.b)
    print("c vector: ", sf.c)
    print(f"Phase-I basis indices: {sf.phase1_basis_indices} ({[sf.variable_names[i] for i in sf.phase1_basis_indices]})")

    bfs = sf.compute_initial_bfs()
    print("Initial BFS:", bfs)
    print("Initial Objective Value:", sf.compute_objective_value(bfs))
    print()


def demo_example_2() -> None:
    print("=" * 60)
    print("EXAMPLE 2: Surplus & Artificial Variables (Phase I)")
    print("minimize x1 + 2·x2 s.t. x1 + x2 >= 3, x1 + 3·x2 >= 6")
    print("=" * 60)

    m = Model(name="example_2")
    m.add_variable(Variable(name="x1", lower_bound=0.0))
    m.add_variable(Variable(name="x2", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, 3.0))
    m.add_constraint(Constraint("c2", {"x1": 1.0, "x2": 3.0}, ConstraintSense.GREATER_EQUAL, 6.0))
    m.set_objective(Objective({"x1": 1.0, "x2": 2.0}, ObjectiveSense.MINIMIZE))

    sf = to_standard_form(m)
    print(sf.summary())
    print("\nA matrix:\n", sf.A)
    print("b vector:        ", sf.b)
    print("Original c:      ", sf.c)
    print("Phase-I c:       ", sf.phase1_c)
    print(f"Phase-I basis:    {sf.phase1_basis_indices} ({[sf.variable_names[i] for i in sf.phase1_basis_indices]})")

    bfs = sf.compute_phase1_bfs()
    print("Phase-I initial BFS:", bfs)
    print("Phase-I auxiliary obj (sum of artificials):", sf.compute_phase1_objective_value(bfs))
    print()


def demo_example_3() -> None:
    print("=" * 60)
    print("EXAMPLE 3: Mixed Equality & <= Constraint")
    print("maximize 2·x1 + x2 s.t. x1 + x2 == 4, x1 <= 3")
    print("=" * 60)

    m = Model(name="example_3")
    m.add_variable(Variable(name="x1", lower_bound=0.0))
    m.add_variable(Variable(name="x2", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.EQUAL, 4.0))
    m.add_constraint(Constraint("c2", {"x1": 1.0}, ConstraintSense.LESS_EQUAL, 3.0))
    m.set_objective(Objective({"x1": 2.0, "x2": 1.0}, ObjectiveSense.MAXIMIZE))

    sf = to_standard_form(m)
    print(sf.summary())
    print("\nA matrix:\n", sf.A)
    print("b vector:        ", sf.b)
    print("Original c:      ", sf.c)
    print("Phase-I c:       ", sf.phase1_c)
    print(f"Phase-I basis:    {sf.phase1_basis_indices} ({[sf.variable_names[i] for i in sf.phase1_basis_indices]})")

    bfs = sf.compute_phase1_bfs()
    print("Phase-I initial BFS:", bfs)
    print("Phase-I auxiliary obj:", sf.compute_phase1_objective_value(bfs))
    print()


def main() -> None:
    demo_example_1()
    demo_example_2()
    demo_example_3()
    print("All standard-form demonstrations executed successfully.")


if __name__ == "__main__":
    main()
