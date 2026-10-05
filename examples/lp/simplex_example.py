"""Revised Simplex solver example.

Demonstrates solving:
1. A canonical 2-variable maximization problem with Phase II.
2. A minimization problem requiring Phase I to find an initial BFS.
"""

from optiforge.model import (
    Constraint,
    ConstraintSense,
    Model,
    Objective,
    ObjectiveSense,
    Variable,
)
from optiforge.lp.simplex import RevisedSimplexSolver


def demo_canonical_max() -> None:
    print("=" * 60)
    print("DEMO 1: Canonical Maximization LP")
    print("maximize 3·x1 + 2·x2")
    print("subject to")
    print("    x1 +  x2 <= 4")
    print("   2·x1 + x2 <= 5")
    print("    x1, x2 >= 0")
    print("=" * 60)

    m = Model(name="canonical_max")
    m.add_variable(Variable("x1", lower_bound=0.0))
    m.add_variable(Variable("x2", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
    m.add_constraint(Constraint("c2", {"x1": 2.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
    m.set_objective(Objective({"x1": 3.0, "x2": 2.0}, ObjectiveSense.MAXIMIZE))

    sol = m.solve()
    print("Solve Status:    ", sol.status.value)
    print("Iterations:      ", sol.iterations)
    print("Objective Value: ", sol.objective_value)
    print("Variable Values: ", sol.variable_values)
    print("Message:         ", sol.message)
    print()


def demo_phase1_min() -> None:
    print("=" * 60)
    print("DEMO 2: Minimization LP with >= Constraints (Phase I + Phase II)")
    print("minimize x1 + 2·x2")
    print("subject to")
    print("    x1 +  x2 >= 3")
    print("    x1 + 3·x2 >= 6")
    print("    x1, x2 >= 0")
    print("=" * 60)

    m = Model(name="phase1_min")
    m.add_variable(Variable("x1", lower_bound=0.0))
    m.add_variable(Variable("x2", lower_bound=0.0))
    m.add_constraint(Constraint("c1", {"x1": 1.0, "x2": 1.0}, ConstraintSense.GREATER_EQUAL, 3.0))
    m.add_constraint(Constraint("c2", {"x1": 1.0, "x2": 3.0}, ConstraintSense.GREATER_EQUAL, 6.0))
    m.set_objective(Objective({"x1": 1.0, "x2": 2.0}, ObjectiveSense.MINIMIZE))

    solver = RevisedSimplexSolver()
    sol = solver.solve(m)
    print("Solve Status:    ", sol.status.value)
    print("Iterations:      ", sol.iterations)
    print("Objective Value: ", sol.objective_value)
    print("Variable Values: ", sol.variable_values)
    print("Message:         ", sol.message)
    print()


def main() -> None:
    demo_canonical_max()
    demo_phase1_min()


if __name__ == "__main__":
    main()
