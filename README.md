# OptiForge

A from-scratch mathematical optimization solver for Linear Programming (LP) and Mixed-Integer Linear Programming (MILP) problems.

## Overview

OptiForge is built entirely from first mathematical principles — it does **not** wrap or delegate to any existing optimization solver (Gurobi, CPLEX, HiGHS, GLPK, SCIP, OR-Tools, SciPy, etc.). NumPy is used only for basic numerical arrays and linear-system solves (`np.linalg.solve`).

## Project Structure

```
OptiForge/
├── optiforge/          # Core package
│   ├── model/          # Variables, constraints, objectives, model container
│   ├── core/           # Solution representation and solve status
│   ├── numerics/       # Numerical utilities and linear algebra core
│   ├── lp/             # LP standard-form converter and Revised Simplex solver
│   ├── milp/           # MILP solver (future)
│   ├── presolve/       # Presolve routines (future)
│   ├── adaptive/       # Adaptive MILP framework (future)
│   └── io/             # Serialization (JSON)
├── examples/           # Example scripts
├── tests/              # Comprehensive test suite (92 tests)
└── benchmarks/         # Performance benchmarks (future)
```

## Current Status

* **Stage 1 — Model Layer**:
  - `Variable`, `Constraint`, `Objective`, and `Model` classes.
  - `Solution` dataclass and `SolutionStatus` enumeration.
  - JSON serialization and deserialization (`model_to_json`, `model_from_json`).

* **Stage 2 & 2.5 — Standard Form & Phase-I Canonicalization**:
  - Exact standard-form conversion: $\min/\max \mathbf{c}^T \mathbf{x}$ s.t. $\mathbf{A}\mathbf{x} = \mathbf{b}, \mathbf{x} \ge \mathbf{0}$.
  - Negative RHS normalization: multiplies rows by $-1$ and flips sense so $\mathbf{b} \ge \mathbf{0}$.
  - Slack, surplus, and artificial variable allocation.
  - Auxiliary Phase-I objective vector ($\min \sum a_i$) and guaranteed $m$-column initial basis matrix $\mathbf{B} = \mathbf{I}_m$.
  - Singular basis detection via `SingularBasisError`.

* **Stage 3 — Baseline Revised Simplex LP Solver**:
  - Two-phase Revised Simplex implementation with transparent matrix operations:
    1. **Basis & BFS**: Solves $\mathbf{B} \mathbf{x}_B = \mathbf{b}$ via `np.linalg.solve`.
    2. **Simplex Multipliers**: Solves $\mathbf{B}^T \mathbf{y} = \mathbf{c}_B$ for dual multipliers.
    3. **Reduced Costs**: $\bar{c}_j = c_j - \mathbf{A}_j^T \mathbf{y}$.
    4. **Bland's Anti-Cycling Rule**: Smallest-index selection for entering variable ($\bar{c}_j < -\text{tol}$ for min, $\bar{c}_j > \text{tol}$ for max) and smallest-index tie-breaking for leaving variable in the ratio test.
    5. **Search Direction**: Solves $\mathbf{B} \mathbf{d} = \mathbf{A}_j$ for entering column.
    6. **Ratio Test**: Minimum ratio test over rows with $d_i > \text{pivot\_tol}$. Detects unboundedness if no $d_i > \text{pivot\_tol}$.
    7. **Phase-I $\to$ Phase-II Transition**: Drives out zero-valued basic artificial variables or eliminates redundant rows before transitioning to Phase II.
    8. **Numerical Tolerances**: Configurable via `SimplexTolerances`.

## Quick Start

```bash
pip install -r requirements.txt
python examples/lp/simplex_example.py
pytest tests/ -v
```

### Python API Example

```python
from optiforge import Model, Variable, Constraint, ConstraintSense, Objective, ObjectiveSense

# Build an LP
m = Model(name="production")
x1 = m.add_variable(Variable("x1", lower_bound=0.0))
x2 = m.add_variable(Variable("x2", lower_bound=0.0))

m.add_constraint(Constraint("capacity", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
m.add_constraint(Constraint("labor", {"x1": 2.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))
m.set_objective(Objective({"x1": 3.0, "x2": 2.0}, ObjectiveSense.MAXIMIZE))

# Solve with Revised Simplex
sol = m.solve()

print("Status:", sol.status.value)
print("Objective:", sol.objective_value)
print("Solution:", sol.variable_values)
# Status: optimal
# Objective: 9.0
# Solution: {'x1': 1.0, 'x2': 3.0}
```

## High-Level Pipeline

```
Model Manager → Presolve → Problem Fingerprint → LP/MILP Solver
→ Search-State Monitor → Adaptive Strategy
→ Branching / Node Selection / Heuristics → Numerical Core → Solution
```

## License

MIT
