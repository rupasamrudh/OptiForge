"""Stage 8.15: Workload generation for structurally diverse LP and MILP benchmarks.

Generates deterministic, reproducible synthetic structured workloads covering
25 structural classes (15 LP + 10 MILP) across Small/Medium/Large size buckets.

All generated problems use fixed random seeds for full reproducibility.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from optiforge.model import (
    Constraint, ConstraintSense, Model, Objective, ObjectiveSense,
    Variable, VariableType,
)


class WorkloadClass(Enum):
    """Structural classification of benchmark workloads."""
    # LP classes
    SMALL_DENSE_LP = "small_dense_lp"
    MEDIUM_DENSE_LP = "medium_dense_lp"
    LARGE_DENSE_LP = "large_dense_lp"
    SPARSE_LP = "sparse_lp"
    HIGHLY_SPARSE_LP = "highly_sparse_lp"
    EQUALITY_HEAVY_LP = "equality_heavy_lp"
    INEQUALITY_HEAVY_LP = "inequality_heavy_lp"
    MIXED_SENSE_LP = "mixed_sense_lp"
    DEGENERATE_LP = "degenerate_lp"
    ILL_CONDITIONED_LP = "ill_conditioned_lp"
    PHASE1_HEAVY_LP = "phase1_heavy_lp"
    LONG_PIVOT_LP = "long_pivot_lp"
    SHORT_PIVOT_LP = "short_pivot_lp"
    BOUND_HEAVY_LP = "bound_heavy_lp"
    REDUNDANT_CONSTRAINT_LP = "redundant_constraint_lp"
    # MILP classes
    KNAPSACK = "knapsack"
    SET_COVERING = "set_covering"
    BINARY_SELECTION = "binary_selection"
    MIXED_INT_RESOURCE = "mixed_int_resource"
    INTEGER_ALLOCATION = "integer_allocation"
    BRANCHING_HEAVY_MILP = "branching_heavy_milp"
    MULTILEVEL_BB_MILP = "multilevel_bb_milp"
    INFEASIBLE_CHILDREN_MILP = "infeasible_children_milp"
    STRONG_RELAXATION_MILP = "strong_relaxation_milp"
    WEAK_RELAXATION_MILP = "weak_relaxation_milp"


class SizeBucket(Enum):
    SMALL = "small"        # m <= 10
    MEDIUM = "medium"      # 10 < m <= 50
    LARGE = "large"        # 50 < m <= 150


@dataclass
class WorkloadSpec:
    """Metadata describing a generated workload."""
    name: str
    workload_class: WorkloadClass
    size_bucket: SizeBucket
    model: Model
    seed: int
    m: int  # constraints
    n: int  # decision variables
    n_integer: int = 0
    n_binary: int = 0
    density: float = 1.0
    constraint_types: str = ""
    objective_sense: str = "MAXIMIZE"
    expected_pivots: str = "unknown"


# ---------------------------------------------------------------------------
# LP Generators
# ---------------------------------------------------------------------------

def _dense_lp(m: int, n: int, seed: int, name: str) -> Model:
    """Dense LP: max c^T x s.t. Ax <= b, x >= 0."""
    rng = np.random.RandomState(seed)
    model = Model(name)
    vnames = [f"x{j}" for j in range(n)]
    for vn in vnames:
        model.add_variable(Variable(vn, lower_bound=0.0))
    A = rng.uniform(0.5, 10.0, (m, n))
    b = A @ np.ones(n) * rng.uniform(0.5, 1.5, m)
    for i in range(m):
        coeffs = {vnames[j]: float(A[i, j]) for j in range(n)}
        model.add_constraint(Constraint(f"c{i}", coeffs, ConstraintSense.LESS_EQUAL, float(b[i])))
    c = rng.uniform(1.0, 10.0, n)
    model.set_objective(Objective({vnames[j]: float(c[j]) for j in range(n)}, ObjectiveSense.MAXIMIZE))
    return model


def _sparse_lp(m: int, n: int, density: float, seed: int, name: str) -> Model:
    """Sparse LP with controlled density."""
    rng = np.random.RandomState(seed)
    model = Model(name)
    vnames = [f"x{j}" for j in range(n)]
    for vn in vnames:
        model.add_variable(Variable(vn, lower_bound=0.0, upper_bound=100.0))
    for i in range(m):
        coeffs = {}
        for j in range(n):
            if rng.rand() < density:
                coeffs[vnames[j]] = float(rng.uniform(0.5, 5.0))
        if not coeffs:
            j = rng.randint(n)
            coeffs[vnames[j]] = float(rng.uniform(1.0, 5.0))
        rhs = float(rng.uniform(10.0, 50.0))
        model.add_constraint(Constraint(f"c{i}", coeffs, ConstraintSense.LESS_EQUAL, rhs))
    c = rng.uniform(1.0, 10.0, n)
    model.set_objective(Objective({vnames[j]: float(c[j]) for j in range(n)}, ObjectiveSense.MAXIMIZE))
    return model


def _equality_heavy_lp(m: int, n: int, seed: int, name: str) -> Model:
    """LP with mostly equality constraints (Phase-I required)."""
    rng = np.random.RandomState(seed)
    model = Model(name)
    vnames = [f"x{j}" for j in range(n)]
    for vn in vnames:
        model.add_variable(Variable(vn, lower_bound=0.0))
    n_eq = max(1, int(0.8 * m))
    for i in range(m):
        coeffs = {vnames[j]: float(rng.uniform(0.5, 5.0)) for j in range(n)}
        rhs = float(rng.uniform(5.0, 20.0))
        sense = ConstraintSense.EQUAL if i < n_eq else ConstraintSense.LESS_EQUAL
        model.add_constraint(Constraint(f"c{i}", coeffs, sense, rhs))
    c = rng.uniform(1.0, 5.0, n)
    model.set_objective(Objective({vnames[j]: float(c[j]) for j in range(n)}, ObjectiveSense.MINIMIZE))
    return model


def _mixed_sense_lp(m: int, n: int, seed: int, name: str) -> Model:
    """LP with mixed <=, >=, == constraints."""
    rng = np.random.RandomState(seed)
    model = Model(name)
    vnames = [f"x{j}" for j in range(n)]
    for vn in vnames:
        model.add_variable(Variable(vn, lower_bound=0.0))
    senses = [ConstraintSense.LESS_EQUAL, ConstraintSense.GREATER_EQUAL, ConstraintSense.EQUAL]
    for i in range(m):
        coeffs = {vnames[j]: float(rng.uniform(0.5, 5.0)) for j in range(n)}
        rhs = float(rng.uniform(5.0, 30.0))
        sense = senses[i % 3]
        model.add_constraint(Constraint(f"c{i}", coeffs, sense, rhs))
    c = rng.uniform(1.0, 10.0, n)
    model.set_objective(Objective({vnames[j]: float(c[j]) for j in range(n)}, ObjectiveSense.MAXIMIZE))
    return model


def _degenerate_lp(m: int, n: int, seed: int, name: str) -> Model:
    """Degenerate LP: many constraints active at optimum."""
    rng = np.random.RandomState(seed)
    model = Model(name)
    vnames = [f"x{j}" for j in range(n)]
    for vn in vnames:
        model.add_variable(Variable(vn, lower_bound=0.0))
    # Create tight constraints around a known feasible point
    x_star = rng.uniform(1.0, 5.0, n)
    for i in range(m):
        a = rng.uniform(0.5, 3.0, n)
        rhs = float(np.dot(a, x_star) + rng.uniform(0.0, 0.1))  # very tight
        coeffs = {vnames[j]: float(a[j]) for j in range(n)}
        model.add_constraint(Constraint(f"c{i}", coeffs, ConstraintSense.LESS_EQUAL, rhs))
    c = rng.uniform(1.0, 5.0, n)
    model.set_objective(Objective({vnames[j]: float(c[j]) for j in range(n)}, ObjectiveSense.MAXIMIZE))
    return model


def _ill_conditioned_lp(m: int, n: int, seed: int, name: str) -> Model:
    """LP with ill-conditioned constraint matrix."""
    rng = np.random.RandomState(seed)
    model = Model(name)
    vnames = [f"x{j}" for j in range(n)]
    for vn in vnames:
        model.add_variable(Variable(vn, lower_bound=0.0, upper_bound=100.0))
    # Scale rows by exponentially different factors
    for i in range(m):
        scale = 10.0 ** (rng.uniform(-3, 3))
        coeffs = {vnames[j]: float(scale * rng.uniform(0.5, 5.0)) for j in range(n)}
        rhs = float(scale * rng.uniform(5.0, 30.0))
        model.add_constraint(Constraint(f"c{i}", coeffs, ConstraintSense.LESS_EQUAL, rhs))
    c = rng.uniform(1.0, 10.0, n)
    model.set_objective(Objective({vnames[j]: float(c[j]) for j in range(n)}, ObjectiveSense.MAXIMIZE))
    return model


def _bound_heavy_lp(m: int, n: int, seed: int, name: str) -> Model:
    """LP where most constraints are simple bounds."""
    rng = np.random.RandomState(seed)
    model = Model(name)
    vnames = [f"x{j}" for j in range(n)]
    for vn in vnames:
        lb = float(rng.uniform(0.0, 2.0))
        ub = float(rng.uniform(5.0, 20.0))
        model.add_variable(Variable(vn, lower_bound=lb, upper_bound=ub))
    # Only a few linking constraints
    n_link = max(1, m // 3)
    for i in range(n_link):
        coeffs = {vnames[j]: float(rng.uniform(0.5, 3.0)) for j in range(n)}
        rhs = float(rng.uniform(10.0, 50.0))
        model.add_constraint(Constraint(f"c{i}", coeffs, ConstraintSense.LESS_EQUAL, rhs))
    # Remaining as simple bound constraints
    for i in range(n_link, m):
        j = i % n
        coeffs = {vnames[j]: 1.0}
        rhs = float(rng.uniform(3.0, 15.0))
        model.add_constraint(Constraint(f"c{i}", coeffs, ConstraintSense.LESS_EQUAL, rhs))
    c = rng.uniform(1.0, 10.0, n)
    model.set_objective(Objective({vnames[j]: float(c[j]) for j in range(n)}, ObjectiveSense.MAXIMIZE))
    return model


def _redundant_constraint_lp(m: int, n: int, seed: int, name: str) -> Model:
    """LP with redundant constraints (some are dominated)."""
    rng = np.random.RandomState(seed)
    model = Model(name)
    vnames = [f"x{j}" for j in range(n)]
    for vn in vnames:
        model.add_variable(Variable(vn, lower_bound=0.0, upper_bound=50.0))
    n_core = max(2, m // 2)
    core_A = rng.uniform(0.5, 5.0, (n_core, n))
    core_b = rng.uniform(10.0, 30.0, n_core)
    for i in range(n_core):
        coeffs = {vnames[j]: float(core_A[i, j]) for j in range(n)}
        model.add_constraint(Constraint(f"c{i}", coeffs, ConstraintSense.LESS_EQUAL, float(core_b[i])))
    # Add redundant constraints (same LHS, larger RHS)
    for i in range(n_core, m):
        k = i % n_core
        coeffs = {vnames[j]: float(core_A[k, j]) for j in range(n)}
        rhs = float(core_b[k] * rng.uniform(1.5, 3.0))
        model.add_constraint(Constraint(f"c{i}", coeffs, ConstraintSense.LESS_EQUAL, rhs))
    c = rng.uniform(1.0, 10.0, n)
    model.set_objective(Objective({vnames[j]: float(c[j]) for j in range(n)}, ObjectiveSense.MAXIMIZE))
    return model


def _long_pivot_lp(m: int, n: int, seed: int, name: str) -> Model:
    """LP expected to require many pivots (Klee-Minty-like structure)."""
    rng = np.random.RandomState(seed)
    model = Model(name)
    vnames = [f"x{j}" for j in range(n)]
    for vn in vnames:
        model.add_variable(Variable(vn, lower_bound=0.0))
    # Create cascading constraints that force long simplex paths
    for i in range(m):
        coeffs = {}
        for j in range(n):
            if j <= i:
                coeffs[vnames[j]] = float(2.0 ** (i - j)) + rng.uniform(0.0, 0.1)
            else:
                coeffs[vnames[j]] = float(rng.uniform(0.0, 0.5))
        rhs = float(5.0 ** (i + 1))
        model.add_constraint(Constraint(f"c{i}", coeffs, ConstraintSense.LESS_EQUAL, rhs))
    c = rng.uniform(1.0, 10.0, n)
    model.set_objective(Objective({vnames[j]: float(c[j]) for j in range(n)}, ObjectiveSense.MAXIMIZE))
    return model


def _short_pivot_lp(m: int, n: int, seed: int, name: str) -> Model:
    """LP expected to solve in few pivots (well-structured)."""
    rng = np.random.RandomState(seed)
    model = Model(name)
    vnames = [f"x{j}" for j in range(n)]
    for vn in vnames:
        model.add_variable(Variable(vn, lower_bound=0.0))
    # Nearly diagonal structure
    for i in range(m):
        coeffs = {vnames[i % n]: float(rng.uniform(1.0, 3.0))}
        for j in range(n):
            if j != i % n and rng.rand() < 0.1:
                coeffs[vnames[j]] = float(rng.uniform(0.0, 0.3))
        rhs = float(rng.uniform(5.0, 20.0))
        model.add_constraint(Constraint(f"c{i}", coeffs, ConstraintSense.LESS_EQUAL, rhs))
    c = rng.uniform(1.0, 5.0, n)
    model.set_objective(Objective({vnames[j]: float(c[j]) for j in range(n)}, ObjectiveSense.MAXIMIZE))
    return model


# ---------------------------------------------------------------------------
# MILP Generators
# ---------------------------------------------------------------------------

def _knapsack(n_items: int, seed: int, name: str) -> Model:
    """Binary knapsack: max Σ v_i x_i s.t. Σ w_i x_i <= C."""
    rng = np.random.RandomState(seed)
    model = Model(name)
    vnames = [f"x{j}" for j in range(n_items)]
    for vn in vnames:
        model.add_variable(Variable(vn, variable_type=VariableType.BINARY))
    weights = rng.uniform(1.0, 20.0, n_items)
    values = rng.uniform(1.0, 30.0, n_items)
    capacity = float(np.sum(weights) * 0.4)
    model.add_constraint(Constraint("cap", {vnames[j]: float(weights[j]) for j in range(n_items)},
                                     ConstraintSense.LESS_EQUAL, capacity))
    model.set_objective(Objective({vnames[j]: float(values[j]) for j in range(n_items)}, ObjectiveSense.MAXIMIZE))
    return model


def _set_covering(n_sets: int, n_elements: int, seed: int, name: str) -> Model:
    """Set covering: min Σ c_j x_j s.t. Σ a_ij x_j >= 1 for all i."""
    rng = np.random.RandomState(seed)
    model = Model(name)
    vnames = [f"x{j}" for j in range(n_sets)]
    for vn in vnames:
        model.add_variable(Variable(vn, variable_type=VariableType.BINARY))
    for i in range(n_elements):
        # Each element covered by ~30% of sets
        coeffs = {}
        for j in range(n_sets):
            if rng.rand() < 0.3:
                coeffs[vnames[j]] = 1.0
        if not coeffs:
            coeffs[vnames[rng.randint(n_sets)]] = 1.0
        model.add_constraint(Constraint(f"cover{i}", coeffs, ConstraintSense.GREATER_EQUAL, 1.0))
    costs = rng.uniform(1.0, 10.0, n_sets)
    model.set_objective(Objective({vnames[j]: float(costs[j]) for j in range(n_sets)}, ObjectiveSense.MINIMIZE))
    return model


def _binary_selection(n_items: int, n_groups: int, seed: int, name: str) -> Model:
    """Binary selection with group constraints."""
    rng = np.random.RandomState(seed)
    model = Model(name)
    vnames = [f"x{j}" for j in range(n_items)]
    for vn in vnames:
        model.add_variable(Variable(vn, variable_type=VariableType.BINARY))
    # Group constraints: select at most k from each group
    items_per_group = max(2, n_items // n_groups)
    for g in range(n_groups):
        start = g * items_per_group
        end = min(start + items_per_group, n_items)
        if start >= n_items:
            break
        coeffs = {vnames[j]: 1.0 for j in range(start, end)}
        k = max(1, (end - start) // 2)
        model.add_constraint(Constraint(f"grp{g}", coeffs, ConstraintSense.LESS_EQUAL, float(k)))
    # Budget constraint
    costs = rng.uniform(1.0, 10.0, n_items)
    budget = float(np.sum(costs) * 0.5)
    model.add_constraint(Constraint("budget", {vnames[j]: float(costs[j]) for j in range(n_items)},
                                     ConstraintSense.LESS_EQUAL, budget))
    values = rng.uniform(1.0, 20.0, n_items)
    model.set_objective(Objective({vnames[j]: float(values[j]) for j in range(n_items)}, ObjectiveSense.MAXIMIZE))
    return model


def _mixed_int_resource(n_cont: int, n_int: int, m: int, seed: int, name: str) -> Model:
    """Mixed-integer resource allocation."""
    rng = np.random.RandomState(seed)
    model = Model(name)
    cont_names = [f"y{j}" for j in range(n_cont)]
    int_names = [f"z{j}" for j in range(n_int)]
    for vn in cont_names:
        model.add_variable(Variable(vn, lower_bound=0.0))
    for vn in int_names:
        model.add_variable(Variable(vn, lower_bound=0.0, upper_bound=10.0, variable_type=VariableType.INTEGER))
    all_names = cont_names + int_names
    n = n_cont + n_int
    for i in range(m):
        coeffs = {all_names[j]: float(rng.uniform(0.5, 5.0)) for j in range(n) if rng.rand() < 0.5}
        if not coeffs:
            coeffs[all_names[rng.randint(n)]] = float(rng.uniform(1.0, 3.0))
        rhs = float(rng.uniform(10.0, 50.0))
        model.add_constraint(Constraint(f"c{i}", coeffs, ConstraintSense.LESS_EQUAL, rhs))
    c = rng.uniform(1.0, 10.0, n)
    model.set_objective(Objective({all_names[j]: float(c[j]) for j in range(n)}, ObjectiveSense.MAXIMIZE))
    return model


def _integer_allocation(n_int: int, m: int, seed: int, name: str) -> Model:
    """Pure integer allocation."""
    rng = np.random.RandomState(seed)
    model = Model(name)
    vnames = [f"x{j}" for j in range(n_int)]
    for vn in vnames:
        model.add_variable(Variable(vn, lower_bound=0.0, upper_bound=10.0, variable_type=VariableType.INTEGER))
    for i in range(m):
        coeffs = {vnames[j]: float(rng.uniform(1.0, 5.0)) for j in range(n_int)}
        rhs = float(rng.uniform(10.0, 40.0))
        model.add_constraint(Constraint(f"c{i}", coeffs, ConstraintSense.LESS_EQUAL, rhs))
    c = rng.uniform(1.0, 10.0, n_int)
    model.set_objective(Objective({vnames[j]: float(c[j]) for j in range(n_int)}, ObjectiveSense.MAXIMIZE))
    return model


def _strong_relaxation_milp(n: int, m: int, seed: int, name: str) -> Model:
    """MILP with tight LP relaxation (close integrality gap)."""
    rng = np.random.RandomState(seed)
    model = Model(name)
    vnames = [f"x{j}" for j in range(n)]
    for vn in vnames:
        model.add_variable(Variable(vn, lower_bound=0.0, upper_bound=1.0, variable_type=VariableType.BINARY))
    # Tight constraints that make LP relaxation nearly integral
    for i in range(m):
        # Each constraint involves ~2-3 variables
        n_vars = min(n, rng.randint(2, 4))
        chosen = rng.choice(n, n_vars, replace=False)
        coeffs = {vnames[j]: 1.0 for j in chosen}
        rhs = float(n_vars - 1)  # tight
        model.add_constraint(Constraint(f"c{i}", coeffs, ConstraintSense.LESS_EQUAL, rhs))
    c = rng.uniform(1.0, 10.0, n)
    model.set_objective(Objective({vnames[j]: float(c[j]) for j in range(n)}, ObjectiveSense.MAXIMIZE))
    return model


def _weak_relaxation_milp(n: int, m: int, seed: int, name: str) -> Model:
    """MILP with loose LP relaxation (large integrality gap)."""
    rng = np.random.RandomState(seed)
    model = Model(name)
    vnames = [f"x{j}" for j in range(n)]
    for vn in vnames:
        model.add_variable(Variable(vn, lower_bound=0.0, upper_bound=1.0, variable_type=VariableType.BINARY))
    # Loose constraints that allow fractional LP optima
    for i in range(m):
        coeffs = {vnames[j]: float(rng.uniform(0.5, 5.0)) for j in range(n)}
        rhs = float(np.sum([coeffs[vnames[j]] for j in range(n)]) * 0.7)
        model.add_constraint(Constraint(f"c{i}", coeffs, ConstraintSense.LESS_EQUAL, rhs))
    c = rng.uniform(1.0, 20.0, n)
    model.set_objective(Objective({vnames[j]: float(c[j]) for j in range(n)}, ObjectiveSense.MAXIMIZE))
    return model


# ---------------------------------------------------------------------------
# Master Generator
# ---------------------------------------------------------------------------

def generate_all_workloads() -> list[WorkloadSpec]:
    """Generate all 25 workload classes across appropriate size buckets."""
    workloads = []
    base_seed = 815_000

    # --- LP WORKLOADS ---
    # 1. Small dense LP (m=5, n=8)
    m, n, s = 5, 8, base_seed + 1
    workloads.append(WorkloadSpec("small_dense_lp", WorkloadClass.SMALL_DENSE_LP, SizeBucket.SMALL,
                                  _dense_lp(m, n, s, "small_dense_lp"), s, m, n, density=1.0,
                                  constraint_types="<=", expected_pivots="<10"))
    # 2. Medium dense LP (m=30, n=50)
    m, n, s = 30, 50, base_seed + 2
    workloads.append(WorkloadSpec("medium_dense_lp", WorkloadClass.MEDIUM_DENSE_LP, SizeBucket.MEDIUM,
                                  _dense_lp(m, n, s, "medium_dense_lp"), s, m, n, density=1.0,
                                  constraint_types="<=", expected_pivots="30-60"))
    # 3. Large dense LP (m=100, n=150)
    m, n, s = 100, 150, base_seed + 3
    workloads.append(WorkloadSpec("large_dense_lp", WorkloadClass.LARGE_DENSE_LP, SizeBucket.LARGE,
                                  _dense_lp(m, n, s, "large_dense_lp"), s, m, n, density=1.0,
                                  constraint_types="<=", expected_pivots="100-300"))
    # 4. Sparse LP (m=40, n=80, density=0.3)
    m, n, s, d = 40, 80, base_seed + 4, 0.3
    workloads.append(WorkloadSpec("sparse_lp", WorkloadClass.SPARSE_LP, SizeBucket.MEDIUM,
                                  _sparse_lp(m, n, d, s, "sparse_lp"), s, m, n, density=d,
                                  constraint_types="<="))
    # 5. Highly sparse LP (m=80, n=200, density=0.1)
    m, n, s, d = 80, 200, base_seed + 5, 0.1
    workloads.append(WorkloadSpec("highly_sparse_lp", WorkloadClass.HIGHLY_SPARSE_LP, SizeBucket.LARGE,
                                  _sparse_lp(m, n, d, s, "highly_sparse_lp"), s, m, n, density=d,
                                  constraint_types="<="))
    # 6. Equality-heavy LP (m=20, n=35)
    m, n, s = 20, 35, base_seed + 6
    workloads.append(WorkloadSpec("equality_heavy_lp", WorkloadClass.EQUALITY_HEAVY_LP, SizeBucket.MEDIUM,
                                  _equality_heavy_lp(m, n, s, "equality_heavy_lp"), s, m, n,
                                  constraint_types="80%==,20%<="))
    # 7. Inequality-heavy LP (m=25, n=40)
    m, n, s = 25, 40, base_seed + 7
    workloads.append(WorkloadSpec("inequality_heavy_lp", WorkloadClass.INEQUALITY_HEAVY_LP, SizeBucket.MEDIUM,
                                  _dense_lp(m, n, s, "inequality_heavy_lp"), s, m, n, density=1.0,
                                  constraint_types="100%<="))
    # 8. Mixed sense LP (m=30, n=45)
    m, n, s = 30, 45, base_seed + 8
    workloads.append(WorkloadSpec("mixed_sense_lp", WorkloadClass.MIXED_SENSE_LP, SizeBucket.MEDIUM,
                                  _mixed_sense_lp(m, n, s, "mixed_sense_lp"), s, m, n,
                                  constraint_types="33%<=,33%>=,33%=="))
    # 9. Degenerate LP (m=20, n=10)
    m, n, s = 20, 10, base_seed + 9
    workloads.append(WorkloadSpec("degenerate_lp", WorkloadClass.DEGENERATE_LP, SizeBucket.MEDIUM,
                                  _degenerate_lp(m, n, s, "degenerate_lp"), s, m, n,
                                  constraint_types="<= (tight)"))
    # 10. Ill-conditioned LP (m=25, n=40)
    m, n, s = 25, 40, base_seed + 10
    workloads.append(WorkloadSpec("ill_conditioned_lp", WorkloadClass.ILL_CONDITIONED_LP, SizeBucket.MEDIUM,
                                  _ill_conditioned_lp(m, n, s, "ill_conditioned_lp"), s, m, n,
                                  constraint_types="<= (scaled)"))
    # 11. Phase-I heavy LP (m=15, n=25)
    m, n, s = 15, 25, base_seed + 11
    workloads.append(WorkloadSpec("phase1_heavy_lp", WorkloadClass.PHASE1_HEAVY_LP, SizeBucket.MEDIUM,
                                  _equality_heavy_lp(m, n, s, "phase1_heavy_lp"), s, m, n,
                                  constraint_types=">=,=="))
    # 12. Long-pivot LP (m=8, n=12)
    m, n, s = 8, 12, base_seed + 12
    workloads.append(WorkloadSpec("long_pivot_lp", WorkloadClass.LONG_PIVOT_LP, SizeBucket.SMALL,
                                  _long_pivot_lp(m, n, s, "long_pivot_lp"), s, m, n,
                                  expected_pivots="many"))
    # 13. Short-pivot LP (m=8, n=15)
    m, n, s = 8, 15, base_seed + 13
    workloads.append(WorkloadSpec("short_pivot_lp", WorkloadClass.SHORT_PIVOT_LP, SizeBucket.SMALL,
                                  _short_pivot_lp(m, n, s, "short_pivot_lp"), s, m, n,
                                  expected_pivots="few"))
    # 14. Bound-heavy LP (m=30, n=20)
    m, n, s = 30, 20, base_seed + 14
    workloads.append(WorkloadSpec("bound_heavy_lp", WorkloadClass.BOUND_HEAVY_LP, SizeBucket.MEDIUM,
                                  _bound_heavy_lp(m, n, s, "bound_heavy_lp"), s, m, n,
                                  constraint_types="bounds+linking"))
    # 15. Redundant-constraint LP (m=30, n=20)
    m, n, s = 30, 20, base_seed + 15
    workloads.append(WorkloadSpec("redundant_constraint_lp", WorkloadClass.REDUNDANT_CONSTRAINT_LP, SizeBucket.MEDIUM,
                                  _redundant_constraint_lp(m, n, s, "redundant_constraint_lp"), s, m, n,
                                  constraint_types="<= (50% redundant)"))

    # --- MILP WORKLOADS ---
    # 16. Knapsack (small)
    s = base_seed + 16
    workloads.append(WorkloadSpec("knapsack_small", WorkloadClass.KNAPSACK, SizeBucket.SMALL,
                                  _knapsack(8, s, "knapsack_small"), s, 1, 8, n_binary=8,
                                  constraint_types="<="))
    # 17. Set covering
    s = base_seed + 17
    workloads.append(WorkloadSpec("set_covering", WorkloadClass.SET_COVERING, SizeBucket.SMALL,
                                  _set_covering(10, 6, s, "set_covering"), s, 6, 10, n_binary=10,
                                  constraint_types=">="))
    # 18. Binary selection
    s = base_seed + 18
    workloads.append(WorkloadSpec("binary_selection", WorkloadClass.BINARY_SELECTION, SizeBucket.SMALL,
                                  _binary_selection(12, 3, s, "binary_selection"), s, 4, 12, n_binary=12,
                                  constraint_types="<="))
    # 19. Mixed-integer resource allocation
    s = base_seed + 19
    workloads.append(WorkloadSpec("mixed_int_resource", WorkloadClass.MIXED_INT_RESOURCE, SizeBucket.MEDIUM,
                                  _mixed_int_resource(5, 5, 8, s, "mixed_int_resource"), s, 8, 10,
                                  n_integer=5, constraint_types="<="))
    # 20. Integer allocation
    s = base_seed + 20
    workloads.append(WorkloadSpec("integer_allocation", WorkloadClass.INTEGER_ALLOCATION, SizeBucket.SMALL,
                                  _integer_allocation(6, 5, s, "integer_allocation"), s, 5, 6,
                                  n_integer=6, constraint_types="<="))
    # 21. Branching-heavy MILP
    s = base_seed + 21
    workloads.append(WorkloadSpec("branching_heavy", WorkloadClass.BRANCHING_HEAVY_MILP, SizeBucket.SMALL,
                                  _knapsack(10, s, "branching_heavy"), s, 1, 10, n_binary=10,
                                  constraint_types="<=", expected_pivots="many nodes"))
    # 22. Multi-level B&B MILP
    s = base_seed + 22
    workloads.append(WorkloadSpec("multilevel_bb", WorkloadClass.MULTILEVEL_BB_MILP, SizeBucket.MEDIUM,
                                  _mixed_int_resource(8, 6, 10, s, "multilevel_bb"), s, 10, 14,
                                  n_integer=6, constraint_types="<="))
    # 23. MILP with frequent infeasible children
    s = base_seed + 23
    workloads.append(WorkloadSpec("infeasible_children", WorkloadClass.INFEASIBLE_CHILDREN_MILP, SizeBucket.SMALL,
                                  _integer_allocation(5, 8, s, "infeasible_children"), s, 8, 5,
                                  n_integer=5, constraint_types="<= (tight)"))
    # 24. MILP with strong LP relaxation
    s = base_seed + 24
    workloads.append(WorkloadSpec("strong_relaxation", WorkloadClass.STRONG_RELAXATION_MILP, SizeBucket.SMALL,
                                  _strong_relaxation_milp(8, 6, s, "strong_relaxation"), s, 6, 8,
                                  n_binary=8, constraint_types="<= (tight)"))
    # 25. MILP with weak LP relaxation
    s = base_seed + 25
    workloads.append(WorkloadSpec("weak_relaxation", WorkloadClass.WEAK_RELAXATION_MILP, SizeBucket.SMALL,
                                  _weak_relaxation_milp(8, 4, s, "weak_relaxation"), s, 4, 8,
                                  n_binary=8, constraint_types="<= (loose)"))

    return workloads


if __name__ == "__main__":
    wls = generate_all_workloads()
    print(f"Generated {len(wls)} workloads:\n")
    for w in wls:
        kind = "MILP" if w.n_integer + w.n_binary > 0 else "LP"
        print(f"  {w.name:30s} | {kind:4s} | {w.size_bucket.value:6s} | m={w.m:3d} n={w.n:3d} | "
              f"int={w.n_integer} bin={w.n_binary} | density={w.density:.1f} | {w.constraint_types}")
