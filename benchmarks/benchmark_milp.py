#!/usr/bin/env python
"""MILP Solver Performance Benchmarks.

Measures solve times, node counts, and iteration counts for MILP problems of varying sizes.
"""

import time
import numpy as np
from typing import List
from dataclasses import dataclass

from optiforge.model import Variable, Constraint, ConstraintSense, Objective, ObjectiveSense, Model, VariableType
from optiforge.milp import solve_milp
from optiforge.milp.solver import MILPTolerances
from optiforge.core.solution import SolutionStatus


@dataclass
class MILPBenchmarkResult:
    """Result of a single MILP benchmark run."""
    problem_name: str
    num_variables: int
    num_constraints: int
    num_integer_vars: int
    status: SolutionStatus
    solve_time_ms: float
    nodes_explored: int
    simplex_iterations: int
    objective_value: float
    success: bool


def generate_random_milp(num_vars: int, num_constraints: int, int_fraction: float = 0.5, seed: int = 42) -> Model:
    """Generate a random MILP problem.

    max c^T x
    s.t. A x <= b
         x >= 0
         x_i integer for some i

    Uses a non-negative RHS (b > 0) and a positive objective direction
    to ensure LP relaxations are typically feasible and bounded.
    """
    rng = np.random.default_rng(seed)

    m = Model(f"random_milp_{num_vars}v_{num_constraints}c")

    # Add variables - some integer, some continuous
    num_int = int(num_vars * int_fraction)
    for i in range(num_vars):
        var_type = VariableType.INTEGER if i < num_int else VariableType.CONTINUOUS
        m.add_variable(Variable(f"x{i}", lower_bound=0.0, variable_type=var_type))

    # Constraint matrix: use abs() values to ensure non-negative coefficients,
    # making the LP relaxation more likely to be feasible and bounded.
    A = np.abs(rng.standard_normal((num_constraints, num_vars)))
    b = rng.random(num_constraints) * 10 + 5  # positive RHS: 5..15

    for i in range(num_constraints):
        coeffs = {f"x{j}": float(A[i, j]) for j in range(num_vars)}
        m.add_constraint(Constraint(f"c{i}", coeffs, ConstraintSense.LESS_EQUAL, float(b[i])))

    # Objective: maximise a positive combination (ensures bounded, meaningful problem)
    c = np.abs(rng.standard_normal(num_vars)) + 0.1
    obj_coeffs = {f"x{j}": float(c[j]) for j in range(num_vars)}
    m.set_objective(Objective(obj_coeffs, ObjectiveSense.MAXIMIZE))

    return m


def generate_knapsack_milp(num_items: int, seed: int = 42) -> Model:
    """Generate a knapsack problem (classic MILP benchmark)."""
    np.random.seed(seed)
    
    m = Model(f"knapsack_{num_items}items")
    
    # Binary variables for each item
    for i in range(num_items):
        m.add_variable(Variable(f"x{i}", lower_bound=0.0, upper_bound=1.0, variable_type=VariableType.BINARY))
    
    # Weights and values
    weights = np.random.randint(1, 20, num_items)
    values = np.random.randint(1, 50, num_items)
    capacity = int(np.sum(weights) * 0.5)
    
    # Knapsack constraint
    coeffs = {f"x{i}": float(weights[i]) for i in range(num_items)}
    m.add_constraint(Constraint("capacity", coeffs, ConstraintSense.LESS_EQUAL, float(capacity)))
    
    # Objective: maximize value
    obj_coeffs = {f"x{i}": float(values[i]) for i in range(num_items)}
    m.set_objective(Objective(obj_coeffs, ObjectiveSense.MAXIMIZE))
    
    return m


def generate_set_covering_milp(num_sets: int, num_elements: int, seed: int = 42) -> Model:
    """Generate a set covering problem."""
    np.random.seed(seed)
    
    m = Model(f"set_cover_{num_sets}s_{num_elements}e")
    
    # Binary variables for each set
    for i in range(num_sets):
        m.add_variable(Variable(f"s{i}", lower_bound=0.0, upper_bound=1.0, variable_type=VariableType.BINARY))
    
    # Each element must be covered by at least one set
    # Randomly assign elements to sets
    for e in range(num_elements):
        coeffs = {}
        for s in range(num_sets):
            if np.random.rand() < 0.3:  # 30% chance each set covers each element
                coeffs[f"s{s}"] = 1.0
        if not coeffs:
            coeffs[f"s{np.random.randint(num_sets)}"] = 1.0
        m.add_constraint(Constraint(f"cover_e{e}", coeffs, ConstraintSense.GREATER_EQUAL, 1.0))
    
    # Objective: minimize number of sets
    obj_coeffs = {f"s{i}": 1.0 for i in range(num_sets)}
    m.set_objective(Objective(obj_coeffs, ObjectiveSense.MINIMIZE))
    
    return m


def run_milp_benchmark(model: Model, problem_name: str, num_runs: int = 3,
                       max_nodes: int = 10000, max_time: float = 30.0) -> MILPBenchmarkResult:
    """Run MILP benchmark on a model multiple times and return best result.

    Outcome classification:
      success=True  : SolutionStatus.OPTIMAL
      success=False : infeasible, unbounded, iteration_limit, unknown

    Only successful (OPTIMAL) results contribute to timing averages.
    """
    best_time = float('inf')
    best_result = None

    tolerances = MILPTolerances(max_nodes=max_nodes)

    for run in range(num_runs):
        start = time.perf_counter()
        sol = solve_milp(model, tolerances=tolerances)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if elapsed_ms < best_time:
            best_time = elapsed_ms
            # Classify outcome — never store NaN objective for non-optimal cases
            is_optimal = sol.status == SolutionStatus.OPTIMAL
            best_result = MILPBenchmarkResult(
                problem_name=problem_name,
                num_variables=len(model.variables),
                num_constraints=len(model.constraints),
                num_integer_vars=sum(
                    1 for v in model.variables.values()
                    if v.variable_type != VariableType.CONTINUOUS
                ),
                status=sol.status,
                solve_time_ms=elapsed_ms,
                nodes_explored=sol.node_count if sol.node_count is not None else 0,
                simplex_iterations=sol.iterations if hasattr(sol, 'iterations') else 0,
                # Only record objective value for genuinely optimal solutions
                objective_value=sol.objective_value if is_optimal else None,
                success=is_optimal,
            )

    return best_result


def main():
    """Run MILP benchmarks."""
    print("=" * 80)
    print("MILP Solver Performance Benchmarks")
    print("=" * 80)

    # Problem sizes (smaller for MILP due to exponential complexity)
    sizes = [
        (5, 5),    # Tiny
        (8, 8),    # Very small
        (10, 10),  # Small
        (12, 12),  # Medium-small
        (15, 15),  # Medium
    ]

    results: List[MILPBenchmarkResult] = []

    print("\n--- Random MILP Problems (50% integer) ---")
    print("    (each problem uses a unique seed to ensure diversity)")
    # Use a distinct seed per problem size to ensure diverse instances.
    # Seed = 100 * num_vars + num_cons avoids per-size seed collision.
    for n_vars, n_cons in sizes:
        seed = 100 * n_vars + n_cons
        model = generate_random_milp(n_vars, n_cons, int_fraction=0.5, seed=seed)
        result = run_milp_benchmark(model, f"random_milp_{n_vars}v_{n_cons}c")
        results.append(result)
        status_str = "OK" if result.success else f"-- {result.status.value}"
        obj_str = f"{result.objective_value:.4f}" if result.objective_value is not None else "N/A"
        print(f"  [{status_str:20s}] {result.problem_name:35s} | "
              f"vars={result.num_variables:2d} cons={result.num_constraints:2d} int={result.num_integer_vars:2d} | "
              f"time={result.solve_time_ms:8.2f}ms | nodes={result.nodes_explored:4d} | "
              f"iters={result.simplex_iterations:4d} | obj={obj_str}")

    print("\n--- Knapsack Problems ---")
    knapsack_sizes = [10, 15, 20, 25, 30]
    for n_items in knapsack_sizes:
        model = generate_knapsack_milp(n_items)
        result = run_milp_benchmark(model, f"knapsack_{n_items}items")
        results.append(result)
        status_str = "OK" if result.success else f"-- {result.status.value}"
        obj_str = f"{result.objective_value:.4f}" if result.objective_value is not None else "N/A"
        print(f"  [{status_str:20s}] {result.problem_name:35s} | "
              f"vars={result.num_variables:2d} cons={result.num_constraints:2d} int={result.num_integer_vars:2d} | "
              f"time={result.solve_time_ms:8.2f}ms | nodes={result.nodes_explored:4d} | "
              f"iters={result.simplex_iterations:4d} | obj={obj_str}")

    print("\n--- Set Covering Problems ---")
    set_cover_sizes = [(10, 15), (15, 20), (20, 25)]
    for n_sets, n_elements in set_cover_sizes:
        model = generate_set_covering_milp(n_sets, n_elements)
        result = run_milp_benchmark(model, f"set_cover_{n_sets}s_{n_elements}e")
        results.append(result)
        status_str = "OK" if result.success else f"-- {result.status.value}"
        obj_str = f"{result.objective_value:.4f}" if result.objective_value is not None else "N/A"
        print(f"  [{status_str:20s}] {result.problem_name:35s} | "
              f"vars={result.num_variables:2d} cons={result.num_constraints:2d} int={result.num_integer_vars:2d} | "
              f"time={result.solve_time_ms:8.2f}ms | nodes={result.nodes_explored:4d} | "
              f"iters={result.simplex_iterations:4d} | obj={obj_str}")

    # Summary statistics — only over successful (OPTIMAL) solves
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Outcome breakdown
    from collections import Counter
    status_counts = Counter(r.status.value for r in results)
    print(f"Total problems: {len(results)}")
    for st, cnt in sorted(status_counts.items()):
        tag = "[SOLVED]" if st == "optimal" else "[NOT SOLVED]"
        print(f"  {tag} {st}: {cnt}")

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    if successful:
        avg_time = np.mean([r.solve_time_ms for r in successful])
        avg_nodes = np.mean([r.nodes_explored for r in successful])
        avg_iters = np.mean([r.simplex_iterations for r in successful])
        max_time = np.max([r.solve_time_ms for r in successful])
        max_nodes = np.max([r.nodes_explored for r in successful])

        print(f"\nSuccessful solves ({len(successful)}/{len(results)}):")
        print(f"  Average solve time : {avg_time:.2f} ms  (excludes non-optimal)")
        print(f"  Average nodes      : {avg_nodes:.1f}")
        print(f"  Average iterations : {avg_iters:.1f}")
        print(f"  Max solve time     : {max_time:.2f} ms")
        print(f"  Max nodes          : {max_nodes}")

    if failed:
        print(f"\nFailed / non-optimal ({len(failed)}/{len(results)}):")
        for r in failed:
            print(f"  {r.problem_name}: status={r.status.value} time={r.solve_time_ms:.2f}ms")

    # Per-type breakdown
    for ptype in ["random_milp", "knapsack", "set_cover"]:
        ptype_ok = [r for r in results if r.problem_name.startswith(ptype) and r.success]
        ptype_all = [r for r in results if r.problem_name.startswith(ptype)]
        if ptype_ok:
            avg_time = np.mean([r.solve_time_ms for r in ptype_ok])
            avg_nodes = np.mean([r.nodes_explored for r in ptype_ok])
            avg_iters = np.mean([r.simplex_iterations for r in ptype_ok])
            label = ptype.replace("_", " ").capitalize()
            print(f"\n{label} problems ({len(ptype_ok)}/{len(ptype_all)} solved):")
            print(f"  Average time: {avg_time:.2f} ms")
            print(f"  Average nodes: {avg_nodes:.1f}")
            print(f"  Average iterations: {avg_iters:.1f}")

    return results


if __name__ == "__main__":
    main()