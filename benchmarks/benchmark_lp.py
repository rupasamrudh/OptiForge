#!/usr/bin/env python
"""LP Solver Performance Benchmarks.

Measures solve times and iteration counts for LP problems of varying sizes.
"""

import time
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass

from optiforge.model import Variable, Constraint, ConstraintSense, Objective, ObjectiveSense, Model
from optiforge.lp.simplex import RevisedSimplexSolver
from optiforge.core.solution import SolutionStatus


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    problem_name: str
    num_variables: int
    num_constraints: int
    status: SolutionStatus
    solve_time_ms: float
    iterations: int
    objective_value: float
    success: bool


def generate_random_lp(num_vars: int, num_constraints: int, seed: int = 42) -> Model:
    """Generate a random LP problem.
    
    max c^T x
    s.t. A x <= b
         0 <= x <= 100
    """
    np.random.seed(seed)
    
    m = Model(f"random_lp_{num_vars}v_{num_constraints}c")
    
    # Add variables with upper bounds to prevent unboundedness
    for i in range(num_vars):
        m.add_variable(Variable(f"x{i}", lower_bound=0.0, upper_bound=100.0))
    
    # Add constraints
    A = np.random.randn(num_constraints, num_vars)
    b = np.random.rand(num_constraints) * 10 + 1  # Positive RHS
    
    for i in range(num_constraints):
        coeffs = {f"x{j}": float(A[i, j]) for j in range(num_vars)}
        m.add_constraint(Constraint(f"c{i}", coeffs, ConstraintSense.LESS_EQUAL, float(b[i])))
    
    # Objective
    c = np.random.randn(num_vars)
    obj_coeffs = {f"x{j}": float(c[j]) for j in range(num_vars)}
    m.set_objective(Objective(obj_coeffs, ObjectiveSense.MAXIMIZE))
    
    return m


def generate_dense_lp(num_vars: int, num_constraints: int, seed: int = 42) -> Model:
    """Generate a dense LP problem with all variables in all constraints."""
    np.random.seed(seed)
    
    m = Model(f"dense_lp_{num_vars}v_{num_constraints}c")
    
    for i in range(num_vars):
        m.add_variable(Variable(f"x{i}", lower_bound=0.0, upper_bound=100.0))
    
    # Dense constraint matrix
    A = np.random.rand(num_constraints, num_vars) * 2 - 1  # [-1, 1]
    b = np.random.rand(num_constraints) * 20 + 5
    
    for i in range(num_constraints):
        coeffs = {f"x{j}": float(A[i, j]) for j in range(num_vars)}
        m.add_constraint(Constraint(f"c{i}", coeffs, ConstraintSense.LESS_EQUAL, float(b[i])))
    
    c = np.random.randn(num_vars)
    obj_coeffs = {f"x{j}": float(c[j]) for j in range(num_vars)}
    m.set_objective(Objective(obj_coeffs, ObjectiveSense.MAXIMIZE))
    
    return m


def generate_sparse_lp(num_vars: int, num_constraints: int, density: float = 0.3, seed: int = 42) -> Model:
    """Generate a sparse LP problem that is likely to be bounded."""
    np.random.seed(seed)
    
    m = Model(f"sparse_lp_{num_vars}v_{num_constraints}c")
    
    for i in range(num_vars):
        m.add_variable(Variable(f"x{i}", lower_bound=0.0, upper_bound=100.0))
    
    # Sparse constraint matrix - ensure each variable appears in at least one constraint
    var_in_constraint = [False] * num_vars
    
    for i in range(num_constraints):
        coeffs = {}
        for j in range(num_vars):
            if np.random.rand() < density:
                coeffs[f"x{j}"] = float(np.random.randn())
                var_in_constraint[j] = True
        if not coeffs:
            j = np.random.randint(num_vars)
            coeffs[f"x{j}"] = 1.0
            var_in_constraint[j] = True
        m.add_constraint(Constraint(f"c{i}", coeffs, ConstraintSense.LESS_EQUAL, float(np.random.rand() * 10 + 1)))
    
    # Ensure all variables appear in at least one constraint
    for j in range(num_vars):
        if not var_in_constraint[j]:
            # Add a simple bound constraint
            m.add_constraint(Constraint(f"bound_x{j}", {f"x{j}": 1.0}, ConstraintSense.LESS_EQUAL, 100.0))
    
    c = np.random.randn(num_vars)
    obj_coeffs = {f"x{j}": float(c[j]) for j in range(num_vars)}
    m.set_objective(Objective(obj_coeffs, ObjectiveSense.MAXIMIZE))
    
    return m


def run_benchmark(model: Model, problem_name: str, num_runs: int = 3) -> BenchmarkResult:
    """Run benchmark on a model multiple times and return best result."""
    best_time = float('inf')
    best_result = None
    
    for run in range(num_runs):
        solver = RevisedSimplexSolver()
        start = time.perf_counter()
        sol = solver.solve(model)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        if elapsed_ms < best_time:
            best_time = elapsed_ms
            best_result = BenchmarkResult(
                problem_name=problem_name,
                num_variables=len(model.variables),
                num_constraints=len(model.constraints),
                status=sol.status,
                solve_time_ms=elapsed_ms,
                iterations=sol.iterations,
                objective_value=sol.objective_value if sol.objective_value is not None else float('nan'),
                success=sol.status == SolutionStatus.OPTIMAL
            )
    
    return best_result


def main():
    """Run LP benchmarks."""
    print("=" * 80)
    print("LP Solver Performance Benchmarks")
    print("=" * 80)
    
    # Problem sizes to test
    sizes = [
        (3, 3),    # Tiny
        (5, 5),    # Very small
        (10, 10),  # Small
        (15, 15),  # Medium-small
        (20, 20),  # Medium
        (25, 25),  # Medium-large
        (30, 30),  # Large
    ]
    
    results: List[BenchmarkResult] = []
    
    print("\n--- Random LP Problems ---")
    for n_vars, n_cons in sizes:
        model = generate_random_lp(n_vars, n_cons)
        result = run_benchmark(model, f"random_{n_vars}v_{n_cons}c")
        results.append(result)
        status_str = "✓" if result.success else "✗"
        print(f"  {status_str} {result.problem_name:30s} | vars={result.num_variables:2d} cons={result.num_constraints:2d} | "
              f"time={result.solve_time_ms:8.2f}ms | iters={result.iterations:3d} | obj={result.objective_value:10.4f}")
    
    print("\n--- Dense LP Problems ---")
    for n_vars, n_cons in sizes:
        model = generate_dense_lp(n_vars, n_cons)
        result = run_benchmark(model, f"dense_{n_vars}v_{n_cons}c")
        results.append(result)
        status_str = "✓" if result.success else "✗"
        print(f"  {status_str} {result.problem_name:30s} | vars={result.num_variables:2d} cons={result.num_constraints:2d} | "
              f"time={result.solve_time_ms:8.2f}ms | iters={result.iterations:3d} | obj={result.objective_value:10.4f}")
    
    print("\n--- Sparse LP Problems (10% density) ---")
    for n_vars, n_cons in sizes:
        model = generate_sparse_lp(n_vars, n_cons, density=0.1)
        result = run_benchmark(model, f"sparse_{n_vars}v_{n_cons}c")
        results.append(result)
        status_str = "✓" if result.success else "✗"
        print(f"  {status_str} {result.problem_name:30s} | vars={result.num_variables:2d} cons={result.num_constraints:2d} | "
              f"time={result.solve_time_ms:8.2f}ms | iters={result.iterations:3d} | obj={result.objective_value:10.4f}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    successful = [r for r in results if r.success]
    if successful:
        avg_time = np.mean([r.solve_time_ms for r in successful])
        avg_iters = np.mean([r.iterations for r in successful])
        max_time = np.max([r.solve_time_ms for r in successful])
        max_iters = np.max([r.iterations for r in successful])
        
        print(f"Successful solves: {len(successful)}/{len(results)}")
        print(f"Average solve time: {avg_time:.2f} ms")
        print(f"Average iterations: {avg_iters:.1f}")
        print(f"Max solve time: {max_time:.2f} ms")
        print(f"Max iterations: {max_iters}")
    
    # By problem type
    for ptype in ["random", "dense", "sparse"]:
        ptype_results = [r for r in results if r.problem_name.startswith(ptype) and r.success]
        if ptype_results:
            avg_time = np.mean([r.solve_time_ms for r in ptype_results])
            avg_iters = np.mean([r.iterations for r in ptype_results])
            print(f"\n{ptype.capitalize()} problems:")
            print(f"  Average time: {avg_time:.2f} ms")
            print(f"  Average iterations: {avg_iters:.1f}")
    
    return results


if __name__ == "__main__":
    main()