"""Stage 8.6 Profiling and Baseline Measurement Suite."""

import cProfile
import pstats
import io
import sys
import os
sys.path.insert(0, os.path.abspath("."))
import time
import math
import numpy as np

from optiforge.model import Model, Variable, VariableType, Constraint, ConstraintSense, Objective, ObjectiveSense
from optiforge.lp.simplex import RevisedSimplexSolver
from optiforge.milp.solver import solve_milp, MILPTolerances
from optiforge.presolve import presolve_model
from optiforge.lp.standard_form import to_standard_form
from optiforge.numerics.basis import BasisFactorization
from benchmarks.benchmark_lp import generate_random_lp, generate_dense_lp, generate_sparse_lp
from benchmarks.benchmark_milp import generate_random_milp, generate_knapsack_milp, generate_set_covering_milp


def generate_phase1_heavy_lp(num_vars: int = 15, num_constraints: int = 15, seed: int = 42) -> Model:
    """Generate an LP with >= and == constraints that strictly requires Phase I."""
    rng = np.random.default_rng(seed)
    m = Model("phase1_heavy_lp")
    for j in range(num_vars):
        m.add_variable(Variable(f"x{j}", lower_bound=0.0, upper_bound=100.0))
    for i in range(num_constraints):
        coeffs = {f"x{j}": float(rng.uniform(-2, 5)) for j in range(num_vars)}
        sense = ConstraintSense.GREATER_EQUAL if i % 2 == 0 else ConstraintSense.EQUAL
        rhs = float(rng.uniform(10, 50))
        m.add_constraint(Constraint(f"c{i}", coeffs, sense, rhs))
    c = {f"x{j}": float(rng.uniform(1, 10)) for j in range(num_vars)}
    m.set_objective(Objective(c, ObjectiveSense.MINIMIZE))
    return m


def run_cprofile_analysis():
    """Run cProfile on medium LP and knapsack MILP workloads to identify top bottlenecks."""
    print("=" * 80)
    print("STAGE 8.6: cProfile Bottleneck Analysis")
    print("=" * 80)

    # 1. Profile LP (Dense 25x25)
    print("\n--- Profiling Dense LP (25 vars, 25 constrs) ---")
    lp_model = generate_dense_lp(25, 25, seed=42)
    profiler = cProfile.Profile()
    profiler.enable()
    solver = RevisedSimplexSolver()
    sol_lp = solver.solve(lp_model)
    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("tottime")
    ps.print_stats(20)
    print(s.getvalue())

    # 2. Profile MILP (Knapsack 25 items)
    print("\n--- Profiling Knapsack MILP (25 items) ---")
    milp_model = generate_knapsack_milp(25, seed=42)
    profiler_milp = cProfile.Profile()
    profiler_milp.enable()
    sol_milp = solve_milp(milp_model)
    profiler_milp.disable()

    s_milp = io.StringIO()
    ps_milp = pstats.Stats(profiler_milp, stream=s_milp).sort_stats("tottime")
    ps_milp.print_stats(25)
    print(s_milp.getvalue())


def run_component_benchmarks():
    """Run precise component breakdowns across standard workloads."""
    print("=" * 80)
    print("STAGE 8.6: Workload Baselines and Component Timings")
    print("=" * 80)

    workloads = [
        ("LP small (5v, 5c)", generate_random_lp(5, 5, seed=42), False),
        ("LP medium (20v, 20c)", generate_random_lp(20, 20, seed=42), False),
        ("LP dense (20v, 20c)", generate_dense_lp(20, 20, seed=42), False),
        ("LP sparse (25v, 25c)", generate_sparse_lp(25, 25, seed=42), False),
        ("LP Phase-I (15v, 15c)", generate_phase1_heavy_lp(15, 15, seed=42), False),
        ("MILP random (10v, 10c)", generate_random_milp(10, 10, int_fraction=0.5, seed=1010), True),
        ("MILP knapsack (20 items)", generate_knapsack_milp(20, seed=42), True),
        ("MILP knapsack (25 items)", generate_knapsack_milp(25, seed=42), True),
        ("MILP set cover (15s, 20e)", generate_set_covering_milp(15, 20, seed=42), True),
    ]

    for name, model, is_milp in workloads:
        # Run 3 times, report best time
        times = []
        last_sol = None
        for _ in range(3):
            t0 = time.perf_counter()
            if is_milp:
                sol = solve_milp(model)
            else:
                sol = RevisedSimplexSolver().solve(model)
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)
            last_sol = sol

        best_t = min(times)
        status = last_sol.status.value
        obj = f"{last_sol.objective_value:.4f}" if last_sol.objective_value is not None else "None"
        iters = getattr(last_sol, "iterations", 0)
        nodes = getattr(last_sol, "node_count", 0) if is_milp else "N/A"

        print(f"{name:30s} | time: {best_t:8.2f} ms | status: {status:10s} | obj: {obj:12s} | iters: {iters:4d} | nodes: {str(nodes):4s}")


if __name__ == "__main__":
    run_cprofile_analysis()
    run_component_benchmarks()
