"""Stage 8.19 Safe Experiment Benchmarks.

Compares:
1. Stage 8.18 Baseline PFI (fast_pivot_update=False)
2. Stage 8.19 Optimized PFI (fast_pivot_update=True)
3. IncrementalBasisFactorization

Collects precise runtime, speedup, memory, and numerical match across the 9 target workloads.
"""

import sys
sys.path.insert(0, ".")
import gc
import json
import time
import tracemalloc
import numpy as np

from optiforge.lp.simplex import RevisedSimplexSolver, SimplexTolerances
from optiforge.lp.dual_simplex import DualRevisedSimplexSolver
from optiforge.milp.solver import MILPSolver, MILPTolerances
from optiforge.core.solution import SolutionStatus
from benchmarks.run_stage_8_18 import generate_scale_sweep_lp
from benchmarks.stage_8_15.generate_workloads import generate_all_workloads


def benchmark_workload(workload_name, model, is_milp=False, reps=5):
    configs = [
        ("Baseline_PFI", {"use_pfi": True, "fast": False, "use_inc": False}),
        ("Optimized_PFI", {"use_pfi": True, "fast": True, "use_inc": False}),
        ("IncrementalBasis", {"use_pfi": False, "fast": False, "use_inc": True}),
    ]

    results = {}

    for cfg_name, cfg in configs:
        times = []
        sol = None
        mem_peak = 0

        gc.collect()
        tracemalloc.start()
        mem_base = tracemalloc.get_traced_memory()[0]

        for rep in range(reps):
            if is_milp:
                solver = MILPSolver(
                    tolerances=MILPTolerances(
                        warm_start_relaxations=True,
                        inherit_basis_factorization=True,
                        use_adaptive_pivot_factorization=cfg["use_pfi"],
                        adaptive_pfi_min_dimension=1 if cfg["use_pfi"] else 9999,
                        use_fast_pivot_update=cfg["fast"],
                    ),
                    simplex_tolerances=SimplexTolerances(
                        use_pivot_factorization=cfg["use_pfi"],
                        use_adaptive_pivot_factorization=cfg["use_pfi"],
                        adaptive_pfi_min_dimension=1 if cfg["use_pfi"] else 9999,
                        use_fast_pivot_update=cfg["fast"],
                    ),
                )
                t0 = time.perf_counter()
                s = solver.solve(model)
                t1 = time.perf_counter()
            else:
                solver = RevisedSimplexSolver(
                    tolerances=SimplexTolerances(
                        use_pivot_factorization=cfg["use_pfi"],
                        use_incremental_basis=cfg["use_inc"],
                        use_fast_pivot_update=cfg["fast"],
                    )
                )
                t0 = time.perf_counter()
                s = solver.solve(model)
                t1 = time.perf_counter()

            times.append((t1 - t0) * 1000)
            if rep == 0:
                sol = s

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_peak = peak - mem_base

        results[cfg_name] = {
            "min_ms": round(min(times), 3),
            "median_ms": round(float(np.median(times)), 3),
            "mean_ms": round(float(np.mean(times)), 3),
            "status": sol.status.value,
            "objective": round(float(sol.objective_value), 6) if sol.objective_value is not None else None,
            "peak_mem_bytes": mem_peak,
        }

    # Speedup calculations
    t_base = results["Baseline_PFI"]["median_ms"]
    t_opt = results["Optimized_PFI"]["median_ms"]
    t_inc = results["IncrementalBasis"]["median_ms"]

    speedup_opt_vs_base = round((t_base - t_opt) / t_base * 100, 2) if t_base > 0 else 0.0
    ratio_opt_to_inc = round(t_inc / t_opt, 3) if t_opt > 0 else 1.0

    return {
        "workload": workload_name,
        "is_milp": is_milp,
        "baseline_pfi_ms": t_base,
        "optimized_pfi_ms": t_opt,
        "incremental_ms": t_inc,
        "speedup_percentage": speedup_opt_vs_base,
        "ratio_inc_to_opt": ratio_opt_to_inc,
        "numerical_match": bool(
            results["Baseline_PFI"]["status"] == results["Optimized_PFI"]["status"] and
            np.isclose(results["Baseline_PFI"]["objective"] or 0, results["Optimized_PFI"]["objective"] or 0, atol=1e-5)
        ),
        "details": results,
    }


def main():
    print("=" * 70)
    print("STAGE 8.19: SAFE OPTIMIZATION EXPERIMENT BENCHMARK")
    print("=" * 70)

    # 1. Scale Sweep LPs
    workloads = [
        ("scale_lp_m6", generate_scale_sweep_lp(6, 15, seed=42), False, 7),
        ("scale_lp_m50", generate_scale_sweep_lp(50, 80, seed=42), False, 5),
        ("scale_lp_m100", generate_scale_sweep_lp(100, 150, seed=42), False, 3),
        ("scale_lp_m200", generate_scale_sweep_lp(200, 200, seed=42), False, 2),
        ("scale_lp_m500", generate_scale_sweep_lp(500, 500, seed=42), False, 1),
    ]

    # 2. Structurally Diverse Workloads
    all_wl = {w.name: w.model for w in generate_all_workloads()}
    workloads.extend([
        ("phase1_heavy_lp", all_wl["phase1_heavy_lp"], False, 5),
        ("highly_sparse_lp", all_wl["highly_sparse_lp"], False, 1),
        ("branching_heavy", all_wl["branching_heavy"], True, 5),
        ("multilevel_bb", all_wl["multilevel_bb"], True, 5),
    ])

    results = []
    print(f"{'Workload':<20} | {'Base PFI (ms)':<14} | {'Opt PFI (ms)':<14} | {'Inc Basis (ms)':<14} | {'Speedup %':<10} | {'Match'}")
    print("-" * 88)

    for name, model, is_milp, reps in workloads:
        res = benchmark_workload(name, model, is_milp, reps=reps)
        results.append(res)
        print(f"{res['workload']:<20} | {res['baseline_pfi_ms']:<14.2f} | {res['optimized_pfi_ms']:<14.2f} | {res['incremental_ms']:<14.2f} | {res['speedup_percentage']:>+8.2f}% | {res['numerical_match']}")

    def json_converter(o):
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, (np.floating, float)):
            return float(o)
        return str(o)

    with open("benchmarks/stage_8_19_experiment_results.json", "w") as f:
        json.dump(results, f, indent=2, default=json_converter)

    print("\nResults successfully saved to benchmarks/stage_8_19_experiment_results.json")


if __name__ == "__main__":
    main()
