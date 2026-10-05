"""Stage 8.16: Comprehensive MILP Benchmark comparing Cold, WarmStart, and WarmStart+Dual PFI.

Configurations:
  Config A: Cold Default (warm_start_relaxations=False)
  Config B: Stage 8.9/8.10 WarmStart Baseline (warm_start_relaxations=True, inherit_basis_factorization=True, use_pivot_factorization=False)
  Config C: Stage 8.16 WarmStart + Dual PFI (warm_start_relaxations=True, inherit_basis_factorization=True, use_pivot_factorization=True)

Workloads: 10 deterministic MILP classes from Stage 8.15.
Repetitions: 5 per workload per configuration.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project root and benchmarks in sys.path
_root = Path(__file__).resolve().parents[1]
_stage_815_dir = _root / "benchmarks" / "stage_8_15"
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_stage_815_dir))

from generate_workloads import generate_all_workloads
from optiforge.core.solution import SolutionStatus
from optiforge.lp.simplex import SimplexTolerances
from optiforge.milp.solver import MILPSolver, MILPTolerances
from optiforge.model import VariableType


@dataclass
class BenchmarkEntry:
    workload: str
    config: str
    status: str
    objective: float | None
    node_count: int
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    ws_attempted: int
    ws_accepted: int
    ws_fallbacks: int
    dual_pivots: int
    cold_pivots: int
    pfi_updates: int
    pfi_refactorizations: int
    pfi_fallbacks: int
    max_constraint_violation: float
    max_integrality_violation: float


def compute_violations(model, sol):
    if sol.status != SolutionStatus.OPTIMAL or not sol.variable_values:
        return 0.0, 0.0
    vals = sol.variable_values
    max_c_viol = 0.0
    for c in model.constraints.values():
        lhs = sum(coeff * vals.get(var, 0.0) for var, coeff in c.coefficients.items())
        rhs = c.rhs
        sense = c.sense.value if hasattr(c.sense, "value") else str(c.sense)
        if sense == "<=":
            viol = max(0.0, lhs - rhs)
        elif sense == ">=":
            viol = max(0.0, rhs - lhs)
        elif sense == "==":
            viol = abs(lhs - rhs)
        else:
            viol = 0.0
        max_c_viol = max(max_c_viol, viol)

    max_int_viol = 0.0
    for v in model.variables.values():
        if v.variable_type in (VariableType.INTEGER, VariableType.BINARY):
            x = vals.get(v.name, 0.0)
            int_viol = abs(x - round(x))
            max_int_viol = max(max_int_viol, int_viol)

    return max_c_viol, max_int_viol


def run_benchmarks(reps: int = 5) -> list[BenchmarkEntry]:
    workloads = [w for w in generate_all_workloads() if w.n_integer + w.n_binary > 0]
    configs = {
        "A_ColdDefault": (
            MILPTolerances(warm_start_relaxations=False, inherit_basis_factorization=False),
            SimplexTolerances(use_pivot_factorization=False),
        ),
        "B_WarmStart": (
            MILPTolerances(warm_start_relaxations=True, inherit_basis_factorization=True),
            SimplexTolerances(use_pivot_factorization=False),
        ),
        "C_WarmStart_DualPFI": (
            MILPTolerances(warm_start_relaxations=True, inherit_basis_factorization=True),
            SimplexTolerances(use_pivot_factorization=True),
        ),
    }

    results: list[BenchmarkEntry] = []

    for w in workloads:
        print(f"\nWorkload: {w.name} (m={w.m}, n={w.n}, ints={w.n_integer+w.n_binary})")
        for cfg_name, (milp_tol, lp_tol) in configs.items():
            times_ms = []
            last_sol = None
            last_solver = None

            for rep in range(reps):
                gc.collect()
                solver = MILPSolver(
                    tolerances=milp_tol,
                    simplex_tolerances=lp_tol,
                    warm_start_relaxations=milp_tol.warm_start_relaxations,
                    inherit_basis_factorization=milp_tol.inherit_basis_factorization,
                )
                t0 = time.perf_counter()
                sol = solver.solve(w.model)
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)
                last_sol = sol
                last_solver = solver

            arr = np.array(times_ms)
            stats = last_solver.get_warm_start_stats()
            c_viol, int_viol = compute_violations(w.model, last_sol)

            entry = BenchmarkEntry(
                workload=w.name,
                config=cfg_name,
                status=last_sol.status.name,
                objective=last_sol.objective_value,
                node_count=last_sol.node_count,
                mean_ms=float(np.mean(arr)),
                median_ms=float(np.median(arr)),
                min_ms=float(np.min(arr)),
                max_ms=float(np.max(arr)),
                std_ms=float(np.std(arr)),
                ws_attempted=stats.attempted,
                ws_accepted=stats.accepted,
                ws_fallbacks=stats.fallbacks,
                dual_pivots=stats.dual_pivots,
                cold_pivots=stats.cold_pivots,
                pfi_updates=stats.pfi_updates,
                pfi_refactorizations=stats.pfi_refactorizations,
                pfi_fallbacks=stats.pfi_fallbacks,
                max_constraint_violation=c_viol,
                max_integrality_violation=int_viol,
            )
            results.append(entry)
            print(f"  {cfg_name:20s}: med={entry.median_ms:6.2f}ms, nodes={entry.node_count:2d}, ws_acc={entry.ws_accepted:2d}, pfi_upd={entry.pfi_updates:2d}, obj={entry.objective}")

    return results


if __name__ == "__main__":
    results = run_benchmarks(reps=5)
    out_path = Path("benchmarks/stage_8_16_results.json")
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nSaved benchmark results to {out_path}")
