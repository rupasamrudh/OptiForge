"""Stage 8.19: Head-to-Head Profiling of PFI vs IncrementalBasisFactorization.

Profiles the 9 target workloads across:
1. scale_lp_m6
2. scale_lp_m50
3. scale_lp_m100
4. scale_lp_m200
5. scale_lp_m500
6. phase1_heavy_lp
7. branching_heavy (MILP)
8. multilevel_bb (MILP)
9. highly_sparse_lp
"""
from __future__ import annotations

import gc
import json
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import sys
_root = str(Path(__file__).resolve().parents[1])
if _root not in sys.path:
    sys.path.insert(0, _root)

from optiforge.core.solution import SolutionStatus
from optiforge.lp.simplex import RevisedSimplexSolver, SimplexTolerances
from optiforge.lp.dual_simplex import DualRevisedSimplexSolver
from optiforge.milp.solver import MILPSolver, MILPTolerances
from optiforge.model import Model
from optiforge.numerics.pivot_basis import PivotUpdateFactorization
from optiforge.numerics.incremental_basis import IncrementalBasisFactorization
from benchmarks.stage_8_15.generate_workloads import generate_all_workloads
from benchmarks.run_stage_8_18 import generate_scale_sweep_lp


@dataclass
class HeadToHeadResult:
    workload: str
    m: int
    n: int
    category: str
    pfi_time_ms: float
    inc_time_ms: float
    pfi_pivots: int
    inc_pivots: int
    pfi_refactors: int
    inc_refactors: int
    pfi_solve_calls: int
    inc_solve_calls: int
    pfi_solve_time_ms: float
    inc_solve_time_ms: float
    pfi_update_time_ms: float
    inc_update_time_ms: float
    speedup_inc_over_pfi: float
    winner: str


def run_head_to_head():
    print("=" * 95)
    print("STAGE 8.19: PFI VS INCREMENTAL HEAD-TO-HEAD COMPARISON PROFILER")
    print("=" * 95)

    # 1. Scale Sweep Models: m=6, 50, 100, 200, 500
    scale_dims = [
        (6, 15, "scale_lp_m6"),
        (50, 80, "scale_lp_m50"),
        (100, 150, "scale_lp_m100"),
        (200, 200, "scale_lp_m200"),
        (500, 500, "scale_lp_m500"),
    ]

    models: list[tuple[Model, str, str]] = []
    for m, n, name in scale_dims:
        models.append((generate_scale_sweep_lp(m, n, seed=819_000 + m), name, "scale_lp"))

    wls = {w.name: w.model for w in generate_all_workloads()}
    models.append((wls["phase1_heavy_lp"], "phase1_heavy_lp", "diverse_lp"))
    models.append((wls["highly_sparse_lp"], "highly_sparse_lp", "diverse_lp"))
    models.append((wls["branching_heavy"], "branching_heavy", "milp"))
    models.append((wls["multilevel_bb"], "multilevel_bb", "milp"))

    results: list[HeadToHeadResult] = []

    for model, name, kind in models:
        m_con = len(model.constraints)
        n_vars = len(model.variables)
        print(f"\nProfiling {name} (m={m_con}, n={n_vars})...", flush=True)

        if kind in ("scale_lp", "diverse_lp"):
            # Profile PFI
            t0 = time.perf_counter()
            sol_pfi = RevisedSimplexSolver().solve(model, tolerances=SimplexTolerances(use_pivot_factorization=True))
            t_pfi = (time.perf_counter() - t0) * 1000

            # Profile Inc
            t0 = time.perf_counter()
            sol_inc = RevisedSimplexSolver().solve(model, tolerances=SimplexTolerances(use_incremental_basis=True))
            t_inc = (time.perf_counter() - t0) * 1000

            pfi_piv = sol_pfi.iterations
            inc_piv = sol_inc.iterations
            refactor_pfi = 0
            refactor_inc = 0

        else:
            # MILP
            # Profile PFI Warm-Start
            solver_pfi = MILPSolver(
                tolerances=MILPTolerances(warm_start_relaxations=True, inherit_basis_factorization=True),
                simplex_tolerances=SimplexTolerances(use_pivot_factorization=True),
            )
            t0 = time.perf_counter()
            sol_pfi = solver_pfi.solve(model)
            t_pfi = (time.perf_counter() - t0) * 1000

            # Profile Inc Warm-Start
            solver_inc = MILPSolver(
                tolerances=MILPTolerances(warm_start_relaxations=True, inherit_basis_factorization=True),
                simplex_tolerances=SimplexTolerances(use_pivot_factorization=False),
            )
            t0 = time.perf_counter()
            sol_inc = solver_inc.solve(model)
            t_inc = (time.perf_counter() - t0) * 1000

            pfi_piv = sol_pfi.iterations
            inc_piv = sol_inc.iterations
            refactor_pfi = solver_pfi.warm_start_stats.pfi_refactorizations
            refactor_inc = 0

        speedup_inc = t_pfi / max(1e-6, t_inc)
        winner = "Incremental" if t_inc < t_pfi else "PFI"

        res = HeadToHeadResult(
            workload=name,
            m=m_con,
            n=n_vars,
            category=kind,
            pfi_time_ms=round(t_pfi, 2),
            inc_time_ms=round(t_inc, 2),
            pfi_pivots=pfi_piv,
            inc_pivots=inc_piv,
            pfi_refactors=refactor_pfi,
            inc_refactors=refactor_inc,
            pfi_solve_calls=0,
            inc_solve_calls=0,
            pfi_solve_time_ms=0.0,
            inc_solve_time_ms=0.0,
            pfi_update_time_ms=0.0,
            inc_update_time_ms=0.0,
            speedup_inc_over_pfi=round(speedup_inc, 3),
            winner=winner,
        )
        results.append(res)
        print(f"  PFI: {t_pfi:8.1f} ms ({pfi_piv} piv) | Inc: {t_inc:8.1f} ms ({inc_piv} piv) | Inc/PFI speedup: {speedup_inc:5.2f}x | Winner: {winner}", flush=True)

    print("\n" + "=" * 95)
    print(f"{'Workload':22s} | {'m':3s} | {'n':3s} | {'PFI (ms)':10s} | {'Inc (ms)':10s} | {'Inc Speedup':12s} | {'Winner':12s}")
    print("=" * 95)
    for r in results:
        print(f"{r.workload:22s} | {r.m:3d} | {r.n:3d} | {r.pfi_time_ms:10.2f} | {r.inc_time_ms:10.2f} | {r.speedup_inc_over_pfi:10.2f}x | {r.winner:12s}")

    out_file = Path(_root) / "benchmarks" / "stage_8_19_head_to_head.json"
    with open(out_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n[OK] Head-to-head results saved to {out_file}")


if __name__ == "__main__":
    run_head_to_head()
