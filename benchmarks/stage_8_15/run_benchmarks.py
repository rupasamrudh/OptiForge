"""Stage 8.15: Comprehensive LP and MILP benchmark runner.

Runs all generated workloads across configuration matrix:
  LP:  A=DenseLU reference, B=Stage 8.13 optimized, C=Stage 8.14 PFI
  MILP: D=Default cold, E=Warm-start enabled

Measures correctness, performance, PFI behavior, and memory.
"""
from __future__ import annotations

import gc
import json
import sys
import time
import traceback
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Add both project root and benchmarks dir to path
_project_root = str(Path(__file__).resolve().parents[2])
_bench_dir = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _bench_dir not in sys.path:
    sys.path.insert(0, _bench_dir)

from generate_workloads import (
    WorkloadSpec, generate_all_workloads,
)
from optiforge.core.solution import SolutionStatus
from optiforge.lp.simplex import RevisedSimplexSolver, SimplexTolerances
from optiforge.milp.solver import MILPSolver, MILPTolerances
from optiforge.numerics.pivot_basis import PivotUpdateFactorization


# ---------------------------------------------------------------------------
# Result Containers
# ---------------------------------------------------------------------------

@dataclass
class LPResult:
    config: str
    workload: str
    status: str
    objective: float | None
    iterations: int
    times_ms: list[float]
    mean_ms: float = 0.0
    median_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    std_ms: float = 0.0
    primal_residual: float | None = None
    constraint_violation: float | None = None
    error: str = ""
    # PFI-specific
    pfi_total_updates: int = 0
    pfi_total_refactorizations: int = 0
    pfi_total_fallbacks: int = 0
    pfi_max_chain_length: int = 0


@dataclass
class MILPResult:
    config: str
    workload: str
    status: str
    objective: float | None
    iterations: int
    nodes: int
    times_ms: list[float]
    mean_ms: float = 0.0
    median_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    std_ms: float = 0.0
    warm_start_attempted: int = 0
    warm_start_accepted: int = 0
    warm_start_fallbacks: int = 0
    error: str = ""


@dataclass
class MemoryResult:
    config: str
    workload: str
    peak_traced_bytes: int
    current_traced_bytes: int


# ---------------------------------------------------------------------------
# Solver Wrappers
# ---------------------------------------------------------------------------

TIMEOUT_S = 30.0
LP_REPS = 10
MILP_REPS = 5


def _solve_lp(model, tol: SimplexTolerances) -> tuple[Any, float]:
    """Solve LP and return (solution, elapsed_ms)."""
    solver = RevisedSimplexSolver()
    t0 = time.perf_counter()
    sol = solver.solve(model, tolerances=tol)
    elapsed = (time.perf_counter() - t0) * 1000
    return sol, elapsed


def _solve_milp(model, milp_tol: MILPTolerances, lp_tol: SimplexTolerances) -> tuple[Any, float, Any]:
    """Solve MILP and return (solution, elapsed_ms, solver_instance)."""
    solver = MILPSolver(
        tolerances=milp_tol,
        simplex_tolerances=lp_tol,
        warm_start_relaxations=milp_tol.warm_start_relaxations,
        inherit_basis_factorization=milp_tol.inherit_basis_factorization,
    )
    t0 = time.perf_counter()
    sol = solver.solve(model)
    elapsed = (time.perf_counter() - t0) * 1000
    return sol, elapsed, solver


def _compute_residuals(model, sol) -> tuple[float | None, float | None]:
    """Compute primal residual and constraint violation."""
    if sol.status != SolutionStatus.OPTIMAL or not sol.variable_values:
        return None, None
    try:
        from optiforge.lp.standard_form import to_standard_form
        sf = to_standard_form(model)
        m, n = sf.A.shape
        x = np.zeros(n)
        for vname, col_idx in sf.original_variable_mapping.items():
            if vname in sol.variable_values:
                x[col_idx] = sol.variable_values[vname]
        residual = float(np.linalg.norm(sf.A @ x[:n] - sf.b, ord=np.inf))
        return residual, residual
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# LP Configurations
# ---------------------------------------------------------------------------

def _lp_configs() -> dict[str, SimplexTolerances]:
    """Return LP configuration matrix."""
    return {
        "A_DenseLU": SimplexTolerances(
            use_incremental_basis=False,
            use_pivot_factorization=False,
        ),
        "B_Optimized": SimplexTolerances(
            use_incremental_basis=False,
            use_pivot_factorization=False,
        ),
        "C_PFI": SimplexTolerances(
            use_incremental_basis=False,
            use_pivot_factorization=True,
            max_basis_updates=30,
        ),
    }


def _milp_configs() -> dict[str, tuple[MILPTolerances, SimplexTolerances]]:
    """Return MILP configuration matrix."""
    return {
        "D_Default": (
            MILPTolerances(warm_start_relaxations=False, inherit_basis_factorization=False),
            SimplexTolerances(use_pivot_factorization=False),
        ),
        "E_WarmStart": (
            MILPTolerances(warm_start_relaxations=True, inherit_basis_factorization=True),
            SimplexTolerances(use_pivot_factorization=False),
        ),
    }


# ---------------------------------------------------------------------------
# LP Benchmark
# ---------------------------------------------------------------------------

def run_lp_benchmarks(workloads: list[WorkloadSpec]) -> list[LPResult]:
    """Run LP workloads across all configurations."""
    lp_workloads = [w for w in workloads if w.n_integer + w.n_binary == 0]
    configs = _lp_configs()
    results = []

    for w in lp_workloads:
        print(f"\n  LP: {w.name} (m={w.m}, n={w.n})")
        for cfg_name, tol in configs.items():
            times = []
            sol = None
            error = ""
            pfi_updates = 0
            pfi_refactors = 0
            pfi_fallbacks = 0
            pfi_max_chain = 0

            try:
                for rep in range(LP_REPS):
                    gc.collect()
                    s, t = _solve_lp(w.model, tol)
                    times.append(t)
                    if rep == 0:
                        sol = s
            except Exception as e:
                error = str(e)[:200]
                sol = None

            if sol is not None:
                # PFI stats from last run (instrument if PFI)
                if cfg_name == "C_PFI" and sol.status == SolutionStatus.OPTIMAL:
                    # Run once more with instrumentation
                    try:
                        _orig_init = PivotUpdateFactorization.__init__
                        _last_pfi = [None]
                        def _track_init(self, *a, **kw):
                            _orig_init(self, *a, **kw)
                            _last_pfi[0] = self
                        PivotUpdateFactorization.__init__ = _track_init
                        s2, _ = _solve_lp(w.model, tol)
                        PivotUpdateFactorization.__init__ = _orig_init
                        if _last_pfi[0] is not None:
                            pf = _last_pfi[0]
                            pfi_updates = pf.total_updates
                            pfi_refactors = pf.total_refactorizations
                            pfi_fallbacks = pf.total_fallbacks
                            pfi_max_chain = pf.num_etas
                    except Exception:
                        PivotUpdateFactorization.__init__ = _orig_init

                pr, cv = _compute_residuals(w.model, sol)
                arr = np.array(times)
                r = LPResult(
                    config=cfg_name, workload=w.name,
                    status=sol.status.name,
                    objective=sol.objective_value,
                    iterations=sol.iterations,
                    times_ms=times,
                    mean_ms=float(np.mean(arr)),
                    median_ms=float(np.median(arr)),
                    min_ms=float(np.min(arr)),
                    max_ms=float(np.max(arr)),
                    std_ms=float(np.std(arr)),
                    primal_residual=pr,
                    constraint_violation=cv,
                    pfi_total_updates=pfi_updates,
                    pfi_total_refactorizations=pfi_refactors,
                    pfi_total_fallbacks=pfi_fallbacks,
                    pfi_max_chain_length=pfi_max_chain,
                )
            else:
                r = LPResult(
                    config=cfg_name, workload=w.name,
                    status="ERROR", objective=None, iterations=0,
                    times_ms=times, error=error,
                )
            results.append(r)
            status_str = r.status
            obj_str = f"{r.objective:.6f}" if r.objective is not None else "N/A"
            print(f"    {cfg_name:15s} | {status_str:10s} | obj={obj_str:>14s} | "
                  f"med={r.median_ms:8.3f}ms | iters={r.iterations}")

    return results


# ---------------------------------------------------------------------------
# MILP Benchmark
# ---------------------------------------------------------------------------

def run_milp_benchmarks(workloads: list[WorkloadSpec]) -> list[MILPResult]:
    """Run MILP workloads across configurations."""
    milp_workloads = [w for w in workloads if w.n_integer + w.n_binary > 0]
    configs = _milp_configs()
    results = []

    for w in milp_workloads:
        print(f"\n  MILP: {w.name} (m={w.m}, n={w.n}, int={w.n_integer}, bin={w.n_binary})")
        for cfg_name, (mtol, ltol) in configs.items():
            times = []
            sol = None
            solver_inst = None
            error = ""
            try:
                for rep in range(MILP_REPS):
                    gc.collect()
                    s, t, si = _solve_milp(w.model, mtol, ltol)
                    times.append(t)
                    if rep == 0:
                        sol = s
                        solver_inst = si
            except Exception as e:
                error = str(e)[:200]
                sol = None

            ws_attempted = 0
            ws_accepted = 0
            ws_fallbacks = 0
            if solver_inst is not None:
                try:
                    ws = solver_inst.get_warm_start_stats()
                    ws_attempted = ws.attempted
                    ws_accepted = ws.accepted
                    ws_fallbacks = ws.fallbacks
                except Exception:
                    pass

            if sol is not None:
                arr = np.array(times)
                r = MILPResult(
                    config=cfg_name, workload=w.name,
                    status=sol.status.name,
                    objective=sol.objective_value,
                    iterations=getattr(sol, 'iterations', 0),
                    nodes=getattr(sol, 'nodes_explored', 0) if hasattr(sol, 'nodes_explored') else 0,
                    times_ms=times,
                    mean_ms=float(np.mean(arr)),
                    median_ms=float(np.median(arr)),
                    min_ms=float(np.min(arr)),
                    max_ms=float(np.max(arr)),
                    std_ms=float(np.std(arr)),
                    warm_start_attempted=ws_attempted,
                    warm_start_accepted=ws_accepted,
                    warm_start_fallbacks=ws_fallbacks,
                )
            else:
                r = MILPResult(
                    config=cfg_name, workload=w.name,
                    status="ERROR", objective=None, iterations=0, nodes=0,
                    times_ms=times, error=error,
                )
            results.append(r)
            status_str = r.status
            obj_str = f"{r.objective:.4f}" if r.objective is not None else "N/A"
            print(f"    {cfg_name:15s} | {status_str:10s} | obj={obj_str:>12s} | "
                  f"med={r.median_ms:8.3f}ms | nodes={r.nodes}")

    return results


# ---------------------------------------------------------------------------
# Memory Benchmark
# ---------------------------------------------------------------------------

def run_memory_benchmarks(workloads: list[WorkloadSpec]) -> list[MemoryResult]:
    """Run memory benchmarks for representative workloads."""
    targets = ["small_dense_lp", "medium_dense_lp", "large_dense_lp"]
    mem_results = []

    for tname in targets:
        w = next((w for w in workloads if w.name == tname), None)
        if w is None:
            continue
        for cfg_name, tol in [("A_DenseLU", SimplexTolerances()),
                               ("C_PFI", SimplexTolerances(use_pivot_factorization=True))]:
            gc.collect()
            tracemalloc.start()
            try:
                _solve_lp(w.model, tol)
                current, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            mem_results.append(MemoryResult(cfg_name, w.name, peak, current))
            print(f"  Memory: {w.name:20s} | {cfg_name:15s} | peak={peak:,} bytes")

    return mem_results


# ---------------------------------------------------------------------------
# Cross-Configuration Equivalence
# ---------------------------------------------------------------------------

def check_equivalence(lp_results: list[LPResult], milp_results: list[MILPResult]) -> list[dict]:
    """Check cross-configuration objective equivalence."""
    issues = []
    OBJ_TOL = 1e-4

    # LP: compare all configs for each workload
    lp_by_wl: dict[str, list[LPResult]] = {}
    for r in lp_results:
        lp_by_wl.setdefault(r.workload, []).append(r)

    for wl, rs in lp_by_wl.items():
        optimal_rs = [r for r in rs if r.status == "OPTIMAL"]
        if len(optimal_rs) < 2:
            continue
        ref = optimal_rs[0]
        for r in optimal_rs[1:]:
            if ref.objective is not None and r.objective is not None:
                diff = abs(ref.objective - r.objective)
                if diff > OBJ_TOL:
                    issues.append({
                        "type": "LP_OBJECTIVE_MISMATCH",
                        "workload": wl,
                        "config_a": ref.config, "obj_a": ref.objective,
                        "config_b": r.config, "obj_b": r.objective,
                        "diff": diff,
                    })

    # MILP: compare configs
    milp_by_wl: dict[str, list[MILPResult]] = {}
    for r in milp_results:
        milp_by_wl.setdefault(r.workload, []).append(r)

    for wl, rs in milp_by_wl.items():
        optimal_rs = [r for r in rs if r.status == "OPTIMAL"]
        if len(optimal_rs) < 2:
            continue
        ref = optimal_rs[0]
        for r in optimal_rs[1:]:
            if ref.objective is not None and r.objective is not None:
                diff = abs(ref.objective - r.objective)
                if diff > OBJ_TOL:
                    issues.append({
                        "type": "MILP_OBJECTIVE_MISMATCH",
                        "workload": wl,
                        "config_a": ref.config, "obj_a": ref.objective,
                        "config_b": r.config, "obj_b": r.objective,
                        "diff": diff,
                    })

    return issues


# ---------------------------------------------------------------------------
# Performance Analysis
# ---------------------------------------------------------------------------

def compute_speedups(lp_results: list[LPResult]) -> list[dict]:
    """Compute speedup of C_PFI vs A_DenseLU."""
    speedups = []
    by_wl: dict[str, dict[str, LPResult]] = {}
    for r in lp_results:
        by_wl.setdefault(r.workload, {})[r.config] = r

    for wl, cfgs in by_wl.items():
        a = cfgs.get("A_DenseLU")
        c = cfgs.get("C_PFI")
        if a and c and a.median_ms > 0 and c.median_ms > 0:
            sp = a.median_ms / c.median_ms
            speedups.append({
                "workload": wl, "baseline_ms": a.median_ms, "pfi_ms": c.median_ms,
                "speedup": sp,
                "classification": "beneficial" if sp > 1.05 else ("slower" if sp < 0.95 else "neutral"),
            })
    return speedups


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("STAGE 8.15: Solver Benchmarking — Workload Diversity")
    print("=" * 70)

    # Generate workloads
    print("\n[1] Generating workloads...")
    workloads = generate_all_workloads()
    print(f"    Generated {len(workloads)} workloads.")

    # LP benchmarks
    print("\n[2] Running LP benchmarks...")
    lp_results = run_lp_benchmarks(workloads)

    # MILP benchmarks
    print("\n[3] Running MILP benchmarks...")
    milp_results = run_milp_benchmarks(workloads)

    # Memory benchmarks
    print("\n[4] Running memory benchmarks...")
    mem_results = run_memory_benchmarks(workloads)

    # Equivalence checks
    print("\n[5] Checking cross-configuration equivalence...")
    issues = check_equivalence(lp_results, milp_results)
    if issues:
        print(f"    WARNING: {len(issues)} equivalence issues found!")
        for iss in issues:
            print(f"    {iss}")
    else:
        print("    All configurations produce equivalent results.")

    # Speedup analysis
    print("\n[6] Computing PFI speedups...")
    speedups = compute_speedups(lp_results)
    for sp in speedups:
        print(f"    {sp['workload']:30s} | {sp['speedup']:.2f}x | {sp['classification']}")

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    def _serialize(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Cannot serialize {type(obj)}")

    with open(results_dir / "lp_results.json", "w") as f:
        json.dump([{k: v for k, v in r.__dict__.items()} for r in lp_results], f, indent=2, default=_serialize)
    with open(results_dir / "milp_results.json", "w") as f:
        json.dump([{k: v for k, v in r.__dict__.items()} for r in milp_results], f, indent=2, default=_serialize)
    with open(results_dir / "memory_results.json", "w") as f:
        json.dump([{k: v for k, v in r.__dict__.items()} for r in mem_results], f, indent=2, default=_serialize)
    with open(results_dir / "equivalence_issues.json", "w") as f:
        json.dump(issues, f, indent=2, default=_serialize)
    with open(results_dir / "speedups.json", "w") as f:
        json.dump(speedups, f, indent=2, default=_serialize)

    print(f"\n[7] Results saved to {results_dir}")
    print("\n" + "=" * 70)
    print("STAGE 8.15 BENCHMARK RUN COMPLETE")
    print("=" * 70)

    # Summary
    n_lp = len([r for r in lp_results if r.config == "A_DenseLU"])
    n_milp = len([r for r in milp_results if r.config == "D_Default"])
    n_correct = len([r for r in lp_results if r.status == "OPTIMAL"])
    n_issues = len(issues)
    beneficial = len([s for s in speedups if s["classification"] == "beneficial"])
    neutral = len([s for s in speedups if s["classification"] == "neutral"])
    slower = len([s for s in speedups if s["classification"] == "slower"])

    print(f"\n  LP workloads:        {n_lp}")
    print(f"  MILP workloads:      {n_milp}")
    print(f"  LP optimal results:  {n_correct}/{len(lp_results)}")
    print(f"  Equivalence issues:  {n_issues}")
    print(f"  PFI beneficial:      {beneficial}")
    print(f"  PFI neutral:         {neutral}")
    print(f"  PFI slower:          {slower}")
    if speedups:
        all_sp = [s["speedup"] for s in speedups]
        print(f"  Median speedup:      {np.median(all_sp):.2f}x")
        print(f"  Min speedup:         {np.min(all_sp):.2f}x")
        print(f"  Max speedup:         {np.max(all_sp):.2f}x")


if __name__ == "__main__":
    main()
