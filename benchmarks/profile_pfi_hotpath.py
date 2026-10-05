"""Stage 8.19/8.26: Detailed Hot-Path Profiler for PivotUpdateFactorization (PFI).

Stage 8.26 update: InstrumentedPFI._raw_solve / _raw_solve_transpose updated to
use the Compact WY kernel (_W, _S, _pivots, _k) introduced in Stage 8.25.
All timing structure is preserved; the sequential _etas loop is removed.
This is a benchmark-only fix; no production code is modified.

Instruments every phase of PFI and IncrementalBasis:
- Base LU solve
- Forward eta application
- Transpose eta application
- Residual checks (GEMV + norm)
- Eta vector construction
- Full refactorization (dense_lu_factor)
- Simplex loop overheads (pricing, ratio test, etc.)
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
from benchmarks.run_stage_8_18 import generate_scale_sweep_lp, fetch_netlib_instance


@dataclass
class PFIProfileResult:
    workload_name: str
    m: int
    n: int
    solver_type: str  # 'primal_lp', 'dual_lp', 'milp'
    config: str       # 'PFI' or 'Incremental'
    total_time_ms: float
    pivots: int
    refactorizations: int
    max_eta_chain: int
    avg_eta_chain: float
    # Breakdown in ms
    base_solve_ms: float
    eta_forward_ms: float
    eta_transpose_ms: float
    residual_check_ms: float
    eta_construct_ms: float
    refactor_ms: float
    other_simplex_ms: float
    peak_mem_mb: float


class InstrumentedPFI(PivotUpdateFactorization):
    def __init__(self, *args, **kwargs):
        self.prof_base_solve = 0.0
        self.prof_eta_forward = 0.0
        self.prof_eta_transpose = 0.0
        self.prof_residual_check = 0.0
        self.prof_eta_construct = 0.0
        self.prof_refactor = 0.0
        self.eta_chain_lengths: list[int] = []
        super().__init__(*args, **kwargs)

    def _factorize_base(self, reason):
        t0 = time.perf_counter()
        super()._factorize_base(reason)
        self.prof_refactor += (time.perf_counter() - t0) * 1000

    def _raw_solve(self, rhs_arr):
        # Stage 8.26 fix: use Compact WY state instead of removed _etas list
        t0 = time.perf_counter()
        if self._base_factorization is not None:
            v = self._base_factorization.solve(rhs_arr)
        else:
            v = lu_solve(self._base_lu, rhs_arr)
        t1 = time.perf_counter()
        self.prof_base_solve += (t1 - t0) * 1000

        k = self._k
        self.eta_chain_lengths.append(k)
        if k == 0:
            return v

        t2 = time.perf_counter()
        pivs = self._pivots[:k]
        W_k  = self._W[:, :k]
        S_k  = self._S[:k, :k]
        if rhs_arr.ndim == 1:
            z = S_k @ v[pivs]
            v = v + W_k @ z
        else:
            z = S_k @ v[pivs, :]
            v = v + W_k @ z
        self.prof_eta_forward += (time.perf_counter() - t2) * 1000
        return v

    def _raw_solve_transpose(self, rhs_arr):
        # Stage 8.26 fix: use Compact WY state instead of removed _etas list
        k = self._k
        self.eta_chain_lengths.append(k)
        if k == 0:
            t0 = time.perf_counter()
            if self._base_factorization is not None:
                res = self._base_factorization.solve_transpose(rhs_arr)
            else:
                res = lu_solve_transpose(self._base_lu, rhs_arr)
            self.prof_base_solve += (time.perf_counter() - t0) * 1000
            return res

        pivs = self._pivots[:k]
        W_k  = self._W[:, :k]
        S_k  = self._S[:k, :k]

        t1 = time.perf_counter()
        if rhs_arr.ndim == 1:
            q = W_k.T @ rhs_arr
            r = S_k.T @ q
            u = rhs_arr.copy()
            import numpy as _np
            _np.add.at(u, pivs, r)
        else:
            q = W_k.T @ rhs_arr
            r = S_k.T @ q
            u = rhs_arr.copy()
            import numpy as _np
            _np.add.at(u, pivs, r)
        self.prof_eta_transpose += (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        if self._base_factorization is not None:
            res = self._base_factorization.solve_transpose(u)
        else:
            res = lu_solve_transpose(self._base_lu, u)
        self.prof_base_solve += (time.perf_counter() - t2) * 1000
        return res

    def residual(self, rhs, x):
        t0 = time.perf_counter()
        res = super().residual(rhs, x)
        self.prof_residual_check += (time.perf_counter() - t0) * 1000
        return res

    def transpose_residual(self, rhs, y):
        t0 = time.perf_counter()
        res = super().transpose_residual(rhs, y)
        self.prof_residual_check += (time.perf_counter() - t0) * 1000
        return res

    def replace_column(self, position, new_column, search_direction=None):
        t0 = time.perf_counter()
        res = super().replace_column(position, new_column, search_direction=search_direction)
        self.prof_eta_construct += (time.perf_counter() - t0) * 1000
        return res


from optiforge.numerics.lu import lu_solve, lu_solve_transpose


def profile_lp_workload(model: Model, name: str) -> PFIProfileResult:
    """Profile LP under PFI with detailed component timer."""
    import optiforge.lp.simplex as simplex_mod
    import optiforge.numerics.pivot_basis as pb_mod

    # Patch PivotUpdateFactorization
    orig_pfi = simplex_mod.PivotUpdateFactorization
    simplex_mod.PivotUpdateFactorization = InstrumentedPFI

    instances: list[InstrumentedPFI] = []
    _orig_init = InstrumentedPFI.__init__
    def _tracking_init(self, *a, **kw):
        _orig_init(self, *a, **kw)
        instances.append(self)
    InstrumentedPFI.__init__ = _tracking_init

    tracemalloc.start()
    t0 = time.perf_counter()
    solver = RevisedSimplexSolver()
    sol = solver.solve(model, tolerances=SimplexTolerances(use_pivot_factorization=True))
    total_ms = (time.perf_counter() - t0) * 1000
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Unpatch
    simplex_mod.PivotUpdateFactorization = orig_pfi
    InstrumentedPFI.__init__ = _orig_init

    base_solve = sum(inst.prof_base_solve for inst in instances)
    eta_fwd = sum(inst.prof_eta_forward for inst in instances)
    eta_trans = sum(inst.prof_eta_transpose for inst in instances)
    res_chk = sum(inst.prof_residual_check for inst in instances)
    eta_const = sum(inst.prof_eta_construct for inst in instances)
    refact = sum(inst.prof_refactor for inst in instances)
    total_refact = sum(inst.total_refactorizations for inst in instances)
    all_chains = [c for inst in instances for c in inst.eta_chain_lengths]
    max_chain = max(all_chains) if all_chains else 0
    avg_chain = float(np.mean(all_chains)) if all_chains else 0.0

    pfi_tracked = base_solve + eta_fwd + eta_trans + res_chk + eta_const + refact
    other_simplex = max(0.0, total_ms - pfi_tracked)

    m = len(model.constraints)
    n = len(model.variables)

    return PFIProfileResult(
        workload_name=name,
        m=m,
        n=n,
        solver_type="primal_lp",
        config="PFI",
        total_time_ms=total_ms,
        pivots=sol.iterations,
        refactorizations=total_refact,
        max_eta_chain=max_chain,
        avg_eta_chain=avg_chain,
        base_solve_ms=base_solve,
        eta_forward_ms=eta_fwd,
        eta_transpose_ms=eta_trans,
        residual_check_ms=res_chk,
        eta_construct_ms=eta_const,
        refactor_ms=refact,
        other_simplex_ms=other_simplex,
        peak_mem_mb=peak_mem / (1024 * 1024),
    )


def profile_milp_workload(model: Model, name: str) -> PFIProfileResult:
    """Profile MILP under Warm-Start PFI with detailed component timer."""
    import optiforge.lp.dual_simplex as ds_mod
    orig_pfi = ds_mod.PivotUpdateFactorization
    ds_mod.PivotUpdateFactorization = InstrumentedPFI

    instances: list[InstrumentedPFI] = []
    _orig_init = InstrumentedPFI.__init__
    def _tracking_init(self, *a, **kw):
        _orig_init(self, *a, **kw)
        instances.append(self)
    InstrumentedPFI.__init__ = _tracking_init

    tracemalloc.start()
    t0 = time.perf_counter()
    solver = MILPSolver(
        tolerances=MILPTolerances(warm_start_relaxations=True, inherit_basis_factorization=True),
        simplex_tolerances=SimplexTolerances(use_pivot_factorization=True),
    )
    sol = solver.solve(model)
    total_ms = (time.perf_counter() - t0) * 1000
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    ds_mod.PivotUpdateFactorization = orig_pfi
    InstrumentedPFI.__init__ = _orig_init

    base_solve = sum(inst.prof_base_solve for inst in instances)
    eta_fwd = sum(inst.prof_eta_forward for inst in instances)
    eta_trans = sum(inst.prof_eta_transpose for inst in instances)
    res_chk = sum(inst.prof_residual_check for inst in instances)
    eta_const = sum(inst.prof_eta_construct for inst in instances)
    refact = sum(inst.prof_refactor for inst in instances)
    total_refact = sum(inst.total_refactorizations for inst in instances)
    all_chains = [c for inst in instances for c in inst.eta_chain_lengths]
    max_chain = max(all_chains) if all_chains else 0
    avg_chain = float(np.mean(all_chains)) if all_chains else 0.0

    pfi_tracked = base_solve + eta_fwd + eta_trans + res_chk + eta_const + refact
    other_simplex = max(0.0, total_ms - pfi_tracked)

    m = len(model.constraints)
    n = len(model.variables)

    return PFIProfileResult(
        workload_name=name,
        m=m,
        n=n,
        solver_type="milp",
        config="PFI",
        total_time_ms=total_ms,
        pivots=sol.iterations,
        refactorizations=total_refact,
        max_eta_chain=max_chain,
        avg_eta_chain=avg_chain,
        base_solve_ms=base_solve,
        eta_forward_ms=eta_fwd,
        eta_transpose_ms=eta_trans,
        residual_check_ms=res_chk,
        eta_construct_ms=eta_const,
        refactor_ms=refact,
        other_simplex_ms=other_simplex,
        peak_mem_mb=peak_mem / (1024 * 1024),
    )


def run_target_profiles():
    print("=" * 80)
    print("STAGE 8.19 HOT-PATH PROFILING EXPERIMENT")
    print("=" * 80)

    # 1. m=6, 50, 100, 200, 500 Scale LPs
    scale_configs = [
        (6, 15, "scale_lp_m6"),
        (50, 80, "scale_lp_m50"),
        (100, 150, "scale_lp_m100"),
        (200, 200, "scale_lp_m200"),
    ]

    all_results: list[PFIProfileResult] = []

    for m, n, name in scale_configs:
        model = generate_scale_sweep_lp(m, n, seed=819000 + m)
        print(f"Profiling Scale LP: {name} (m={m}, n={n})...", flush=True)
        res = profile_lp_workload(model, name)
        all_results.append(res)

    # Diverse LPs: phase1_heavy_lp, highly_sparse_lp
    from benchmarks.stage_8_15.generate_workloads import generate_all_workloads
    wls = {w.name: w.model for w in generate_all_workloads()}

    print("Profiling phase1_heavy_lp...", flush=True)
    all_results.append(profile_lp_workload(wls["phase1_heavy_lp"], "phase1_heavy_lp"))

    print("Profiling highly_sparse_lp...", flush=True)
    all_results.append(profile_lp_workload(wls["highly_sparse_lp"], "highly_sparse_lp"))

    # MILPs: branching_heavy, multilevel_bb
    print("Profiling branching_heavy (MILP)...", flush=True)
    all_results.append(profile_milp_workload(wls["branching_heavy"], "branching_heavy"))

    print("Profiling multilevel_bb (MILP)...", flush=True)
    all_results.append(profile_milp_workload(wls["multilevel_bb"], "multilevel_bb"))

    # Print Summary Table
    print("\n" + "=" * 105)
    print(f"{'Workload':20s} | {'m':3s} | {'Total ms':9s} | {'Base LU':8s} | {'Eta Fwd':8s} | {'Eta Back':8s} | {'Res Chk':8s} | {'Eta Make':8s} | {'Refactor':8s} | {'Other':8s}")
    print("=" * 105)
    for r in all_results:
        print(
            f"{r.workload_name:20s} | {r.m:3d} | {r.total_time_ms:9.1f} | "
            f"{r.base_solve_ms:8.1f} | {r.eta_forward_ms:8.1f} | {r.eta_transpose_ms:8.1f} | "
            f"{r.residual_check_ms:8.1f} | {r.eta_construct_ms:8.1f} | {r.refactor_ms:8.1f} | {r.other_simplex_ms:8.1f}"
        )

    # Save to JSON
    out_file = Path(_root) / "benchmarks" / "stage_8_19_profile_results.json"
    with open(out_file, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\n[OK] Profiling data written to {out_file}")


if __name__ == "__main__":
    run_target_profiles()
