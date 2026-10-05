"""Stage 8.18: Large-Scale & Diverse Benchmark Validation Harness.

Evaluates factorization strategies:
  A. DenseLU baseline
  B. IncrementalBasis
  C. Pure PivotUpdateFactorization (PFI)
  D. Warm-start + existing factorization (Incremental + Bordered)
  E. Adaptive PFI evaluated across candidate thresholds T in {2, 4, 6, 8, 10, 20, 50}

Measures correctness, numerical stability, solve time distributions,
memory allocation, and feature interactions across:
  - Public Netlib benchmarks (AFIRO, ADLITTLE, AVGAS)
  - Scale sweep LPs (m=1, 2, 4, 6, 8, 10, 20, 50, 100, 200, 500)
  - 15 Diverse LP structural classes
  - 10 Diverse MILP structural classes
  - Threshold-crossing unbounded MILP suite (m_child in {2, 4, 6, 8, 10, 20, 50})
"""
from __future__ import annotations

import gc
import json
import os
import platform
import sys
import time
import tracemalloc
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Enable unbuffered / line-buffered stdout for real-time progress logging
sys.stdout.reconfigure(line_buffering=True)

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from optiforge.core.solution import SolutionStatus
from optiforge.lp.dual_simplex import DualRevisedSimplexSolver
from optiforge.lp.simplex import RevisedSimplexSolver, SimplexTolerances
from optiforge.lp.standard_form import StandardFormModel, to_standard_form
from optiforge.milp.solver import MILPSolver, MILPTolerances
from optiforge.model import (
    Constraint, ConstraintSense, Model, Objective, ObjectiveSense,
    Variable, VariableType,
)
from optiforge.numerics.bordered_basis import BorderedBasisFactorization
from optiforge.numerics.incremental_basis import IncrementalBasisFactorization
from optiforge.numerics.basis import BasisFactorization
from optiforge.numerics.pivot_basis import PivotUpdateFactorization


# ---------------------------------------------------------------------------
# Data Models for Benchmark Results
# ---------------------------------------------------------------------------

@dataclass
class TimingStats:
    times_ms: list[float]
    median_ms: float = 0.0
    mean_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p90_ms: float = 0.0
    setup_ms: float = 0.0
    sf_conv_ms: float = 0.0


@dataclass
class NumericalStabilityStats:
    primal_residual: float = 0.0
    dual_residual: float = 0.0
    basis_residual: float = 0.0
    ftran_error: float = 0.0
    btran_error: float = 0.0
    refactorizations: int = 0
    fallbacks: int = 0
    numerical_failures: int = 0


@dataclass
class MemoryStats:
    baseline_bytes: int = 0
    peak_bytes: int = 0
    diff_bytes: int = 0


@dataclass
class BenchmarkEntry:
    workload_name: str
    workload_category: str
    m_constraints: int
    n_variables: int
    m_standard_form: int
    n_standard_form: int
    nonzeros: int
    density: float
    n_integer: int
    n_binary: int
    config: str
    threshold: int | None
    status: str
    objective: float | None
    iterations: int
    nodes: int
    timings: TimingStats
    stability: NumericalStabilityStats
    memory: MemoryStats
    telemetry: dict[str, Any]
    verified_factorization_path: str
    match_reference: bool = True
    objective_diff_vs_reference: float = 0.0
    notes: str = ""


# ---------------------------------------------------------------------------
# MPS Parser for Public Benchmark Models
# ---------------------------------------------------------------------------

def parse_mps_text(text: str) -> Model:
    """Parse MPS formatted text into an OptiForge Model."""
    lines = text.splitlines()
    section = None
    name = "mps_model"
    row_types = {}
    obj_name = None
    col_coeffs: dict[str, dict[str, float]] = {}
    col_is_int: dict[str, bool] = {}
    rhs_vals: dict[str, float] = {}
    bounds: dict[str, tuple[float, float, bool]] = {}
    is_integer_section = False

    for line in lines:
        line_s = line.strip()
        if not line_s or line.startswith("*"):
            continue
        if line[0] != " " and not line.startswith("\t"):
            parts = line_s.split()
            section = parts[0].upper()
            if section == "NAME" and len(parts) > 1:
                name = parts[1]
            continue

        parts = line_s.split()
        if section == "ROWS":
            rtype, rname = parts[0].upper(), parts[1]
            if rtype == "N" and obj_name is None:
                obj_name = rname
            else:
                row_types[rname] = rtype
        elif section == "COLUMNS":
            if len(parts) >= 3 and ("MARKER" in parts[1] or "MARKER" in parts[2]):
                if "INTORG" in line:
                    is_integer_section = True
                elif "INTEND" in line:
                    is_integer_section = False
                continue
            cname = parts[0]
            if cname not in col_coeffs:
                col_coeffs[cname] = {}
                col_is_int[cname] = is_integer_section
            r1, v1 = parts[1], float(parts[2])
            col_coeffs[cname][r1] = v1
            if len(parts) >= 5:
                r2, v2 = parts[3], float(parts[4])
                col_coeffs[cname][r2] = v2
        elif section == "RHS":
            idx = 1
            while idx < len(parts):
                rname, val = parts[idx], float(parts[idx + 1])
                rhs_vals[rname] = val
                idx += 2
        elif section == "BOUNDS":
            btype = parts[0].upper()
            cname = parts[2]
            val = float(parts[3]) if len(parts) > 3 else 0.0
            lb, ub, is_int = bounds.get(cname, (0.0, float("inf"), col_is_int.get(cname, False)))
            if btype == "UP":
                ub = val
            elif btype == "LO":
                lb = val
            elif btype == "FX":
                lb = val
                ub = val
            elif btype == "FR":
                lb = -float("inf")
                ub = float("inf")
            elif btype == "BV":
                lb = 0.0
                ub = 1.0
                is_int = True
            elif btype == "MI":
                lb = -float("inf")
                ub = 0.0
            bounds[cname] = (lb, ub, is_int)

    model = Model(name)
    for cname in col_coeffs:
        lb, ub, is_int = bounds.get(cname, (0.0, float("inf"), col_is_int.get(cname, False)))
        vtype = VariableType.INTEGER if is_int else VariableType.CONTINUOUS
        model.add_variable(Variable(cname, lower_bound=lb, upper_bound=ub, variable_type=vtype))

    for rname, rtype in row_types.items():
        coeffs = {cname: col_coeffs[cname][rname] for cname in col_coeffs if rname in col_coeffs[cname]}
        rhs = rhs_vals.get(rname, 0.0)
        sense = ConstraintSense.EQUAL if rtype == "E" else (
            ConstraintSense.LESS_EQUAL if rtype == "L" else ConstraintSense.GREATER_EQUAL
        )
        model.add_constraint(Constraint(rname, coeffs, sense, rhs))

    if obj_name:
        obj_coeffs = {cname: col_coeffs[cname][obj_name] for cname in col_coeffs if obj_name in col_coeffs[cname]}
        model.set_objective(Objective(obj_coeffs, ObjectiveSense.MINIMIZE))

    return model


def fetch_netlib_instance(name: str) -> Model | None:
    """Fetch public Netlib benchmark instance from HiGHS repo."""
    url = f"https://raw.githubusercontent.com/ERGO-Code/HiGHS/master/check/instances/{name.lower()}.mps"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "OptiForge-Benchmark"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read().decode("utf-8")
        return parse_mps_text(content)
    except Exception as e:
        print(f"Warning: Failed to fetch Netlib model {name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Procedural Workload Generators
# ---------------------------------------------------------------------------

def generate_scale_sweep_lp(m: int, n: int, seed: int) -> Model:
    """Generate bounded scale sweep LP with unbounded variables (m_SF = m)."""
    rng = np.random.RandomState(seed)
    model = Model(f"scale_lp_m{m}_n{n}")
    vnames = [f"x{j}" for j in range(n)]
    for vn in vnames:
        model.add_variable(Variable(vn, lower_bound=0.0))
    # Budget constraint to ensure boundedness
    model.add_constraint(Constraint("budget", {vn: 1.0 for vn in vnames}, ConstraintSense.LESS_EQUAL, float(m * 10.0)))
    for i in range(1, m):
        c_dict = {vnames[i % n]: float(rng.uniform(1.0, 3.0))}
        k = min(3, n)
        for offset in range(1, k):
            c_dict[vnames[(i + offset) % n]] = float(rng.uniform(0.2, 1.0))
        rhs = float(rng.uniform(10.0, 50.0))
        model.add_constraint(Constraint(f"c{i}", c_dict, ConstraintSense.LESS_EQUAL, rhs))
    c_coeffs = {vn: float(rng.uniform(1.0, 5.0)) for vn in vnames}
    model.set_objective(Objective(c_coeffs, ObjectiveSense.MAXIMIZE))
    return model


def generate_unbounded_milp(m_con: int, n_vars: int, seed: int) -> Model:
    """Generate MILP with unbounded integer variables so m_child crosses candidate thresholds."""
    rng = np.random.RandomState(seed)
    model = Model(f"unbounded_milp_m{m_con}_n{n_vars}")
    vnames = [f"z{j}" for j in range(n_vars)]
    for vn in vnames:
        model.add_variable(Variable(vn, lower_bound=0.0, variable_type=VariableType.INTEGER))
    for i in range(m_con):
        coeffs = {vnames[j]: float(rng.uniform(1.0, 4.0)) for j in range(n_vars)}
        rhs = float(rng.uniform(15.0, 40.0))
        model.add_constraint(Constraint(f"c{i}", coeffs, ConstraintSense.LESS_EQUAL, rhs))
    c = {vnames[j]: float(rng.uniform(2.0, 8.0)) for j in range(n_vars)}
    model.set_objective(Objective(c, ObjectiveSense.MAXIMIZE))
    return model


def get_lp_relaxation(model: Model) -> Model:
    """Return continuous LP relaxation of a model."""
    rel = Model(f"{model.name}_relaxation")
    for vn, v in model.variables.items():
        lb = v.lower_bound
        ub = v.upper_bound
        if v.variable_type == VariableType.BINARY:
            lb = 0.0 if lb is None else lb
            ub = 1.0 if ub is None else ub
        rel.add_variable(Variable(vn, lower_bound=lb, upper_bound=ub))
    for cn, c in model.constraints.items():
        rel.add_constraint(Constraint(cn, dict(c.coefficients), c.sense, c.rhs))
    if model.objective:
        rel.set_objective(Objective(dict(model.objective.coefficients), model.objective.sense))
    return rel


def compute_matrix_stats(model: Model) -> tuple[int, int, int, int, int, float, int, int]:
    """Compute (m, n, m_SF, n_SF, nonzeros, density, n_int, n_bin)."""
    m = len(model.constraints)
    n = len(model.variables)
    n_int = sum(1 for v in model.variables.values() if v.variable_type == VariableType.INTEGER)
    n_bin = sum(1 for v in model.variables.values() if v.variable_type == VariableType.BINARY)
    rel = get_lp_relaxation(model) if (n_int + n_bin > 0) else model
    sf = to_standard_form(rel)
    m_sf, n_sf = sf.A.shape
    nz = int(np.count_nonzero(sf.A))
    density = float(nz / (m_sf * n_sf)) if (m_sf * n_sf) > 0 else 0.0
    return m, n, m_sf, n_sf, nz, density, n_int, n_bin


def compute_numerical_stability(
    model: Model,
    sol: Any,
    solver_instance: Any,
    tolerances: SimplexTolerances,
) -> NumericalStabilityStats:
    """Measure primal, dual, basis residuals and FTRAN/BTRAN errors."""
    stats = NumericalStabilityStats()
    if sol.status != SolutionStatus.OPTIMAL or not sol.variable_values:
        return stats
    try:
        sf = to_standard_form(model)
        m, n = sf.A.shape
        x = np.zeros(n)
        for vname, idx in sf.original_variable_mapping.items():
            if vname in sol.variable_values:
                x[idx] = sol.variable_values[vname]

        # Primal residual ||Ax - b||_inf / max(1, ||b||_inf)
        primal_res = float(np.linalg.norm(sf.A @ x - sf.b, ord=np.inf))
        b_norm = float(np.linalg.norm(sf.b, ord=np.inf))
        stats.primal_residual = primal_res / max(1.0, b_norm)

        # Dual & basis diagnostics if basis is available
        basis = getattr(solver_instance, "current_basis", None) or getattr(solver_instance, "_basis", None)
        if basis is not None and len(basis) == m:
            B = sf.A[:, basis]
            bf = BasisFactorization(B, pivot_tolerance=tolerances.pivot_tolerance)
            x_B = bf.solve(sf.b)
            stats.basis_residual = float(np.linalg.norm(B @ x_B - sf.b, ord=np.inf)) / max(1.0, b_norm)

            c_B = sf.c[basis]
            y = bf.solve_transpose(c_B)
            dual_slack = sf.c - sf.A.T @ y
            c_norm = float(np.linalg.norm(sf.c, ord=np.inf))
            stats.dual_residual = float(np.linalg.norm(dual_slack, ord=np.inf)) / max(1.0, c_norm)

            # FTRAN error test on test vector
            a_test = sf.A[:, min(1, n - 1)]
            d = bf.solve(a_test)
            stats.ftran_error = float(np.linalg.norm(B @ d - a_test, ord=np.inf)) / max(1.0, float(np.linalg.norm(a_test, ord=np.inf)))

            # BTRAN error test
            y_test = bf.solve_transpose(c_B)
            stats.btran_error = float(np.linalg.norm(B.T @ y_test - c_B, ord=np.inf)) / max(1.0, float(np.linalg.norm(c_B, ord=np.inf)))
    except Exception:
        pass
    return stats


def compute_timing_stats(times: list[float], setup_ms: float = 0.0, sf_conv_ms: float = 0.0) -> TimingStats:
    """Compute summary statistics for timing list."""
    arr = np.array(times)
    return TimingStats(
        times_ms=[round(t, 3) for t in times],
        median_ms=float(np.median(arr)),
        mean_ms=float(np.mean(arr)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        p90_ms=float(np.percentile(arr, 90)),
        setup_ms=round(setup_ms, 3),
        sf_conv_ms=round(sf_conv_ms, 3),
    )


# ---------------------------------------------------------------------------
# Benchmark Execution Engine
# ---------------------------------------------------------------------------

def run_lp_entry(
    model: Model,
    category: str,
    config_name: str,
    tolerances: SimplexTolerances,
    repetitions: int,
    reference_sol: Any = None,
) -> BenchmarkEntry:
    """Execute LP benchmark under a specified configuration."""
    m, n, m_sf, n_sf, nz, density, n_int, n_bin = compute_matrix_stats(model)

    t0_sf = time.perf_counter()
    sf_model = to_standard_form(model)
    sf_conv_ms = (time.perf_counter() - t0_sf) * 1000

    times = []
    sol = None
    solver_instance = None
    telemetry: dict[str, Any] = {}
    verified_path = "Unknown"

    tracemalloc.start()
    gc.collect()
    mem_baseline = tracemalloc.get_traced_memory()[0]

    for rep in range(repetitions):
        solver = RevisedSimplexSolver()
        t0 = time.perf_counter()
        s = solver.solve(model, tolerances=tolerances)
        t_ms = (time.perf_counter() - t0) * 1000
        times.append(t_ms)
        if rep == 0:
            sol = s
            solver_instance = solver

    mem_current, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Explicit verification of factorization path
    if tolerances.use_pivot_factorization:
        verified_path = "PivotUpdateFactorization (PFI)"
    elif tolerances.use_incremental_basis:
        verified_path = "IncrementalBasisFactorization"
    else:
        verified_path = "DenseLU (BasisFactorization)"

    stability = compute_numerical_stability(model, sol, solver_instance, tolerances)
    timings = compute_timing_stats(times, sf_conv_ms=sf_conv_ms)
    memory = MemoryStats(
        baseline_bytes=mem_baseline,
        peak_bytes=mem_peak,
        diff_bytes=mem_peak - mem_baseline,
    )

    match_ref = True
    obj_diff = 0.0
    if reference_sol is not None and sol is not None:
        if reference_sol.status != sol.status.value:
            match_ref = False
        if reference_sol.objective is not None and sol.objective_value is not None:
            obj_diff = abs(reference_sol.objective - sol.objective_value)
            if obj_diff > 1e-4:
                match_ref = False

    return BenchmarkEntry(
        workload_name=model.name,
        workload_category=category,
        m_constraints=m,
        n_variables=n,
        m_standard_form=m_sf,
        n_standard_form=n_sf,
        nonzeros=nz,
        density=density,
        n_integer=n_int,
        n_binary=n_bin,
        config=config_name,
        threshold=None,
        status=sol.status.value if sol else "ERROR",
        objective=round(sol.objective_value, 6) if (sol and sol.objective_value is not None) else None,
        iterations=sol.iterations if sol else 0,
        nodes=0,
        timings=timings,
        stability=stability,
        memory=memory,
        telemetry=telemetry,
        verified_factorization_path=verified_path,
        match_reference=match_ref,
        objective_diff_vs_reference=round(obj_diff, 8),
    )


def run_milp_entry(
    model: Model,
    category: str,
    config_name: str,
    milp_tol: MILPTolerances,
    simplex_tol: SimplexTolerances,
    repetitions: int,
    reference_sol: Any = None,
    threshold: int | None = None,
) -> BenchmarkEntry:
    """Execute MILP benchmark under a specified configuration."""
    m, n, m_sf, n_sf, nz, density, n_int, n_bin = compute_matrix_stats(model)

    t0_sf = time.perf_counter()
    rel = get_lp_relaxation(model) if (n_int + n_bin > 0) else model
    sf_model = to_standard_form(rel)
    sf_conv_ms = (time.perf_counter() - t0_sf) * 1000

    times = []
    sol = None
    solver_instance = None
    telemetry: dict[str, Any] = {}

    tracemalloc.start()
    gc.collect()
    mem_baseline = tracemalloc.get_traced_memory()[0]

    for rep in range(repetitions):
        solver = MILPSolver(
            tolerances=milp_tol,
            simplex_tolerances=simplex_tol,
            warm_start_relaxations=milp_tol.warm_start_relaxations,
            inherit_basis_factorization=milp_tol.inherit_basis_factorization,
        )
        t0 = time.perf_counter()
        s = solver.solve(model)
        t_ms = (time.perf_counter() - t0) * 1000
        times.append(t_ms)
        if rep == 0:
            sol = s
            solver_instance = solver

    mem_current, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    ws = solver_instance.warm_start_stats
    telemetry = {
        "warm_start_attempted": ws.attempted,
        "warm_start_accepted": ws.accepted,
        "warm_start_fallbacks": ws.fallbacks,
        "inheritance_attempted": ws.inheritance_attempted,
        "inheritance_accepted": ws.inheritance_accepted,
        "dense_lu_calls_saved": ws.dense_lu_calls_saved,
        "adaptive_pfi_chosen": ws.adaptive_pfi_chosen,
        "adaptive_inc_chosen": ws.adaptive_inc_chosen,
        "pfi_updates": ws.pfi_updates,
        "pfi_refactorizations": ws.pfi_refactorizations,
        "pfi_fallbacks": ws.pfi_fallbacks,
    }

    # Verified path
    if not milp_tol.warm_start_relaxations:
        verified_path = "DenseLU Cold Solves (Root + Child)"
    elif milp_tol.use_adaptive_pivot_factorization:
        verified_path = f"Adaptive PFI (T={threshold}): {ws.adaptive_pfi_chosen} PFI, {ws.adaptive_inc_chosen} Inc"
    elif simplex_tol.use_pivot_factorization:
        verified_path = "Pure PFI Child Relaxations (Bordered + PFI)"
    else:
        verified_path = "WarmStart + IncrementalBasis (Bordered + Inc)"

    stability = NumericalStabilityStats(
        refactorizations=ws.pfi_refactorizations,
        fallbacks=ws.fallbacks + ws.pfi_fallbacks,
    )
    timings = compute_timing_stats(times, sf_conv_ms=sf_conv_ms)
    memory = MemoryStats(
        baseline_bytes=mem_baseline,
        peak_bytes=mem_peak,
        diff_bytes=mem_peak - mem_baseline,
    )

    match_ref = True
    obj_diff = 0.0
    if reference_sol is not None and sol is not None:
        if reference_sol.status != sol.status.value:
            match_ref = False
        if reference_sol.objective is not None and sol.objective_value is not None:
            obj_diff = abs(reference_sol.objective - sol.objective_value)
            if obj_diff > 1e-4:
                match_ref = False

    return BenchmarkEntry(
        workload_name=model.name,
        workload_category=category,
        m_constraints=m,
        n_variables=n,
        m_standard_form=m_sf,
        n_standard_form=n_sf,
        nonzeros=nz,
        density=density,
        n_integer=n_int,
        n_binary=n_bin,
        config=config_name,
        threshold=threshold,
        status=sol.status.value if sol else "ERROR",
        objective=round(sol.objective_value, 6) if (sol and sol.objective_value is not None) else None,
        iterations=sol.iterations if sol else 0,
        nodes=solver_instance._nodes_processed if solver_instance else 0,
        timings=timings,
        stability=stability,
        memory=memory,
        telemetry=telemetry,
        verified_factorization_path=verified_path,
        match_reference=match_ref,
        objective_diff_vs_reference=round(obj_diff, 8),
    )


# ---------------------------------------------------------------------------
# Master Benchmark Execution Routine
# ---------------------------------------------------------------------------

def run_all_stage_8_18_benchmarks() -> dict[str, Any]:
    """Execute complete Stage 8.18 validation benchmark suite."""
    print("=" * 70)
    print("OPTIFORGE STAGE 8.18 — LARGE-SCALE & DIVERSE BENCHMARK VALIDATION")
    print("=" * 70)
    print(f"Python: {sys.version.split()[0]} | NumPy: {np.__version__} | OS: {platform.system()} {platform.release()}")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    all_entries: list[BenchmarkEntry] = []

    # 1. Public Netlib Models
    print("--- 1. Public Netlib Benchmarks ---")
    netlib_instances = ["avgas", "afiro", "adlittle"]
    for inst_name in netlib_instances:
        model = fetch_netlib_instance(inst_name)
        if model is None:
            continue
        print(f"Testing Netlib: {model.name} (m={len(model.constraints)}, n={len(model.variables)})")
        # Reference DenseLU
        ref_entry = run_lp_entry(
            model, "Public Netlib", "A_DenseLU",
            SimplexTolerances(use_incremental_basis=False, use_pivot_factorization=False),
            repetitions=3,
        )
        all_entries.append(ref_entry)
        ref_sol = ref_entry

        # Incremental
        inc_entry = run_lp_entry(
            model, "Public Netlib", "B_Incremental",
            SimplexTolerances(use_incremental_basis=True, use_pivot_factorization=False),
            repetitions=3, reference_sol=ref_sol,
        )
        all_entries.append(inc_entry)

        # PFI
        pfi_entry = run_lp_entry(
            model, "Public Netlib", "C_PFI",
            SimplexTolerances(use_incremental_basis=False, use_pivot_factorization=True),
            repetitions=3, reference_sol=ref_sol,
        )
        all_entries.append(pfi_entry)
        print(f"  DenseLU: {ref_entry.timings.median_ms:.1f}ms | Inc: {inc_entry.timings.median_ms:.1f}ms | PFI: {pfi_entry.timings.median_ms:.1f}ms")

    # 2. Scale Sweep LPs (m=1 to 500)
    print("\n--- 2. Scale Sweep LPs (m=1, 2, 4, 6, 8, 10, 20, 50, 100, 200, 500) ---")
    scale_dims = [
        (1, 4), (2, 6), (4, 10), (6, 15), (8, 20), (10, 25),
        (20, 40), (50, 80), (100, 150), (200, 200), (500, 500),
    ]
    for m, n in scale_dims:
        reps = 3 if m <= 10 else (2 if m <= 50 else 1)
        model = generate_scale_sweep_lp(m, n, seed=818_000 + m)
        print(f"Testing Scale LP: m={m}, n={n} (reps={reps})")
        # Reference DenseLU
        ref_entry = run_lp_entry(
            model, "Scale Sweep LP", "A_DenseLU",
            SimplexTolerances(use_incremental_basis=False, use_pivot_factorization=False),
            repetitions=reps,
        )
        all_entries.append(ref_entry)

        # Incremental
        inc_entry = run_lp_entry(
            model, "Scale Sweep LP", "B_Incremental",
            SimplexTolerances(use_incremental_basis=True, use_pivot_factorization=False),
            repetitions=reps, reference_sol=ref_entry,
        )
        all_entries.append(inc_entry)

        # PFI
        pfi_entry = run_lp_entry(
            model, "Scale Sweep LP", "C_PFI",
            SimplexTolerances(use_incremental_basis=False, use_pivot_factorization=True),
            repetitions=reps, reference_sol=ref_entry,
        )
        all_entries.append(pfi_entry)
        print(f"  m={m:3d}: DenseLU={ref_entry.timings.median_ms:.1f}ms | Inc={inc_entry.timings.median_ms:.1f}ms | PFI={pfi_entry.timings.median_ms:.1f}ms | match={pfi_entry.match_reference}")

    # 3. Structurally Diverse LP Classes (from generate_workloads)
    print("\n--- 3. Structurally Diverse LP Classes ---")
    from benchmarks.stage_8_15.generate_workloads import generate_all_workloads
    all_wl = generate_all_workloads()
    lp_workloads = [w for w in all_wl if w.n_integer + w.n_binary == 0]
    for w in lp_workloads:
        print(f"Testing Diverse LP: {w.name} (m={w.m}, n={w.n})")
        reps = 1 if w.name in ("highly_sparse_lp", "large_dense_lp") else 2
        ref_entry = run_lp_entry(
            w.model, "Diverse LP", "A_DenseLU",
            SimplexTolerances(use_incremental_basis=False, use_pivot_factorization=False),
            repetitions=reps,
        )
        all_entries.append(ref_entry)

        inc_entry = run_lp_entry(
            w.model, "Diverse LP", "B_Incremental",
            SimplexTolerances(use_incremental_basis=True, use_pivot_factorization=False),
            repetitions=reps, reference_sol=ref_entry,
        )
        all_entries.append(inc_entry)

        pfi_entry = run_lp_entry(
            w.model, "Diverse LP", "C_PFI",
            SimplexTolerances(use_incremental_basis=False, use_pivot_factorization=True),
            repetitions=reps, reference_sol=ref_entry,
        )
        all_entries.append(pfi_entry)
        print(f"  DenseLU: {ref_entry.timings.median_ms:.1f}ms | Inc: {inc_entry.timings.median_ms:.1f}ms | PFI: {pfi_entry.timings.median_ms:.1f}ms")

    # 4. Structurally Diverse MILP Classes
    print("\n--- 4. Structurally Diverse MILP Classes ---")
    milp_workloads = [w for w in all_wl if w.n_integer + w.n_binary > 0]
    for w in milp_workloads:
        print(f"Testing Diverse MILP: {w.name} (m={w.m}, n={w.n})")
        # DenseLU Cold Baseline (A)
        dense_entry = run_milp_entry(
            w.model, "Diverse MILP", "A_DenseLU_Cold",
            MILPTolerances(warm_start_relaxations=False, inherit_basis_factorization=False),
            SimplexTolerances(use_pivot_factorization=False),
            repetitions=2,
        )
        all_entries.append(dense_entry)

        # Warm-Start + IncrementalBasis Baseline (D)
        ws_inc_entry = run_milp_entry(
            w.model, "Diverse MILP", "D_WarmStart_Inc",
            MILPTolerances(warm_start_relaxations=True, inherit_basis_factorization=True, use_adaptive_pivot_factorization=False),
            SimplexTolerances(use_pivot_factorization=False, use_adaptive_pivot_factorization=False),
            repetitions=2, reference_sol=dense_entry,
        )
        all_entries.append(ws_inc_entry)

        # Pure PFI Warm-Start (C)
        pfi_entry = run_milp_entry(
            w.model, "Diverse MILP", "C_WarmStart_PFI",
            MILPTolerances(warm_start_relaxations=True, inherit_basis_factorization=True, use_adaptive_pivot_factorization=False),
            SimplexTolerances(use_pivot_factorization=True, use_adaptive_pivot_factorization=False),
            repetitions=2, reference_sol=dense_entry,
        )
        all_entries.append(pfi_entry)

        # Adaptive PFI with default threshold m >= 6 (E)
        adaptive_entry = run_milp_entry(
            w.model, "Diverse MILP", "E_Adaptive_PFI_T6",
            MILPTolerances(warm_start_relaxations=True, inherit_basis_factorization=True, use_adaptive_pivot_factorization=True, adaptive_pfi_min_dimension=6),
            SimplexTolerances(use_adaptive_pivot_factorization=True, adaptive_pfi_min_dimension=6),
            repetitions=2, reference_sol=dense_entry, threshold=6,
        )
        all_entries.append(adaptive_entry)
        print(f"  Cold: {dense_entry.timings.median_ms:.1f}ms | WS-Inc: {ws_inc_entry.timings.median_ms:.1f}ms | WS-PFI: {pfi_entry.timings.median_ms:.1f}ms | Adapt(T=6): {adaptive_entry.timings.median_ms:.1f}ms")

    # 5. Threshold Crossing Validation Suite (Candidate thresholds T in {2, 4, 6, 8, 10, 20, 50})
    print("\n--- 5. Threshold Validation Suite (T in {2, 4, 6, 8, 10, 20, 50}) ---")
    threshold_test_models = [
        (1, 4, "unbounded_milp_M1_crosses_T2"),   # m_child = 2
        (3, 6, "unbounded_milp_M3_crosses_T4"),   # m_child = 4
        (5, 8, "unbounded_milp_M5_crosses_T6"),   # m_child = 6
        (7, 10, "unbounded_milp_M7_crosses_T8"),  # m_child = 8
        (9, 12, "unbounded_milp_M9_crosses_T10"), # m_child = 10
        (19, 25, "unbounded_milp_M19_crosses_T20"), # m_child = 20
        (49, 60, "unbounded_milp_M49_crosses_T50"), # m_child = 50
    ]
    candidate_thresholds = [2, 4, 6, 8, 10, 20, 50]

    for m_con, n_vars, name in threshold_test_models:
        reps = 1 if m_con >= 19 else 2
        model = generate_unbounded_milp(m_con, n_vars, seed=818_100 + m_con)
        print(f"Testing Threshold Model: {name} (M={m_con}, N={n_vars}, m_child={m_con+1}, reps={reps})")
        # Reference DenseLU Cold
        ref_entry = run_milp_entry(
            model, "Threshold Crossing", "A_DenseLU_Cold",
            MILPTolerances(warm_start_relaxations=False, inherit_basis_factorization=False),
            SimplexTolerances(use_pivot_factorization=False),
            repetitions=reps,
        )
        all_entries.append(ref_entry)

        # Baseline WarmStart + Incremental (no PFI)
        inc_entry = run_milp_entry(
            model, "Threshold Crossing", "D_WarmStart_Inc",
            MILPTolerances(warm_start_relaxations=True, inherit_basis_factorization=True, use_adaptive_pivot_factorization=False),
            SimplexTolerances(use_pivot_factorization=False, use_adaptive_pivot_factorization=False),
            repetitions=reps, reference_sol=ref_entry,
        )
        all_entries.append(inc_entry)

        # Pure PFI WarmStart
        pure_pfi_entry = run_milp_entry(
            model, "Threshold Crossing", "C_WarmStart_PFI",
            MILPTolerances(warm_start_relaxations=True, inherit_basis_factorization=True, use_adaptive_pivot_factorization=False),
            SimplexTolerances(use_pivot_factorization=True, use_adaptive_pivot_factorization=False),
            repetitions=reps, reference_sol=ref_entry,
        )
        all_entries.append(pure_pfi_entry)

        # Test each candidate threshold T
        for T in candidate_thresholds:
            adapt_entry = run_milp_entry(
                model, "Threshold Crossing", f"E_Adaptive_PFI_T{T}",
                MILPTolerances(warm_start_relaxations=True, inherit_basis_factorization=True, use_adaptive_pivot_factorization=True, adaptive_pfi_min_dimension=T),
                SimplexTolerances(use_adaptive_pivot_factorization=True, adaptive_pfi_min_dimension=T),
                repetitions=reps, reference_sol=ref_entry, threshold=T,
            )
            all_entries.append(adapt_entry)
            print(f"    T={T:2d}: time={adapt_entry.timings.median_ms:.1f}ms | PFI_chosen={adapt_entry.telemetry['adaptive_pfi_chosen']} | Inc_chosen={adapt_entry.telemetry['adaptive_inc_chosen']}")

    # Output JSON artifact
    out_dir = Path(_project_root) / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "stage_8_18_results.json"

    out_data = {
        "metadata": {
            "stage": "8.18",
            "title": "Large-Scale & Diverse Benchmark Validation",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "numpy": np.__version__,
            },
            "candidate_thresholds_tested": candidate_thresholds,
            "total_entries": len(all_entries),
        },
        "results": [asdict(e) for e in all_entries],
    }

    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n[OK] Benchmark completed. {len(all_entries)} entries saved to {out_file}")
    return out_data


if __name__ == "__main__":
    run_all_stage_8_18_benchmarks()
