"""Stage 8.21 Performance Validation Benchmark.

Measures:
1. Triangular substitution microbenchmarks across m = [1, 4, 6, 10, 12, 20, 50, 100, 200, 500]
   - NumPy Reference
   - SciPy Provider (no threshold, m_min=0)
   - SciPy with Threshold Policy (m_min=10)
   Operations: Forward solve, Backward solve, Transpose solve
2. End-to-End LP Solvers (Dense Revised Simplex & PFI) across m = [10, 20, 50, 100]
3. Representative MILP Workload (Knapsack / Mixed Binary)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from optiforge.numerics.lu import (
    DenseLU,
    dense_lu_factor,
    forward_substitution,
    back_substitution,
    lu_solve,
    lu_solve_transpose,
)
from optiforge.numerics.triangular_provider import (
    is_scipy_available,
    forward_solve,
    backward_solve,
    triangular_lu_solve,
    triangular_lu_solve_transpose,
    get_provider_stats,
    reset_provider_stats,
)
from optiforge.lp.simplex import RevisedSimplexSolver, SimplexTolerances
from optiforge.milp.solver import MILPSolver, MILPTolerances
from optiforge.model.model import Model
from optiforge.model.variable import Variable, VariableType
from optiforge.model.constraint import Constraint, ConstraintSense
from optiforge.model.objective import Objective, ObjectiveSense

def run_microbenchmarks():
    print("=" * 115)
    print("STAGE 8.21 MICROBENCHMARK: NUMPY vs SCIPY (DIRECT) vs SCIPY (POLICY m>=10)")
    print("=" * 115)
    print(f"{'m':<5} | {'Fwd NP (us)':<12} | {'Fwd SP (us)':<12} | {'Fwd Pol(us)':<12} | {'Fwd Spd':<8} | {'Back NP(us)':<12} | {'Back SP(us)':<12} | {'Back Spd':<8} | {'Tr NP (us)':<11} | {'Tr SP (us)':<11} | {'Tr Spd'}")
    print("-" * 115)

    dims = [1, 4, 6, 10, 12, 20, 50, 100, 200, 500]

    for m in dims:
        rng = np.random.default_rng(42 + m)
        L = np.tril(rng.standard_normal((m, m)) * 0.1, -1) + np.eye(m)
        U = np.triu(rng.standard_normal((m, m)) * 0.1, 1) + np.eye(m) * 2.0
        lu_ref = DenseLU(L=L, U=U, perm=np.arange(m), det_sign=1, pivot_tolerance=1e-12, use_scipy_triangular_solve=False)
        lu_sp = DenseLU(L=L, U=U, perm=np.arange(m), det_sign=1, pivot_tolerance=1e-12, use_scipy_triangular_solve=True, scipy_triangular_min_dimension=0)
        lu_pol = DenseLU(L=L, U=U, perm=np.arange(m), det_sign=1, pivot_tolerance=1e-12, use_scipy_triangular_solve=True, scipy_triangular_min_dimension=10)
        b = rng.standard_normal(m)

        # Warmup
        for _ in range(5):
            _ = forward_substitution(L, b, unit_diag=True)
            _ = forward_solve(L, b, unit_diag=True, use_scipy=True, min_dimension=0)
            _ = back_substitution(U, b)
            _ = backward_solve(U, b, use_scipy=True, min_dimension=0)
            _ = lu_solve_transpose(lu_ref, b)
            _ = lu_solve_transpose(lu_sp, b)

        if m <= 6:
            n_reps, inner = 20, 2000
        elif m <= 20:
            n_reps, inner = 20, 1000
        elif m <= 100:
            n_reps, inner = 15, 300
        elif m <= 200:
            n_reps, inner = 10, 100
        else:
            n_reps, inner = 10, 30

        # Forward solve
        fwd_np = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            for _ in range(inner):
                _ = forward_substitution(L, b, unit_diag=True)
            fwd_np.append((time.perf_counter() - t0) * 1e6 / inner)

        fwd_sp = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            for _ in range(inner):
                _ = forward_solve(L, b, unit_diag=True, use_scipy=True, min_dimension=0)
            fwd_sp.append((time.perf_counter() - t0) * 1e6 / inner)

        fwd_pol = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            for _ in range(inner):
                _ = forward_solve(L, b, unit_diag=True, use_scipy=True, min_dimension=10)
            fwd_pol.append((time.perf_counter() - t0) * 1e6 / inner)

        # Backward solve
        back_np = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            for _ in range(inner):
                _ = back_substitution(U, b)
            back_np.append((time.perf_counter() - t0) * 1e6 / inner)

        back_sp = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            for _ in range(inner):
                _ = backward_solve(U, b, use_scipy=True, min_dimension=0)
            back_sp.append((time.perf_counter() - t0) * 1e6 / inner)

        # Transpose solve (via DenseLU.solve_transpose)
        tr_np = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            for _ in range(inner):
                _ = lu_solve_transpose(lu_ref, b)
            tr_np.append((time.perf_counter() - t0) * 1e6 / inner)

        tr_sp = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            for _ in range(inner):
                _ = lu_solve_transpose(lu_sp, b)
            tr_sp.append((time.perf_counter() - t0) * 1e6 / inner)

        med_f_np = float(np.median(fwd_np))
        med_f_sp = float(np.median(fwd_sp))
        med_f_pol = float(np.median(fwd_pol))
        spd_f = med_f_np / med_f_sp if med_f_sp > 0 else 1.0

        med_b_np = float(np.median(back_np))
        med_b_sp = float(np.median(back_sp))
        spd_b = med_b_np / med_b_sp if med_b_sp > 0 else 1.0

        med_t_np = float(np.median(tr_np))
        med_t_sp = float(np.median(tr_sp))
        spd_t = med_t_np / med_t_sp if med_t_sp > 0 else 1.0

        print(f"{m:<5} | {med_f_np:<12.2f} | {med_f_sp:<12.2f} | {med_f_pol:<12.2f} | {spd_f:<8.2f}x | {med_b_np:<12.2f} | {med_b_sp:<12.2f} | {spd_b:<8.2f}x | {med_t_np:<11.2f} | {med_t_sp:<11.2f} | {spd_t:.2f}x")

def run_end_to_end_lp():
    print("\n" + "=" * 115)
    print("STAGE 8.21 END-TO-END LP BENCHMARK: DENSE LP WORKLOADS")
    print("=" * 115)
    print(f"{'m x n':<12} | {'DenseLU (NP) ms':<18} | {'DenseLU (SP) ms':<18} | {'LU Spd':<8} | {'PFI (NP) ms':<15} | {'PFI (SP) ms':<15} | {'PFI Spd'}")
    print("-" * 115)

    cases = [(10, 20), (20, 40), (50, 100), (100, 150), (150, 200)]

    for m, n in cases:
        rng = np.random.default_rng(123 + m)
        A = rng.uniform(0.1, 1.0, size=(m, n))
        b = rng.uniform(10.0, 50.0, size=m)
        c = rng.uniform(1.0, 10.0, size=n)

        # Build Model
        model = Model(f"dense_lp_{m}x{n}")
        for j in range(n):
            model.add_variable(Variable(f"x_{j}", lower_bound=0.0))
        for i in range(m):
            coeffs = {f"x_{j}": float(A[i, j]) for j in range(n)}
            model.add_constraint(Constraint(f"c_{i}", coeffs, ConstraintSense.LESS_EQUAL, float(b[i])))
        obj = {f"x_{j}": float(c[j]) for j in range(n)}
        model.set_objective(Objective(obj, ObjectiveSense.MAXIMIZE))

        # 1. DenseLU NumPy
        tol_lu_np = SimplexTolerances(use_scipy_triangular_solve=False)
        # 2. DenseLU SciPy
        tol_lu_sp = SimplexTolerances(use_scipy_triangular_solve=True, scipy_triangular_min_dimension=10)
        # 3. PFI NumPy
        tol_pfi_np = SimplexTolerances(use_pivot_factorization=True, use_fast_pivot_update=True, use_scipy_triangular_solve=False)
        # 4. PFI SciPy
        tol_pfi_sp = SimplexTolerances(use_pivot_factorization=True, use_fast_pivot_update=True, use_scipy_triangular_solve=True, scipy_triangular_min_dimension=10)

        n_reps = 7 if m <= 50 else 5

        # Measure DenseLU NP
        t_lu_np = []
        res = None
        for _ in range(n_reps):
            s = RevisedSimplexSolver(tolerances=tol_lu_np)
            t0 = time.perf_counter()
            res = s.solve(model)
            t_lu_np.append((time.perf_counter() - t0) * 1000)

        # Measure DenseLU SP
        t_lu_sp = []
        res_sp = None
        for _ in range(n_reps):
            s = RevisedSimplexSolver(tolerances=tol_lu_sp)
            t0 = time.perf_counter()
            res_sp = s.solve(model)
            t_lu_sp.append((time.perf_counter() - t0) * 1000)

        # Measure PFI NP
        t_pfi_np = []
        res_pfi = None
        for _ in range(n_reps):
            s = RevisedSimplexSolver(tolerances=tol_pfi_np)
            t0 = time.perf_counter()
            res_pfi = s.solve(model)
            t_pfi_np.append((time.perf_counter() - t0) * 1000)

        # Measure PFI SP
        t_pfi_sp = []
        res_pfi_sp = None
        for _ in range(n_reps):
            s = RevisedSimplexSolver(tolerances=tol_pfi_sp)
            t0 = time.perf_counter()
            res_pfi_sp = s.solve(model)
            t_pfi_sp.append((time.perf_counter() - t0) * 1000)

        med_lu_np = float(np.median(t_lu_np))
        med_lu_sp = float(np.median(t_lu_sp))
        spd_lu = med_lu_np / med_lu_sp if med_lu_sp > 0 else 1.0

        med_pfi_np = float(np.median(t_pfi_np))
        med_pfi_sp = float(np.median(t_pfi_sp))
        spd_pfi = med_pfi_np / med_pfi_sp if med_pfi_sp > 0 else 1.0

        # Verify numerical equivalence
        assert np.isclose(res.objective_value, res_sp.objective_value, atol=1e-7), f"Objective mismatch at m={m}"
        assert np.isclose(res.objective_value, res_pfi_sp.objective_value, atol=1e-7), f"PFI objective mismatch at m={m}"

        dim_str = f"{m} x {n}"
        print(f"{dim_str:<12} | {med_lu_np:<18.2f} | {med_lu_sp:<18.2f} | {spd_lu:<8.2f}x | {med_pfi_np:<15.2f} | {med_pfi_sp:<15.2f} | {spd_pfi:.2f}x")

def create_knapsack_model(n_items=15, seed=999):
    rng = np.random.default_rng(seed)
    weights = rng.integers(5, 50, size=n_items).astype(np.float64)
    values = rng.integers(10, 100, size=n_items).astype(np.float64)
    capacity = float(np.sum(weights) * 0.45)

    model = Model("knapsack_bench")
    for i in range(n_items):
        model.add_variable(Variable(f"x_{i}", variable_type=VariableType.BINARY))
    
    coeffs = {f"x_{i}": float(weights[i]) for i in range(n_items)}
    model.add_constraint(Constraint("capacity", coeffs, ConstraintSense.LESS_EQUAL, capacity))
    
    obj_coeffs = {f"x_{i}": float(values[i]) for i in range(n_items)}
    model.set_objective(Objective(obj_coeffs, ObjectiveSense.MAXIMIZE))
    return model

def create_multiconstraint_milp(m=12, n=20, seed=2026):
    rng = np.random.default_rng(seed)
    A = rng.uniform(1.0, 10.0, size=(m, n))
    b = rng.uniform(20.0, 60.0, size=m)
    c = rng.uniform(2.0, 15.0, size=n)

    model = Model("mixed_binary_bench")
    for j in range(n):
        vtype = VariableType.BINARY if j < 12 else VariableType.CONTINUOUS
        model.add_variable(Variable(f"x_{j}", lower_bound=0.0, upper_bound=1.0 if j < 12 else 100.0, variable_type=vtype))

    for i in range(m):
        coeffs = {f"x_{j}": float(A[i, j]) for j in range(n)}
        model.add_constraint(Constraint(f"c_{i}", coeffs, ConstraintSense.LESS_EQUAL, float(b[i])))

    obj_coeffs = {f"x_{j}": float(c[j]) for j in range(n)}
    model.set_objective(Objective(obj_coeffs, ObjectiveSense.MAXIMIZE))
    return model

def run_representative_milp():
    print("\n" + "=" * 115)
    print("STAGE 8.21 REPRESENTATIVE MILP BENCHMARK")
    print("=" * 115)
    print(f"{'Workload':<30} | {'NumPy Time (ms)':<18} | {'SciPy Time (ms)':<18} | {'Speedup':<10} | {'Node Count (NP / SP)'}")
    print("-" * 115)

    tol_np = MILPTolerances(use_scipy_triangular_solve=False, warm_start_relaxations=True)
    tol_sp = MILPTolerances(use_scipy_triangular_solve=True, scipy_triangular_min_dimension=10, warm_start_relaxations=True)

    # Workload 1: 15-item 0-1 Knapsack
    model1 = create_knapsack_model(15, seed=999)
    t_np1, t_sp1 = [], []
    res_np1, res_sp1 = None, None
    for _ in range(5):
        s_np = MILPSolver(tolerances=tol_np)
        t0 = time.perf_counter()
        res_np1 = s_np.solve(model1)
        t_np1.append((time.perf_counter() - t0) * 1000)

    for _ in range(5):
        s_sp = MILPSolver(tolerances=tol_sp)
        t0 = time.perf_counter()
        res_sp1 = s_sp.solve(model1)
        t_sp1.append((time.perf_counter() - t0) * 1000)

    med_np1 = float(np.median(t_np1))
    med_sp1 = float(np.median(t_sp1))
    spd1 = med_np1 / med_sp1 if med_sp1 > 0 else 1.0

    assert np.isclose(res_np1.objective_value, res_sp1.objective_value, atol=1e-7)
    assert res_np1.node_count == res_sp1.node_count
    name1 = "Knapsack (15 vars, 1 con)"
    nodes1 = f"{res_np1.node_count} / {res_sp1.node_count}"
    print(f"{name1:<30} | {med_np1:<18.2f} | {med_sp1:<18.2f} | {spd1:<10.2f}x | {nodes1}")

    # Workload 2: Multi-constraint Mixed Binary (m=12, n=20)
    model2 = create_multiconstraint_milp(12, 20, seed=2026)
    t_np2, t_sp2 = [], []
    res_np2, res_sp2 = None, None
    for _ in range(5):
        s_np = MILPSolver(tolerances=tol_np)
        t0 = time.perf_counter()
        res_np2 = s_np.solve(model2)
        t_np2.append((time.perf_counter() - t0) * 1000)

    for _ in range(5):
        s_sp = MILPSolver(tolerances=tol_sp)
        t0 = time.perf_counter()
        res_sp2 = s_sp.solve(model2)
        t_sp2.append((time.perf_counter() - t0) * 1000)

    med_np2 = float(np.median(t_np2))
    med_sp2 = float(np.median(t_sp2))
    spd2 = med_np2 / med_sp2 if med_sp2 > 0 else 1.0

    assert np.isclose(res_np2.objective_value, res_sp2.objective_value, atol=1e-7)
    assert res_np2.node_count == res_sp2.node_count
    name2 = "Mixed Binary (12 con, 20 vars)"
    nodes2 = f"{res_np2.node_count} / {res_sp2.node_count}"
    print(f"{name2:<30} | {med_np2:<18.2f} | {med_sp2:<18.2f} | {spd2:<10.2f}x | {nodes2}")

def main():
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {np.__version__}")
    print(f"SciPy available: {is_scipy_available()}")
    run_microbenchmarks()
    run_end_to_end_lp()
    run_representative_milp()

if __name__ == "__main__":
    main()
