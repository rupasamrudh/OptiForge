"""Isolated Microbenchmark Script for scipy.linalg.solve_triangular vs OptiForge.

Can be run if scipy is installed in the target environment:
    python benchmarks/benchmark_scipy_triangular.py
or in an isolated environment:
    scratch/scipy_env/Scripts/python benchmarks/benchmark_scipy_triangular.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from optiforge.numerics.lu import (
    dense_lu_factor,
    forward_substitution,
    back_substitution,
    lu_solve,
    lu_solve_transpose,
    DenseLU,
)

def main():
    try:
        from scipy.linalg import solve_triangular
    except ImportError:
        print("=" * 80)
        print("SCIPY IS NOT INSTALLED IN CURRENT ENVIRONMENT")
        print("OptiForge operates under zero-external-dependency baseline (standard library + numpy).")
        print("This script is preserved for optional future evaluation if scipy is made available.")
        print("=" * 80)
        return

    print("=" * 115)
    print("EMPIRICAL BENCHMARK: OPTIFORGE PURE-NUMPY BASELINE VS SCIPY LAPACK (solve_triangular)")
    print("=" * 115)
    print(f"{'m':<4} | {'Fwd Base (med)':<15} | {'Fwd SciPy (med)':<15} | {'Fwd Spd':<7} | {'Back Base':<15} | {'Back SciPy':<15} | {'Back Spd':<8} | {'Trans Spd'}")
    print("-" * 115)

    dims = [1, 2, 4, 6, 10, 20, 50, 100, 200, 500]

    for m in dims:
        rng = np.random.default_rng(42)
        L = np.tril(rng.standard_normal((m, m)) * 0.1, -1) + np.eye(m)
        U = np.triu(rng.standard_normal((m, m)) * 0.1, 1) + np.eye(m) * 2.0
        lu = DenseLU(L=L, U=U, perm=np.arange(m), det_sign=1, pivot_tolerance=1e-12)
        b = rng.standard_normal(m)

        # Warmup
        for _ in range(5):
            _ = forward_substitution(L, b, unit_diag=True)
            _ = solve_triangular(L, b, lower=True, unit_diagonal=True)
            _ = back_substitution(U, b)
            _ = solve_triangular(U, b, lower=False, unit_diagonal=False)

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

        # 1. Forward Substitution
        fwd_base = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            for _ in range(inner):
                _ = forward_substitution(L, b, unit_diag=True)
            fwd_base.append((time.perf_counter() - t0) * 1e6 / inner)

        fwd_sp = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            for _ in range(inner):
                _ = solve_triangular(L, b, lower=True, unit_diagonal=True)
            fwd_sp.append((time.perf_counter() - t0) * 1e6 / inner)

        # 2. Back Substitution
        back_base = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            for _ in range(inner):
                _ = back_substitution(U, b)
            back_base.append((time.perf_counter() - t0) * 1e6 / inner)

        back_sp = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            for _ in range(inner):
                _ = solve_triangular(U, b, lower=False, unit_diagonal=False)
            back_sp.append((time.perf_counter() - t0) * 1e6 / inner)

        # 3. Transpose solve
        trans_base = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            for _ in range(inner):
                _ = lu_solve_transpose(lu, b)
            trans_base.append((time.perf_counter() - t0) * 1e6 / inner)

        trans_sp = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            for _ in range(inner):
                z = solve_triangular(U, b, lower=False, unit_diagonal=False, trans='T')
                w = solve_triangular(L, z, lower=True, unit_diagonal=True, trans='T')
            trans_sp.append((time.perf_counter() - t0) * 1e6 / inner)

        med_fb, med_fs = np.median(fwd_base), np.median(fwd_sp)
        med_bb, med_bs = np.median(back_base), np.median(back_sp)
        med_tb, med_ts = np.median(trans_base), np.median(trans_sp)

        spd_f = med_fb / med_fs
        spd_b = med_bb / med_bs
        spd_t = med_tb / med_ts

        print(f"{m:<4} | {med_fb:8.2f} µs     | {med_fs:8.2f} µs      | {spd_f:6.2f}x | {med_bb:8.2f} µs     | {med_bs:8.2f} µs      | {spd_b:7.2f}x  | {spd_t:6.2f}x")

if __name__ == "__main__":
    main()
