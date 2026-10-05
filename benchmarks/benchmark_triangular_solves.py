"""Stage 8.20 Baseline Microbenchmark for Triangular Solves in lu.py.
Measures forward_substitution, back_substitution, transpose solves, and complete lu_solve / lu_solve_transpose.
"""

import sys
sys.path.insert(0, ".")
import time
import numpy as np

from optiforge.numerics.lu import (
    dense_lu_factor,
    forward_substitution,
    back_substitution,
    lu_solve,
    lu_solve_transpose,
    DenseLU,
)

def generate_triangular_system(m, seed=42):
    rng = np.random.RandomState(seed)
    A = rng.randn(m, m)
    # Make well-conditioned and diagonally dominant
    A = A + np.eye(m) * (m * 2.0 + 5.0)
    lu = dense_lu_factor(A)
    b = rng.randn(m)
    return lu, b

def benchmark_op(func, reps=100):
    times = []
    # Warmup
    for _ in range(max(5, reps // 10)):
        func()
    # Timed runs
    for _ in range(reps):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6) # in microseconds
    arr = np.array(times)
    return {
        "median_us": float(np.median(arr)),
        "mean_us": float(np.mean(arr)),
        "std_us": float(np.std(arr)),
        "min_us": float(np.min(arr)),
        "max_us": float(np.max(arr)),
    }

def main():
    dimensions = [1, 2, 4, 6, 10, 20, 50, 100, 200, 500]
    print("=" * 95)
    print("STAGE 8.20 BASELINE TRIANGULAR SOLVE MICROBENCHMARK (Time in Microseconds µs)")
    print("=" * 95)
    print(f"{'m':<5} | {'Fwd Subs (L)':<14} | {'Back Subs (U)':<14} | {'Complete Solve':<16} | {'Complete Transpose':<18} | {'Total Solve (ms)'}")
    print("-" * 95)

    for m in dimensions:
        lu, b = generate_triangular_system(m, seed=100 + m)
        y = forward_substitution(lu.L, b[lu.perm])
        reps = 1000 if m <= 10 else (300 if m <= 50 else (100 if m <= 200 else 20))

        # Benchmarks
        fwd_res = benchmark_op(lambda: forward_substitution(lu.L, b, unit_diag=True), reps=reps)
        back_res = benchmark_op(lambda: back_substitution(lu.U, y), reps=reps)
        solve_res = benchmark_op(lambda: lu_solve(lu, b), reps=reps)
        solve_t_res = benchmark_op(lambda: lu_solve_transpose(lu, b), reps=reps)

        tot_ms = solve_res["median_us"] / 1000.0
        print(f"{m:<5} | {fwd_res['median_us']:10.2f} µs | {back_res['median_us']:10.2f} µs | {solve_res['median_us']:12.2f} µs | {solve_t_res['median_us']:14.2f} µs | {tot_ms:8.4f} ms")

if __name__ == "__main__":
    main()
