"""Stage 8.48 Performance Benchmark.

Measures:
1. Normal solve without verification (mode="none")
2. Solve with basic verification (mode="basic")
3. Solve with strict verification (mode="strict")
4. Warm-start with verification
5. Cold fallback
"""

import time
import numpy as np

from optiforge.core.solution import SolutionStatus
from optiforge.lp.simplex import SimplexTolerances
from optiforge.model.constraint import Constraint, ConstraintSense
from optiforge.model.model import Model
from optiforge.model.objective import Objective, ObjectiveSense
from optiforge.model.variable import Variable


def build_benchmark_lp(m: int = 50, n: int = 100, seed: int = 42) -> Model:
    """Generate a consistent benchmark LP."""
    rng = np.random.default_rng(seed)
    model = Model(name=f"bench_{m}x{n}")
    for j in range(n):
        model.add_variable(Variable(f"x_{j}", lower_bound=0.0))

    c_coeffs = {f"x_{j}": float(rng.uniform(1.0, 10.0)) for j in range(n)}
    model.set_objective(Objective(c_coeffs, sense=ObjectiveSense.MINIMIZE))

    for i in range(m):
        row_coeffs = {f"x_{j}": float(rng.uniform(0.1, 1.0)) for j in range(n)}
        rhs_val = float(rng.uniform(20.0, 50.0))
        model.add_constraint(Constraint(f"c_{i}", row_coeffs, ConstraintSense.GREATER_EQUAL, rhs_val))

    return model


def run_benchmark(reps: int = 30):
    m_base = build_benchmark_lp(m=30, n=60, seed=123)

    # Warmup
    sol_base = m_base.solve(presolve=False, numerical_verification="none")
    assert sol_base.status == SolutionStatus.OPTIMAL
    binfo = sol_base.basis_info

    # 1. Normal solve without verification
    t0 = time.perf_counter()
    for _ in range(reps):
        m = build_benchmark_lp(m=30, n=60, seed=123)
        sol = m.solve(presolve=False, numerical_verification="none")
        assert sol.status == SolutionStatus.OPTIMAL
    t_none = (time.perf_counter() - t0) / reps * 1000.0

    # 2. Basic verification
    t0 = time.perf_counter()
    for _ in range(reps):
        m = build_benchmark_lp(m=30, n=60, seed=123)
        sol = m.solve(presolve=False, numerical_verification="basic")
        assert sol.status == SolutionStatus.OPTIMAL
    t_basic = (time.perf_counter() - t0) / reps * 1000.0

    # 3. Strict verification
    t0 = time.perf_counter()
    for _ in range(reps):
        m = build_benchmark_lp(m=30, n=60, seed=123)
        sol = m.solve(presolve=False, numerical_verification="strict")
        assert sol.status == SolutionStatus.OPTIMAL
    t_strict = (time.perf_counter() - t0) / reps * 1000.0

    # 4. Warm-start with verification
    t0 = time.perf_counter()
    for _ in range(reps):
        m = build_benchmark_lp(m=30, n=60, seed=123)
        m._constraints["c_0"].rhs += 1.0  # slight perturbation
        sol = m.solve(basis=binfo, presolve=False, numerical_verification="strict")
        assert sol.status == SolutionStatus.OPTIMAL
    t_warm = (time.perf_counter() - t0) / reps * 1000.0

    # 5. Cold fallback
    t0 = time.perf_counter()
    for _ in range(reps):
        m = build_benchmark_lp(m=30, n=60, seed=123)
        m._constraints["c_0"].rhs += 1.0
        sol = m.solve(presolve=False, numerical_verification="strict")
        assert sol.status == SolutionStatus.OPTIMAL
    t_cold = (time.perf_counter() - t0) / reps * 1000.0

    print("=" * 80)
    print("STAGE 8.48 NUMERICAL ROBUSTNESS & VERIFICATION BENCHMARK")
    print("=" * 80)
    print(f"Benchmark Problem: 30 constraints x 60 variables, {reps} repetitions")
    print("-" * 80)
    print(f"1. Normal Solve (no verification):   {t_none:8.3f} ms")
    print(f"2. Basic Verification:              {t_basic:8.3f} ms (overhead: +{t_basic - t_none:6.3f} ms, +{(t_basic - t_none) / t_none * 100:5.1f}%)")
    print(f"3. Strict Verification:             {t_strict:8.3f} ms (overhead: +{t_strict - t_none:6.3f} ms, +{(t_strict - t_none) / t_none * 100:5.1f}%)")
    print(f"4. Warm-Start with Verification:    {t_warm:8.3f} ms (speedup vs cold: {t_cold / t_warm:5.2f}x)")
    print(f"5. Cold Fallback with Verification: {t_cold:8.3f} ms")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark(reps=30)
