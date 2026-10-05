"""Stage 8.49 Performance Benchmark.

Sparse Iterative Condition Estimation (Hager-Higham) & Dynamic Ruiz Scaling.

Measures:
1. SVD condition estimation vs. Hager-Higham 1-norm across increasing matrix dimensions:
   - runtime (ms)
   - estimated condition
   - Hager-Higham iteration count
   - speedup factor
2. Unscaled solve vs. Explicitly scaled solve:
   - runtime (ms)
   - iteration count
   - solver status
   - objective difference
   - KKT residuals
"""

import time
import numpy as np

from optiforge.core.solution import SolutionStatus
from optiforge.model.constraint import Constraint, ConstraintSense
from optiforge.model.model import Model
from optiforge.model.objective import Objective, ObjectiveSense
from optiforge.model.variable import Variable
from optiforge.numerics.condition import estimate_condition_1norm
from optiforge.numerics.lu import dense_lu_factor
from optiforge.numerics.policy import NumericalPolicy


def build_ill_scaled_lp(m: int = 40, n: int = 80, seed: int = 42) -> Model:
    """Generate an ill-scaled benchmark LP with coefficient disparities spanning 1e-4 to 1e4."""
    rng = np.random.default_rng(seed)
    model = Model(name=f"ill_scaled_{m}x{n}")

    # Column scale factors spanning 1e-2 to 1e2
    col_scales = rng.uniform(0.01, 100.0, size=n)
    # Row scale factors spanning 1e-2 to 1e2
    row_scales = rng.uniform(0.01, 100.0, size=m)

    for j in range(n):
        model.add_variable(Variable(f"x_{j}", lower_bound=0.0))

    c_coeffs = {f"x_{j}": float(rng.uniform(1.0, 5.0) * col_scales[j]) for j in range(n)}
    model.set_objective(Objective(c_coeffs, sense=ObjectiveSense.MINIMIZE))

    for i in range(m):
        row_coeffs = {
            f"x_{j}": float(rng.uniform(0.1, 1.0) * row_scales[i] * col_scales[j])
            for j in range(n)
        }
        rhs_val = float(rng.uniform(20.0, 50.0) * row_scales[i])
        model.add_constraint(
            Constraint(f"c_{i}", row_coeffs, ConstraintSense.GREATER_EQUAL, rhs_val)
        )

    return model


def benchmark_condition_estimation(dimensions=(25, 50, 100, 200), reps=15):
    """Benchmark SVD vs Hager-Higham 1-norm condition estimation."""
    print("=" * 88)
    print("STAGE 8.49 CONDITION ESTIMATION BENCHMARK: SVD vs. HAGER-HIGHAM 1-NORM")
    print("=" * 88)
    print(f"{'Dim (m x m)':<12} | {'SVD Time (ms)':<14} | {'Hager-Higham (ms)':<18} | {'Speedup':<9} | {'Cond (SVD)':<12} | {'Cond (1-norm)':<14} | {'Iters'}")
    print("-" * 88)

    rng = np.random.default_rng(99)

    for dim in dimensions:
        # Generate well-conditioned dense matrix
        M = rng.standard_normal((dim, dim))
        B = M + np.diag(np.sum(np.abs(M), axis=1) + 2.0)

        # Factorize LU once to simulate basis evaluation during simplex
        lu = dense_lu_factor(B)

        # 1. Hager-Higham 1-norm (reusing LU factors)
        # Warmup
        _ = estimate_condition_1norm(B, solver=lu)
        t0 = time.perf_counter()
        for _ in range(reps):
            res_hh = estimate_condition_1norm(B, solver=lu)
        t_hh = (time.perf_counter() - t0) / reps * 1000.0

        # 2. SVD 2-norm
        t0 = time.perf_counter()
        for _ in range(reps):
            s = np.linalg.svdvals(B)
            cond_svd = float(s[0] / s[-1]) if s[-1] > 1e-15 else float("inf")
        t_svd = (time.perf_counter() - t0) / reps * 1000.0

        speedup = t_svd / t_hh if t_hh > 0 else 1.0
        print(
            f"{dim:<12} | {t_svd:<14.3f} | {t_hh:<18.3f} | {speedup:<9.2f}x | "
            f"{cond_svd:<12.2e} | {res_hh.condition_estimate:<14.2e} | {res_hh.iterations}"
        )

    print("-" * 88)


def benchmark_solver_scaling(m=30, n=60, reps=10):
    """Benchmark solver with and without Ruiz dynamic scaling on ill-scaled LP."""
    print()
    print("=" * 88)
    print("STAGE 8.49 SOLVER BENCHMARK: UNSCALED SOLVE vs. RUIZ DYNAMIC SCALING")
    print("=" * 88)

    # 1. Unscaled solve
    t0 = time.perf_counter()
    for _ in range(reps):
        model_unscaled = build_ill_scaled_lp(m=m, n=n, seed=42)
        sol_unscaled = model_unscaled.solve(
            scaling=False,
            presolve=False,
            numerical_verification="strict",
        )
        assert sol_unscaled.status == SolutionStatus.OPTIMAL
    t_unscaled = (time.perf_counter() - t0) / reps * 1000.0

    # 2. Scaled solve
    t0 = time.perf_counter()
    for _ in range(reps):
        model_scaled = build_ill_scaled_lp(m=m, n=n, seed=42)
        sol_scaled = model_scaled.solve(
            scaling=True,
            presolve=False,
            numerical_verification="strict",
        )
        assert sol_scaled.status == SolutionStatus.OPTIMAL
    t_scaled = (time.perf_counter() - t0) / reps * 1000.0

    obj_diff = abs((sol_unscaled.objective_value or 0.0) - (sol_scaled.objective_value or 0.0))
    p_res_u = sol_unscaled.diagnostics.primal_residual if sol_unscaled.diagnostics else 0.0
    p_res_s = sol_scaled.diagnostics.primal_residual if sol_scaled.diagnostics else 0.0
    d_res_u = sol_unscaled.diagnostics.dual_residual if sol_unscaled.diagnostics else 0.0
    d_res_s = sol_scaled.diagnostics.dual_residual if sol_scaled.diagnostics else 0.0

    print(f"Problem Size: {m} constraints x {n} variables, {reps} repetitions")
    print("-" * 88)
    print(f"Unscaled Solve Runtime:      {t_unscaled:8.3f} ms | Iterations: {sol_unscaled.iterations}")
    print(f"Scaled Solve Runtime:        {t_scaled:8.3f} ms | Iterations: {sol_scaled.iterations}")
    print(f"Objective Difference:        {obj_diff:8.2e}")
    print(f"Unscaled KKT Residuals:      Primal={p_res_u:.2e}, Dual={d_res_u:.2e}")
    print(f"Scaled KKT Residuals:        Primal={p_res_s:.2e}, Dual={d_res_s:.2e}")
    print(f"Verification Status:         Unscaled: {sol_unscaled.diagnostics.verification_passed if sol_unscaled.diagnostics else None}, Scaled: {sol_scaled.diagnostics.verification_passed if sol_scaled.diagnostics else None}")
    print("=" * 88)


if __name__ == "__main__":
    benchmark_condition_estimation()
    benchmark_solver_scaling()
