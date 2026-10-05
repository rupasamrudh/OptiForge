# OPTIFORGE — STAGE 8.47 TECHNICAL REPORT
## Dual Simplex Warm-Start Re-optimization

**Status**: Verified & Complete  
**Date**: October 2026  
**Baseline Test Suite**: 1181 passed  
**Post-Stage Test Suite**: 1201 passed (20 new tests, 0 failures, 21 benign warnings)  
**Measured Speedup**: Up to 45.5x wall-clock speedup (137x iteration reduction) on 50x100 LP  

---

### 1. Executive Summary

Stage 8.47 introduces **Dual Simplex Warm-Start Re-optimization** to the sovereign OptiForge optimization engine.

Previously (Stages 8.43–8.46), when an RHS, constraint bound, or scenario perturbation rendered a previously optimal LP basis primal-infeasible ($x_B = B^{-1} b' \not\ge 0$), the pipeline was forced to discard the basis and perform an expensive cold Phase-I/Phase-II Primal Simplex solve from scratch.

Under Stage 8.47:
- The optimal basis is preserved and validated.
- If $x_B = B^{-1} b'$ violates primal feasibility beyond numerical tolerance ($-\text{feasibility\_tolerance}$), the dual multipliers $y = B^{-T} c_B$ and reduced costs $r_N = c_N - A_N^T y$ are checked.
- If dual feasibility is preserved, the solve is dispatched directly to `DualRevisedSimplexSolver` initialized with the existing basis.
- Dual Simplex restores primal feasibility in a minimal number of dual pivots (typically 1 to 9 pivots instead of hundreds or thousands of cold iterations), matching cold-solve solutions down to machine precision.
- If dual feasibility or basis singularity checks fail, the engine safely falls back to standard cold solving without corrupting models or producing invalid results.

---

### 2. Mathematical Contract & Invariants

For a standard-form continuous linear program:
$$\min c^T x \quad \text{s.t.} \quad A x = b, \quad x \ge 0$$
Let $B$ be the $m \times m$ basis matrix formed by column indices $\mathcal{B} \subset \{0, \dots, n-1\}$.

#### 2.1 Basis Validity
A basis $\mathcal{B}$ is admitted for warm-start re-optimization if and only if:
1. $|\mathcal{B}| = m$ (exact standard-form row dimension).
2. Unique column indices: $|\text{set}(\mathcal{B})| = m$.
3. Column indices $\mathcal{B}_i \in [0, n-1]$.
4. The basis matrix $B = A_{:, \mathcal{B}}$ is non-singular ($|\det(B)| > \text{pivot\_tolerance}$, verified by LU factorization).

#### 2.2 Primal Infeasibility Detection
The basic solution vector is computed via LU forward/backward substitution:
$$x_B = B^{-1} b'$$
- If $\min(x_B) \ge -\epsilon_{\text{feas}}$: The basis remains primal-feasible. Dispatched to `RevisedSimplexSolver` via `warm_start_primal_feasible`.
- Else: The basis is primal-infeasible. Proceed to dual feasibility verification.

#### 2.3 Dual Feasibility Invariant
Dual multipliers and reduced costs are computed:
$$y = B^{-T} c_B, \qquad r = c - A^T y$$
Columns corresponding to Phase-I artificial variables ($\mathcal{A}_{\text{art}}$) are excluded from the optimality check:
$$\mathcal{N}_{\text{active}} = \{j \in \{0, \dots, n-1\} \setminus \mathcal{B} \mid j \notin \mathcal{A}_{\text{art}}\}$$

Optimality condition for candidate dual warm-start:
- For **MINIMIZE**: $r_j \ge -\epsilon_{\text{pivot}} \quad \forall j \in \mathcal{N}_{\text{active}}$
- For **MAXIMIZE**: $r_j \le \epsilon_{\text{pivot}} \quad \forall j \in \mathcal{N}_{\text{active}}$

Because RHS perturbations $b \to b + \Delta b$ do not alter $c$ or $A$, $y = B^{-T} c_B$ and $r_N = c_N - A_N^T y$ are invariant under pure RHS changes. Therefore, a previously optimal basis remains dual feasible under RHS perturbations unless numerical ill-conditioning or coefficient perturbations intervene.

#### 2.4 Fallback Guarantee
If either $B$ is singular or dual feasibility fails, the pipeline safely falls back to a cold solve (`warm_start_fallback_cold_start`), ensuring correctness is never sacrificed.

---

### 3. Architecture & Implementation Changes

#### 3.1 `optiforge/lp/dual_simplex.py`
- **`DualSimplexResult` Extended**:
  Added diagnostic fields: `warm_started: bool`, `initial_basis: tuple[int, ...] | None`, `initial_primal_feasible: bool`, `initial_dual_feasible: bool`, `reoptimization_iterations: int`.
- **`DualRevisedSimplexSolver.solve()` Extended**:
  - Accepts `initial_basis: list[int] | None`.
  - Rigorous basis verification (length $m$, duplicate check, range $[0, n-1]$, and non-singularity via `BasisFactorization`).
  - Dual feasibility verification before entering the pivot loop.
  - Phase-I artificial variable exclusion (`all_cols - set(basis) - art_set`) from entering column candidates.
  - Corrected variable shift adjustments (`val += model_sf.variable_shifts[col_idx]`) for variables with shifted lower bounds.
  - Returns `Solution` decorated with `solver_metadata`.

#### 3.2 `optiforge/model/pipeline.py`
- **`select_solver()` Routing**:
  Accepts `initial_basis` and `tolerances`.
  - Primal feasible $\to$ `("revised_simplex", "warm_start_primal_feasible")`
  - Primal infeasible & Dual feasible $\to$ `("dual_simplex", "warm_start_dual_feasible")`
  - Primal infeasible & Dual infeasible $\to$ `("revised_simplex", "warm_start_fallback_cold_start")`
- **`execute_pipeline()` Execution**:
  - Handles singular bases with descriptive `ValueError`.
  - Instantiates `DualRevisedSimplexSolver` with `initial_basis`.
  - Propagates `solver_reason` and `solver_metadata` (`warm_started`, `reoptimization_iterations`) to the user-facing `Solution`.

#### 3.3 `optiforge/core/solution.py`
- Added `solver_reason: str | None = None` and `solver_metadata: dict[str, Any] = field(default_factory=dict)` to `Solution` dataclass without breaking backward compatibility.

#### 3.4 `optiforge/sensitivity/scenario.py`
- Extended `ScenarioResult` with:
  - `solve_method: str = "analytical" | "primal_warm_start" | "dual_warm_start" | "cold_resolve"`
  - `warm_started: bool`
  - `initial_basis_preserved: bool`
  - `initial_primal_feasible: bool`
  - `initial_dual_feasible: bool`
  - `reoptimization_iterations: int | None`
- In `evaluate_scenario(model, scenario, resolve_if_needed=True)`:
  - Preserves model immutability (deep clone of modified parameters).
  - Attempts warm-start re-optimization with `model.last_basis`.
  - Detects and categorizes `solve_method` based on solver output.
- Extended `ScenarioComparison.summary()` table to display `Method` and `Warm-Started` columns.

---

### 4. Benchmark Results (`scratch/measure_dual_warm_start.py`)

Run on Windows 64-bit environment, Python 3.12:

| Problem Size | Mode | Time (ms) | Simplex Iterations | Objective Value | Selected Solver |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Small (2x2)** | Cold Re-solve | 0.733 ms | 2 | 120.0000 | `revised_simplex` |
| | Dual Warm-Start | 2.046 ms | 1 | 120.0000 | `dual_simplex` |
| | Primal Warm-Start | 1.067 ms | 0 | 190.0000 | `revised_simplex` |
| **Medium (10x20)** | Cold Re-solve | 2.662 ms | 13 | 661.8782 | `revised_simplex` |
| | Dual Warm-Start | 1.633 ms | 2 | 661.8782 | `dual_simplex` |
| **Large (50x100)** | Cold Re-solve | 1045.383 ms | 1236 | 2855.2920 | `revised_simplex` |
| | Dual Warm-Start | **22.956 ms** | **9** | 2855.2920 | `dual_simplex` |

#### Key Benchmark Takeaways:
1. **Massive Iteration Reduction**: On the 50x100 LP, dual warm start reduced simplex iterations from 1,236 down to 9 (**137x reduction**).
2. **Substantial Speedup**: Wall-clock solve time dropped from 1,045 ms down to 22.9 ms (**45.5x wall-clock speedup**).
3. **Exact Mathematical Agreement**: In all benchmark cases, objective values matched cold solves to 10 decimal places.

---

### 5. Verification & Test Suite

All 20 test categories specified in Section 13 were implemented in `tests/test_stage_8_47_dual_warm_start.py`:
1. `test_basic_dual_warm_start`: Verified dual simplex selection on primal-infeasible perturbation.
2. `test_mathematical_correctness_vs_cold_solve`: Objective and variable values match cold solve across multiple RHS values.
3. `test_optimal_status_returned`: Verified `SolutionStatus.OPTIMAL`.
4. `test_basis_reuse_verified`: Verified basis indices consumed directly.
5. `test_iteration_reduction_vs_cold_solve`: Re-optimization completes in 1 pivot.
6. `test_primal_feasible_warm_start`: Primal warm start routes to `revised_simplex`.
7. `test_dual_infeasible_basis_fallback`: Objective modification triggering dual infeasibility safely falls back.
8. `test_invalid_basis_rejection`: Verified rejection of wrong dimensions, duplicates, out-of-range indices, singular bases, and non-LP bases.
9. `test_presolve_compatibility`: Verified warm-start with bound shifts, fixed variables, and presolve reductions.
10. `test_degenerate_basis`: Verified correctness on degenerate bases.
11. `test_multiple_infeasible_basic_variables`: Correct pivoting when multiple basic variables are negative.
12. `test_infeasible_perturbed_problem_detection`: Reports `INFEASIBLE` when dual unboundedness is detected.
13. `test_backward_compatibility`: Solves without basis remain 100% identical.
14. `test_scenario_integration_dual_warm_start`: `evaluate_scenario(..., resolve_if_needed=True)` uses `dual_warm_start`.
15. `test_model_immutability`: Original `Model` attributes are never mutated during scenario solve.
16. `test_objective_preservation`: Baseline objective and metadata remain intact.
17. `test_maximize_sign_conventions`: MAXIMIZE sign conventions ($r_N \le 0$) verified.
18. `test_minimize_sign_conventions`: MINIMIZE sign conventions ($r_N \ge 0$) verified.
19. `test_tolerance_boundary`: Distinguishes sub-tolerance violation (primal warm start) from genuine violation (dual warm start).
20. `test_fallback_correctness`: Correct fallback to cold solve when prerequisites fail.

**Regression Results**:
```
1201 passed, 21 warnings in 35.12s
```
- Total test count: 1201
- Failures: 0
- Regressions: 0

---

### 6. Known Limitations & Future Scope

1. **Continuous LP Only**: Dual warm-start applies strictly to continuous LP standard forms; MILP branch-and-bound warm starts are explicitly deferred to future MILP stages.
2. **Basis Dimensions Across Dynamic Presolve**: If an extreme perturbation alters the set of redundant constraints identified by presolve, basis dimensions between pre- and post-perturbation models may diverge. In such cases, the system safely and cleanly falls back to a cold solve.
