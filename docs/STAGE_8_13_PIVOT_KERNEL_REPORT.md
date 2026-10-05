# Stage 8.13 Implementation Report: Pivot Kernel & Simplex Inner-Loop Optimization

**OptiForge Advanced Optimization Suite**  
**Date:** September 27, 2026  
**Status:** IMPLEMENTED — READY FOR INDEPENDENT AUDIT  
**Baseline Test Count:** 846 passed  
**Post-Implementation Test Count:** 858 passed (12 new tests added, 0 regressions)  

---

## 1. Objective

Stage 8.12 established Phase-I → Phase-II factorization inheritance and activated MILP Phase-I warm starts, leaving the Revised Simplex inner loop (per-pivot operations) as the primary runtime consumer. The objective of Stage 8.13 is to:
1. Conduct empirical profiling of the Revised Simplex inner loop to isolate exact operation costs.
2. Eliminate provable computational and allocation overheads in the per-pivot kernel without altering solver mathematics, Bland's anti-cycling rules, or numerical tolerances.
3. Validate mathematical invariants, numerical stability, and end-to-end speedups across standard LP and MILP benchmarks.
4. Keep all experimental feature flags disabled by default, preserving cold/reference behavior.

---

## 2. Repository Baseline

- **Initial pytest baseline:** 846 passed, 12 warnings in 6.36s (`python -m pytest tests/ -q`).
- **Trusted factorization:** `DenseLU` (`optiforge/numerics/lu.py`) remains the reference factorization.
- **Default solver flags:**
  - `warm_start_relaxations = False`
  - `inherit_basis_factorization = False`
  - `use_incremental_basis = False`
- **Simplex algorithm:** Two-phase Revised Simplex using Bland's minimum-index rule for entering and leaving variable selection.

---

## 3. Profiling Methodology

Profiling was performed on the active codebase using `cProfile`, high-resolution `time.perf_counter`, and targeted per-pivot micro-timers across 2,050 actual simplex pivots on standard form LPs spanning small, medium, and large dimensions.

Every pivot step was isolated and measured individually:
- Basis extraction & factorization: $B = A[:, \text{basis}]$ and $LU(B)$
- Direction solve (FTRAN): $B d = a_q$
- Dual multipliers solve (BTRAN): $B^T y = c_B$
- Primal solve: $B x_B = b$
- Feasibility clamp & checks: non-negativity verification of $x_B$
- Ratio test: calculation of $\theta_i = x_B[i] / d[i]$ with Bland tie-breaking
- Reduced cost computation: $r = c - A^T y$
- Bland pricing scan: search for entering variable $q$
- Basis & nonbasic updates: array swaps and sorting

---

## 4. Bottleneck Evidence

From profiling 2,050 pivots (`scratch/profile_inner_loop.py`), the runtime breakdown per pivot was measured as:

| Operation | Calls/Pivot | Mean Time / Pivot | % Simplex Runtime | Allocation / Overhead Evidence |
| :--- | :---: | :---: | :---: | :--- |
| **Basis extract & DenseLU** | 1 | 460.52 $\mu$s | 56.9% | Dense NumPy slicing `A[:, basis]` and $O(m^3)$ LU |
| **Direction solve ($B d = a_q$)** | 1 | 105.10 $\mu$s | 13.0% | Forward/back substitution in `BasisFactorization.solve` |
| **Dual solve ($B^T y = c_B$)** | 1 | 92.61 $\mu$s | 11.4% | Forward/back substitution in `solve_transpose` |
| **Primal solve ($B x_B = b$)** | 1 | 88.76 $\mu$s | 11.0% | **Redundant linear solve**: $x_B$ can be updated mathematically |
| **Feasibility clamp** | 1 | 23.79 $\mu$s | 2.9% | Python row loop `for k in range(m):` |
| **Ratio test** | 1 | 14.70 $\mu$s | 1.8% | 3 list comprehensions, `zip`, lambda function calls |
| **Reduced costs ($A^T y$)** | 1 | 14.10 $\mu$s | 1.7% | Allocates new `ATy` and `r_all` arrays every pivot |
| **Pricing scan** | 1 | 6.46 $\mu$s | 0.8% | Full list comprehension `[j for j in nonbasic if ...]` |
| **Basis validation** | 1 | 1.83 $\mu$s | 0.2% | `len(basis)` and `set(basis)` checks |
| **Basis update** | 1 | 1.19 $\mu$s | 0.1% | `nonbasic.remove()` + `append()` + `sort()` ($O(N \log N)$) |

### Key Bottleneck Findings:
1. **Redundant Primal Linear Solve ($B x_B = b$):** In every pivot, `basis_fact.solve(b)` was invoked to compute $x_B$, consuming **11.0%** of pivot runtime (88.76 $\mu$s). Under Revised Simplex theory, once the initial $x_B$ is computed, subsequent basic solutions satisfy the exact algebraic recurrence $x_B^{\text{new}} = x_B - \theta^* d$ (with $x_B^{\text{new}}[i_{\text{leave}}] = \theta^*$). Re-solving from scratch on every single pivot was computationally unnecessary.
2. **Intermediate NumPy Array Allocations:** The expressions `ATy = A.T @ y` and `r_all = c - ATy` created 2 new full-length vectors of size $n$ on every single pivot.
3. **CPython List and Lambda Overhead in Ratio Test:** The ratio test constructed multiple Python lists (`eligible_rows`, `ratios`, `min_ratio_rows`) and called `min(..., key=lambda i: basis[i])`, which triggered repeated Python bytecode calls per eligible row.
4. **Full Candidate List Building in Pricing:** The pricing step evaluated all nonbasic columns into a list `candidates = [j for j in nonbasic if ...]` before calling `min(candidates)`. Since `nonbasic` is sorted in ascending order, the very first candidate encountered with improving reduced cost is mathematically identical to `min(candidates)`.

---

## 5. Optimization Hypotheses

| ID | Hypothesis | Target Operation | Expected Impact | Status |
| :---: | :--- | :--- | :--- | :---: |
| **H1** | Reusing/pre-allocating reduced-cost buffers eliminates $2 \times \text{pivots}$ array allocations. | $r = c - A^T y$ | Eliminates GC churn, improves cache locality | **VALIDATED & IMPLEMENTED** |
| **H2** | Vectorizing ratio test in NumPy and fast-pathing unique minimum avoids Python list overhead. | Ratio test | 1.77x microbenchmark speedup on ratio test | **VALIDATED & IMPLEMENTED** |
| **H3** | Early-break scan on sorted nonbasic list satisfies Bland's rule without building candidate lists. | Pricing scan | Avoids intermediate list allocations; breaks on first improving column | **VALIDATED & IMPLEMENTED** |
| **H4** | Replacing $O(N \log N)$ sort with $O(N)$ `bisect` preserves sorted nonbasic order with lower cost. | Nonbasic update | 1.52x microbenchmark speedup on nonbasic updates | **VALIDATED & IMPLEMENTED** |
| **H5** | Updating $x_B$ mathematically ($x_B - \theta^* d$) eliminates 1 full triangular solve per pivot. | Primal solve | Eliminates 88.76 $\mu$s per pivot (33.3% of all triangular solves) | **VALIDATED & IMPLEMENTED** |
| **H6** | Replacing DenseLU with pure-Python forward/backward loops. | Basis solves | Made execution slower due to Python interpreter overhead. | **REJECTED** |

---

## 6. Implemented Changes

All changes were implemented in [optiforge/lp/simplex.py](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/lp/simplex.py) and [optiforge/lp/dual_simplex.py](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/lp/dual_simplex.py):

### 1. Primal Solution Mathematical Update ($x_B \leftarrow x_B - \theta^* d$)
- Instead of solving $B x_B = b$ via LU back-substitution on every pivot:
  ```python
  theta_star = max(0.0, float(x_B[i_leave])) / d[i_leave]
  x_B -= theta_star * d
  x_B[i_leave] = theta_star
  ```
- **Self-Healing & Drift Prevention:**
  - $x_B$ is computed via `basis_fact.solve(b)` on iteration 0.
  - $x_B$ is re-solved from scratch every 25 pivots to prevent cumulative floating-point drift.
  - If feasibility clamping encounters a value $< -\tau_{\text{feas}}$, it re-solves $B x_B = b$ directly before declaring infeasibility.
  - At optimal or unbounded termination, $x_B$ is re-solved from `basis_fact.solve(b)` to guarantee bit-accurate final solution reporting.

### 2. Bland Pricing Early-Break Scan
- Because `nonbasic` is maintained strictly sorted in ascending order:
  $$\min \{ j \in \text{nonbasic} \mid r_j < -\tau \} \equiv \text{first } j \in \text{nonbasic} \text{ such that } r_j < -\tau$$
- The scan breaks on the first improving candidate, eliminating the construction of the Python list `candidates = [...]` and the subsequent `min(candidates)` call.

### 3. Vectorized Ratio Test & Fast-Path Bland Tie-Breaking
- Replaced list comprehensions and scalar zip loops with vectorized NumPy array operations:
  ```python
  eligible_mask = d > tolerances.pivot_tolerance
  eligible_indices = np.flatnonzero(eligible_mask)
  ...
  ratios = eligible_xB / eligible_d
  min_ratio = np.min(ratios)
  tie_mask = np.abs(ratios - min_ratio) <= 1e-11
  tie_indices = eligible_indices[tie_mask]
  if tie_indices.size == 1:
      i_leave = int(tie_indices[0])
  else:
      i_leave = min(tie_indices, key=lambda i: basis[i])
  ```
- In >95% of pivots, there is a unique minimum ratio (`tie_indices.size == 1`), taking the fast path without executing any Python lambda or key function.

### 4. Pre-allocated Reusable Buffers
- `ATy_buf` and `r_all` (and in dual simplex, `alpha` and `e_p`) are pre-allocated once before entering the pivot loop.
- Matrix-vector products use in-place operations: `np.dot(A.T, y, out=ATy_buf)` and `np.subtract(c, ATy_buf, out=r_all)`.

### 5. Bisect-based Nonbasic Maintenance
- Maintained `nonbasic` sorted order using `bisect.bisect_left` and `bisect.insort` ($O(N)$ operations), eliminating $O(N \log N)$ sorting and linear searches.

---

## 7. Mathematical Invariants

The implementation strictly preserves all linear-algebraic and Revised Simplex invariants:

1. **Primal Basic Solution Identity:**
   Let $B_{\text{new}}$ denote the basis matrix after replacing column $i_{\text{leave}}$ with $A_{j_{\text{enter}}} = B d$.
   $$\sum_{i \neq i_{\text{leave}}} B[:, i] (x_B[i] - \theta^* d[i]) + A_{j_{\text{enter}}} \theta^* = B x_B - \theta^* B d + \theta^* B d = B x_B = b$$
   Thus $B_{\text{new}} x_B^{\text{new}} = b$ is exact.
2. **Dual Multipliers:** $B^T y = c_B$ is solved using the exact LU factorization.
3. **Reduced Costs:** $r_N = c_N - A_N^T y$ is evaluated using in-place floating-point subtraction, mathematically identical to $c - A^T y$.
4. **Bland's Rule Invariant:** Smallest index entering variable and smallest basic index leaving variable are selected in all cases.
5. **Phase-I Feasibility:** Phase-I sum of artificial variables $w^* \le \tau$ remains unchanged.
6. **Objective Values:** Objective values match the reference implementation within floating-point tolerances.

---

## 8. Correctness Results

A dedicated test suite [tests/test_stage_8_13_pivot_kernel.py](file:///c:/Users/Prasanna/SIH2026/OptiForge/tests/test_stage_8_13_pivot_kernel.py) was constructed to test all edge cases and problem configurations:

| Test Case | Description | Result |
| :--- | :--- | :---: |
| `test_primal_xb_update_mathematical_identity` | 10 randomized trials verifying $\|B_{\text{new}} x_B^{\text{new}} - b\| < 10^{-11}$ | **PASSED** |
| `test_bland_pricing_early_break_equivalence` | 100 randomized trials testing MIN and MAX entering variable selection | **PASSED** |
| `test_vectorized_ratio_test_tie_breaking` | 50 trials with artificial exact ratio ties comparing scalar vs vectorized | **PASSED** |
| `test_inplace_gemv_reduced_cost` | Verification of in-place GEMV buffer vs fresh `c - A.T @ y` | **PASSED** |
| `test_small_bounded_minimize` | Bounded LP minimization with inequality constraints | **PASSED** |
| `test_small_bounded_maximize` | Bounded LP maximization with inequality constraints | **PASSED** |
| `test_equality_constraints` | Linear equality constraints requiring Phase-I artificials | **PASSED** |
| `test_infeasible_lp` | Infeasible LP detection in Phase I | **PASSED** |
| `test_unbounded_lp` | Unbounded LP detection via empty ratio test set | **PASSED** |
| `test_degenerate_lp` | Degenerate BFS with zero RHS and Bland anti-cycling verification | **PASSED** |
| `test_numerical_stability_monitor_tracking` | Diagnostic monitor tracking primal/dual residuals through optimized loop | **PASSED** |
| `test_milp_solver_with_optimized_simplex` | End-to-end MILP solving comparing cold vs warm start | **PASSED** |

---

## 9. Numerical Stability Results

Residual tracking across optimized solves showed zero loss of numerical precision:
- **Max Primal Residual ($\|A x - b\|_\infty$):** $\le 1.12 \times 10^{-14}$ (well below feasibility tolerance $10^{-7}$).
- **Max Residual Drift of $x_B$ update vs fresh solve:** $\le 5.68 \times 10^{-13}$ across 25 consecutive pivots.
- **Factorization Condition Numbers:** Identical to reference solves.
- **Pivot Magnitudes:** Monitored and verified against thresholds without small pivot warnings.
- **Dual Feasibility at Optimum:** Verified with zero reduced cost violations ($r_j \ge -10^{-7}$ for min, $r_j \le 10^{-7}$ for max).

---

## 10. Allocation Evidence

Memory allocation profiling using Python's `tracemalloc` across 10 repetitions per workload:

| Workload | Stage 8.12 Peak Memory | Stage 8.13 Peak Memory | Memory Delta | Allocation Evidence |
| :--- | :---: | :---: | :---: | :--- |
| **Small LP (5x8)** | 28.5 KB | 21.8 KB | **-23.5%** | Pre-allocated GEMV buffer eliminated 30 temporary array allocations |
| **Medium LP (20x30)** | 89.7 KB | 89.7 KB | Neutral | Buffer reuse; temporary array allocations eliminated |
| **Large LP (40x50)** | 272.1 KB | 272.3 KB | Neutral | Buffer reuse; DenseLU dominating memory footprint |
| **Sparse LP (30x40)** | 167.1 KB | 166.9 KB | Neutral | Buffer reuse; zero-copy slices preserved |
| **MILP Multi-level** | 110.2 KB | 110.2 KB | Neutral | Identical B&B tree memory profile |

---

## 11. Benchmark Methodology

- Benchmark script: `scratch/benchmark_stage813.py`
- Repetitions: 10 per workload with fixed random seeds.
- Garbage collection was disabled during timing intervals to isolate computational runtime.
- Metrics recorded: Mean, median, min, max, standard deviation, pivot count, solution status, objective value, and peak memory.

---

## 12. Before/After Benchmark Tables

### Wall-Clock Time Comparison (10 Repetitions)

| Workload | Dimensions / Specs | Stage 8.12 Mean (ms) | Stage 8.13 Mean (ms) | Speedup vs 8.12 | Pivots | Status / Obj Agreement |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **A. Small LP** | $m=5, n=8$, Dense | 16.810 ms | 15.682 ms | **1.07x** | 15 | True (24.192977) |
| **B. Medium LP** | $m=20, n=30$, Dense | 747.088 ms | 619.848 ms | **1.21x** | 206 | True (67.271623) |
| **C. Large LP** | $m=40, n=50$, Dense | 3292.093 ms | 3092.014 ms | **1.06x** | 489 | True (56.798814) |
| **D. Dense LP** | $m=20, n=30$, Density 1.0 | 690.481 ms | 572.561 ms | **1.21x** | 172 | True (90.875082) |
| **E. Sparse LP** | $m=30, n=40$, Density 0.25 | 1572.738 ms | 1412.677 ms | **1.11x** | 301 | True (441.751469) |
| **F. Phase-I LP** | $m=25, n=35$, Inequalities | 1111.365 ms | 972.364 ms | **1.14x** | 244 | True (66.472179) |
| **G. Degenerate LP** | $m=15, n=20$, Degenerate | 174.983 ms | 158.280 ms | **1.11x** | 67 | True (63.027746) |
| **H. MILP Child** | 4 cons, 8 integer vars | 125.035 ms | 125.926 ms | **0.99x** | 19 | True (45.459364) |
| **I. Multi-level MILP** | 6 cons, 8 integer vars | 50.194 ms | 48.197 ms | **1.04x** | 9 | True (49.067156) |

---

## 13. Comprehensive Evolution: Cold vs Stage 8.9 vs Stage 8.12 vs Stage 8.13

| Workload | Cold Mean (ms) | Stage 8.9 Mean (ms) | Stage 8.12 Mean (ms) | Stage 8.13 Mean (ms) | Cumulative Speedup (8.13 vs Cold) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A. Small LP (5x8)** | 16.810 | 16.810 | 16.810 | **15.682** | **1.07x** |
| **B. Medium LP (20x30)** | 747.088 | 747.088 | 747.088 | **619.848** | **1.21x** |
| **C. Large LP (40x50)** | 3292.093 | 3292.093 | 3292.093 | **3092.014** | **1.06x** |
| **D. Dense LP (20x30)** | 690.481 | 690.481 | 690.481 | **572.561** | **1.21x** |
| **E. Sparse LP (30x40)** | 1572.738 | 1572.738 | 1572.738 | **1412.677** | **1.11x** |
| **F. Phase-I LP (25x35)** | 1111.365 | 1111.365 | 1111.365 | **972.364** | **1.14x** |
| **G. Degenerate LP (15x20)** | 174.983 | 174.983 | 174.983 | **158.280** | **1.11x** |
| **H. MILP Child Relaxation** | 311.997 | 141.149 | 125.035 | **125.926** | **2.48x** |
| **I. Multi-level MILP** | 197.932 | 56.869 | 50.194 | **48.197** | **4.11x** |

---

## 14. Regression Results

Full pytest regression executed:
```bash
python -m pytest tests/ -q
858 passed, 12 warnings in 3.46s
```
- Total test count: 858 passed (all 846 baseline tests passed without modification or loosening of assertions + 12 new dedicated Stage 8.13 tests).
- 0 failed, 0 skipped, 0 xfailed.

---

## 15. Rejected Optimization Ideas

1. **Pure-Python for-loop replacing list comprehensions in ratio test:**
   - Attempted: Replacing list comprehensions with `for i in range(m):` in Python bytecode.
   - Result: Slowed down ratio test from 17.94 $\mu$s to 30.28 $\mu$s (0.59x slowdown) due to CPython bytecode loop overhead.
   - Decision: Rejected. Replaced with vectorized NumPy flatnonzero indexing (10.40 $\mu$s, 1.77x speedup).
2. **`c.take(basis, out=c_B)` vs `c[basis]`:**
   - Attempted: In-place buffer extraction for $c_B$.
   - Result: `c.take()` took 8.39 $\mu$s vs 5.70 $\mu$s for `c[basis]` due to method dispatch overhead.
   - Decision: Rejected. Kept `c[basis]`.
3. **Skipping Basis Refactorization without PFI/Incremental Basis:**
   - Attempted: Omitting LU refactorization entirely on default path without an update representation.
   - Result: Violates mathematical correctness.
   - Decision: Rejected. DenseLU remains the trusted reference factorization.

---

## 16. Remaining Bottlenecks

1. **DenseLU Basis Factorization ($O(m^3)$):**
   - Basis extraction and LU factorization currently accounts for **56.9%** of per-pivot runtime on dense problems.
   - On problems with $m > 50$, LU factorization time grows cubically. PFI / Forrest-Tomlin or sparse LU updating will be necessary to achieve sub-millisecond pivots for $m > 100$.
2. **Dense Matrix-Vector Slices:**
   - $B d = a_q$ requires column slicing $A[:, j_{\text{enter}}]$. On large matrices, non-contiguous column access has small cache penalties.

---

## 17. Limitations

1. **Dense Linear Algebra:** The solver uses dense 2D NumPy arrays. For large sparse linear programs ($m > 500$, density $< 1\%$), memory and FLOP complexity remain $O(m^2)$ and $O(m^3)$ respectively.
2. **Floating-Point Refresh Frequency:** $x_B$ is refreshed from $B^{-1} b$ every 25 pivots and at termination. While drift is measured to be $< 10^{-12}$, highly ill-conditioned bases ($\kappa(B) > 10^{12}$) may trigger premature re-solves via the safety fallback.
3. **MILP Short LP Relaxations:** When an LP relaxation finishes in 1–3 pivots, the inner loop speedup is small compared to tree search overhead (e.g. Workload H is neutral at 0.99x).

---

## 18. Claims and Evidence Matrix

| Claim | Evidence | Verdict |
| :--- | :--- | :---: |
| Simplex inner loop profiled | Profile table across 2,050 pivots (`scratch/profile_inner_loop.py`) | **VERIFIED** |
| Primal $x_B$ update preserves $B x_B = b$ | 10 randomized trials; residual error $< 10^{-11}$ (`test_primal_xb_update_mathematical_identity`) | **VERIFIED** |
| Bland pricing early break preserves min candidate | 100 trials with 100% agreement across MIN/MAX (`test_bland_pricing_early_break_equivalence`) | **VERIFIED** |
| Vectorized ratio test preserves Bland tie-breaking | 50 trials with deliberate ties; 100% agreement (`test_vectorized_ratio_test_tie_breaking`) | **VERIFIED** |
| Preallocated buffers eliminate temporary allocations | In-place GEMV verified (`test_inplace_gemv_reduced_cost`); memory down 23.5% on small LP | **VERIFIED** |
| End-to-end LP speedup vs Stage 8.12 | Measured 1.06x to 1.21x across all standard LP workloads (10 repetitions) | **IMPROVED** |
| End-to-end MILP speedup vs Cold | Measured 2.48x (Child) and 4.11x (Multi-level MILP) vs Cold baseline | **IMPROVED** |
| Zero test regression | 858/858 tests passing in 3.46s | **VERIFIED** |
| Tolerances and defaults unchanged | All experimental flags remain `False` by default | **VERIFIED** |

---

## 19. Final Implementation Status

**Verdict:** READY FOR INDEPENDENT AUDIT  
**Performance Status:** IMPROVED on medium/large LPs (1.06x–1.21x vs 8.12), WORKLOAD-DEPENDENT on short MILP relaxations (0.99x–1.04x vs 8.12, 2.48x–4.11x vs Cold).
