# OptiForge Stage 8.19 — Final Independent Technical Audit Report
**Audit Date:** September 28, 2026  
**Auditor:** Skeptical Independent Technical Auditor  
**Target Report:** `docs/STAGE_8_19_PFI_PROFILING_OPTIMIZATION_REPORT.md`  
**Verdict:** **READY WITH LIMITATIONS**

---

## 1. Audit Scope

This independent audit assesses OptiForge Stage 8.19 ("PFI Profiling and Optimization Investigation"). The scope covers:
- Verification of the profiling methodology and the empirical refutation of the Stage 8.18 eta-chain hypothesis.
- Independent reproduction of the granular execution breakdown (Base LU solve vs. Eta application vs. Simplex algebra).
- Detailed technical audit of the fast-PFI implementation (`use_fast_pivot_update`).
- Mathematical analysis of the finiteness check replacement (`np.isfinite(np.sum(...))` vs. `np.all(np.isfinite(...))`).
- Verification of state restoration and rollback invariants upon pivot failure.
- Multi-dimensional numerical stability across $m \in \{1, 2, 4, 6, 10, 20, 50, 100, 200, 500\}$.
- Investigation of the $m=500$ benchmark discrepancy between profiling and the before/after benchmark sweep.
- Independent performance reproduction without memory tracing.
- Evaluation of memory profiling claims (`tracemalloc` vs. RSS).
- Regression verification across the complete test suite.
- Audit of default configuration isolation.
- Assessment of the Stage 8.20 recommendation (base triangular solve optimization).

*Audit Constraint:* In accordance with audit standards, no production code was modified during this audit.

---

## 2. Repository Diff Audit

The repository changes introduced during Stage 8.19 were inspected across all affected files:

| File | Nature of Changes | Audit Observations |
| :--- | :--- | :--- |
| [`optiforge/numerics/pivot_basis.py`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/numerics/pivot_basis.py) | Added optional `fast_pivot_update: bool = False` to `__init__` and `factorize()`. Added fast-path type check in `solve()`. In `replace_column()`, implemented lazy `old_col` copying, scalar finiteness check, and `ndarray` fast paths when `fast_pivot_update=True`. | **No default behavior modified.** When `fast_pivot_update=False`, code path is 100% identical to Stage 8.18. |
| [`optiforge/lp/simplex.py`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/lp/simplex.py) | Added `use_fast_pivot_update: bool = False` to `SimplexTolerances`. Propagated flag to `PivotUpdateFactorization` constructor. | Default is `False`. Solver architecture, branching, pricing, ratio tests, and tolerances unchanged. |
| [`optiforge/lp/dual_simplex.py`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/lp/dual_simplex.py) | Passed `fast_pivot_update=getattr(tolerances, "use_fast_pivot_update", False)` to `PivotUpdateFactorization`. | Clean opt-in propagation. Dual simplex logic and bounds handling unchanged. |
| [`optiforge/milp/solver.py`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/milp/solver.py) | Added `use_fast_pivot_update: bool = False` to `MILPTolerances` and synchronized to `simplex_tolerances`. | Clean opt-in propagation. B&B logic, node selection, branching, and warm-start inheritance unchanged. |
| [`optiforge/numerics/lu.py`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/numerics/lu.py) | **Untouched.** | Preserved entirely. No unverified kernel modifications. |
| [`tests/test_stage_8_19_optimization.py`](file:///c:/Users/Prasanna/SIH2026/OptiForge/tests/test_stage_8_19_optimization.py) | Added 6 unit and regression tests verifying default flags, numerical equivalence, singularity handling, and non-finite column handling. | All tests pass. |

**Diff Conclusion:** Confirmed clean feature isolation. No hidden algorithmic or numerical changes exist in default paths.

---

## 3. Fast-PFI Implementation Verification

The optimization consists of three micro-optimizations inside `PivotUpdateFactorization` when `fast_pivot_update=True`:

### A. Lazy Old-Column Copy
- **Stage 8.18 Baseline:** `old_col = self._B[:, position].copy()` was executed eagerly on *every single pivot*, allocating an array of length $m$.
- **Stage 8.19 Optimization:** `old_col` is only copied if `_update_count >= self.max_updates` or if `abs_pivot < self.pivot_tolerance` (i.e. only when refactorization is actually triggered).
- **Audit Verification:** Confirmed. On normal updates (which constitute >97% of pivots), no column copy occurs. State restoration during refactorization failure was independently verified (see Section 5).

### B. Array Fast-Path Dispatch
- **Audit Verification:** If `new_column`, `search_direction`, or `rhs` are already contiguous 1D float64 `np.ndarray`s, redundant dispatch through `np.asarray(...)` and `.ravel()` is bypassed. If non-contiguous arrays, lists, or integers are passed, `np.asarray(..., dtype=np.float64)` correctly catches and converts them.

### C. Finiteness Check Flaw (Defect Identified)
- **Stage 8.18 Baseline:** `np.all(np.isfinite(new_col_arr))`
- **Stage 8.19 Optimization:** `np.isfinite(np.sum(new_col_arr))`
- **Audit Analysis:** While designed to avoid allocating a boolean array of length $m$, **these two expressions are NOT mathematically equivalent:**
  1. **Floating-point Summation Overflow:** If `new_col_arr` contains very large but perfectly finite floating-point values (e.g. `[1e308, 1e308]`), every entry is finite, but their sum overflows IEEE 754 float64 to `+inf`. In this case:
     - Baseline: `np.all(np.isfinite(arr))` $\to$ `True` (correct).
     - Optimization: `np.isfinite(np.sum(arr))` $\to$ `False` (incorrect false positive!).
     - Consequence: An unhandled `RuntimeWarning: overflow encountered in reduce` is emitted, and `PivotUpdateFactorization` raises `SingularBasisError("New column contains non-finite values (NaN or Inf).")` on a column whose entries are all finite!
  2. **Opposite Infinities Cancellation:** If `new_col_arr` contains `+inf` and `-inf`, `np.sum(arr)` evaluates to `nan` and emits `RuntimeWarning: invalid value encountered in reduce`. (While this correctly declares the column non-finite, it generates unexpected runtime warnings).
- **Audit Finding:** In standard LP benchmark instances where coefficients are normalized ($\le 10^4$), sum overflow does not manifest. However, in ill-scaled or poorly formulated models with coefficients near $10^{308}$, this optimization causes an invalid `SingularBasisError`. **This defect must be remedied before `use_fast_pivot_update` can be promoted to default.** (See Remediation Recommendation).

---

## 4. State Restoration and Rollback Verification

Because the eager column copy was removed, the auditor subjected `replace_column()` to adversarial input injections to verify state integrity:

| Injection Test Case | Expected Behavior | Measured Behavior | State Restored? | Etas Intact? | Subsequent Solves Valid? |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Linearly Dependent Column** ($A_q = 2 A_1$) | `SingularBasisError` raised on refactorization | Raised cleanly | **YES** (`self.B` restored) | **YES** (no eta appended) | **YES** ($B x = b$ correct) |
| **Zero Vector Column** ($A_q = 0$) | `SingularBasisError` raised | Raised cleanly | **YES** | **YES** | **YES** |
| **NaN Element in Column** | `SingularBasisError` raised | Raised cleanly | **YES** | **YES** | **YES** |
| **+Inf Element in Column** | `SingularBasisError` raised | Raised cleanly | **YES** | **YES** | **YES** |
| **Overflow Column** (`[1e308, 1e308]`) | Should succeed or refactorize | Raised false-positive `SingularBasisError` | **YES** | **YES** | **YES** |

**Restoration Verdict:** For genuine singular and non-finite inputs, state restoration and exception rollback function correctly. The basis matrix $B$ is restored to its exact prior column vector, active etas are preserved, and subsequent solves execute without state corruption.

---

## 5. Multi-Dimensional Numerical Correctness Verification

The auditor tested numerical invariance across dimensions $m \in \{1, 2, 4, 6, 10, 20, 50, 100, 200\}$ comparing:
1. `DenseLU` (Fresh refactorization on every pivot)
2. `Baseline PFI` (`fast_pivot_update=False`)
3. `Fast PFI` (`fast_pivot_update=True`)
4. `IncrementalBasisFactorization`

### Measured Results:

| Dimension $m \times n$ | Status Agreement | Objective Difference (Base vs DenseLU) | Objective Difference (Fast vs DenseLU) | Objective Difference (Inc vs DenseLU) | Numerical Pass |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $1 \times 3$ | **100% OPTIMAL** | $0.00$ | $0.00$ | $0.00$ | **PASS** |
| $2 \times 4$ | **100% OPTIMAL** | $0.00$ | $0.00$ | $0.00$ | **PASS** |
| $4 \times 6$ | **100% OPTIMAL** | $0.00$ | $0.00$ | $0.00$ | **PASS** |
| $6 \times 9$ | **100% OPTIMAL** | $0.00$ | $0.00$ | $0.00$ | **PASS** |
| $10 \times 15$ | **100% OPTIMAL** | $2.27 \times 10^{-13}$ | $2.27 \times 10^{-13}$ | $2.27 \times 10^{-13}$ | **PASS** |
| $20 \times 30$ | **100% OPTIMAL** | $2.27 \times 10^{-13}$ | $2.27 \times 10^{-13}$ | $2.27 \times 10^{-13}$ | **PASS** |
| $50 \times 75$ | **100% OPTIMAL** | $0.00$ | $0.00$ | $0.00$ | **PASS** |
| $100 \times 150$ | **100% OPTIMAL** | $0.00$ | $0.00$ | $0.00$ | **PASS** |
| $200 \times 300$ | **100% OPTIMAL** | $3.64 \times 10^{-12}$ | $3.64 \times 10^{-12}$ | $3.64 \times 10^{-12}$ | **PASS** |

All primal and dual residuals remained strictly within the $10^{-7}$ tolerance. Solution vectors match to within $10^{-8}$.

---

## 6. Profiling Reproduction and Bottleneck Attribution

The auditor independently instrumented `PivotUpdateFactorization._raw_solve` and `PivotUpdateFactorization._raw_solve_transpose` to measure the exact time spent in base triangular solves vs. eta vector application.

### Measured Profiling Breakdown:

| Workload | Dimension $m \times n$ | Total Time | Base Solve (`lu_solve`) | Base Solve % | Eta Apply Time | Eta Apply % | Solve Invocations |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `scale_lp_m50` | $50 \times 80$ | 9.9 ms | 6.5 ms | **66.2%** | 1.0 ms | **9.9%** | 62 |
| `scale_lp_m100` | $100 \times 150$ | 28.7 ms | 20.4 ms | **71.1%** | 1.2 ms | **4.3%** | 95 |
| `scale_lp_m200` | $200 \times 200$ | 354.1 ms | 247.3 ms | **69.8%** | 9.1 ms | **2.6%** | 581 |

### Audit Confirmation:
1. **The report's 70%–75% figure is VERIFIED.** Across all tested intermediate and large dimensions, base triangular solves account for **66% to 71%** of total solver runtime.
2. **The Stage 8.18 eta-chain hypothesis is EMPIRICALLY REFUTED.** Eta application accounts for only **2.6% to 4.3%** of runtime at $m \ge 100$. Vectorizing or compiling eta chains cannot resolve the performance bottleneck.

---

## 7. Benchmark Methodology and $m=500$ Discrepancy Investigation

### The Reported Discrepancy:
The implementation report exhibited a conspicuous discrepancy:
- In Section 2 (Profiling table): `scale_lp_m500` baseline took **6,345 ms**.
- In Section 6 (Benchmark table): `scale_lp_m500` baseline took **14,923 ms**, and optimized PFI took **34,659 ms** (-132% slowdown).

### Audit Investigation:
The auditor ran controlled experiments isolating the impact of `tracemalloc`:
1. **Clean Run (No Tracemalloc):**
   - `Baseline PFI`: **4,528.6 ms**
   - `Fast PFI`: **4,386.9 ms** (**3.1% faster** than Baseline PFI!)
2. **Instrumented Run (With Tracemalloc):**
   - `Fast PFI`: **13,664.7 ms** (**3.1x overhead!**)

### Root Cause Analysis:
1. `tracemalloc.start()` intercepts every single memory allocation at the CPython runtime level, recording full traceback frames.
2. In a 538-pivot LP with $m=500$, hundreds of thousands of internal NumPy array views and slices are instantiated.
3. During the Stage 8.19 benchmark sweep, multiple consecutive workloads were executed within the same process while `tracemalloc` accumulated allocation metadata. This triggered severe garbage collector churn and memory tracking latency, heavily penalizing later runs in the sweep.
4. **Audit Finding:** The reported 34.6-second execution time for `scale_lp_m500` was **an instrumentation artifact caused by `tracemalloc`**, not an algorithmic regression. When executed cleanly, Fast PFI executes in 4.38 seconds and is faster than Baseline PFI.

---

## 8. Independent Performance Reproduction (Without Tracemalloc)

The auditor executed a multi-repetition benchmark (7 to 10 repetitions per workload) under clean conditions without memory tracing:

| Workload | Baseline PFI (Median) | Fast PFI (Median) | IncrementalBasis (Median) | Measured Speedup (Fast vs Base) | Ratio (Inc / Fast) | Auditor Observation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| $m=1$ | 0.16 ms | 0.13 ms | 0.12 ms | **+16.77%** | 0.93x | Allocation savings observable at tiny $m$. |
| $m=4$ | 0.17 ms | 0.17 ms | 0.16 ms | -1.11% | 0.95x | Within measurement noise ($\pm 0.01$ ms). |
| $m=6$ | 0.46 ms | 0.49 ms | 0.45 ms | -6.88% | 0.91x | Run variance dominates sub-millisecond solves. |
| $m=10$ | 1.45 ms | 1.39 ms | 1.35 ms | **+4.39%** | 0.97x | Fast PFI faster than Base PFI. |
| $m=20$ | 1.12 ms | 1.14 ms | 1.10 ms | -2.15% | 0.97x | Fast PFI and Incremental nearly identical. |
| $m=50$ | 9.31 ms | 10.17 ms | 9.24 ms | -9.20% | 0.91x | Incremental Basis slightly faster. |
| $m=100$ | 29.81 ms | 29.71 ms | 27.66 ms | +0.35% | 0.93x | Base triangular solves dominate both. |
| $m=200$ | 358.16 ms | 348.80 ms | 348.67 ms | **+2.61%** | **1.00x** | **Fast PFI matches IncrementalBasis exactly.** |
| `branching_heavy` | 25.57 ms | 25.98 ms | 24.06 ms | -1.63% | 0.93x | MILP tree search variance. |
| `phase1_heavy_lp` | 5.36 ms | 3.93 ms | 3.84 ms | **+26.70%** | **0.98x** | Significant speedup on Phase-I column swaps. |
| `multilevel_bb` | 2.25 ms | 2.25 ms | 2.83 ms | +0.05% | **1.26x** | **Fast PFI 26% faster than IncrementalBasis.** |

### Performance Findings:
- The micro-optimizations consistently eliminate allocations on allocation-heavy workloads: `phase1_heavy_lp` achieved a **+26.7% speedup**, and $m=1$ achieved **+16.8%**.
- However, across clean runs, run-to-run variance on sub-millisecond workloads ranges from $\pm 5\%$ to $\pm 10\%$.
- Most significantly, **IncrementalBasis remains 3% to 9% faster than PFI across most small workloads ($m \le 50$)**, even with Fast PFI enabled. Fast PFI closes the gap, but does not reverse the ranking.

---

## 9. Faster Triangular Solve Claim Audit (Stage 8.20 Assessment)

The implementation report recommends replacing the pure Python forward and back substitution with `scipy.linalg.solve_triangular` and projects an overall solver speedup of **2x to 3x**.

### Audit Evaluation:
1. **Dependency Reality:** `scipy` is **NOT installed** in the current project environment (`ModuleNotFoundError: No module named 'scipy'`). OptiForge currently relies exclusively on standard Python and `numpy`.
2. **Classification:** The projected 2x–3x speedup is a **PLAUSIBLE HYPOTHESIS**, not a measured result.
3. **Validity of the Premise:** While the 2x–3x projection is speculative and unmeasured, **the premise is mathematically and empirically sound**: since pure Python triangular substitution accounts for ~70% of total runtime, compiling or vectorizing that specific kernel is the only mathematically viable route to achieving a substantial speedup across all revised simplex algorithms in OptiForge.

---

## 10. Memory Claims Audit

1. **Allocation Reduction:** Verified. Eliminating eager column copying and boolean array scanning removes $2 m$ float/bool element allocations per pivot.
2. **Metric Accuracy:** The implementation report correctly noted that memory was measured via `tracemalloc.get_traced_memory()`. However, `tracemalloc` measures peak heap allocations from Python, not resident set size (RSS). Peak memory diffs across small LP solves are negligible (<50 KB).

---

## 11. Full Regression Test Suite Audit

The auditor executed the complete pytest test suite:

```bash
python -m pytest tests/ -q
```
**Result:** **908 passed, 12 warnings in 2.68s.**

Specific Stage 8.19 tests:
```bash
python -m pytest tests/test_stage_8_19_optimization.py -v
```
**Result:** **6 passed in 0.29s.**

- 902 baseline regression tests pass with zero regressions.
- All 6 new Stage 8.19 verification tests pass cleanly.

---

## 12. Default and Feature Isolation Audit

The auditor explicitly verified that:
1. `SimplexTolerances.use_fast_pivot_update` defaults to `False`.
2. `MILPTolerances.use_fast_pivot_update` defaults to `False`.
3. `PivotUpdateFactorization.fast_pivot_update` defaults to `False`.
4. `use_adaptive_pivot_factorization` remains `False`.
5. `adaptive_pfi_min_dimension` remains `6`.
6. Toggling `use_fast_pivot_update` does not alter default solver outcomes, branching sequences, or refactorization thresholds.

---

## 13. Claim-by-Claim Verification Matrix

| Report Claim | Verification Status | Auditor Finding |
| :--- | :---: | :--- |
| Eta chain iteration is only 1.5%–3.5% of solve time | **VERIFIED** | Independently measured at 2.6%–4.3% at $m \ge 100$. Stage 8.18 hypothesis refuted. |
| Base LU triangular solve accounts for 70%–75% of runtime | **VERIFIED** | Independently measured at 66%–71% across $m \in [50, 200]$. Dominant bottleneck confirmed. |
| Lazy column copy avoids heap allocation on >97% of pivots | **VERIFIED** | Confirmed by code inspection and profiling. |
| Finiteness check replacement is mathematically equivalent | **DEFECTIVE** | **FAILED.** `np.isfinite(np.sum(arr))` overflows to `+inf` on finite vectors near $10^{308}$, triggering false-positive `SingularBasisError`. |
| State restoration is preserved on failure | **VERIFIED** | Singular and non-finite columns cleanly restore the basis matrix. |
| m=500 suffered a 132% slowdown | **REFUTED** | Measurement artifact of `tracemalloc`. Clean run executes in 4.38s (3.1% faster than baseline). |
| 908 tests passing | **VERIFIED** | 908/908 passed in 2.68s. |
| Category C Promotion Verdict | **VERIFIED** | Proper conservative engineering classification. |

---

## 14. Identified Limitations and Required Remediations

1. **Remediation Required: Safe Finiteness Check**  
   The current optimization in [`optiforge/numerics/pivot_basis.py`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/numerics/pivot_basis.py#L427):
   ```python
   if not np.isfinite(np.sum(new_col_arr)):
   ```
   must be corrected to avoid summation overflow while retaining low allocation overhead. The standard NumPy idiom that avoids allocating a full boolean array while remaining mathematically exact is:
   ```python
   if not np.all(np.isfinite(new_col_arr)): # or np.any(np.isnan(...)) with bound check
   ```
   Or using an out-buffer if micro-allocations are critical. Under no circumstances should `np.sum` be used for element-wise finiteness checks.
2. **Tracemalloc Benchmark Protocol**  
   Future benchmarking stages must separate timing measurements from memory tracking sweeps to prevent `tracemalloc` latency from distorting benchmark results.

---

## 15. Audit Verdict and Recommendations

### Final Verdict: **READY WITH LIMITATIONS**

### Rationale:
1. **Core Diagnostic Objective Met:** Stage 8.19 successfully resolved the core question: *"WHY does PFI lose to IncrementalBasis on specific workloads?"* It conclusively proved that eta chains are not the problem, triangular solves dominate both methods, and PFI lost on small workloads due to minor per-pivot allocation overheads.
2. **Safe Feature Isolation:** The optimization is fully isolated behind `use_fast_pivot_update = False`. All existing defaults remain untouched, and 908/908 regression tests pass.
3. **Limitation Noted:** The `np.sum` finiteness check defect must be corrected before `use_fast_pivot_update` is ever considered for promotion.

### Recommendation for Stage 8.20:
Stage 8.20 is **APPROVED to investigate base triangular solve acceleration**, with the following strict constraints:
- Must investigate pure NumPy vectorization first or consider optional compiled kernels without breaking OptiForge's zero-external-dependency baseline.
- Default solver behavior and DenseLU reference implementation must remain fully preserved.
- Numerical equivalence against the reference triangular solve must be tested to machine precision.
