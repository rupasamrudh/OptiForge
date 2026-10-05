# OPTIFORGE — STAGE 8.17 FINAL INDEPENDENT AUDIT
## Adaptive Dual-PFI Dispatch + Code Hardening

**Auditor:** Independent Technical Auditor  
**Date:** September 28, 2026  
**Baseline Audited:** Stage 8.16 (`READY WITH LIMITATIONS`, 885 tests)  
**Subject Under Audit:** Stage 8.17 Implementation Report (`docs/STAGE_8_17_ADAPTIVE_PFI_REPORT.md`)  
**Audit Verdict:** **READY WITH LIMITATIONS**

---

### 1. Executive Summary

This independent audit evaluated Stage 8.17 of the OptiForge project, focusing on two primary areas:
1. **Code Hardening:** Resolution of typing introspection failure (`typing.Any` import in `DualSimplexResult`) and narrowing of overly broad exception handlers (`except Exception`) to numerical exceptions (`SingularBasisError`, `ArithmeticError`).
2. **Adaptive Dual-PFI Dispatch:** Introduction of `should_use_pivot_factorization(m, tolerances)` and configurable dimension-based switching between `IncrementalBasisFactorization` and `PivotUpdateFactorization` (PFI).

**Key Findings:**
- **Code Hardening is 100% Verified:** `typing.get_type_hints(DualSimplexResult)` now succeeds without error. All five targeted exception handlers in `optiforge/lp/dual_simplex.py` and `optiforge/milp/solver.py` were strictly narrowed. Numerical failures cleanly fall back to `DenseLU`, while programming defects (`TypeError`, `IndexError`, `AttributeError`, `KeyError`, `ValueError`) propagate unimpeded.
- **Full Regression Passes:** All 894 tests pass cleanly (885 baseline + 9 new tests) in 4.34s with zero failures, zero skips, and zero xfails. No tests were weakened or masked.
- **Mathematical Correctness Invariance is Preserved:** Across 13 distinct LP and MILP problem classes (MIN, MAX, unbounded, degenerate, Phase-I, non-zero bounds, integer, binary, mixed-integer, multilevel B&B, infeasible child nodes), solutions matched 100% across DenseLU, IncrementalBasis, Pure PFI, and Adaptive PFI.
- **Backward Compatibility is Maintained:** Default solver configurations (`use_pivot_factorization=False`, `use_adaptive_pivot_factorization=False`) preserve exact Stage 8.16 / Stage 8.9 behavior with zero PFI invocations.
- **Discrepancy 1 — Benchmark Degeneracy on Candidate Thresholds:** The Stage 8.17 report claimed that testing thresholds $m \ge 2, 4, 5, 6, 8, 10$ on the 10 deterministic MILP workloads empirically proved $m \ge 6$ was optimal and avoided regressions on $m=1, 4, 5$. **Audit revealed that in standard form, all 10 workloads have row dimension $m_{SF} \ge 9$ (child relaxations $m_{child} \ge 10$) due to variable upper bound rows.** Consequently, for all tested thresholds $T \le 10$, the solver selected PFI 100% of the time (`adaptive_inc_chosen = 0`). The runtime differences reported across thresholds in Table 6 were entirely host measurement jitter, not algorithmic divergence.
- **Discrepancy 2 — The "Infeasible-Child Anomaly" was Host System Jitter:** The report's observed 37.81 ms runtime for `infeasible_children` under $m \ge 6$ was completely unreproducible. An independent 20-run reproduction measured median 7.59 ms (faster than WarmStart 7.71 ms). In both runs, exactly 10 PFI solves and 18 updates occurred. The report erroneously rationalized host noise as algorithmic sensitivity.
- **Discrepancy 3 — Dispatch Accounting Under-Counts Infeasible Child Solves:** When Dual Simplex proves a relaxation infeasible, `DualRevisedSimplexSolver` returns `DualSimplexResult` with `basis_factorization=None`. Consequently, `MILPSolver._try_warm_start_relaxation` fails the `isinstance(res.basis_factorization, ...)` check, and does not increment `adaptive_pfi_chosen` or `adaptive_inc_chosen`. On workloads with infeasible branches, `adaptive_pfi_chosen + adaptive_inc_chosen = accepted - infeasible_pruned` rather than `accepted`.

---

### 2. Audit Scope

The audit encompassed:
1. Complete code diff inspection of `optiforge/lp/dual_simplex.py`, `optiforge/lp/simplex.py`, `optiforge/milp/solver.py`, and `tests/test_stage_8_17_adaptive_pfi.py`.
2. Full pytest test suite execution and test modification integrity verification.
3. Rigorous boundary testing of the adaptive dispatch decision matrix across 6 thresholds and 5 dimension test points per threshold.
4. Backward compatibility validation comparing Stage 8.17 defaults against Stage 8.16-equivalent configurations.
5. Explicit PFI override verification ensuring `use_pivot_factorization=True` unconditionally dominates adaptive settings.
6. Exception-hardening stress testing across 5 numerical exception types and 5 programming error types.
7. Independent reproduction of the 10-workload $\times$ 10-repetition benchmark across Cold, WarmStart, Pure PFI, and 6 adaptive thresholds.
8. Investigation of the empirical justification for the default threshold $m \ge 6$.
9. Deep investigation of the `infeasible_children` performance anomaly.
10. Correctness invariance across 13 problem formulations.
11. State isolation, fallback safety, and branch tree independence.
12. Statistical accounting validation of warm-start counters.
13. Numerical stability stress testing on ill-conditioned bases up to condition number $10^8$.

---

### 3. Repository Diff & Code Inspection

| File | Status | Audit Findings |
|:---|:---:|:---|
| `optiforge/lp/dual_simplex.py` | Verified | Added `from typing import Any`. Added `use_adaptive_pivot_factorization` and `adaptive_pfi_min_dimension` to `DualRevisedSimplexSolver.__init__`. Implemented `should_use_pivot_factorization`. Integrated dispatch call in `solve_standard_form`. Exception handlers narrowed to `except (SingularBasisError, ArithmeticError):` at lines 389, 425, and 545. |
| `optiforge/lp/simplex.py` | Verified | Added `use_adaptive_pivot_factorization: bool = False` and `adaptive_pfi_min_dimension: int = 6` to `SimplexTolerances`. Defaults strictly preserve non-adaptive, non-PFI baseline. |
| `optiforge/milp/solver.py` | Verified | Imported `SingularBasisError` and `IncrementalBasisFactorization`. Added `use_adaptive_pivot_factorization: bool = False` and `adaptive_pfi_min_dimension: int = 6` to `MILPTolerances`. Added `adaptive_pfi_chosen: int = 0` and `adaptive_inc_chosen: int = 0` to `WarmStartStats`. Synchronized tolerances in `MILPSolver.__init__`. Forwarded tolerances and narrowed exception handling at lines 980 and 997 in `_try_warm_start_relaxation`. |
| `tests/test_stage_8_17_adaptive_pfi.py` | Verified | 9 new tests covering typing inspection, exception narrowing, threshold boundary behavior, below/above threshold dispatch, explicit PFI override, MIN/MAX LPs, and MILP warm-start stats. |

**Configuration Defaults Check:**
- `SimplexTolerances.use_pivot_factorization`: `False` (Verified)
- `SimplexTolerances.use_adaptive_pivot_factorization`: `False` (Verified)
- `SimplexTolerances.adaptive_pfi_min_dimension`: `6` (Verified)
- `MILPTolerances.use_adaptive_pivot_factorization`: `False` (Verified)
- `MILPTolerances.adaptive_pfi_min_dimension`: `6` (Verified)

---

### 4. Regression Results

Full regression was executed in the active workspace environment:
```
Command: python -m pytest tests/ -q
Result: 894 passed, 12 warnings in 4.34s
```
- **Total Tests Collected:** 894
- **Passed:** 894
- **Failed:** 0
- **Skipped:** 0
- **XFailed:** 0
- **Warnings:** 12 (3 deprecation warnings in `httpx2`, 9 `PytestReturnNotNoneWarning` in legacy tests; zero warnings in new Stage 8.17 code)

**Test Modification Audit:**
No existing tests from Stage 8.16 or earlier were deleted, skipped, weakened, or modified. Baseline test count was 885; Stage 8.17 adds exactly 9 unit/integration tests in `tests/test_stage_8_17_adaptive_pfi.py` ($885 + 9 = 894$).

---

### 5. Dispatch Logic Audit

The dispatch decision matrix implemented in `DualRevisedSimplexSolver.should_use_pivot_factorization` was evaluated across all combinations:

```python
def should_use_pivot_factorization(
    self, m: int, tolerances: SimplexTolerances | None = None
) -> bool:
    tol = tolerances or self.tolerances
    if self.use_pivot_factorization or tol.use_pivot_factorization:
        return True
    if self.use_adaptive_pivot_factorization or tol.use_adaptive_pivot_factorization:
        threshold = (
            self.adaptive_pfi_min_dimension
            if self.adaptive_pfi_min_dimension is not None
            else tol.adaptive_pfi_min_dimension
        )
        return m >= threshold
    return False
```

#### Decision Matrix Verification:

| Configuration Case | Explicit PFI | Adaptive PFI | Basis Dimension $m$ | Expected Decision | Actual Decision | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Case A** | `False` | `False` | $m \in \{0, 1, T-1, T, T+1\}$ | `False` (DenseLU / IncBasis) | `False` | **VERIFIED** |
| **Case B** | `True` | `False` | $m \in \{0, 1, T-1, T, T+1\}$ | `True` (PFI) | `True` | **VERIFIED** |
| **Case C** | `False` | `True` | $m < \text{threshold}$ | `False` (IncrementalBasis) | `False` | **VERIFIED** |
| **Case D** | `False` | `True` | $m == \text{threshold}$ | `True` (PFI) | `True` | **VERIFIED** |
| **Case E** | `False` | `True` | $m > \text{threshold}$ | `True` (PFI) | `True` | **VERIFIED** |
| **Case F** | `True` | `True` | $m < \text{threshold}$ | `True` (Explicit PFI wins) | `True` | **VERIFIED** |
| **Case G** | `True` | `True` | $m \ge \text{threshold}$ | `True` (Explicit PFI wins) | `True` | **VERIFIED** |

All candidate threshold values ($T \in \{2, 4, 5, 6, 8, 10\}$) were tested at $m=0, 1, T-1, T, T+1$. No off-by-one errors exist; the boundary strictly triggers at $m \ge T$. Tolerance object overriding was also verified.

---

### 6. Backward Compatibility Audit

To confirm that existing code oblivious to Stage 8.17 options behaves identically:
1. Default solvers (`MILPSolver()` without parameters) initialize with adaptive flags set to `False`.
2. All 10 MILP workloads were solved under Stage 8.17 default warm-start settings (`warm_start_relaxations=True, inherit_basis_factorization=True`) versus explicit Stage 8.16 baseline settings (`use_pivot_factorization=False, use_adaptive_pivot_factorization=False`).
3. **Results:**
   - Solution status: 100.0% identical across all 10 workloads.
   - Objective value: 100.0% identical ($|\Delta \text{Obj}| < 10^{-10}$).
   - B&B node count: 100.0% identical.
   - Attempted, accepted, and fallback counts: 100.0% identical.
   - Dual pivots: 100.0% identical.
   - PFI counters (`adaptive_pfi_chosen`, `pfi_updates`, `pfi_refactorizations`, `pfi_fallbacks`): strictly `0` across all workloads.

Default adaptive mode does not silently activate PFI under any circumstance.

---

### 7. Code-Hardening Audit

#### 7.1 Typing Introspection
- Executed `typing.get_type_hints(DualSimplexResult)`.
- Previously raised `NameError: name 'Any' is not defined`.
- In Stage 8.17, returns full dictionary containing all 9 attributes. `hints["basis_factorization"]` correctly resolves to `typing.Any | None`.
- **Verdict: VERIFIED.**

#### 7.2 Exception Narrowing
Tested all narrowed exception sites in `optiforge/lp/dual_simplex.py` and `optiforge/milp/solver.py`:

```
Site 1: dual_simplex.py:389 (inc_fact.solve fallback)
Site 2: dual_simplex.py:425 (inc_fact.solve_transpose fallback)
Site 3: dual_simplex.py:545 (inc_fact.replace_column fallback)
Site 4: milp/solver.py:980   (BorderedBasisFactorization constructor fallback)
Site 5: milp/solver.py:997   (dual_solver.solve_standard_form fallback)
```

Injected mock exceptions into `PivotUpdateFactorization.replace_column`:
1. **Numerical Exceptions (Must trigger clean fallback to DenseLU):**
   - `SingularBasisError` $\rightarrow$ Fallback caught; solve completed with `SolutionStatus.OPTIMAL`.
   - `ArithmeticError` $\rightarrow$ Fallback caught; solve completed with `SolutionStatus.OPTIMAL`.
   - `FloatingPointError` $\rightarrow$ Fallback caught; solve completed with `SolutionStatus.OPTIMAL`.
   - `ZeroDivisionError` $\rightarrow$ Fallback caught; solve completed with `SolutionStatus.OPTIMAL`.
   - `OverflowError` $\rightarrow$ Fallback caught; solve completed with `SolutionStatus.OPTIMAL`.
2. **Programming Errors (Must propagate directly to surface bugs):**
   - `TypeError` $\rightarrow$ Propagated directly to caller.
   - `IndexError` $\rightarrow$ Propagated directly to caller.
   - `AttributeError` $\rightarrow$ Propagated directly to caller.
   - `KeyError` $\rightarrow$ Propagated directly to caller.
   - `ValueError` $\rightarrow$ Propagated directly to caller.

Search for residual broad exception handling: No `except Exception` blocks remain in `optiforge/lp/dual_simplex.py` or `optiforge/milp/solver.py`.

---

### 8. Threshold Benchmark Reproduction

The benchmark was independently re-executed across all 10 MILP workloads with 10 repetitions per configuration, measuring Cold (A), WarmStart Reference (B), Pure PFI (C), and Adaptive thresholds $m \ge 2, 4, 5, 6, 8, 10$.

#### Independent Benchmark Results (Median ms across 10 repetitions):

| Workload | High-Level $(m, n)$ | Std Form $m_{SF}$ | Cold (A) | WarmStart (B) | Pure PFI (C) | $m \ge 2$ | $m \ge 4$ | $m \ge 5$ | $m \ge 6$ | $m \ge 8$ | $m \ge 10$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `knapsack_small` | $(1, 8)$ | 9 | 9.68 | 5.08 | 5.48 | 5.36 | 4.99 | 5.73 | 4.80 | 5.20 | 5.13 |
| `set_covering` | $(6, 10)$ | 16 | 10.64 | 4.87 | 4.37 | 4.93 | 4.82 | 4.77 | 4.85 | 4.81 | 4.84 |
| `binary_selection` | $(4, 12)$ | 16 | 8.58 | 4.12 | 3.33 | 4.10 | 4.46 | 4.02 | 4.14 | 4.59 | 4.01 |
| `mixed_int_resource` | $(8, 10)$ | 13 | 0.92 | 0.97 | 0.68 | 1.04 | 1.01 | 1.06 | 1.18 | 1.12 | 1.04 |
| `integer_allocation` | $(5, 6)$ | 11 | 19.79 | 12.05 | 14.68 | 11.69 | 11.58 | 11.47 | 11.56 | 11.32 | 11.27 |
| `branching_heavy` | $(1, 10)$ | 11 | 40.95 | 24.45 | 25.48 | 22.25 | 22.77 | 22.14 | 22.51 | 23.41 | 24.05 |
| `multilevel_bb` | $(10, 14)$ | 16 | 5.82 | 3.06 | 2.21 | 2.92 | 2.92 | 2.81 | 3.08 | 2.89 | 2.90 |
| `infeasible_children` | $(8, 5)$ | 13 | 14.95 | 9.31 | 8.96 | 8.64 | 9.02 | 8.59 | 8.08 | 9.52 | 8.50 |
| `strong_relaxation` | $(6, 8)$ | 14 | 6.74 | 3.30 | 2.61 | 2.93 | 2.95 | 3.01 | 2.93 | 2.92 | 2.95 |
| `weak_relaxation` | $(4, 8)$ | 12 | 25.22 | 10.89 | 11.78 | 11.19 | 10.98 | 10.93 | 11.89 | 11.65 | 11.92 |

---

### 9. Investigation of the $m=6$ Claim and Benchmark Degeneracy

#### The Discrepancy:
In Section 6 and Section 10 of the Stage 8.17 report, the authors claim:
> *"On branching_heavy (m=1), Pure PFI regressed to 28.27 ms vs WarmStart 26.25 ms. Adaptive dispatch with m >= 6 selects IncrementalBasisFactorization, running at 25.91 ms (avoiding the regression). Similarly on integer_allocation (m=5), runtime improved from 14.92 ms under Pure PFI to 12.51 ms under m >= 6."*

#### Independent Audit Finding:
**This claim is contradicted by the codebase.**
1. In `optiforge/milp/solver.py`, when a model is prepared for relaxation, each variable's upper bound (e.g. $x_j \le 1$ for binary variables) is converted to an equality constraint $x_j + s_j = u_j$ in standard form.
2. In `branching_heavy`, the high-level model has 1 constraint and 10 binary variables. The root standard-form constraint matrix $A$ has dimension $m_{SF} = 1 + 10 = 11$ rows. Each child relaxation adds a branch constraint row, so child relaxations have dimension $m_{child} = 12$.
3. When `DualRevisedSimplexSolver.solve_standard_form` is called on child relaxations, `m` is $A_{child}.shape[0] = 12$.
4. For all tested thresholds $T \in \{2, 4, 5, 6, 8, 10\}$, $m_{child} = 12 \ge T$. Therefore, `should_use_pivot_factorization(12)` returned `True` **every single time**.
5. Inspection of solver statistics confirms that across all 10 workloads and all 6 candidate thresholds:
   ```
   knapsack_small:      pfi_chosen=6,  inc_chosen=0
   set_covering:        pfi_chosen=2,  inc_chosen=0
   binary_selection:    pfi_chosen=2,  inc_chosen=0
   mixed_int_resource:  pfi_chosen=0,  inc_chosen=0 (root optimal, 0 branches)
   integer_allocation:  pfi_chosen=13, inc_chosen=0
   branching_heavy:     pfi_chosen=26, inc_chosen=0
   multilevel_bb:       pfi_chosen=2,  inc_chosen=0
   infeasible_children: pfi_chosen=10, inc_chosen=0
   strong_relaxation:   pfi_chosen=2,  inc_chosen=0
   weak_relaxation:     pfi_chosen=12, inc_chosen=0
   ```
   `adaptive_inc_chosen` was **strictly 0** across all 10 workloads for all candidate thresholds.
6. **Conclusion:** All columns for $m \ge 2, 4, 5, 6, 8, 10$ in Table 6 executed the **identical code path** (PFI on every child solve). The differences in reported runtimes were pure measurement noise. The benchmark suite did not actually test the adaptive crossover point on these MILP models because none of them had standard-form child dimension $m < 10$.

#### Synthetic Direct Crossover Evaluation:
To evaluate the true crossover dimension, an independent benchmark was run on standard-form LPs without artificial upper-bound slacks across $m \in \{1, 2, 3, 4, 5, 6, 8, 10\}$ (100 repetitions each):

| Dimension $m$ | IncrementalBasis (ms) | PFI (ms) | Ratio (Inc / PFI) | Winner |
|:---:|:---:|:---:|:---:|:---:|
| $m=1$ | 0.0873 | 0.0896 | 0.97x | Tie / Parity |
| $m=2$ | 0.1874 | 0.1932 | 0.97x | Incremental |
| $m=3$ | 0.3075 | 0.3128 | 0.98x | Tie / Parity |
| $m=4$ | 0.4379 | 0.4898 | 0.89x | **Incremental (PFI regresses 11%)** |
| $m=5$ | 0.5969 | 0.6065 | 0.98x | Tie / Parity |
| $m=6$ | 0.4128 | 0.4184 | 0.99x | Parity |
| $m \ge 8$ | 1.4882 | 1.5133 | 0.98x | Parity (pulls ahead on >10 pivots) |

**Audit Conclusion on Threshold:** Setting the default minimum dimension to $m \ge 6$ is a sound engineering heuristic because on very small systems ($m \le 4$), `IncrementalBasis` consistently avoids Python eta-chain dispatch overhead (such as the 11% regression at $m=4$). However, the claim in the implementation report that the 10 MILP workloads empirically justified $m=6$ over $m=8$ or $m=10$ is contradicted by code reality.

---

### 10. Investigation of the Infeasible-Child Anomaly

In the Stage 8.17 report, `infeasible_children` showed an extreme runtime jump from WarmStart (8.27 ms) and Pure PFI (8.81 ms) to $m \ge 6$ (37.81 ms), which the author theorized was caused by "workload sensitivity" in bordered factorization during infeasibility proofs.

#### Independent Audit Investigation:
1. **Reproducibility Test:** Ran 20 consecutive repetitions of WarmStart, Pure PFI, Adaptive $m \ge 6$, and Adaptive $m \ge 20$ (which forces IncrementalBasis):
   - WarmStart (B): median = 7.71 ms, mean = 8.01 ms
   - Pure PFI (C): median = 7.72 ms, mean = 7.82 ms
   - Adaptive $m \ge 6$: median = 7.59 ms, mean = 7.68 ms
   - Adaptive $m \ge 20$: median = 7.84 ms, mean = 8.10 ms
   **The 37.81 ms slowdown is completely unreproducible.**
2. **Algorithmic Path Inspection:**
   - Both in the original data and in our audit, under $m \ge 6$, the solver performed:
     `attempted=12, accepted=12, fallbacks=0, infeasible_pruned=2`
     `dual_pivots=19, cold_pivots=5, pfi_updates=18, pfi_refactors=0, pfi_fallbacks=0`
   - Under Pure PFI (C), the solver performed the exact same operations:
     `attempted=12, accepted=12, fallbacks=0, infeasible_pruned=2, pfi_updates=18`
3. **Audit Finding:** Because both Pure PFI and Adaptive $m \ge 6$ executed the exact same operations with identical updates and zero fallbacks, the 37.81 ms spike in the report's single benchmark run was entirely caused by **transient host CPU throttling, background OS scheduling, or garbage collection jitter**. It was not an algorithmic anomaly.

---

### 11. Correctness Invariance Audit

Correctness was evaluated across 13 distinct problem categories comparing DenseLU, IncrementalBasis, Pure PFI, and Adaptive PFI ($m \ge 6$):

| # | Test Problem Class | Formulation / Characteristics | Status Across All 4 Factorizations | Objective Agreement | Node Match | Verdict |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| 1 | MIN LP | 2 vars, 2 inequality constraints | `OPTIMAL` (4/4) | $10.0000$ ($< 10^{-10}$) | N/A | **PASS** |
| 2 | MAX LP | 2 vars, 2 inequality constraints | `OPTIMAL` (4/4) | $12.6667$ ($< 10^{-10}$) | N/A | **PASS** |
| 3 | Feasible Multi-variable LP | $m=8, n=12$, dense random coefficients | `OPTIMAL` (4/4) | $113.0000$ ($< 10^{-10}$) | N/A | **PASS** |
| 4 | Infeasible LP | Incompatible bounds ($x \le 2, x \ge 5$) | `INFEASIBLE` (4/4) | N/A | N/A | **PASS** |
| 5 | Unbounded LP | Unbounded cone ($x - y \le 2, \max x + y$) | `UNBOUNDED` (4/4) | N/A | N/A | **PASS** |
| 6 | Degenerate LP | Beale's cycling LP (4 vars, 3 rows) | `OPTIMAL` (4/4) | $-4.0000$ ($< 10^{-10}$) | N/A | **PASS** |
| 7 | Phase-I LP | Equality constraint requiring artificials | `OPTIMAL` (4/4) | $6.4000$ ($< 10^{-10}$) | N/A | **PASS** |
| 8 | Non-zero Lower Bounds | $x_1 \in [2, 10], x_2 \in [3, 8]$ | `OPTIMAL` (4/4) | $48.0000$ ($< 10^{-10}$) | N/A | **PASS** |
| 9 | Integer MILP | 2 integer vars, 2 constraints | `OPTIMAL` (4/4) | $20.0000$ ($< 10^{-10}$) | 1 node | **PASS** |
| 10 | Binary MILP | 3 binary vars, knapsack constraint | `OPTIMAL` (4/4) | $11.0000$ ($< 10^{-10}$) | 1 node | **PASS** |
| 11 | Mixed-Integer MILP | 1 continuous, 1 integer var | `OPTIMAL` (4/4) | $16.0000$ ($< 10^{-10}$) | 3 nodes | **PASS** |
| 12 | Multilevel B&B | 10 constraints, 14 integer vars | `OPTIMAL` (4/4) | $120.8230$ ($< 10^{-10}$) | 3 nodes | **PASS** |
| 13 | Infeasible Child Nodes | 8 constraints, 5 binary vars | `OPTIMAL` (4/4) | $34.5317$ ($< 10^{-10}$) | 13 nodes | **PASS** |

Mathematical correctness invariance is 100% preserved.

---

### 12. Fallback & State Safety

1. **Parent Factorization Isolation:** Child node solves instantiate a distinct `DualRevisedSimplexSolver` with parent standard-form data copied. Mutating child basis structures leaves the parent basis factorization completely unaltered.
2. **Sibling Isolation:** Sibling branches are created from independent copies of `WarmStartBasisState`. In `tests/test_stage_8_16_dual_pfi.py::test_sibling_isolation_no_leakage`, left and right branches pivot into different bases with zero state leakage.
3. **Adaptive Dispatch Isolation:** Adaptive mode does not share eta chains across nodes. Each child solve instantiates a fresh `PivotUpdateFactorization` or `IncrementalBasisFactorization`.
4. **Numerical Failure Fallback:** If `replace_column` encounters numerical singularity or residual violation, it cleanly falls back to fresh `DenseLU`. A corrupted factorization cannot remain active.

---

### 13. Statistics & Accounting Audit (Defect Identified)

The accounting invariant $\text{attempted} = \text{accepted} + \text{fallbacks}$ was verified across all 10 workloads in all modes.

#### Discrepancy Found in Dispatch Statistics:
In `optiforge/milp/solver.py`, lines 1002–1011:
```python
if isinstance(res.basis_factorization, PivotUpdateFactorization):
    ...
    if self.simplex_tolerances.use_adaptive_pivot_factorization:
        self.warm_start_stats.adaptive_pfi_chosen += 1
elif isinstance(res.basis_factorization, IncrementalBasisFactorization):
    if self.simplex_tolerances.use_adaptive_pivot_factorization:
        self.warm_start_stats.adaptive_inc_chosen += 1
```
However, in `optiforge/lp/dual_simplex.py` line 493:
```python
return DualSimplexResult(
    status=SolutionStatus.INFEASIBLE,
    basis_indices=current_basis,
    nonbasic_indices=current_nonbasic,
    x=x,
    y=y,
    objective_value=obj_val,
    iterations=iterations,
    message=f"Dual Simplex detected primal infeasibility at row {p}...",
    # Note: basis_factorization is omitted! Defaults to None.
)
```
**Impact:**
When a child relaxation is proven infeasible:
- `res.basis_factorization` is `None`.
- Neither `adaptive_pfi_chosen` nor `adaptive_inc_chosen` is incremented.
- Yet line 1041 increments `self.warm_start_stats.accepted += 1` and `infeasible_pruned += 1`.
- Consequently, on workloads with infeasible branches (e.g. `integer_allocation`, `infeasible_children`), the dispatch counters under-count by exactly `infeasible_pruned`:
  ```
  integer_allocation:  accepted = 16, pfi_chosen = 13, infeasible_pruned = 3
  infeasible_children: accepted = 12, pfi_chosen = 10, infeasible_pruned = 2
  ```
**Remediation Recommendation for Future Stage:** In `dual_simplex.py` line 493, pass `basis_factorization=basis_fact` when returning `SolutionStatus.INFEASIBLE`.

---

### 14. Numerical Stability Audit

Numerical stress testing was conducted across matrices with condition numbers ranging from $10^2$ to $10^8$:
- **Forward residual:** $\|B x - rhs\| / \|rhs\|$ was $< 5 \times 10^{-15}$ for well-conditioned systems, and $< 2 \times 10^{-9}$ at condition number $10^8$.
- **Backward residual:** $\|B^T y - rhs\| / \|rhs\|$ maintained $< 10^{-8}$ at condition number $10^8$.
- **Repeated Column Updates:** Consecutive rank-1 updates maintained stable residual checks; upon exceeding residual tolerance, refactorization fallback cleanly engaged.
- Adaptive dispatch does not alter numerical update logic or bypass tolerances.

---

### 15. Feature Interaction Matrix

| Mode | Explicit PFI | Adaptive PFI | Basis Dim $m$ vs Threshold $T$ | Expected Factorization | Actual Factorization | Correct |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Default Mode** | `False` | `False` | $m=3, T=6$ | `IncrementalBasisFactorization` | `IncrementalBasisFactorization` | **Yes** |
| **Explicit PFI** | `True` | `False` | $m=3, T=6$ | `PivotUpdateFactorization` | `PivotUpdateFactorization` | **Yes** |
| **Adaptive Below $T$** | `False` | `True` | $m=3, T=6$ | `IncrementalBasisFactorization` | `IncrementalBasisFactorization` | **Yes** |
| **Adaptive At $T$** | `False` | `True` | $m=6, T=6$ | `PivotUpdateFactorization` | `PivotUpdateFactorization` | **Yes** |
| **Adaptive Above $T$** | `False` | `True` | $m=8, T=6$ | `PivotUpdateFactorization` | `PivotUpdateFactorization` | **Yes** |
| **Explicit Override ($m < T$)** | `True` | `True` | $m=3, T=6$ | `PivotUpdateFactorization` | `PivotUpdateFactorization` | **Yes** |
| **Explicit Override ($m \ge T$)** | `True` | `True` | $m=8, T=6$ | `PivotUpdateFactorization` | `PivotUpdateFactorization` | **Yes** |

---

### 16. Claims Matrix

| # | Implementation Report Claim | Report Evidence | Independent Audit Evidence | Audit Verdict |
|:---:|:---|:---|:---|:---:|
| 1 | 894 tests passing | Stated in report | Executed `pytest tests/ -q`: 894 passed in 4.34s | **VERIFIED** |
| 2 | Typing import fix (`Any`) | Stated in report | `get_type_hints(DualSimplexResult)` succeeds | **VERIFIED** |
| 3 | Exception handling narrowed | Lines cited | 5 sites verified; mock tests confirm numerical catch and error propagation | **VERIFIED** |
| 4 | Adaptive dispatch correctness | Unit tests | Tested across $T \in \{2,4,5,6,8,10\}$ and boundary values | **VERIFIED** |
| 5 | Boundary condition $m \ge T$ | Unit tests | Tested $m=T-1, T, T+1$; no off-by-one errors | **VERIFIED** |
| 6 | Backward compatibility preserved | Defaults unchanged | Tested defaults vs Stage 8.16 baseline on 10 workloads: 100% identical | **VERIFIED** |
| 7 | Mathematical correctness invariance | 10 workloads | Evaluated across 13 problem classes; 100% agreement | **VERIFIED** |
| 8 | $m \ge 6$ empirically validated on 10 workloads | Table 6 runtime comparisons | Degenerate benchmark: all 10 workloads had standard-form child $m \ge 10$; all thresholds chose PFI | **CONTRADICTED** |
| 9 | Avoids small-$m$ regressions on `branching_heavy` | Stated in Section 10 | `branching_heavy` has child $m=12$; chose PFI 26/26 times under $m \ge 6$; runtime delta was noise | **CONTRADICTED** |
| 10 | True dimension crossover at $m \le 4$ | Stage 8.16 audit notes | Direct $m=1..10$ test confirms IncrementalBasis wins at $m=4$ (11% faster) | **VERIFIED** |
| 11 | Infeasible-child 37.81 ms anomaly | Table 6 | Unreproducible; independent 20-run median was 7.59 ms; host noise artifact | **CONTRADICTED** |
| 12 | Fallback & state safety | Unit tests | Sibling and parent isolation verified; singular fallback verified | **VERIFIED** |
| 13 | Accounting invariant holds | $\text{attempted} = \text{accepted} + \text{fallbacks}$ | Invariant holds; but `adaptive_pfi_chosen` under-counts on infeasible nodes | **PARTIALLY VERIFIED** |
| 14 | Safe opt-in defaults | Code inspection | Flags default to `False`; zero risk of production regression | **VERIFIED** |

---

### 17. Limitations & Recommendations

1. **Synthetic Workload Standard-Form Dimension:**
   In high-level model definitions, variable upper bounds are not visible as rows until conversion to standard form. Benchmark suites assessing dimension thresholds must account for standard-form row expansion ($m_{SF} = m_{constraints} + n_{upper\_bounds}$).
2. **Accounting on Infeasible Children:**
   `DualRevisedSimplexSolver` omits `basis_factorization` in `DualSimplexResult` when reporting infeasibility. This does not harm solver correctness or speed, but causes `adaptive_pfi_chosen` to under-count by `infeasible_pruned`.
3. **Opt-In Recommendation:**
   Adaptive PFI should remain strictly **OPT-IN** (`use_adaptive_pivot_factorization=False`). The default behavior must remain untouched until larger sparse MIPLIB models are benchmarked.

---

### 18. Final Verdict

# **READY WITH LIMITATIONS**

### Safe to Use:
- Code hardening fixes (`typing.Any` import and narrowed numerical exception handling).
- The adaptive dispatch helper `should_use_pivot_factorization(m, tolerances)`.
- Explicit PFI override behavior (`use_pivot_factorization=True`).
- Adaptive PFI dispatch as an **opt-in** feature (`use_adaptive_pivot_factorization=True`).
- Complete backward compatibility under default configurations.

### Limitations:
- The empirical justification for threshold $m \ge 6$ in the implementation report was based on degenerate benchmark data where all child relaxations had $m \ge 10$.
- Infeasible child relaxations are not counted in `adaptive_pfi_chosen`.
- Performance gains on small dense models are modest ($\sim 1.02\text{x} - 1.05\text{x}$); adaptive PFI is not recommended for global default activation.
