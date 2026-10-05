# OPTIFORGE — STAGE 8.17 REMEDIATION REPORT
## Targeted Remediation: Adaptive-PFI Dispatch Statistics Accounting

**Stage:** 8.17 Remediation  
**Status:** IMPLEMENTED & VALIDATED  
**Baseline:** Stage 8.17 Independent Audit (`READY WITH LIMITATIONS`, 894 tests passing)  
**Tests Passing:** **902 passed**, 0 failed, 0 skipped, 0 xfailed (894 baseline + 8 new remediation tests)  
**Scope:** Observability / Statistics Accounting Only (Zero mathematical or threshold alterations)

---

### 1. Audit Finding

The Stage 8.17 Final Independent Audit (`docs/STAGE_8_17_FINAL_AUDIT.md`) identified an accounting discrepancy in `WarmStartStats` when solving MILP models containing child LP relaxations that prove infeasible:
- In `optiforge/milp/solver.py`, when a child relaxation is proven infeasible, it is accepted and pruned (`accepted += 1`, `infeasible_pruned += 1`).
- However, `adaptive_pfi_chosen` and `adaptive_inc_chosen` were only incremented if `isinstance(res.basis_factorization, ...)` succeeded.
- Because `DualRevisedSimplexSolver` returned `DualSimplexResult(status=SolutionStatus.INFEASIBLE, ..., basis_factorization=None)` without attaching the terminal basis factorization, `res.basis_factorization` was `None`.
- Consequently, on workloads with infeasible leaf nodes:
  $$\text{adaptive\_pfi\_chosen} + \text{adaptive\_inc\_chosen} = \text{accepted} - \text{infeasible\_pruned}$$
  rather than accounting for all accepted warm-start child solves.
- Examples identified during the audit:
  - `integer_allocation`: `accepted = 16`, `pfi_chosen = 13`, `infeasible_pruned = 3` (under-count of 3)
  - `infeasible_children`: `accepted = 12`, `pfi_chosen = 10`, `infeasible_pruned = 2` (under-count of 2)

The audit confirmed that this issue did **not** affect solver correctness, branching decisions, or runtime performance; it affected observability and telemetry accounting only.

---

### 2. Root Cause Analysis

Two interacting factors caused the under-count:

1. **Omission of Terminal Basis Factorization in Dual Simplex Result:**  
   In `optiforge/lp/dual_simplex.py` (line 493), when Dual Simplex detects primal infeasibility (dual unboundedness in the leaving row direction $v = B^{-T} e_p$ with all tableau row elements $\alpha_j \ge -\text{pivot\_tolerance}$), the active basis matrix $B$ is completely non-singular and validly represented by `basis_fact` (`PivotUpdateFactorization` or `IncrementalBasisFactorization`). However, while line 454 passed `basis_factorization=basis_fact` for `SolutionStatus.OPTIMAL`, line 493 omitted `basis_factorization`, causing it to default to `None`.
   Furthermore, any PFI updates (`total_updates`) performed during pivots prior to encountering the infeasible certificate were lost to the warm-start stats recorder because `res.basis_factorization` was `None`.

2. **Accounting Gating in MILP Solver:**  
   In `optiforge/milp/solver.py` (lines 1002–1011), `adaptive_pfi_chosen` and `adaptive_inc_chosen` were strictly conditioned on `isinstance(res.basis_factorization, ...)`. Without a factorization attached to the result, the dispatcher choice was lost on infeasible child solves.

---

### 3. Chosen Design: Option A + Robust Fallback Dispatch Recording

Two designs were evaluated:
- **Option A (Primary):** Retain and return the genuine, mathematically valid terminal basis factorization `basis_fact` in `DualSimplexResult` when returning `SolutionStatus.INFEASIBLE`.
- **Option B (Secondary / Defense-in-Depth):** In `_try_warm_start_relaxation`, record adaptive dispatch choices for all accepted child solves (`res.status in (OPTIMAL, INFEASIBLE)`), consulting `dual_solver.should_use_pivot_factorization(m + 1, tolerances)` if an internal numerical fallback to DenseLU occurred.

#### Mathematical Justification for Option A:
- **Invariance of Invertibility:** In Dual Simplex, every iteration maintains an invertible basis matrix $B$. When the ratio test finds no candidate entering variable, the terminal basis matrix $B = A[:, \text{current\_basis}]$ is non-singular. The vector $v = B^{-T} e_p$ satisfies $A^T v \ge -\text{tol}$ and certifies dual unboundedness / primal infeasibility.
- **Genuine Representation:** The factorization `basis_fact` is the exact representation of $B^{-1}$ (either with accumulated eta matrices in `PivotUpdateFactorization` or row updates in `IncrementalBasisFactorization`). It is not a placeholder or fake object.
- **Safety / Containment:** When `res.status == SolutionStatus.INFEASIBLE`, `MILPSolver._try_warm_start_relaxation` returns `sol, None` (the second element `child_state` is strictly `None`). The node is pruned; no child states are spawned, and the factorization is never stored in the B&B tree or passed to sibling nodes.
- **Telemetry Completeness:** Retaining `basis_fact` allows `MILPSolver` to accurately capture `pfi_updates`, `pfi_refactorizations`, and `pfi_fallbacks` accumulated during iterations leading to infeasibility detection (e.g. `infeasible_children` correctly reports 19 PFI updates instead of 18).

---

### 4. Files Changed

1. `optiforge/lp/dual_simplex.py`:
   - Updated line 493 (`solve_standard_form` infeasibility return) to include `basis_factorization=basis_fact`.
2. `optiforge/milp/solver.py`:
   - Updated lines 1001–1015 (`_try_warm_start_relaxation`) to decouple PFI telemetry recording from adaptive choice recording, ensuring all accepted child solves (`OPTIMAL` or `INFEASIBLE`) increment either `adaptive_pfi_chosen` or `adaptive_inc_chosen`.
3. `tests/test_stage_8_17_remediation.py`:
   - Created 8 focused accounting tests verifying feasible children, infeasible children, fallback invariants, multiple infeasible nodes, default mode, and explicit PFI.

---

### 5. Implementation Details

#### Diff in `optiforge/lp/dual_simplex.py`:
```diff
@@ -493,6 +493,7 @@
                 return DualSimplexResult(
                     status=SolutionStatus.INFEASIBLE,
                     basis_indices=current_basis,
                     nonbasic_indices=current_nonbasic,
                     x=x,
                     y=y,
                     objective_value=obj_val,
                     iterations=iterations,
                     message=f"Dual Simplex detected primal infeasibility at row {p} (all tableau entries non-negative).",
+                    basis_factorization=basis_fact,
                 )
```

#### Diff in `optiforge/milp/solver.py`:
```diff
@@ -1001,11 +1001,17 @@
         self.warm_start_stats.dual_pivots += res.iterations
         if isinstance(res.basis_factorization, PivotUpdateFactorization):
             self.warm_start_stats.pfi_updates += res.basis_factorization.total_updates
             self.warm_start_stats.pfi_refactorizations += res.basis_factorization.total_refactorizations
             self.warm_start_stats.pfi_fallbacks += res.basis_factorization.total_fallbacks
-            if self.simplex_tolerances.use_adaptive_pivot_factorization:
-                self.warm_start_stats.adaptive_pfi_chosen += 1
-        elif isinstance(res.basis_factorization, IncrementalBasisFactorization):
-            if self.simplex_tolerances.use_adaptive_pivot_factorization:
-                self.warm_start_stats.adaptive_inc_chosen += 1
+
+        if res.status in (SolutionStatus.OPTIMAL, SolutionStatus.INFEASIBLE):
+            if self.simplex_tolerances.use_adaptive_pivot_factorization:
+                if isinstance(res.basis_factorization, PivotUpdateFactorization):
+                    self.warm_start_stats.adaptive_pfi_chosen += 1
+                elif isinstance(res.basis_factorization, IncrementalBasisFactorization):
+                    self.warm_start_stats.adaptive_inc_chosen += 1
+                elif dual_solver.should_use_pivot_factorization(m + 1, self.simplex_tolerances):
+                    self.warm_start_stats.adaptive_pfi_chosen += 1
+                else:
+                    self.warm_start_stats.adaptive_inc_chosen += 1
```

---

### 6. Accounting Invariants

The accounting semantics are rigorously defined and validated:

1. **Attempted Invariant:**
   $$\text{attempted} = \text{accepted} + \text{fallbacks}$$
   Every warm-start attempt either succeeds (accepted into the B&B tree as an optimal solution or an infeasibility proof) or fails (fallback to cold solve).

2. **Adaptive Dispatch Invariant (under `use_adaptive_pivot_factorization = True`):**
   $$\text{adaptive\_pfi\_chosen} + \text{adaptive\_inc\_chosen} = \text{accepted}$$
   Every accepted child relaxation is accounted for by exactly one adaptive dispatch choice.

3. **Default / Explicit Mode Invariant (under `use_adaptive_pivot_factorization = False`):**
   $$\text{adaptive\_pfi\_chosen} = 0, \quad \text{adaptive\_inc\_chosen} = 0$$
   Adaptive dispatch counters strictly remain zero when adaptive mode is not enabled.

---

### 7. Before / After Workload Statistics

Evaluated under `use_adaptive_pivot_factorization=True, adaptive_pfi_min_dimension=6`:

| Workload | Metric | Before Remediation | After Remediation | Audit Assessment |
|:---|:---|:---:|:---:|:---|
| `integer_allocation` | `accepted` | 16 | 16 | Unchanged |
| | `infeasible_pruned` | 3 | 3 | Unchanged |
| | `adaptive_pfi_chosen` | **13** (Under-count) | **16** | **Fixed** ($16 = \text{accepted}$) |
| | `adaptive_inc_chosen` | 0 | 0 | Correct |
| | `pfi_updates` | 29 | 29 | Correct |
| | `objective` | 46.0589 | 46.0589 | 100% Invariant |
| | `node_count` | 17 | 17 | 100% Invariant |
| `infeasible_children` | `accepted` | 12 | 12 | Unchanged |
| | `infeasible_pruned` | 2 | 2 | Unchanged |
| | `adaptive_pfi_chosen` | **10** (Under-count) | **12** | **Fixed** ($12 = \text{accepted}$) |
| | `adaptive_inc_chosen` | 0 | 0 | Correct |
| | `pfi_updates` | **18** (Missed 1 pivot) | **19** | **Complete** (Infeasible pivot captured) |
| | `objective` | 34.5317 | 34.5317 | 100% Invariant |
| | `node_count` | 13 | 13 | 100% Invariant |
| `branching_heavy` | `accepted` | 26 | 26 | Unchanged |
| | `adaptive_pfi_chosen` | 26 | 26 | Correct ($26 = \text{accepted}$) |
| | `objective` | 100.8109 | 100.8109 | 100% Invariant |
| `multilevel_bb` | `accepted` | 2 | 2 | Unchanged |
| | `adaptive_pfi_chosen` | 2 | 2 | Correct ($2 = \text{accepted}$) |
| | `objective` | 120.8230 | 120.8230 | 100% Invariant |
| `strong_relaxation` | `accepted` | 2 | 2 | Unchanged |
| | `adaptive_pfi_chosen` | 2 | 2 | Correct ($2 = \text{accepted}$) |
| | `objective` | 34.0220 | 34.0220 | 100% Invariant |
| `weak_relaxation` | `accepted` | 12 | 12 | Unchanged |
| | `adaptive_pfi_chosen` | 12 | 12 | Correct ($12 = \text{accepted}$) |
| | `objective` | 64.5268 | 64.5268 | 100% Invariant |

---

### 8. Regression & Verification Results

1. **Dedicated Remediation Tests:**
   `python -m pytest tests/test_stage_8_17_remediation.py -v`
   - TEST 1 (`test_adaptive_pfi_feasible_child`): **PASSED**
   - TEST 2 (`test_adaptive_inc_feasible_child`): **PASSED**
   - TEST 3 (`test_adaptive_pfi_infeasible_child`): **PASSED**
   - TEST 4 (`test_adaptive_inc_infeasible_child`): **PASSED**
   - TEST 5 (`test_fallback_accounting_invariant`): **PASSED**
   - TEST 6 (`test_multiple_infeasible_children`): **PASSED**
   - TEST 7 (`test_default_mode_counters_remain_zero`): **PASSED**
   - TEST 8 (`test_explicit_pfi_statistics`): **PASSED**
   - **Result:** 8 passed in 0.30s.

2. **Full Regression Suite:**
   `python -m pytest tests/ -q`
   - **Result:** **902 passed**, 12 warnings in 2.51s.
   - 0 failed, 0 skipped, 0 xfailed.

3. **Mathematical Correctness Invariance:**
   - Evaluated across 13 problem classes (MIN, MAX, feasible, infeasible, unbounded, degenerate, Phase-I, non-zero bounds, integer, binary, mixed-integer, multilevel B&B, infeasible child nodes).
   - Solutions matched 100% across DenseLU, IncrementalBasis, Pure PFI, and Adaptive PFI ($|\Delta \text{Obj}| < 10^{-10}$, identical node counts).

---

### 9. Remaining Limitations

1. **Threshold Empirical Evidence Scope:**
   As discovered during the Stage 8.17 independent audit, standard-form variable upper bounds expand the row count of all 10 synthetic benchmark workloads to $m_{SF} \ge 9$ (child relaxations $m_{child} \ge 10$). Consequently, candidate thresholds $T \le 10$ all selected PFI on these specific models. While the threshold $m \ge 6$ is analytically sound and microbenchmarks confirm that `IncrementalBasis` is 11% faster at $m=4$, testing across large-scale sparse MIPLIB models remains necessary before considering any global default promotion.
2. **Opt-In Policy Preserved:**
   Adaptive PFI remains strictly **OPT-IN** (`use_adaptive_pivot_factorization = False` by default).

---

### 10. Conclusion

The accounting under-count on infeasible child LP relaxations has been completely resolved. The dispatch decision is accurately recorded, PFI updates on infeasible nodes are fully captured, mathematical correctness is 100% preserved, and the test suite passes with 902 tests. The repository is ready for independent re-audit.
