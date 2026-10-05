# OPTIFORGE — STAGE 8.36 REPORT
## Documentation & Behavioral-Compatibility Contract Hardening

**Author**: Lead Numerical-Software Engineer  
**Stage**: 8.36 (Contract Hardening)  
**Target Code**: `optiforge/lp/simplex.py`, `optiforge/lp/dual_simplex.py`  
**Test Suites**: `tests/test_stage_8_36_contract_hardening.py` (8 tests), Full Regression (1008 tests)  
**Verdict**: **COMPLETE — DOCUMENTATION AND CONTRACT HARDENED**

---

## 1. Objective

The objective of this stage is to close the two documentation and contract issues identified in the Stage 8.35 independent audit:
1. **Backward-Compatibility Clarification**: Rigorously distinguish 100% API compatibility from the intentional behavioral default activation for PFI-enabled solves.
2. **Primal vs Dual Simplex Clarification**: Formally document that incremental reduced-cost updating is utilized exclusively by `DualRevisedSimplexSolver` and does not alter the pricing or pivot mathematics of the primal revised simplex algorithm.

No numerical mathematics, tolerances, factorization algorithms, or solver heuristics were altered in this stage.

---

## 2. Audit Findings Addressed

| Audit Finding | Prior Status | Resolution in Stage 8.36 |
|---|---|---|
| **1. Backward Compatibility Claim** | Prior report claimed "100% backward compatible" without qualification. | Explicitly separated **API compatibility** (100% backward compatible) from **Behavioral compatibility** (intentional default shift for PFI-enabled solves from OFF to AUTO $\rightarrow$ ON). |
| **2. Primal vs Dual Simplex Scope** | Docstrings did not explicitly state whether incremental reduced costs applied to primal simplex. | Formally documented that incremental reduced-cost updating ($\Delta y = -\theta_D v, \Delta r = \theta_D \alpha$) is exclusive to Dual Revised Simplex. Primal Revised Simplex continues to evaluate $B^T y = c_B$ directly for column pricing. |

---

## 3. Exact Behavioral Compatibility Distinction

The tri-state configuration semantics:
```python
use_incremental_reduced_costs: bool | None = None
```
are governed by the following contract:

1. **API Compatibility: 100% Backward Compatible**
   - The type annotation accepts `bool`, `None`, or default omission.
   - Positional arguments for `SimplexTolerances` are preserved.
   - All serialization, `copy.copy()`, and `copy.deepcopy()` operations function identically.

2. **Default Unconfigured Solves: 100% Identical Behavior**
   - Global OptiForge defaults remain conservative:
     ```python
     use_pivot_factorization = False
     use_adaptive_pivot_factorization = False
     use_incremental_reduced_costs = None
     ```
   - On unconfigured solves, the solver utilizes DenseLU, where AUTO resolves to **OFF**. Solves execute full $B^T y = c_B$ solves on every iteration, identically to prior versions.

3. **Explicit User Settings: 100% Identical Behavior**
   - Callers passing `use_incremental_reduced_costs=False` receive explicit OFF under all conditions.
   - Callers passing `use_incremental_reduced_costs=True` receive explicit ON under all conditions (with safe full-solve fallback if DenseLU is active).

4. **PFI-Enabled Solves with Unspecified Reduced Costs: Intentional Behavioral Enhancement**
   - **Old Behavior (Stage 8.33 and earlier)**: Default was `False` $\rightarrow$ Incremental reduced costs were **OFF** (redundant $B^T y = c_B$ solves performed on every pivot).
   - **New Behavior (Stage 8.34+)**: Default is `None` (AUTO) $\rightarrow$ Incremental reduced costs are automatically **ON** when PFI is active.
   - **Impact**: Delivers a verified **$1.15\times - 1.35\times$ end-to-end speedup** with bit-exact objective equality ($\Delta obj = 0.0$) and bounded residual drift ($\|r\|_\infty < 10^{-14}$).

---

## 4. Effective Tri-State Resolution Table

The precedence hierarchy between instance-level solver configuration, tolerance-level configuration, and actual runtime factorization activation is defined by the following contract:

| Solver Instance Setting | Tolerance Setting | Actual Runtime PFI Active | Effective Reduced-Cost Mode | Behavioral Description |
|:---:|:---:|:---:|:---:|:---|
| `None` | `None` | **No** (DenseLU) | **OFF** | Full $B^T y = c_B$ solve on every pivot (unconfigured default). |
| `None` | `None` | **Yes** (PFI active) | **AUTO $\rightarrow$ ON** | Incremental updates ($\Delta y = -\theta_D v, \Delta r = \theta_D \alpha$). |
| `None` | `False` | Yes or No | **OFF** | Explicit tolerance-level OFF strictly overrides AUTO. |
| `None` | `True` | Yes or No | **ON** | Explicit tolerance-level ON strictly overrides AUTO. |
| `False` | *Any* | Yes or No | **OFF** | Explicit solver-level OFF strictly overrides all other settings. |
| `True` | *Any* | **Yes** (PFI active) | **ON** | Explicit solver-level ON strictly overrides all other settings. |
| `True` | *Any* | **No** (DenseLU) | **Safe Fallback** | `inc_fact is None` forces `need_solve_y = True` each pivot; full solves execute safely. |

> **Critical Runtime Distinction**:  
> AUTO couples to **actual runtime PFI activation** (`m >= threshold` under adaptive PFI, or explicit PFI), rather than merely the user's intent. If adaptive PFI selects DenseLU for a small problem ($m < 6$), AUTO correctly resolves to **OFF**.

---

## 5. Primal vs Dual Simplex Clarification

### Dual Revised Simplex (`DualRevisedSimplexSolver`):
- Dual simplex maintains dual feasibility ($r \ge 0$) while driving out primal infeasibilities ($x_B < 0$).
- At each dual pivot $(p, q)$, the dual step length $\theta_D$ and the dual direction row $v = B^{-T} e_p$ provide the exact rank-1 update relations:
  $$y \leftarrow y - \theta_D v$$
  $$r \leftarrow r + \theta_D \alpha$$
- Updating $y$ and $r$ incrementally eliminates the expensive $B^T y = c_B$ BTRAN solve and the subsequent $A^T y$ dense matrix-vector product.

### Primal Revised Simplex (`RevisedSimplexSolver`):
- Primal simplex maintains primal feasibility ($x_B \ge 0$) while driving out dual infeasibilities ($r_q < 0$).
- Column pricing requires identifying entering variable $q = \arg\min_j r_j$ via dual multipliers $y = B^{-T} c_B$.
- In the primal algorithm, dual multipliers do not admit the simple rank-1 ratio test update formula used by dual simplex.
- Therefore, `RevisedSimplexSolver` continues to perform its normal, robust $B^T y = c_B$ solve for pricing on every iteration.
- Configuring `use_incremental_reduced_costs` on `SimplexTolerances` has **zero effect** on primal simplex execution or numerical behavior.

---

## 6. Contract Verification Test Suite

A dedicated test suite was implemented in [`tests/test_stage_8_36_contract_hardening.py`](file:///c:/Users/Prasanna/SIH2026/OptiForge/tests/test_stage_8_36_contract_hardening.py):

| Test Case | Method | Contract Requirement Verified | Behavioral Assertion | Result |
|:---:|:---|:---|:---|:---:|
| **A** | `test_case_a_default_denselu_rc_disabled` | Default DenseLU resolves AUTO to OFF. | `btran_y_count == iterations + 1` | **PASSED** |
| **B** | `test_case_b_explicit_false_pfi_rc_disabled` | Explicit False remains OFF under active PFI. | `btran_y_count == iterations + 1` | **PASSED** |
| **C** | `test_case_c_explicit_true_pfi_rc_enabled` | Explicit True enables incremental RC under PFI. | `btran_y_count < iterations + 1` | **PASSED** |
| **D** | `test_case_d_auto_pfi_rc_enabled` | AUTO enables incremental RC under active PFI. | `btran_y_count < iterations + 1` | **PASSED** |
| **E** | `test_case_e_auto_denselu_rc_disabled` | AUTO disables incremental RC under DenseLU. | `btran_y_count == iterations + 1` | **PASSED** |
| **F** | `test_case_f_solver_level_precedence` | Solver explicit settings override tolerance settings. | Solver=False $\rightarrow$ full solves; Solver=True $\rightarrow$ incremental. | **PASSED** |
| **G** | `test_case_g_auto_inheritance_precedence` | Tolerance explicit settings override AUTO. | Tol=False $\rightarrow$ full solves; Tol=True $\rightarrow$ incremental. | **PASSED** |
| **H** | `test_case_h_primal_simplex_pricing_unaffected` | Primal simplex pricing is unaffected by RC config. | Identical iterations, basis sequence, and objective across None/False/True. | **PASSED** |

Execution of the dedicated suite:
```text
scratch\scipy_env\Scripts\python.exe -m pytest tests/test_stage_8_36_contract_hardening.py -v
============================== 8 passed in 0.29s ==============================
```

---

## 7. Full Regression Results

Full repository regression was executed across the complete test suite:
```text
scratch\scipy_env\Scripts\python.exe -m pytest tests/ -q
======================== 1008 passed, 16 warnings in 22.10s ========================
```
- Total tests: **1008** (990 baseline + 10 Stage 8.36 triangular prototype tests + 8 Stage 8.36 contract tests)
- Failures: **0**
- Regressions: **0**

---

## 8. Files Modified

1. **[`optiforge/lp/simplex.py`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/lp/simplex.py)**:
   - Updated `SimplexTolerances` docstrings with full contract definitions, tri-state resolution rules, backward compatibility distinctions, and dual-simplex algorithm scope.
2. **[`optiforge/lp/dual_simplex.py`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/lp/dual_simplex.py)**:
   - Updated `DualRevisedSimplexSolver` class docstrings and `should_use_incremental_reduced_costs` docstrings with the exact resolution table, precedence contracts, and dual vs primal simplex explanations.
3. **[`tests/test_stage_8_36_contract_hardening.py`](file:///c:/Users/Prasanna/SIH2026/OptiForge/tests/test_stage_8_36_contract_hardening.py)**:
   - Added 8 dedicated behavioral tests covering Items A through H.

---

## 9. Mathematical Integrity Declaration

**No numerical algorithms, tolerances, or factorization routines were modified in this stage.**
- Simplex ratio testing and pricing rules remain identical.
- PivotUpdateFactorization and Compact WY mathematics remain identical.
- DenseLU factorization remains identical.
- Global defaults (`use_pivot_factorization=False`, `use_adaptive_pivot_factorization=False`, `use_incremental_reduced_costs=None`) remain identical.

---

## 10. Final Verdict

### **COMPLETE — DOCUMENTATION AND CONTRACT HARDENED**

All audit findings from Stage 8.35 have been resolved with formal documentation, explicit contract tables, and 100% passing behavioral regression tests.
