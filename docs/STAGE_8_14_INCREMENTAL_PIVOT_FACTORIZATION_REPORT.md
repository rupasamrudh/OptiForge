# OPTIFORGE — STAGE 8.14 IMPLEMENTATION REPORT
## Incremental Basis Factorization for Revised Simplex Pivot Updates

**Date:** September 27, 2026  
**Status:** IMPLEMENTED (Opt-in Prototype Verified; Reference Fallback Retained)  
**Baseline Test Count:** 858 passed  
**Final Test Count:** 875 passed (17 new dedicated tests; zero regressions)  
**Primary Bottleneck Targeted:** DenseLU basis factorization (~56.9% of pivot runtime in Stage 8.13)  

---

### 1. Executive Summary & Objective

In Stage 8.13, profiling identified that full dense LU factorizations (`DenseLU`) accounted for approximately 56.9% of Revised Simplex per-pivot runtime. Because the standard Revised Simplex rebuilds and re-factorizes the basis matrix $B$ from scratch on every iteration, it performs repeated $O(m^3)$ floating-point operations.

**Stage 8.14 Objective:**
Research, prototype, mathematically validate, and integrate an incremental basis factorization mechanism for Revised Simplex pivot updates, reducing the per-pivot factorization cost from $O(m^3)$ to $O(m)$ while preserving `DenseLU` as the trusted numerical reference and fallback.

**Key Achievements:**
1. **Mathematical Update Representation:** Implemented Product Form of the Inverse (PFI) using elementary eta matrices $E_k^{-1} = I + (\eta_k - e_p) e_p^T$.
2. **Prototype Class:** Created [`optiforge/numerics/pivot_basis.py`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/numerics/pivot_basis.py) defining [`PivotUpdateFactorization`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/numerics/pivot_basis.py#L77) with complete primal/transpose solve, column replacement, structured refactorization reasons, and deterministic fallback.
3. **Structured Refactorization Triggers:** Implemented [`RefactorizationReason`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/numerics/pivot_basis.py#L47) supporting `UPDATE_LIMIT`, `RESIDUAL_LIMIT`, `PIVOT_INSTABILITY`, `NUMERICAL_DRIFT`, `EXPLICIT_REQUEST`, and `SINGULAR_BASE`.
4. **Strict Opt-In Feature Flag:** Added `use_pivot_factorization: bool = False` to `SimplexTolerances`. When disabled (default), OptiForge behaves identically to Stage 8.13.
5. **Measured Performance Speedup:** Benchmarked across $m \in \{5, 10, 20, 40, 50, 75, 100\}$ over 10 repetitions per configuration:
   - $m=40$: **2.77x speedup** (50.79 ms $\to$ 18.31 ms)
   - $m=50$: **2.40x speedup** (303.34 ms $\to$ 126.20 ms)
   - $m=75$: **3.33x speedup** (1,126.24 ms $\to$ 338.38 ms)
   - $m=100$: **3.35x speedup** (3,203.25 ms $\to$ 955.34 ms; 1,084 DenseLU factorizations eliminated)
6. **Zero Regression:** All 858 baseline tests pass; 17 new tests pass (875 total).

---

### 2. Stage 8.13 Baseline Review

Stage 8.13 was independently audited and closed with a verdict of **READY WITH LIMITATIONS**.
- Verified baseline: 858/858 tests passing.
- Revised Simplex inner-loop optimizations verified: primal $x_B$ recurrence, Bland early-break pricing, ratio-test tie-breaking, preallocated reduced-cost buffers, and bisect nonbasic maintenance.
- End-to-end LP improvement: 1.06x–1.21x vs Stage 8.12.
- Profiling revealed that after inner-loop overhead was reduced, basis refactorization (`dense_lu_factor`) constituted ~56.9% of pivot runtime.

---

### 3. Existing Infrastructure Inspection

Before designing the Stage 8.14 prototype, existing codebase components were audited:
- [`optiforge/numerics/lu.py`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/numerics/lu.py): Full dense LU factorization $P B = L U$ with partial pivoting.
- [`optiforge/numerics/incremental_basis.py`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/numerics/incremental_basis.py): Stage 8.7 research prototype implementing basic PFI eta updates.
- [`optiforge/numerics/bordered_basis.py`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/numerics/bordered_basis.py): Stage 8.10/8.11 bordered Schur-complement inheritance for child B&B relaxations.
- [`optiforge/lp/simplex.py`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/lp/simplex.py): Two-phase Revised Simplex solver.
- [`optiforge/lp/dual_simplex.py`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/lp/dual_simplex.py): Dual Simplex solver used for B&B warm starts.

**Key Findings:**
1. In Stage 8.7, `IncrementalBasisFactorization` was created as an isolated prototype. It maintained an explicit basis matrix and an eta list, but lacked structured reason codes, memory rollback protection, and full integration with the Stage 8.13 $x_B$ recurrence.
2. In Stage 8.7, `_simplex_loop` had an experimental branch `if tolerances.use_incremental_basis:` that was solving $x_B = B^{-1} b$ on every pivot, defeating the $x_B$ recurrence introduced in Stage 8.13.
3. In Stage 8.14, a clean, production-grade abstraction [`PivotUpdateFactorization`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/numerics/pivot_basis.py#L77) was developed in [`optiforge/numerics/pivot_basis.py`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/numerics/pivot_basis.py), ensuring seamless cooperation with the Stage 8.13 $x_B$ recurrence.

---

### 4. Mathematical Derivation of Pivot Updates

In the Revised Simplex algorithm, each pivot replaces column $p$ of current basis matrix $B$ with entering column $a_q$:
$$B_{\text{new}} = B + (a_q - B e_p) e_p^T$$
where $e_p$ is the $p$-th standard unit basis vector in $\mathbb{R}^m$.

Let $d = B^{-1} a_q$ be the search direction computed during the pricing step. Since $B d = a_q$:
$$B_{\text{new}} = B + B (d - e_p) e_p^T = B [I + (d - e_p) e_p^T] = B E$$
where $E$ is an elementary matrix differing from the identity matrix only in column $p$, with $E_{:, p} = d$.

To invert $E$, define the eta vector $\eta \in \mathbb{R}^m$:
$$\eta_p = \frac{1}{d_p}, \quad \eta_i = -\frac{d_i}{d_p} \quad (\forall i \neq p)$$
Then:
$$E^{-1} = I + (\eta - e_p) e_p^T$$
Verification:
$$E^{-1} E_{:, p} = E^{-1} d = d + (\eta - e_p) d_p = d + \eta d_p - d_p e_p = e_p$$
$$E^{-1} E_{:, j} = E^{-1} e_j = e_j \quad (\forall j \neq p)$$
Thus, $E^{-1} E = I$.

#### Primal Solve (FTRAN):
Given right-hand side $b$, solve $B_k x = b$ where $B_k = B_0 E_1 E_2 \dots E_k$:
$$x = B_k^{-1} b = E_k^{-1} \dots E_1^{-1} (B_0^{-1} b)$$
For any vector $v$, applying $E^{-1} v$:
$$E^{-1} v = v + (\eta - e_p) v_p$$
- Component $p$: $(E^{-1} v)_p = v_p + (\eta_p - 1) v_p = \eta_p v_p = \frac{v_p}{d_p}$
- Component $i \neq p$: $(E^{-1} v)_i = v_i + \eta_i v_p$
Each eta multiplication requires exactly $m$ multiply-add operations ($O(m)$ work).

#### Transpose / Dual Solve (BTRAN):
Given vector $c_B$, solve $B_k^T y = c_B$:
$$y = B_k^{-T} c_B = B_0^{-T} E_1^{-T} \dots E_k^{-T} c_B$$
Applied in reverse chronological order:
$$u_k = c_B, \quad u_{j-1} = E_j^{-T} u_j \quad (j = k, \dots, 1), \quad y = B_0^{-T} u_0$$
For any vector $u$, applying $E^{-T} u$:
$$E^{-T} u = [I + e_p (\eta - e_p)^T] u = u + e_p [(\eta - e_p)^T u] = u + e_p [\eta^T u - u_p]$$
- Component $p$: $(E^{-T} u)_p = u_p + \eta^T u - u_p = \eta^T u$ (inner product of length $m$)
- Component $i \neq p$: $(E^{-T} u)_i = u_i$
Each transpose eta update requires exactly one dot product of length $m$ ($O(m)$ work).

---

### 5. Candidate Update Methods Comparison

Two primary mathematical update representations were evaluated:

| Criterion | Option A: Product Form of the Inverse (PFI) | Option B: Sherman-Morrison Formula |
| :--- | :--- | :--- |
| **Mathematical Formulation** | $B_k^{-1} = E_k^{-1} \dots E_1^{-1} B_0^{-1}$ | $B_{\text{new}}^{-1} = B^{-1} - \frac{(B^{-1} u)(v^T B^{-1})}{1 + v^T B^{-1} u}$ |
| **Storage per Update** | $m$ floats (one vector $\eta$ + index $p$) | $m \times m$ dense matrix (or accumulated $u, v$ pairs) |
| **Per-Pivot Solve Cost** | $O(m)$ per active eta ($k \times m$ FLOPs) | $O(m^2)$ matrix-vector or $O(m^2)$ dense rank-1 update |
| **Base Factorization Compatibility** | Compatible with any base (`DenseLU`, `BorderedBasis`) | Destroys sparsity/structure; requires dense inverse |
| **Numerical Drift** | Base $B_0$ remains pristine; periodic refactor resets error | Rank-one drift accumulates in dense inverse; loss of precision |
| **Selected** | **YES (Chosen)** | **NO (Rejected)** |

**Decision:** Option A (PFI) was chosen because it stores only $O(m)$ data per pivot, performs $O(m)$ work per solve, wraps cleanly around any existing base factorization, and allows trivial numerical resets upon refactorization.

---

### 6. Prototype Architecture: `PivotUpdateFactorization`

Implemented in [`optiforge/numerics/pivot_basis.py`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/numerics/pivot_basis.py):

```python
class PivotUpdateFactorization:
    def __init__(self, B, pivot_tolerance=1e-12, max_updates=30, residual_tolerance=1e-7, base_factorization=None)
    def factorize(cls, A, basis, ...) -> PivotUpdateFactorization
    def solve(self, rhs, check_residual=False) -> np.ndarray
    def solve_transpose(self, rhs, check_residual=False) -> np.ndarray
    def replace_column(self, position, new_column, search_direction=None) -> bool
    def refactorize(self, reason=RefactorizationReason.EXPLICIT_REQUEST) -> None
    def residual(self, rhs, x) -> float
    def transpose_residual(self, rhs, y) -> float
    def needs_refactorization(self) -> bool
    def get_update_count(self) -> int
    def get_total_refactorizations(self) -> int
    def get_total_fallbacks(self) -> int
    def get_refactorization_reason(self) -> RefactorizationReason
```

**Key Architectural Features:**
1. **PFI Chain:** Maintains `self._etas: list[EtaMatrixUpdate]` storing $(p, \eta, d_p)$.
2. **Explicit Basis Matrix:** Holds `self._B` in full precision to allow instant fresh `DenseLU` refactorization at any time.
3. **Rollback Protection:** If an update attempts to insert a singular or non-finite column, `self._B[:, position]` is automatically restored to its previous value, preventing state corruption.
4. **Fast 2D Solves:** Vectorized forward and transpose applications for matrix right-hand sides $(m, k)$.
5. **Zero-Eta Fast Path:** Solves when $\text{len}(\text{etas}) = 0$ directly bypass loop overhead.

---

### 7. Refactorization Policy

Rather than an arbitrary fixed schedule, refactorization is governed by evidence-based numerical triggers:

| Trigger Reason | Condition | Action |
| :--- | :--- | :--- |
| `UPDATE_LIMIT` | `update_count >= max_updates` (default 30) | Full `DenseLU` refactorization of current $B$; resets eta chain. |
| `PIVOT_INSTABILITY` | $\|d_p\| < \text{pivot\_tolerance}$ or $\text{not } \text{isfinite}(d_p)$ | Rejects ill-conditioned eta; forces full DenseLU or rollback. |
| `RESIDUAL_LIMIT` | $\text{rel\_res} > \text{residual\_tolerance}$ | Automatically detects error drift during solve; resets to fresh DenseLU. |
| `NUMERICAL_DRIFT` | Condition degradation $\kappa(B) > 10^8 \kappa(B_0)$ | Recomputes fresh LU to restore conditioning. |
| `EXPLICIT_REQUEST` | Caller explicitly triggers `refactorize()` | E.g. Phase-I $\to$ Phase-II boundary or simplex cancellation. |
| `SINGULAR_BASE` | `dense_lu_factor(B)` fails | Raises `SingularBasisError` cleanly. |

---

### 8. Reference Fallback Mechanism

`DenseLU` remains the authoritative numerical reference.
If any numerical anomaly occurs:
1. When `replace_column` detects $\|d_p\| < \text{pivot\_tolerance}$:
   - Does not construct an ill-conditioned eta matrix.
   - Refactorizes the basis with `dense_lu_factor(B)`.
   - If $B$ is genuinely singular, rolls back column replacement and raises `SingularBasisError`.
2. When `solve` or `solve_transpose` with `check_residual=True` observes a relative residual $> 10^{-7}$:
   - Increments `_total_fallbacks`.
   - Discards the accumulated eta chain.
   - Computes fresh `DenseLU` on $B$.
   - Solves with the fresh LU and returns the verified solution.
3. Determinism: All fallbacks are logged with precise reason codes in `RefactorizationReason`.

---

### 9. Revised Simplex Integration (Opt-In)

Integration in [`optiforge/lp/simplex.py`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/lp/simplex.py):
- Feature flag added to `SimplexTolerances`:
  ```python
  use_pivot_factorization: bool = False  # STRICTLY OPT-IN
  ```
- Cooperation with Stage 8.13 $x_B$ Recurrence:
  When `use_pivot_factorization=True`, $x_B$ continues to be updated via linear recurrence $x_B \leftarrow x_B - \theta^* d$ on every pivot. Linear solves $B^{-1} b$ are only executed when `need_solve_xB` is set (initial iteration, periodic 25-pivot reset, refactorization, or final optimality check).
- Search direction $d = B^{-1} a_{\text{enter}}$ and simplex multipliers $y = B^{-T} c_B$ use `basis_fact.solve` and `basis_fact.solve_transpose`.
- Column update:
  ```python
  updated = inc_fact.replace_column(i_leave, a_enter, search_direction=d)
  if not updated:
      need_solve_xB = True
  ```

---

### 10. Dual Simplex Status

**Status: LEFT ON EXISTING FACTORIZATION PATH.**

**Rationale:**
Dual Simplex in OptiForge (used primarily for branch-and-bound warm starts) currently interacts with `BorderedBasisFactorization` (Stage 8.10/8.11). In branch-and-bound child solves, the basis modification is typically a row addition (bound tightening) rather than a column replacement, which is handled via Schur-complement bordering. Integrating PFI column updates on top of recursive bordered bases introduces non-trivial interactions between Schur complements and eta chains. Per Section 13 instructions, Dual Simplex remains on its existing verified path.

---

### 11. Phase-I / Phase-II Interaction

- Fully compatible with Stage 8.12 Phase-I basis inheritance.
- If Phase-I terminates with no basic artificial variables, `phase1_res.basis_factorization` is passed as `initial_p2_fact` to Phase-II.
- `PivotUpdateFactorization` accepts `base_factorization=initial_basis_factorization`, wrapping the inherited factorization without re-factorizing $B$.
- Verified in `test_16_phase1_compatibility`.

---

### 12. MILP / Branch-and-Bound Interaction

- MILP branching, node selection, incumbent tracking, and pruning are completely untouched.
- `use_pivot_factorization` defaults to `False`. When MILP root relaxation solves with `use_pivot_factorization=True`, optimal solutions and bounds match reference DenseLU identically.

---

### 13. Comprehensive Performance Benchmark Results

Benchmarked across 7 problem dimensions ($m \in \{5, 10, 20, 40, 50, 75, 100\}$) on 10 independent repetitions per configuration using standard dense synthetic linear programs ($m \times 2m$).

#### Four Configurations Compared:
- **Config A:** DenseLU Reference (re-factorize $B$ from scratch on every pivot).
- **Config B:** Stage 8.13 Baseline (optimized inner loop with DenseLU reference).
- **Config C:** Stage 8.14 Incremental Factorization (`use_pivot_factorization=True`, `max_updates=30`).
- **Config D:** Stage 8.14 with Fallback Validation (`use_pivot_factorization=True`, residual checking active).

#### Benchmark Summary Table (Mean Runtime over 10 Repetitions):

| Dimension $m$ | Pivots | Config A (Ref DenseLU) | Config B (Stage 8.13) | Config C (Stage 8.14 PFI) | Config D (Stage 8.14 Fallback) | Speedup (C vs A) | Speedup (D vs A) | DenseLU Calls Eliminated |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5** | 4 | 1.12 ms | 0.71 ms | **0.35 ms** | 0.34 ms | **3.22x** | 3.29x | 4 / 5 (80%) |
| **10** | 16 | 3.03 ms | 3.08 ms | **2.01 ms** | 1.88 ms | **1.51x** | 1.61x | 16 / 17 (94%) |
| **20** | 31 | 11.32 ms | 11.74 ms | **6.54 ms** | 6.65 ms | **1.73x** | 1.70x | 30 / 32 (94%) |
| **40** | 59 | 50.79 ms | 44.82 ms | **18.31 ms** | 21.41 ms | **2.77x** | 2.37x | 58 / 60 (97%) |
| **50** | 295 | 303.34 ms | 353.49 ms | **126.20 ms** | 131.27 ms | **2.40x** | 2.31x | 286 / 296 (97%) |
| **75** | 578 | 1,126.24 ms | 1,089.20 ms | **338.38 ms** | 357.48 ms | **3.33x** | 3.15x | 559 / 579 (97%) |
| **100** | 1,121 | 3,203.25 ms | 3,701.71 ms | **955.34 ms** | 982.26 ms | **3.35x** | 3.26x | 1,084 / 1,122 (97%) |

#### Detailed Statistics (Mean, Median, Min, Max, StDev):

- **$m=40$ (59 pivots):**
  - Config A: Mean 50.79 ms | Med 46.37 ms | Min 41.30 ms | Max 69.20 ms | Std 9.61 ms
  - Config C: Mean 18.31 ms | Med 17.29 ms | Min 16.14 ms | Max 25.53 ms | Std 2.68 ms (**2.77x speedup**)
  - Config D: Mean 21.41 ms | Med 19.47 ms | Min 17.55 ms | Max 36.40 ms | Std 5.49 ms (**2.37x speedup**)
- **$m=50$ (295 pivots):**
  - Config A: Mean 303.34 ms | Med 302.87 ms | Min 282.13 ms | Max 335.70 ms | Std 14.58 ms
  - Config C: Mean 126.20 ms | Med 113.38 ms | Min 108.07 ms | Max 208.05 ms | Std 29.68 ms (**2.40x speedup**)
  - Config D: Mean 131.27 ms | Med 115.28 ms | Min 107.34 ms | Max 227.09 ms | Std 36.57 ms (**2.31x speedup**)
- **$m=75$ (578 pivots):**
  - Config A: Mean 1,126.24 ms | Med 1,094.15 ms | Min 1,016.65 ms | Max 1,275.07 ms | Std 96.54 ms
  - Config C: Mean 338.38 ms | Med 338.53 ms | Min 301.26 ms | Max 381.55 ms | Std 23.32 ms (**3.33x speedup**)
  - Config D: Mean 357.48 ms | Med 350.35 ms | Min 306.84 ms | Max 432.67 ms | Std 35.30 ms (**3.15x speedup**)
- **$m=100$ (1,121 pivots):**
  - Config A: Mean 3,203.25 ms | Med 3,109.06 ms | Min 2,641.69 ms | Max 3,603.62 ms | Std 304.43 ms
  - Config C: Mean 955.34 ms | Med 923.34 ms | Min 799.78 ms | Max 1,298.39 ms | Std 146.03 ms (**3.35x speedup**)
  - Config D: Mean 982.26 ms | Med 965.14 ms | Min 777.87 ms | Max 1,241.48 ms | Std 153.39 ms (**3.26x speedup**)

#### Solution Quality & Numerical Verification:
Across all 7 dimensions and all 4 configurations:
- **Optimal Objective Values:** Identical to machine precision (e.g. $m=100$: 16,378.1328 across all runs).
- **Pivot Sequences:** Identical iteration counts across all configurations.
- **Primal Residuals:** $\|Ax - b\|_\infty < 10^{-12}$ for all solves.
- **Fallbacks:** 0 unexpected fallbacks required on standard workloads.

---

### 14. Dedicated Test Suite Results

Created [`tests/test_stage_8_14_pivot_factorization.py`](file:///c:/Users/Prasanna/SIH2026/OptiForge/tests/test_stage_8_14_pivot_factorization.py) covering all 17 requirements:
1. `test_01_one_update_primal_solve`: Verified (relative error $< 10^{-11}$).
2. `test_02_one_update_transpose_solve`: Verified (relative error $< 10^{-11}$).
3. `test_03_repeated_updates`: Verified across 15 consecutive updates against fresh LU.
4. `test_04_5_updates`: Verified.
5. `test_05_10_updates`: Verified.
6. `test_06_25_updates`: Verified.
7. `test_07_50_updates`: Verified across periodic refactorization boundary.
8. `test_08_refactorization`: Explicit refactorization and reason tracking verified.
9. `test_09_residual_trigger`: Fallback triggered on residual degradation verified.
10. `test_10_singular_fallback`: Singular column rejected and state safely rolled back.
11. `test_11_nan_inf_fallback`: Non-finite column rejected with `SingularBasisError`.
12. `test_12_near_singular_pivot`: Pivot element $< 10^{-8}$ rejected with `PIVOT_INSTABILITY`.
13. `test_13_memory_isolation`: External array mutation does not corrupt internal state.
14. `test_14_2d_rhs`: Vectorized 2D primal and transpose solves verified.
15. `test_15_primal_simplex_equivalence`: Revised Simplex equivalence verified.
16. `test_16_phase1_compatibility`: Phase-I $\to$ Phase-II factorization inheritance verified.
17. `test_17_feature_flag_isolation`: Default `use_pivot_factorization=False` verified.

**Pytest Execution:**
- Stage 8.14 suite: **17 passed in 0.33s**.
- Entire repository suite: **875 passed, 12 warnings in 3.47s**.

---

### 15. Claims / Evidence Matrix

| Claim | Status | Evidence |
| :--- | :---: | :--- |
| PFI update representation mathematically valid | **VERIFIED** | Derivation in Section 4; tests 1–7 pass with residual $< 10^{-11}$. |
| Transpose solve correct for dual simplex multipliers | **VERIFIED** | Tests 2, 3, 4, 5, 6, 7 pass; exact match with `lu_solve_transpose`. |
| Rollback and fallback protect against corrupted state | **VERIFIED** | Tests 9, 10, 11, 12 pass; singular columns roll back without side effects. |
| Feature flag strictly opt-in (`False` by default) | **VERIFIED** | `use_pivot_factorization = False` in `SimplexTolerances`; test 17 passes. |
| Zero regression on existing test suite | **VERIFIED** | 858 baseline tests $\to$ 875 final tests passing (100%). |
| Eliminates repeated $O(m^3)$ DenseLU refactorizations | **VERIFIED** | For $m=100$, DenseLU calls reduced from 1,122 down to 38 (-97%). |
| End-to-end performance improvement on medium/large bases | **VERIFIED** | $m=40$: 2.77x; $m=50$: 2.40x; $m=75$: 3.33x; $m=100$: 3.35x speedup. |
| Dual Simplex left on existing path | **VERIFIED** | Documented in Section 10; zero change to Dual Simplex. |

---

### 16. Limitations & Scope Restrictions

1. **Dual Simplex Integration Deferred:** Dual Simplex remains on the existing path to prevent interference with Stage 8.10/8.11 `BorderedBasisFactorization`.
2. **Dense Eta Storage:** The current implementation uses dense numpy arrays for eta vectors ($O(m)$ storage per pivot). On extremely large sparse LPs ($m > 1,000$), sparse eta vectors would be required.
3. **Periodic Refactorization Threshold:** A conservative limit of `max_basis_updates = 30` is maintained. While 50 updates pass unit tests, 30 maintains an optimal balance between solve speed and numerical precision.
4. **Independent Audit Pending:** Stage 8.14 implementation is complete; independent audit must be conducted separately before declaring production readiness.
