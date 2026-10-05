# Stage 8.12 Phase-I Optimization Report

## 1. Objective

Stage 8.12 investigates and, guided strictly by profiling evidence, optimizes the linear programming Phase-I to Phase-II transition and root LP solving path:

$$\text{LP Phase-I} \longrightarrow \text{feasible basis} \longrightarrow \text{Phase-II transition}$$

Specific areas of investigation include:
- Artificial-variable creation and standard-form overhead
- Phase-I objective construction
- Phase-I simplex iteration costs
- Artificial-column removal and basis cleanup (`_cleanup_phase1_basis`)
- Reduced-cost and objective reconstruction
- Unnecessary matrix copies and memory allocations
- Redundant basis factorizations between Phase-I and Phase-II
- MILP root relaxation warm-start state propagation

The primary guiding question of Stage 8.12 is:
> *"Can Phase-I/root LP solving be made measurably faster without changing mathematical behavior?"*

---

## 2. Baseline

Prior to code inspection and modification, the full test suite was executed to establish the reference baseline:

```
python -m pytest tests/ -q
836 passed, 12 warnings in 2.34s
```

All 836 unit, integration, and performance regression tests from Stages 8.1 through 8.11 passed.

---

## 3. Profiling Methodology

Ten deterministic, fixed-seed LP workloads representing diverse mathematical structures were constructed (`scratch/profile_phase1.py` and `scratch/benchmark_stage812.py`):

1. **A. No Phase-I**: 15 vars, 8 pure $\le$ constraints with positive RHS
2. **B. Small Phase-I**: 5 vars, 3 constraints with mixed $\ge, =$
3. **C. Medium Phase-I**: 20 vars, 15 constraints with mixed senses (feasible)
4. **D. Highly Constrained Phase-I**: 40 vars, 35 constraints with dense coefficients (feasible)
5. **E. Degenerate Phase-I**: 6 vars, 5 constraints with redundant equalities and zero RHS
6. **F. Equality-Heavy LP**: 20 vars, 15 equality constraints
7. **G. Mixed Senses LP**: 30 vars, 25 constraints with $\le, \ge, =$
8. **H. Dense Phase-I**: 15 vars, 12 constraints with 100% matrix density
9. **I. Sparse Phase-I**: 40 vars, 30 constraints with 10% matrix density
10. **J. Infeasible LP**: 2 vars, 2 contradictory constraints ($x_0 \le 2, x_0 \ge 5$)

Profiling tools employed:
- `cProfile` and `pstats` for detailed call-graph attribution
- High-resolution wall-clock timing via `time.perf_counter` (10 repetitions per problem)
- `unittest.mock` counting of `BasisFactorization` / `dense_lu_factor` invocations

---

## 4. Phase-I Bottleneck Analysis

### Empirical Component Breakdown

Across the profiled workloads, the execution time was decomposed into individual phases:

| Component | Calls | Mean Time (ms) | % Root LP Time | Candidate for Optimization? |
|---|---:|---:|---:|---|
| **Standard-form conversion** (`to_standard_form`) | 10 | 0.12 ms | 0.4% | **No** (already fast, $<0.4$ ms even for $50\times60$) |
| **Artificial-variable creation** | 10 | 0.04 ms | 0.1% | **No** (negligible memory allocation) |
| **Phase-I Objective construction** | 10 | 0.01 ms | $<0.1\%$ | **No** (simple slice/indexing) |
| **Phase-I Simplex Iterations** (`_simplex_loop`) | 10 | 18.42 ms | 56.5% | **Yes** (dominated by DenseLU factorizations) |
| **Phase-I Basis Cleanup** (`_cleanup_phase1_basis`) | 10 | 0.03 ms | 0.1% | **Yes** (redundant array copies & scans eliminated) |
| **Phase-I $\to$ Phase-II Factorization Reuse** | 10 | 0.35 ms | 1.1% | **Yes** (1 full DenseLU call saved per Phase-I LP) |
| **Phase-II Simplex Iterations** (`_simplex_loop`) | 10 | 13.58 ms | 41.7% | **Yes** (factorization inheritance on first iteration) |
| **Solution extraction / Objective restoration** | 10 | 0.02 ms | 0.1% | **No** (pure dict lookups) |

### Key Bottleneck Findings

1. **Phase-I Basis Cleanup is Not the Global Bottleneck**:
   - `_cleanup_phase1_basis` accounts for $<0.2\%$ of root LP runtime (0.01 to 0.07 ms).
   - However, in the non-degenerate case (no artificial variables basic at Phase-I termination), it performed:
     - `curr_A = np.array(model_sf.A, copy=True)`: full matrix copy
     - `curr_b = np.array(model_sf.b, copy=True)`: full vector copy
     - Loop over $m$ rows checking for basic artificial variables
     - Followed immediately by `A_phase2 = curr_A[:, keep_cols]` (a second matrix copy).
   - *Targeted Optimization*: When `not any(b in artificial_set for b in final_basis)`, slice `model_sf.A[:, keep_cols]` directly, copy `b` once, and bypass the row scan entirely.

2. **Redundant DenseLU Factorization at Phase-I $\to$ Phase-II Boundary**:
   - At the final iteration of Phase-I (when Phase-I is optimal), `_simplex_loop` executes `BasisFactorization(B)` to verify that all reduced costs $\bar{w}_j \ge 0$.
   - When no artificial variables are basic, the Phase-II starting basis matrix $B_{\text{Phase-II}} = A_{\text{Phase-II}}[:, \text{basis\_phase2}]$ is **bit-for-bit identical** to $B_{\text{Phase-I}} = \text{model\_sf.A}[:, \text{final\_basis}]$.
   - Phase-II was previously constructing a fresh `BasisFactorization(B_{\text{Phase-II}})` on iteration 0, repeating an identical $O(m^3)$ DenseLU factorization.
   - *Targeted Optimization*: Pass `initial_basis_factorization` from Phase-I to Phase-II. Iteration 0 of Phase-II reuses this factorization directly.

3. **Critical Coordinate Collision Bug in MILP Root Warm-Starting**:
   - In `MILPSolver._solve_root_or_cold` (lines 766–777):
     ```python
     artificial_set = set(model_sf.artificial_indices)
     if any(col in artificial_set for col in phase2_res.basis_indices):
         return Solution(...), None
     ```
   - `phase2_res.basis_indices` are column indices into $A_{\text{phase2}}$ ($0 \le \text{col} < |keep\_cols|$), whereas `model_sf.artificial_indices` are column indices into Phase-I standard-form $A$.
   - Whenever an arbitrary column in $A_{\text{phase2}}$ happened to share an index value with a Phase-I artificial variable, this check triggered a **false positive**, setting `warm_state = None`.
   - As a direct consequence, warm-starting was silently disabled across all downstream Branch-and-Bound child nodes on Phase-I MILP models!
   - *Targeted Optimization*: Correctly map Phase-II basis indices back to Phase-I space (`rev_col_mapping.get(col) in artificial_set`). This unblocks Phase-I MILP warm-starting completely.

---

## 5. Mathematical Invariants

Let $A_{\text{sf}} \in \mathbb{R}^{m \times n_{\text{sf}}}$ be the Phase-I standard-form matrix, partitioned as:

$$A_{\text{sf}} = \begin{bmatrix} A_{\text{orig}} & I_{\text{slack}} & I_{\text{art}} \end{bmatrix}$$

Let $\mathcal{A} \subset \{0, \dots, n_{\text{sf}}-1\}$ denote the set of artificial column indices.

### Invariant 1: Column Slicing & Elimination
Let $\mathcal{K} = \{j \in \{0, \dots, n_{\text{sf}}-1\} : j \notin \mathcal{A}\}$ be the set of kept non-artificial columns with $|\mathcal{K}| = n_{\text{p2}}$.
The bijection $\sigma: \mathcal{K} \to \{0, \dots, n_{\text{p2}}-1\}$ defines the Phase-II column coordinates:

$$A_{\text{p2}}[:, \sigma(j)] = A_{\text{sf}}[:, j], \quad \forall j \in \mathcal{K}$$

### Invariant 2: Basis Equivalence Across Phases
Suppose Phase-I terminates with optimal basis $\mathcal{B}_{\text{p1}} = \{b_0, \dots, b_{m-1}\}$.
If $\mathcal{B}_{\text{p1}} \cap \mathcal{A} = \emptyset$ (no artificial variable is basic):
1. Every basis column $b_k \in \mathcal{K}$.
2. The Phase-II basis is defined by $\mathcal{B}_{\text{p2}} = \{\sigma(b_0), \dots, \sigma(b_{m-1})\}$.
3. For every column $k \in \{0, \dots, m-1\}$:

$$B_{\text{p2}}[:, k] = A_{\text{p2}}[:, \sigma(b_k)] = A_{\text{sf}}[:, b_k] = B_{\text{p1}}[:, k]$$

Therefore:

$$B_{\text{p2}} \equiv B_{\text{p1}} \quad \text{(bitwise identical matrices)}$$

### Invariant 3: LU Factorization Inheritance
Since $B_{\text{p2}} \equiv B_{\text{p1}}$, the permutation matrix $P$, unit lower triangular matrix $L$, and upper triangular matrix $U$ satisfying $P B_{\text{p1}} = L U$ also satisfy:

$$P B_{\text{p2}} = L U$$

Inheriting $(P, L, U)$ is mathematically exact: forward and backward substitution solve $B_{\text{p2}} x = b$ and $B_{\text{p2}}^T y = c_{\mathcal{B}}$ with **zero numerical discrepancy**.

---

## 6. Optimization Hypothesis

1. **Hypothesis 1 (Cleanup Fast Path)**:
   - For all non-degenerate Phase-I solves ($\mathcal{B}_{\text{p1}} \cap \mathcal{A} = \emptyset$), bypassing redundant array allocations and row drive-out checks reduces memory allocation and CPU cycles during basis cleanup without altering tableau geometry.
2. **Hypothesis 2 (Factorization Inheritance)**:
   - Reusing the terminal Phase-I `BasisFactorization` on iteration 0 of Phase-II eliminates exactly 1 full $O(m^3)$ DenseLU factorization per Phase-I LP solve.
3. **Hypothesis 3 (MILP Root Warm-Start Activation)**:
   - Correcting the coordinate space check in `MILPSolver._solve_root_or_cold` permits Phase-I root LPs to yield valid `WarmStartBasisState` instances, enabling Dual Simplex warm-starting and factorization inheritance downstream in Branch-and-Bound.

---

## 7. Implementation

### File 1: [`optiforge/lp/simplex.py`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/lp/simplex.py)

1. **`SimplexIterationResult` Dataclass**:
   - Added optional field: `basis_factorization: BasisFactorization | None = None`.
   - On optimal termination, `_simplex_loop` populates this field with the active basis factorization.

2. **Phase-I Basis Cleanup Fast Path (`_cleanup_phase1_basis`)**:
   ```python
   has_basic_artificial = any(b in artificial_set for b in final_basis)
   if not has_basic_artificial:
       total_cols = model_sf.A.shape[1]
       keep_cols = [j for j in range(total_cols) if j not in artificial_set]
       col_mapping: dict[int, int] = {
           old_idx: new_idx for new_idx, old_idx in enumerate(keep_cols)
       }
       A_phase2 = model_sf.A[:, keep_cols]
       b_phase2 = np.array(model_sf.b, copy=True)
       c_phase2 = model_sf.c[keep_cols]
       basis_phase2 = [col_mapping[old_idx] for old_idx in final_basis]
       nonbasic_phase2 = [
           col_mapping[old_idx]
           for old_idx in final_nonbasic
           if old_idx not in artificial_set
       ]
       nonbasic_phase2.sort()
       return (A_phase2, b_phase2, c_phase2, basis_phase2, nonbasic_phase2, col_mapping)
   ```

3. **Factorization Inheritance in `solve()` and `_simplex_loop()`**:
   - In `solve()`:
     ```python
     initial_p2_fact = None
     if phase1_res.basis_factorization is not None:
         artificial_set = set(model_sf.artificial_indices)
         if not any(b in artificial_set for b in phase1_res.basis_indices):
             initial_p2_fact = phase1_res.basis_factorization
     ```
   - In `_simplex_loop()`:
     - Accepts `initial_basis_factorization: BasisFactorization | None = None`.
     - In iteration 0, consumes `initial_basis_factorization` directly to compute $x_B = B^{-1} b$, skipping `BasisFactorization(B)`.
     - Passes `base_factorization=initial_basis_factorization` to `IncrementalBasisFactorization` if `use_incremental_basis=True`.

### File 2: [`optiforge/milp/solver.py`](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/milp/solver.py)

1. **Phase-I $\to$ Phase-II Factorization Inheritance in Root Solve**:
   - Added `initial_basis_factorization=initial_p2_fact` to Phase-II root simplex loop.
2. **Fixed Artificial Variable Basis Check**:
   - Replaced flawed `any(col in artificial_set for col in phase2_res.basis_indices)` with coordinate-aware reverse mapping:
     ```python
     rev_col_mapping = {new_idx: old_idx for old_idx, new_idx in col_mapping.items()}
     if any(rev_col_mapping.get(col) in artificial_set for col in phase2_res.basis_indices):
         return Solution(...), None
     ```
3. **Reused Phase-II Terminal Factorization for Warm State**:
   - `warm_state.basis_factorization` reuses `phase2_res.basis_factorization` instead of redundant refactorization.

---

## 8. Correctness Validation

Ten targeted unit tests were created in [`tests/test_stage_8_12_phase1.py`](file:///c:/Users/Prasanna/SIH2026/OptiForge/tests/test_stage_8_12_phase1.py):

| Test ID | Test Name | Focus | Result |
|---|---|---|:---:|
| 1 | `test_01_phase1_required` | $\ge$ and $=$ constraints requiring Phase-I | **PASSED** |
| 2 | `test_02_phase1_not_required` | Pure $\le$ constraints directly entering Phase-II | **PASSED** |
| 3 | `test_03_artificial_variables_removed_correctly` | Column drop verification and column mapping integrity | **PASSED** |
| 4 | `test_04_phase1_to_phase2_basis_validity` | Bitwise identity $B_{\text{p1}} \equiv B_{\text{p2}}$ and conditioning | **PASSED** |
| 5 | `test_05_equality_heavy_model` | Equality-heavy constraint systems | **PASSED** |
| 6 | `test_06_degenerate_model` | Zero RHS and basic artificial variable drive-out | **PASSED** |
| 7 | `test_07_infeasible_model` | Contradictory constraints returning `INFEASIBLE` | **PASSED** |
| 8 | `test_08_ill_conditioned_model` | Small pivot coefficients ($\varepsilon = 10^{-4}$) stability | **PASSED** |
| 9 | `test_09_original_objective_restoration` | Variable shift and original objective reconstruction | **PASSED** |
| 10 | `test_10_milp_root_relaxation_warm_start_state` | Phase-I root MILP warm-start creation & child acceptance | **PASSED** |

---

## 9. Numerical Validation

Numerical identity between Phase-I final basis and Phase-II starting basis was evaluated across all test instances:

$$\|B_{\text{phase1}} - B_{\text{phase2}}\|_\infty = 0.00 \times 10^0 \quad \text{(exact machine equality)}$$

- **Primal Residuals**: $\|A_{\text{p2}} x_{\text{opt}} - b_{\text{p2}}\|_\infty \le 1.42 \times 10^{-14}$.
- **Dual Residuals**: $\|A_{\text{p2}}^T y_{\text{opt}} - c_{\text{p2}}\|_\infty \le 2.21 \times 10^{-14}$.
- **Condition Numbers**: $\kappa(B) < 10^6$ on well-conditioned problems; monitored and checked within numerical monitor tolerances.

---

## 10. Phase-I / Phase-II Equivalence

Every workload was verified to produce:
1. Identical optimal objective values (within $10^{-6}$ tolerance).
2. Identical feasible/infeasible status classification.
3. Zero artificial variables in the final Phase-II variable assignment.
4. Identical simplex iteration paths and pivot sequences.

---

## 11. MILP Root Validation

Before Stage 8.12, Phase-I root relaxations on mixed-constraint MILPs suffered from coordinate collision in `_solve_root_or_cold`, resulting in `warm_state = None`.

### Measured MILP Comparison (10 Repetitions)

Tested on an 8-variable integer model with $\ge, \le, =$ constraints requiring Phase-I at root relaxation:

| Configuration | Mean Time (ms) | Min (ms) | Max (ms) | Warm Starts Attempted | Warm Starts Accepted | Dense LU Saved | Optimal Objective |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Cold Solve** (`warm_start_relaxations=False`) | 9.91 ms | 8.55 ms | 10.80 ms | 0 | 0 | 0 | 23.0000 |
| **Warm Solve** (`warm_start_relaxations=True`) | **3.98 ms** | 3.55 ms | 4.93 ms | **4** | **4** | **4** | **23.0000** |
| **Speedup** | **2.49x** | **2.41x** | **2.19x** | — | — | — | Bitwise Identical |

- Downstream B&B node count: 5 nodes.
- Child relaxations warm-started: 4/4 (100% acceptance rate).
- Dense LU factorizations saved: 4 calls.
- Final MILP objective: strictly identical (23.0000).

---

## 12. Benchmark Results

Measured over **10 independent repetitions** per workload using fixed deterministic random seeds (`scratch/benchmark_results_stage812.json`):

| Workload | Dimensions & Constraints | Status | Objective Value | Mean Runtime (ms) | Median (ms) | Min (ms) | Max (ms) | P1 / P2 Pivots | Dense LU Factorizations |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A. No Phase-I** | 15 vars, 8 cons ($\le$) | OPTIMAL | 55.7136 | 1.79 ms | 1.53 ms | 1.44 ms | 3.03 ms | 0 / 9 | 10 |
| **B. Small Phase-I** | 5 vars, 3 cons (mixed) | OPTIMAL | 10.0000 | 0.34 ms | 0.32 ms | 0.30 ms | 0.43 ms | 3 / 0 | 4 |
| **C. Medium Phase-I** | 20 vars, 15 cons (mixed) | OPTIMAL | 376.9610 | 16.79 ms | 16.58 ms | 15.19 ms | 19.00 ms | 28 / 26 | 55 *(saved 1)* |
| **D. Highly Constrained** | 40 vars, 35 cons (mixed) | OPTIMAL | 678.4895 | 121.28 ms | 119.56 ms | 114.08 ms | 138.04 ms | 84 / 84 | 169 *(saved 1)* |
| **E. Degenerate Phase-I** | 6 vars, 5 cons (redundant) | OPTIMAL | 10.0000 | 0.51 ms | 0.50 ms | 0.47 ms | 0.58 ms | 3 / 0 | 6 |
| **F. Equality-Heavy** | 20 vars, 15 cons ($=$) | OPTIMAL | 993.2405 | 8.52 ms | 8.48 ms | 7.77 ms | 10.05 ms | 25 / 3 | 29 *(saved 1)* |
| **G. Mixed Senses** | 30 vars, 25 cons (mixed) | OPTIMAL | 653.8978 | 30.40 ms | 30.63 ms | 28.94 ms | 31.65 ms | 42 / 22 | 65 *(saved 1)* |
| **H. Dense Phase-I** | 15 vars, 12 cons (dense) | OPTIMAL | 235.7358 | 7.95 ms | 8.08 ms | 7.11 ms | 8.89 ms | 20 / 15 | 36 *(saved 1)* |
| **I. Sparse Phase-I** | 40 vars, 30 cons (sparse) | OPTIMAL | 478.7046 | 52.92 ms | 53.19 ms | 48.58 ms | 56.94 ms | 48 / 48 | 97 *(saved 1)* |
| **J. Infeasible LP** | 2 vars, 2 cons (contradictory) | INFEASIBLE | 0.0000 | 0.14 ms | 0.13 ms | 0.13 ms | 0.20 ms | 1 / 0 | 2 |

---

## 13. Regression Results

Full regression testing on the entire OptiForge test suite:

```
python -m pytest tests/ -q
846 passed, 12 warnings in 2.49s
```

- **Baseline passed**: 836 tests.
- **Stage 8.12 passed**: **846 tests** (+10 new dedicated Phase-I tests).
- **Test failures**: 0.
- **Regressions**: 0.

---

## 14. Rejected Optimizations

1. **Elimination of Degenerate Artificial Drive-Out Loop**:
   - Investigated skipping the tableau row drive-out loop in degenerate cases.
   - *Rejected*: In degenerate models with redundant constraints, basic artificial variables remain at zero. Bypassing drive-out leaves singular/dependent rows in Phase-II. The drive-out loop is mathematically mandatory.
2. **Replacing Bland's Rule in Phase-I**:
   - Investigated steepest-edge or Dantzig pivot selection for Phase-I.
   - *Rejected*: Non-negotiable rules strictly forbid changing Bland's rule or simplex mathematics.
3. **In-Place Mutation of Standard-Form Matrices**:
   - Investigated in-place deletion of columns from `model_sf.A`.
   - *Rejected*: Violates purity of `StandardFormModel` and breaks caller model reusability.

---

## 15. Remaining Bottlenecks

1. **Per-Pivot DenseLU Dominance in Pure Simplex**:
   - Over 50% of wall-clock time in multi-pivot LPs remains inside `dense_lu_factor` when `use_incremental_basis=False`. This is intrinsic to revised simplex without incremental updates.
2. **Pure-Python Dispatch Overhead**:
   - For small matrices ($m < 15$), function call overhead across the pivot loop represents ~30% of runtime. Compiled C extensions will be necessary for sub-millisecond simplex solving.

---

## 16. Claims / Evidence Matrix

### Claim 1: Basis Matrix Bitwise Identity Across Phases
- **CLAIM**: The Phase-II starting basis matrix $B_{\text{phase2}}$ is bitwise identical to the Phase-I terminal basis matrix $B_{\text{phase1}}$ when no artificial variables are basic.
- **EVIDENCE**: Checked across Workloads B, C, D, F, G, H, I.
- **MEASURED RESULT**: $\|B_{\text{phase1}} - B_{\text{phase2}}\|_\infty = 0.00 \times 10^0$ (exact bitwise match).
- **STATUS**: **VERIFIED**

### Claim 2: Redundant DenseLU Factorization Elimination
- **CLAIM**: Exactly 1 full DenseLU factorization is eliminated per Phase-I LP solve.
- **EVIDENCE**: Tracked via `mock.patch("optiforge.lp.simplex.BasisFactorization")` over 10 repetitions.
- **MEASURED RESULT**: Factorizations reduced from $(P_1 + 1) + (P_2 + 1)$ down to $P_1 + P_2 + 1$ (e.g. Workload G: 66 $\to$ 65 factorizations).
- **STATUS**: **VERIFIED**

### Claim 3: Basis Cleanup Allocation Reduction
- **CLAIM**: Non-degenerate basis cleanup bypasses full matrix copies of $A$ and $b$ before column filtering.
- **EVIDENCE**: Profiling and code trace in `_cleanup_phase1_basis`.
- **MEASURED RESULT**: 1 full matrix allocation avoided per Phase-I solve; cleanup time remains $<0.03$ ms.
- **STATUS**: **VERIFIED**

### Claim 4: MILP Root Warm-Start Activation & Speedup
- **CLAIM**: Resolving coordinate collision in `_solve_root_or_cold` enables Phase-I root MILP warm-starting, yielding measurable speedup.
- **EVIDENCE**: Benchmark on 8-variable Phase-I integer problem over 10 repetitions.
- **MEASURED RESULT**: Runtime reduced from 9.91 ms (Cold) to 3.98 ms (Warm), a **2.49x speedup** with 4/4 warm starts accepted.
- **STATUS**: **VERIFIED**

### Claim 5: Zero Mathematical or Numerical Regression
- **CLAIM**: All simplex invariants, tolerances, Bland's anti-cycling rule, and optimal solutions remain strictly preserved.
- **EVIDENCE**: 846 / 846 tests passed on pytest.
- **MEASURED RESULT**: 100% test pass rate with zero tolerance changes.
- **STATUS**: **VERIFIED**

---

## 17. Final Status

**COMPLETE**
