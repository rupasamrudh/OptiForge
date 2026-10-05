# OPTIFORGE — STAGE 8.44 AUDIT REPORT
## Post-Presolve Basis, Dual & Re-Optimization Readiness Audit

**Date:** 2026-09-30  
**Status:** AUDIT COMPLETE  
**Repository State:** Stage 8.43 complete (1099 passing tests, 0 failures, 21 benign warnings)

---

### 1. Purpose of Audit

Stage 8.44 prepares OptiForge for future Sensitivity & Scenario Analysis by exposing structured optimal-basis results, dual multipliers (shadow prices), reduced costs, presolve-to-original index mappings, and warm-start re-optimization hooks.

In accordance with engineering guidelines, this audit inspects all relevant subsystems prior to code modifications:
1. Basis representation in `RevisedSimplexSolver` and `DualRevisedSimplexSolver`.
2. Availability of $B$, $B^{-1}$, basic/nonbasic column indices, dual vector $y$, and reduced costs $r$.
3. Standard-form to original model variable mapping.
4. Presolve elimination and substitution metadata.
5. MILP solver boundary regarding LP relaxation bases.
6. `Solution` object structure and constructor compatibility.
7. Existing warm-start infrastructure and re-optimization points.

---

### 2. Detailed Subsystem Audit

#### 2.1 Basis Representation in Simplex Solvers

##### A. RevisedSimplexSolver (`optiforge/lp/simplex.py`)
- **Internal Loop Result:** `SimplexIterationResult` (`line 172-187`):
  - `status: SolutionStatus`
  - `basis_indices: list[int]` (length $m$, ordered indices of basic columns in Phase II matrix)
  - `nonbasic_indices: list[int]` (length $n-m$, ordered indices of nonbasic columns)
  - `x: np.ndarray` (length $n$, primal basic solution vector with $x_B = B^{-1}b$ and $x_N = 0$)
  - `objective_value: float`
  - `iterations: int`
  - `basis_factorization: BasisFactorization | None`
- **Dual Vector $y$:** Computed at every pivot via BTRAN (`line 854`):
  $$y = B^{-T} c_B$$
  using `basis_fact.solve_transpose(c_B)`.
- **Reduced Costs $r$:** Computed at every pivot via GEMV (`line 871-872`):
  $$r = c - A^T y$$
- **Phase I vs. Phase II Basis Cleanup:**
  If artificial variables were required, `_cleanup_phase1_basis` (`lines 1220-1320`) removes artificial columns from the system and generates `col_mapping: dict[int, int]` mapping original standard-form column indices to Phase II column indices.

##### B. DualRevisedSimplexSolver (`optiforge/lp/dual_simplex.py`)
- **Internal Result:** `DualSimplexResult` (`line 73-90`):
  - `status: SolutionStatus`
  - `basis_indices: list[int]`
  - `nonbasic_indices: list[int]`
  - `x: np.ndarray`
  - `y: np.ndarray` (dual vector $B^{-T} c_B$ explicitly retained in the result container)
  - `objective_value: float`
  - `iterations: int`
  - `basis_factorization: Any | None`
- **Reduced Costs $r$:** In `solve_standard_form`, maintained either incrementally or via GEMV $r = c - A^T y$.

##### Summary of Factorization Access:
Both solvers maintain `basis_factorization` (`BasisFactorization`, `PivotUpdateFactorization`, or `IncrementalBasisFactorization`). Neither exposes raw $B^{-1}$ as a dense matrix, but both can perform $B^{-1} v$ via `.solve(v)` and $B^{-T} v$ via `.solve_transpose(v)`.

---

#### 2.2 Standard-Form to Model Mapping (`optiforge/lp/standard_form.py`)

`StandardFormModel` provides comprehensive structural metadata:
- `original_variable_mapping: dict[str, int]`: Maps original variable name $\to$ column index in $A_{\text{sf}}$.
- `variable_info: list[VariableInfo]`: Per-column metadata:
  - `name: str` (e.g. `x1`, `s1`, `e2`, `a2`)
  - `role: VariableRole` (`ORIGINAL`, `SLACK`, `SURPLUS`, `ARTIFICIAL`)
  - `column_index: int`
  - `original_lower_bound: float`, `original_upper_bound: float`
- `variable_shifts: dict[int, float]`: Lower-bound shift $L_j$ for variables with $l_j \ne 0$ ($x_j = y_j + L_j, y_j \ge 0$).
- `objective_constant: float`: Accumulated shift $c^T L$.
- Row-to-Constraint Correspondence:
  - Rows $0 \dots m_{\text{orig}}-1$ correspond 1-to-1 with `model.constraints` in order.
  - Rows $m_{\text{orig}} \dots m_{\text{orig}} + n_{\text{ub}} - 1$ correspond to finite upper bound rows ($y_j \le u_j - l_j$).
  - Constraint normalization records sign flips if $b_i < 0$ originally.

---

#### 2.3 Presolve Elimination & Mapping (`optiforge/presolve/presolve.py`)

`PresolveResult` (`lines 72-117`) records:
- `reduced_model: Model | None`
- `fixed_variables: dict[str, float]`: Variables eliminated due to bound fixing ($l_j == u_j$), singleton rows, or zero columns.
- `original_to_reduced: dict[str, str]` & `reduced_to_original: dict[str, str]`: Mapping of surviving variable names.
- `removed_constraints: list[dict[str, Any]]`: Constraints removed as redundant or singleton.
- `objective_constant: float`: Constant shift added to reduced objective.
- `postsolve_solution()`: Restores original variable values from `reduced_solution` and `fixed_variables`.

---

#### 2.4 MILP Boundary

- `MILPSolver` solves mixed-integer problems through branch-and-bound tree search.
- When `has_integer_vars` is true:
  - The final integer solution is an incumbent found at an integer leaf node or via heuristics.
  - It does **not** represent a valid continuous sensitivity basis.
  - Classical linear programming shadow prices and reduced costs are mathematically undefined for integer-constrained optimums without fixing integer variables.
- When `has_integer_vars` is false:
  - `MILPSolver.solve(model)` delegates directly to `self.lp_solver.solve(model)`.

---

#### 2.5 `Solution` Container (`optiforge/core/solution.py`)

Current fields on `Solution`:
- `status: SolutionStatus`
- `variable_values: dict[str, float]`
- `objective_value: float | None`
- `message: str`
- `iterations: int`
- `node_count`, `numerical_snapshots`, `incumbent_objective`, `best_bound`, `optimality_gap`, `absolute_gap`, `nodes_remaining`, `max_depth`, `feasible_solutions_found`, `profile_stats`
- Stage 8.43 additions:
  - `problem_fingerprint: Any | None = None`
  - `solver_selected: str | None = None`

**Integration Point:** Adding `basis_info: BasisInfo | None = None` preserves 100% backward compatibility.

---

#### 2.6 Existing Warm-Start Infrastructure

- `DualRevisedSimplexSolver` accepts `initial_basis: list[int] | None` in `solve()` (`optiforge/lp/dual_simplex.py:301`).
- `MILPSolver` uses this extensively when re-optimizing child relaxations after bound-tightening branches (`optiforge/milp/solver.py:1013-1020`).
- `RevisedSimplexSolver` accepts `basis_indices` and `nonbasic_indices` in `_simplex_loop`, but its public `solve()` method did not expose `initial_basis`.

---

### 3. Architecture & Design Decisions for Stage 8.44

#### 3.1 BasisInfo Data Structure
Create `optiforge/core/basis.py`:
- `BasisStatus` enum (`BASIC`, `NONBASIC_LOWER`, `NONBASIC_UPPER`, `FIXED`, `FREE`, etc.)
- `BasisInfo` (immutable frozen dataclass):
  - `basic_variable_indices: list[int]`
  - `nonbasic_variable_indices: list[int]`
  - `variable_names: list[str]` (names in standard-form space)
  - `original_variable_names: list[str]` (names in original model space)
  - `primal_basic_values: dict[str, float]` (values of basic variables)
  - `reduced_costs: dict[str, float]` (reduced costs mapped to original variable names and slack/surplus names)
  - `dual_multipliers: dict[str, float]` (shadow prices mapped to original constraint names)
  - `objective_value: float`
  - `standard_form_dimension: tuple[int, int]` ($(m, n)$)
  - `solver_selected: str`
  - `is_lp: bool`
  - `is_optimal: bool`
  - `metadata: dict[str, Any]`

#### 3.2 Dual Multiplier (Shadow Price) Sign Conventions
Let optimal objective be $z^*$. Shadow price for constraint $i$ is defined as $\lambda_i = \frac{\partial z^*}{\partial b_i}$:
- For $\min c^T x$ subject to $A_i x \le b_i$:
  Relaxing $b_i$ increases the feasible set $\implies z^*$ decreases or stays equal: $\lambda_i \le 0$.
- For $\min c^T x$ subject to $A_i x \ge b_i$:
  Tightening $b_i$ restricts the feasible set $\implies z^*$ increases or stays equal: $\lambda_i \ge 0$.
- For $\max c^T x$ subject to $A_i x \le b_i$:
  Relaxing $b_i$ increases the feasible set $\implies z^*$ increases or stays equal: $\lambda_i \ge 0$.
- For $\max c^T x$ subject to $A_i x \ge b_i$:
  Tightening $b_i$ restricts the feasible set $\implies z^*$ decreases or stays equal: $\lambda_i \le 0$.
- For equality constraints $A_i x = b_i$:
  $\lambda_i$ can be positive or negative depending on the direction of constraint gradient.

#### 3.3 Presolve Variable & Constraint Recovery
- If presolve eliminated fixed variables ($x_k = \bar{x}_k$):
  - The reduced costs for fixed variables can be computed from the dual multipliers:
    $$r_k = c_k - \sum_i A_{i, k} \lambda_i$$
  - Presolved eliminated redundant constraints have shadow price $\lambda_i = 0.0$.
  - Fixed variables have status `BasisStatus.FIXED` or `NONBASIC_LOWER`.

#### 3.4 Warm-Start Re-Optimization Hook
- `Model.solve(basis=...)`:
  - Validates basis compatibility:
    - `basis.standard_form_dimension` must match standard-form dimensions $(m, n)$.
    - Basis indices must be valid column indices in $[0, n-1]$ and of length $m$.
  - If compatible and solver is `DualRevisedSimplexSolver` or `RevisedSimplexSolver`, injects the warm-start basis.
  - If incompatible, raises `ValueError` with clear diagnostics or falls back safely if requested.

---

### 4. Implementation Plan

1. **`optiforge/core/basis.py`**: Implement `BasisStatus` and `BasisInfo`.
2. **`optiforge/core/solution.py`**: Add `basis_info: BasisInfo | None = None` to `Solution`.
3. **`optiforge/lp/simplex.py` & `optiforge/lp/dual_simplex.py`**: Capture dual vector $y$, reduced costs $r$, and basis indices on optimal termination.
4. **`optiforge/model/pipeline.py`**:
   - Extract dual multipliers and reduced costs.
   - Map from standard-form/presolve space back to original constraints and variables.
   - Construct `BasisInfo` and attach to `Solution` and `model.last_basis`.
   - Support `basis: BasisInfo | list[int] | None` parameter for warm-started re-optimization.
5. **`optiforge/model/model.py`**:
   - Update `Model.solve(..., basis=None)` and add `Model.get_last_basis()`.
6. **Tests & Verification**:
   - Create `tests/test_stage_8_44_basis_readiness.py` covering all 20 required test categories.
   - Run full regression suite (1099+ tests).
7. **Documentation**:
   - Create `docs/STAGE_8_44_BASIS_READINESS_REPORT.md`.
