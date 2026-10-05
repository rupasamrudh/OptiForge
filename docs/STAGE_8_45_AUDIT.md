# STAGE 8.45 AUDIT: LP SENSITIVITY ANALYSIS & PRESOLVE-INVERTED SENSITIVITY
**OptiForge Optimization Engine**  
**Phase**: Pre-Implementation Audit  
**Target Scope**: Continuous LP Sensitivity (RHS & Objective Coefficients)

---

## 1. Existing Infrastructure Audit

### 1.1 Basis Representation (`optiforge/core/basis.py`)
- `BasisInfo` is a frozen dataclass holding:
  - `basic_variable_indices`: List of $m$ column indices in standard-form problem.
  - `nonbasic_variable_indices`: List of $n - m$ column indices.
  - `basic_variables`, `nonbasic_variables`: Column names in standard form.
  - `variable_status`: Mapping from model variable name to `BasisStatus` (`BASIC`, `NONBASIC_LOWER`, `NONBASIC_UPPER`, `NONBASIC_ZERO`, `FIXED`, `FREE`).
  - `primal_basic_values`: Mapping from basic variable name to primal value $x_B$.
  - `reduced_costs`: Mapping from variable name to reduced cost $r_j$.
  - `dual_multipliers`: Mapping from constraint name to shadow price $\lambda_i$.
  - `standard_form_dimension`: $(m, n)$.
  - `is_lp`, `is_optimal`: Booleans ensuring validity.
  - `metadata`: Presolve and solver metadata (`presolved`, `fixed_variable_count`, `removed_constraint_count`).

### 1.2 Simplex Solvers (`RevisedSimplexSolver` & `DualRevisedSimplexSolver`)
- Both solvers record:
  - `self.last_standard_form_model`: The `StandardFormModel` instance ($A, b, c$).
  - `self.last_standard_form_basis_indices`: List of column indices forming basis matrix $B = A_{:, \text{basis}}$.
  - `self.last_iteration_result`: Iteration result with solution vector $x$, dual vector $y = B^{-T} c_B$, and reduced costs $r = c - A^T y$.

### 1.3 Standard Form Representation (`StandardFormModel` in `optiforge/lp/standard_form.py`)
- **Variables**:
  - Original variables: columns $0, \dots, n_{\text{orig}}-1$.
  - Variable shifts for lower bounds $x_j = y_j + L_j$ ($L_j \ne 0$).
  - Finite upper bounds converted into explicit rows: $y_j + s_{\text{ub}} = U_j - L_j$.
  - Slack variables for $\le$ constraints ($+1$).
  - Surplus variables for $\ge$ constraints ($-1$) and artificial variables ($+1$).
- **Constraints & RHS Normalization**:
  - Variable shifts applied first: $\text{raw\_rhs}_i = b_{\text{orig}, i} - \sum_j A_{i, j} L_j$.
  - Normalization: If $\text{raw\_rhs}_i < 0$, row is multiplied by $-1$, and constraint sense is flipped ($<= \leftrightarrow >=$).
  - Equality constraints retain equality sense; if RHS $< 0$, row is multiplied by $-1$.
  - Upper bound constraints added as trailing rows.

### 1.4 Presolve Representation (`optiforge/presolve/presolve.py`)
- `PresolveResult` records:
  - `reduced_model`: The model passed to the simplex solver.
  - `fixed_variables`: Variables fixed to constant values.
  - `removed_constraints`: Constraints removed during reduction (e.g., redundant constraints or singleton rows transferred to variable bounds).
  - `original_to_reduced`, `reduced_to_original`: Name mappings between spaces.
  - `objective_constant`: Shift in objective value due to fixed variables and lower bound shifts.

### 1.5 Model API (`optiforge/model/model.py`)
- `model.solve()` returns `Solution`.
- `solution.basis_info` contains `BasisInfo` for continuous LPs, or `None` for MILP/infeasible/unbounded.
- `model.get_last_basis()` returns `solution.basis_info`.

---

## 2. Sensitivity Analysis Mathematical Formulation

### 2.1 Classical RHS Sensitivity (Standard Form)
For standard form system $Ax = b, x \ge 0$:
$$B x_B = b \implies x_B = B^{-1} b$$
Under perturbation of RHS element $k$ by $\delta$:
$$b' = b + \delta e_k \implies x_B(\delta) = x_B + \delta \beta_k, \quad \text{where } \beta_k = B^{-1} e_k$$
The current basis remains primal feasible iff:
$$x_{B, i} + \delta \beta_{k, i} \ge 0, \quad \forall i = 0, \dots, m-1$$
- For $\beta_{k, i} > 0$: $\delta \ge -\frac{x_{B, i}}{\beta_{k, i}}$ (lower bound).
- For $\beta_{k, i} < 0$: $\delta \le -\frac{x_{B, i}}{\beta_{k, i}}$ (upper bound).
- For $\beta_{k, i} = 0$: no bound.

Allowable standard-form perturbation interval:
$$\delta_{\min} = \max_{i: \beta_{k, i} > 0} \left( -\frac{x_{B, i}}{\beta_{k, i}} \right) \quad (\text{or } -\infty)$$
$$\delta_{\max} = \min_{i: \beta_{k, i} < 0} \left( -\frac{x_{B, i}}{\beta_{k, i}} \right) \quad (\text{or } +\infty)$$

Allowable decrease $= -\delta_{\min}$; Allowable increase $= \delta_{\max}$.

### 2.2 Presolve Inversion & Original Model Mapping for RHS
For an original constraint $C_i$:
1. **Surviving Constraint**:
   - Let row $i$ in standard form correspond to $C_i$.
   - Normalization check: Let $\text{flip\_sign}_i = -1$ if row was multiplied by $-1$ during negative RHS normalization, else $+1$.
   - If $\text{flip\_sign}_i = +1$:
     $$\Delta b_{\text{orig}} \in [\delta_{\min}, \delta_{\max}]$$
     $$\text{allowable\_decrease} = -\delta_{\min}, \quad \text{allowable\_increase} = \delta_{\max}$$
   - If $\text{flip\_sign}_i = -1$:
     $$\Delta b_{\text{sf}} \in [\delta_{\min}, \delta_{\max}] \implies -\Delta b_{\text{orig}} \in [\delta_{\min}, \delta_{\max}] \implies \Delta b_{\text{orig}} \in [-\delta_{\max}, -\delta_{\min}]$$
     $$\text{allowable\_decrease} = \delta_{\max}, \quad \text{allowable\_increase} = -\delta_{\min}$$
   - Current RHS value: $b_{\text{orig}, i}$.
   - Lower allowable RHS: $b_{\text{orig}, i} - \text{allowable\_decrease}$.
   - Upper allowable RHS: $b_{\text{orig}, i} + \text{allowable\_increase}$.

2. **Redundant Constraint Removed in Presolve**:
   - The constraint was inactive/redundant across the entire feasible region.
   - For a $\le$ constraint $a_i^T x \le b_i$: The maximum activity $\max a_i^T x = U_i^{\text{act}} \le b_i$.
   - Increasing $b_i$ only makes the constraint more redundant ($\text{allowable\_increase} = +\infty$).
   - Decreasing $b_i$ remains optimal until $b_i = U_i^{\text{act}}$ (slack $= b_i - U_i^{\text{act}}$).
   - If presolve activity bounds are available, report $[U_i^{\text{act}}, +\infty)$; otherwise mark as explicitly unbounded above with documented limitation.

3. **Singleton Constraint Transferred to Variable Bound**:
   - If constraint $a_i x_j \le b_i$ was converted to $x_j \le b_i / a_i$, its sensitivity maps directly to the bound sensitivity of $x_j$.

### 2.3 Classical Objective Coefficient Sensitivity (Standard Form)
Reduced cost formula:
$$r = c - A^T y, \quad y = B^{-T} c_B$$

#### Case 1: Nonbasic Variable $k \in N$
Perturbation $c_k \to c_k + \delta$:
- $c_B$ is unchanged, so $y$ is unchanged.
- $r_k(\delta) = r_k + \delta$.
- All other reduced costs $r_j$ ($j \ne k$) are unchanged.

Optimality condition:
- **MINIMIZE**: $r_k + \delta \ge 0 \implies \delta \ge -r_k$.
  - Allowable decrease $= r_k \ge 0$.
  - Allowable increase $= +\infty$.
  - Range: $[c_k - r_k, +\infty)$.
- **MAXIMIZE**: $r_k + \delta \le 0 \implies \delta \le -r_k$.
  - (With OptiForge standard convention where $r \le 0$ at max optimum):
  - Allowable decrease $= +\infty$.
  - Allowable increase $= -r_k \ge 0$.
  - Range: $(-\infty, c_k - r_k]$.

#### Case 2: Basic Variable $k \in B$, where $k = B(p)$
Perturbation $c_k \to c_k + \delta$:
- $c_B \to c_B + \delta e_p$.
- Dual vector: $y(\delta) = B^{-T} (c_B + \delta e_p) = y + \delta B^{-T} e_p = y + \delta \gamma_p$, where $\gamma_p = B^{-T} e_p$ ($p$-th row of $B^{-1}$).
- Reduced cost for each nonbasic variable $j \in N$:
  $$r_j(\delta) = c_j - A_{:, j}^T y(\delta) = r_j - \delta (A_{:, j}^T \gamma_p) = r_j - \delta \alpha_{p, j}$$
  where $\alpha_{p, j} = (B^{-1} A_{:, j})_p$.

Optimality condition:
- **MINIMIZE** ($r_j(\delta) \ge 0$ for all $j \in N$):
  - For $\alpha_{p, j} > 0$: $\delta \le \frac{r_j}{\alpha_{p, j}}$ (upper bound on $\delta$).
  - For $\alpha_{p, j} < 0$: $\delta \ge \frac{r_j}{\alpha_{p, j}}$ (lower bound on $\delta$).
  $$\delta_{\max} = \min_{j \in N: \alpha_{p, j} > 0} \frac{r_j}{\alpha_{p, j}} \quad (\text{or } +\infty)$$
  $$\delta_{\min} = \max_{j \in N: \alpha_{p, j} < 0} \frac{r_j}{\alpha_{p, j}} \quad (\text{or } -\infty)$$
  Allowable increase $= \delta_{\max} \ge 0$; Allowable decrease $= -\delta_{\min} \ge 0$.
  Range: $[c_k + \delta_{\min}, c_k + \delta_{\max}]$.

- **MAXIMIZE** ($r_j(\delta) \le 0$ for all $j \in N$):
  - For $\alpha_{p, j} > 0$: $\delta \ge \frac{r_j}{\alpha_{p, j}}$ (lower bound on $\delta$).
  - For $\alpha_{p, j} < 0$: $\delta \le \frac{r_j}{\alpha_{p, j}}$ (upper bound on $\delta$).
  $$\delta_{\max} = \min_{j \in N: \alpha_{p, j} < 0} \frac{r_j}{\alpha_{p, j}} \quad (\text{or } +\infty)$$
  $$\delta_{\min} = \max_{j \in N: \alpha_{p, j} > 0} \frac{r_j}{\alpha_{p, j}} \quad (\text{or } -\infty)$$
  Allowable increase $= \delta_{\max} \ge 0$; Allowable decrease $= -\delta_{\min} \ge 0$.
  Range: $[c_k + \delta_{\min}, c_k + \delta_{\max}]$.

### 2.4 Degeneracy
If any basic variable $x_{B, i} = 0$, then $-\frac{x_{B, i}}{\beta_{k, i}} = 0$.
Thus $\delta_{\min} = 0$ or $\delta_{\max} = 0$.
The allowable increase or decrease is strictly $0.0$, indicating a one-sided or degenerate sensitivity range where any perturbation in that direction immediately causes an alternative basis or basis change.
This is mathematically sound and must not be altered with arbitrary offsets.

### 2.5 Presolve Inversion for Variables
1. **Surviving Variables**:
   - Column index in standard form maps directly to the variable.
   - Reduced costs and tableau coefficients map directly.
   - Variable shifts $x_j = y_j + L_j$ only shift the origin of $x_j$ by a constant $L_j$; objective coefficient $c_j$ multiplies $(y_j + L_j)$, so the sensitivity of $c_j$ with respect to the optimal basis is identical in $x$-space and $y$-space!
2. **Fixed Variables**:
   - A variable fixed by presolve ($x_k = v_k$) is eliminated from the optimization problem. Its objective coefficient cannot alter the optimal basis of the remaining reduced LP.
   - Range: $(-\infty, +\infty)$ with explicit metadata `is_fixed=True`.

---

## 3. Architecture Plan & Module Layout

Create a dedicated module:
`optiforge/sensitivity/`
- `__init__.py`: Public exports.
- `range.py`: `SensitivityRange` dataclass.
- `report.py`: `SensitivityReport` dataclass.
- `calculator.py`: Core LP sensitivity solver operating on `Model`, `BasisInfo`, and `StandardFormModel`.

Expose on `Solution`:
- `solution.sensitivity: SensitivityReport | None`
- `solution.get_sensitivity() -> SensitivityReport`

Expose on `Model`:
- `model.sensitivity_analysis() -> SensitivityReport`

Strict Error Handling:
- For MILP, infeasible, unbounded, or missing basis: raises `ValueError` or returns `None` with explicit status (no fabrication).

---

## 4. Verification Strategy

Create `tests/test_stage_8_45_sensitivity.py` covering all 30 required categories:
1. <= RHS sensitivity.
2. >= RHS sensitivity.
3. Equality constraint RHS sensitivity.
4. Multiple constraints.
5. Negative RHS.
6. Row sign-flipped constraint.
7. Binding constraint.
8. Non-binding constraint.
9. Degenerate optimal basis RHS sensitivity.
10. Infinite allowable direction for RHS.
11. Nonbasic variable objective sensitivity.
12. Basic variable objective sensitivity.
13. Multiple nonbasic variables affected by basic-cost change.
14. MAXIMIZE objective sensitivity.
15. MINIMIZE objective sensitivity.
16. Zero reduced-cost variable.
17. Degenerate optimum objective sensitivity.
18. Fixed-variable elimination mapping.
19. Redundant constraint mapping.
20. Surviving constraint mapping.
21. Surviving variable mapping.
22. Bound transformation mapping.
23. MILP rejected.
24. Infeasible rejected.
25. Unbounded rejected.
26. Missing basis rejected.
27. Sensitivity report references original model names.
28. Infinite ranges represented cleanly (`math.inf`, `is_unbounded_above=True`).
29. Existing Stage 8.44 warm-start remains functional.
30. Existing Stage 8.43 metadata remains intact.

Cross-check testing:
- Perturb parameter by $\pm 0.99 \times \delta$ (inside range) $\implies$ re-solve $\implies$ basis remains optimal.
- Perturb parameter by $\pm 1.05 \times \delta$ (outside range) $\implies$ basis is no longer optimal.
