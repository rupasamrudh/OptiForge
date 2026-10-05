# STAGE 8.45: LP SENSITIVITY ANALYSIS & PRESOLVE-INVERTED SENSITIVITY REPORT
**OptiForge Optimization Engine**  
**Status**: COMPLETE and VERIFIED  
**Regression Baseline**: 1151 passed, 0 failures, 21 benign warnings (100% green)

---

## 1. Executive Summary

Stage 8.45 implements a mathematically rigorous, presolve-inverted post-optimality sensitivity analysis engine for continuous Linear Programs (LPs). Built on top of the verified Stage 8.44 `BasisInfo` infrastructure, this engine derives exact allowable variation intervals for:
1. **Right-Hand-Side (RHS) values** of original constraints while preserving the current optimal basis.
2. **Objective function coefficients** of original variables while preserving the current optimal basis.

> [!IMPORTANT]
> **"Basis remains optimal" vs. "Solution remains numerically unchanged":**  
> A sensitivity interval defines the precise parameter domain over which the **current set of basic variables remains optimal**.  
> - For RHS perturbations, the basis remains optimal, but the numerical basic variable values $x_B(\delta) = B^{-1}(b + \delta e_k)$ and objective value $z^*(\delta) = z^* + \delta \lambda_k$ change linearly with $\delta$.  
> - For objective coefficient perturbations of a basic variable, the primal solution $x^*$ remains numerically identical, but the dual vector $y(\delta) = B^{-T}(c_B + \delta e_p)$ and objective value change.  
> - Sensitivity ranges **do not** predict the new optimal basis or solution outside the allowable interval.

---

## 2. Mathematical Formulations & Derivations

### 2.1 Right-Hand-Side (RHS) Sensitivity
In standard form ($Ax = b, x \ge 0$), the basic solution is given by:
$$x_B = B^{-1} b$$
Perturbing standard-form RHS component $k$ by $\delta$:
$$b' = b + \delta e_k \implies x_B(\delta) = x_B + \delta \beta_k, \quad \text{where } \beta_k = B^{-1} e_k$$
Dual feasibility ($r = c - A^T y$) is independent of $b$. Therefore, the basis $B$ remains optimal if and only if primal feasibility is preserved:
$$x_{B, i} + \delta \beta_{k, i} \ge 0, \quad \forall i = 0, \dots, m-1$$
- For $\beta_{k, i} > 0$: $\delta \ge -\frac{x_{B, i}}{\beta_{k, i}}$ (lower bound).
- For $\beta_{k, i} < 0$: $\delta \le -\frac{x_{B, i}}{\beta_{k, i}}$ (upper bound).
- For $\beta_{k, i} = 0$: no bound imposed.

Allowable standard-form perturbation range $[\delta_{\min}, \delta_{\max}]$:
$$\delta_{\min} = \max_{i: \beta_{k, i} > \epsilon} \left( -\frac{x_{B, i}}{\beta_{k, i}} \right) \quad (\text{or } -\infty)$$
$$\delta_{\max} = \min_{i: \beta_{k, i} < -\epsilon} \left( -\frac{x_{B, i}}{\beta_{k, i}} \right) \quad (\text{or } +\infty)$$

#### Negative RHS Normalization & Row Sign Flips
If constraint $k$ had $\text{raw\_rhs}_k < 0$, `to_standard_form` multiplied the row by $-1$ and flipped the constraint sense:
$$b_{\text{sf}, k} = -b_{\text{orig}, k} \implies \Delta b_{\text{sf}, k} = -\Delta b_{\text{orig}, k}$$
Therefore:
$$\Delta b_{\text{orig}, k} \in [-\delta_{\max}, -\delta_{\min}]$$
- $\text{allowable\_decrease} = \delta_{\max}$
- $\text{allowable\_increase} = -\delta_{\min}$

### 2.2 Objective Coefficient Sensitivity
Standard-form reduced costs are:
$$r = c - A^T y, \quad y = B^{-T} c_B$$

#### Nonbasic Variable $k \in N$
Perturbation $c_k \to c_k + \delta$ does not change $c_B$, so $y$ is unchanged. Only $r_k$ changes:
$$r_k(\delta) = r_k + \delta$$
- **MINIMIZE** ($r_k(\delta) \ge 0$): $\delta \ge -r_k \implies$ Allowable decrease $= r_k$, Allowable increase $= +\infty$. Interval: $[c_k - r_k, +\infty)$.
- **MAXIMIZE** ($r_k(\delta) \le 0$): $\delta \le -r_k \implies$ Allowable decrease $= +\infty$, Allowable increase $= -r_k$. Interval: $(-\infty, c_k - r_k]$.

#### Basic Variable $k \in B$, where $k = B(p)$
Perturbing $c_k \to c_k + \delta$ changes $c_B \to c_B + \delta e_p$. The dual vector shifts:
$$y(\delta) = y + \delta B^{-T} e_p = y + \delta \gamma_p, \quad \text{where } \gamma_p = B^{-T}[:, p]$$
For every nonbasic variable $j \in N$ (excluding Phase-I artificial columns):
$$r_j(\delta) = c_j - A_{:, j}^T y(\delta) = r_j - \delta (A_{:, j}^T \gamma_p) = r_j - \delta \alpha_{p, j}, \quad \text{where } \alpha_{p, j} = (B^{-1} A_{:, j})_p$$
- **MINIMIZE** ($r_j(\delta) \ge 0$):
  - $\alpha_{p, j} > 0 \implies \delta \le \frac{r_j}{\alpha_{p, j}}$
  - $\alpha_{p, j} < 0 \implies \delta \ge \frac{r_j}{\alpha_{p, j}}$
- **MAXIMIZE** ($r_j(\delta) \le 0$):
  - $\alpha_{p, j} > 0 \implies \delta \ge \frac{r_j}{\alpha_{p, j}}$
  - $\alpha_{p, j} < 0 \implies \delta \le \frac{r_j}{\alpha_{p, j}}$

### 2.3 Degeneracy Handling
When a basic variable has value $x_{B, i} = 0$, $-\frac{x_{B, i}}{\beta_{k, i}} = 0$. This correctly yields a one-sided sensitivity range:
$$\text{allowable\_decrease} = 0.0 \quad \text{or} \quad \text{allowable\_increase} = 0.0$$
The engine correctly exposes exact zero-width limits without fabricating arbitrary non-zero tolerances.

### 2.4 Representation of Infinite Ranges
Unbounded variation is explicitly represented:
- `allowable_decrease = math.inf`, `is_unbounded_below = True`, `lower_bound = -math.inf`
- `allowable_increase = math.inf`, `is_unbounded_above = True`, `upper_bound = math.inf`

---

## 3. Presolve Inversion & Original Coordinates Mapping

Sensitivity is reported in the user's original model space:

1. **Surviving Constraints**: Standard-form ranges are adjusted for variable shifts ($\sum a_{ij} L_j$) and negative RHS row inversions ($\text{flip\_sign} = -1$).
2. **Redundant Constraints Removed by Presolve**:
   - For $\le$ constraints: Can increase to $+\infty$ (`is_unbounded_above = True`); can decrease down to activity $a^T x^*$ (`allowable_decrease = b - a^T x^*`).
   - For $\ge$ constraints: Can decrease to $-\infty$ (`is_unbounded_below = True`); can increase up to activity $a^T x^*$.
3. **Fixed Variables**:
   - A variable fixed in presolve ($x_k = v_k$) is eliminated from the LP simplex solve. Its objective coefficient variation cannot change the basis of the remaining reduced model.
   - Reported with `allowable_decrease = math.inf`, `allowable_increase = math.inf`, and metadata `presolved_fixed = True`.

---

## 4. Architecture & File Map

| Component | File Path | Description |
| :--- | :--- | :--- |
| **Sensitivity Range** | [optiforge/sensitivity/range.py](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/sensitivity/range.py) | Immutable `SensitivityRange` dataclass. |
| **Sensitivity Report** | [optiforge/sensitivity/report.py](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/sensitivity/report.py) | Container holding RHS & objective ranges with `.summary()`. |
| **Calculator Engine** | [optiforge/sensitivity/calculator.py](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/sensitivity/calculator.py) | `compute_sensitivity(model, solution, basis_info)`. |
| **Package Exports** | [optiforge/sensitivity/__init__.py](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/sensitivity/__init__.py) | Module exports. |
| **Solution API** | [optiforge/core/solution.py](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/core/solution.py) | Added `solution.sensitivity` and `solution.get_sensitivity()`. |
| **Model API** | [optiforge/model/model.py](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/model/model.py) | Added `model.sensitivity_analysis()`, `model.last_sensitivity`. |
| **Pipeline Integration**| [optiforge/model/pipeline.py](file:///c:/Users/Prasanna/SIH2026/OptiForge/optiforge/model/pipeline.py) | Automatic sensitivity calculation on optimal continuous LPs. |
| **Test Suite** | [tests/test_stage_8_45_sensitivity.py](file:///c:/Users/Prasanna/SIH2026/OptiForge/tests/test_stage_8_45_sensitivity.py) | 32 verification & cross-check tests. |

---

## 5. Usage Example

```python
from optiforge.model import Model, Variable, Constraint, Objective, ObjectiveSense, ConstraintSense

model = Model("production_plan")
x1 = model.add_variable(Variable("x1", lower_bound=0.0))
x2 = model.add_variable(Variable("x2", lower_bound=0.0))

model.set_objective(Objective({"x1": 3.0, "x2": 2.0}, sense=ObjectiveSense.MAXIMIZE))
model.add_constraint(Constraint("labor", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 4.0))
model.add_constraint(Constraint("material", {"x1": 2.0, "x2": 1.0}, ConstraintSense.LESS_EQUAL, 5.0))

solution = model.solve()
report = solution.sensitivity

# Query constraint RHS sensitivity:
labor_range = report.get_rhs_range("labor")
print(labor_range.lower_bound, labor_range.upper_bound)  # 2.5, 5.0
print(labor_range.allowable_decrease, labor_range.allowable_increase)  # 1.5, 1.0

# Query objective coefficient sensitivity:
profit_x1 = report.get_objective_range("x1")
print(profit_x1.lower_bound, profit_x1.upper_bound)  # 2.0, 4.0

# Formatted console report:
print(report.summary())
```

---

## 6. Verification & Test Evidence

Test file: [tests/test_stage_8_45_sensitivity.py](file:///c:/Users/Prasanna/SIH2026/OptiForge/tests/test_stage_8_45_sensitivity.py) (32/32 tests passed in 0.33s).

### Test Matrix
1. `test_01_simple_le_rhs`: <= RHS sensitivity matches analytical range $[2.5, 5.0]$.
2. `test_02_simple_ge_rhs`: >= RHS sensitivity.
3. `test_03_equality_constraint_rhs`: Equality constraint RHS.
4. `test_04_multiple_constraints`: Multiple constraints simultaneously evaluated.
5. `test_05_negative_rhs`: Negative RHS normalizations inverted cleanly.
6. `test_06_row_sign_flipped_constraint`: Row sign flips handled.
7. `test_07_binding_constraint`: Binding constraints flag `binding=True`.
8. `test_08_non_binding_constraint`: Non-binding constraints allow infinite relaxation.
9. `test_09_degenerate_optimal_basis_rhs`: Degenerate optimal basis handles one-sided intervals.
10. `test_10_infinite_allowable_direction`: Explicit unbounded direction verification.
11. `test_11_nonbasic_variable_coefficient`: Nonbasic cost allows one-sided infinite change.
12. `test_12_basic_variable_coefficient`: Basic cost matches analytical $[2.0, 4.0]$ range.
13. `test_13_multiple_nonbasics_affected`: Basic cost checks all nonbasic reduced costs.
14. `test_14_maximize_objective_sensitivity`: MAXIMIZE sign conventions.
15. `test_15_minimize_objective_sensitivity`: MINIMIZE sign conventions.
16. `test_16_zero_reduced_cost_variable`: Zero reduced cost / alternate optima.
17. `test_17_degenerate_optimum_obj`: Degenerate optimum objective bounds.
18. `test_18_fixed_variable_elimination`: Presolve fixed variables mapped to $(-\infty, +\infty)$.
19. `test_19_redundant_constraint_handling`: Presolve redundant constraints mapped.
20. `test_20_surviving_constraint_mapping`: Surviving constraints mapped through presolve.
21. `test_21_surviving_variable_mapping`: Surviving variables mapped through presolve.
22. `test_22_bound_transformation_mapping`: Bound shifts mapped through presolve.
23. `test_23_milp_rejected`: MILP models strictly reject sensitivity analysis.
24. `test_24_infeasible_rejected`: Infeasible models reject sensitivity analysis.
25. `test_25_unbounded_rejected`: Unbounded models reject sensitivity analysis.
26. `test_26_missing_basis_rejected`: Unsolved models reject sensitivity analysis.
27. `test_27_references_original_model_names`: Reports reference original user names.
28. `test_28_infinite_ranges_representation`: Clean `math.inf` and boolean representation.
29. `test_29_existing_stage_8_44_warm_start_functional`: Stage 8.44 warm start remains green.
30. `test_30_stage_8_43_metadata_intact`: Stage 8.43 solver-selection metadata intact.
31. `test_31_cross_check_inside_interval`: Perturbations inside range preserve basis.
32. `test_32_cross_check_outside_interval`: Perturbations outside range change basis.

### Regression Suite Status
- **Total Tests**: **1151 passed**, 0 failures, 21 benign warnings (100% green).
- **Execution Time**: 19.63s.

---

## 7. Performance & Overhead Measurement

Microbenchmark on a 50-row $\times$ 100-column LP ([scratch/measure_sensitivity_performance.py](file:///c:/Users/Prasanna/SIH2026/OptiForge/scratch/measure_sensitivity_performance.py)):

| Metric | Measured Value | Notes |
| :--- | :--- | :--- |
| **Cold Solve Time** | 886.93 ms | 986 simplex iterations |
| **Sensitivity Analysis Time** | **4.35 ms** | Evaluates all 50 constraints and 100 variables |
| **Overhead vs Solve Time** | **0.49%** | Fast vector operations; no refactorization |

---

## 8. Limitations & Deferred Work

- **Continuous LP Only**: Mixed-Integer Linear Programs (MILP) are strictly excluded from post-optimality sensitivity analysis. Integer shadow prices are mathematically non-existent without branch-and-bound tree sensitivity.
- **Simultaneous Multi-Parameter Changes**: 1D allowable intervals evaluate changes to a single parameter at a time ($\Delta b_k$ or $\Delta c_j$ holding all others fixed). The 100% Rule or tolerance regions for simultaneous multi-parameter variations are deferred to Stage 8.46.

---

## 9. Recommendation for Stage 8.46

With 1D sensitivity intervals complete, verified, and benchmarked:
1. **Implement the 100% Rule / Parametric Programming**:
   - Simultaneous RHS variations ($\sum \frac{\Delta b_i}{allowable_i} \le 1$).
   - Simultaneous cost variations.
2. **Scenario & What-If Analysis Engine**:
   - Expose higher-level scenario comparisons and break-even analysis hooks.
