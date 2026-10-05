# OPTIFORGE — STAGE 8.46 REPORT
## Multi-Parameter Sensitivity, 100% Rule & Scenario/What-If Readiness

**Date**: 2026-09-30  
**Status**: COMPLETE and VERIFIED  
**Regression Baseline**: 1181 passed, 0 failures, 21 benign warnings (from 1151 baseline, +30 new tests)

---

## 1. Executive Summary

Stage 8.46 extends OptiForge's sensitivity engine from one-dimensional parameter variations (Stage 8.45) into mathematically controlled **multi-parameter sensitivity, 100% Rule evaluation, exact basis preservation, scenario what-if analysis, and 1D break-even analysis**.

The implementation enables users to evaluate simultaneous changes to constraint right-hand sides ($\Delta b$) and objective coefficients ($\Delta c$) in fractions of a millisecond without incurring full simplex re-solves, while strictly maintaining the LP/MILP boundary and preserving the immutability of the user's original `Model`.

### Measured Performance Highlights:
- **Test Problem**: 50 rows $\times$ 100 columns continuous LP
- **Cold baseline solve time**: $885.98\text{ ms}$ (986 simplex iterations)
- **Analytical scenario evaluation**: **$0.437\text{ ms}$**
- **Analytical speedup over cold solve**: **$2025.3\times$**
- **Analytical speedup over re-solve**: **$2562.0\times$**

---

## 2. Architecture & Design

### New & Modified Modules:
1. `optiforge/sensitivity/multi.py`:
   - `RuleStatus` enum: `WITHIN_REGION`, `OUTSIDE_REGION`, `INDETERMINATE`.
   - `MultiParameterSensitivityResult`: Immutable dataclass reporting aggregate ratio sums, limiting parameters, per-parameter ratios, and optimality guarantee flag.
   - `evaluate_100_percent_rule()`: Vectorized 100% Rule evaluation for RHS and objective perturbations.

2. `optiforge/sensitivity/scenario.py`:
   - `Scenario`: Immutable specification of simultaneous perturbations (`rhs_changes`, `objective_changes`, `metadata`).
   - `ScenarioResult`: Comprehensive report containing exact primal/dual feasibility status, 100% Rule status, analytical or re-solved objective value and delta, reconstructed variable values, and limiting parameters.
   - `ScenarioComparison`: Multi-scenario aggregation container with tabular formatting (`summary()`).
   - `BreakEvenResult`: 1D target reachability, required parameter delta, and allowable sensitivity range.
   - `evaluate_scenario()`: Core analytical scenario engine with exact $B^{-1}(b + \Delta b)$ and $B^{-T}(c_B + \Delta c_B)$ verification and non-destructive re-solve dispatch (`resolve_if_needed=True`).
   - `compare_scenarios()`: High-level multi-scenario comparison helper.
   - `compute_break_even()`: Closed-form 1D linear break-even calculator for binding constraints and basic variables.

3. `optiforge/model/model.py`:
   - `Model.evaluate_scenario(scenario, resolve_if_needed=False)`
   - `Model.compare_scenarios(scenarios, resolve_if_needed=False)`
   - `Model.break_even(parameter, target_value, target="objective_value", parameter_type="auto", direction=None)`

4. `optiforge/sensitivity/__init__.py` & `optiforge/__init__.py`:
   - Seamless top-level exports preserving 100% backward compatibility with all Stage 8.43, 8.44, and 8.45 APIs.

---

## 3. Mathematical Foundations

### 3.1 The 100% Rule (Sufficient Condition)
For simultaneous changes to multiple parameters $p_i$, the 100% Rule states that if the sum of percentage changes relative to their respective 1D allowable limits does not exceed 100%, the current basis is guaranteed to remain optimal:

$$\sum_{i} \frac{|\Delta p_i|}{\text{allowable\_direction}_i} \le 1.0$$

Where:
- For $\Delta p_i > 0$: $\text{allowable\_direction}_i = \text{allowable\_increase}_i$.
- For $\Delta p_i < 0$: $\text{allowable\_direction}_i = \text{allowable\_decrease}_i$.
- If $\Delta p_i = 0$: ratio is $0.0$.
- If $\text{allowable\_direction}_i = \infty$: ratio is $0.0$ (perturbation along an unconstrained direction).
- If $\text{allowable\_direction}_i = 0$ (degenerate basis): any non-zero perturbation in that direction results in an infinite ratio, properly flagging the sufficient condition as not satisfied.

#### Critical Mathematical Assumption:
The 100% Rule is a **sufficient condition**, not a necessary condition. The actual stability region in parameter space is a convex polyhedron (the projection of the primal or dual feasibility cone). Points outside the 100% Rule simplex can still be strictly basis-feasible. Hence, OptiForge evaluates the 100% Rule as an indicator and performs an **exact basis-feasibility check**.

### 3.2 Exact Basis Preservation Checking
Given standard-form problem $Ax = b, x \ge 0$, and optimal basis index set $\mathcal{B}$ with basis matrix $B = A_{:, \mathcal{B}}$:

#### 1. RHS Perturbations ($b' = b + \Delta b$):
Dual feasibility is independent of $b$ and remains satisfied. The perturbed basic variables are:
$$x_B' = B^{-1}(b + \Delta b)$$
Primal feasibility requires:
$$x_B' \ge -\varepsilon_{\text{feas}}$$
If all basic variables remain non-negative, the basis remains primal feasible and therefore optimal. The new optimal objective is:
$$z' = c_B^T x_B' = z^* + y^{*T} \Delta b$$

#### 2. Objective Perturbations ($c' = c + \Delta c$):
Primal basic variables $x_B = B^{-1} b$ remain unchanged. Dual multipliers update according to basic costs:
$$y' = B^{-T} c_B' = B^{-T} (c_B + \Delta c_B)$$
Updated reduced costs across all variables:
$$r' = c' - A^T y'$$
Dual feasibility requires:
- For MINIMIZATION: $r_j' \ge -\varepsilon_{\text{tol}} \quad \forall j \notin \mathcal{B}$
- For MAXIMIZATION: $r_j' \le \varepsilon_{\text{tol}} \quad \forall j \notin \mathcal{B}$
*(Artificial Phase-I columns are excluded from Phase-II optimality checks).*

#### 3. Combined Perturbations ($b + \Delta b, c + \Delta c$):
The current basis remains optimal if and only if **both** primal feasibility ($x_B' \ge 0$) and dual feasibility ($r'$ optimal) hold simultaneously:
$$\text{basis\_preserved} = \text{primal\_feasible} \land \text{dual\_feasible}$$

---

## 4. Presolve Reconciliation

When presolve has removed redundant constraints, fixed variables, shifted variable bounds, or normalized negative RHS values:
1. **Parameter Name Mapping**:
   The public API takes original model names (`labor`, `x1`).
2. **All-Fixed Presolve Models**:
   When presolve fixes all variables (e.g. tightly bounded systems), the engine directly validates constraint activities against updated RHS values without attempting singular basis factorizations.
3. **Removed Redundant Constraints**:
   For constraints eliminated during presolve, their activity $a_i^T x^*$ is computed against the reconstructed optimal solution $x^*$. If the perturbed RHS $b_i + \Delta b_i$ violates the constraint, primal feasibility is correctly marked as violated.
4. **Sign-Flipped Rows**:
   Rows with negative RHS values normalized via $-1$ in standard form are inverted so $\Delta b_i$ is mapped with the correct sign.
5. **Non-Destructive Re-Solving**:
   When `resolve_if_needed=True` is requested, `_clone_model_with_scenario` performs deep replication of variables, constraints, and objectives. The original model instance is never mutated.

---

## 5. Break-Even Analysis Hooks

OptiForge provides closed-form 1D break-even analysis for target objective values $Z_{\text{target}}$:
$$\Delta Z = Z_{\text{target}} - z^*$$

- **Constraint RHS ($b_i$)**:
  $$\Delta b_i = \frac{\Delta Z}{\lambda_i}$$
  If $\lambda_i = 0$ (non-binding constraint), changing $b_i$ has zero marginal impact on $z^*$; the target is reported as unreachable.
- **Objective Coefficient ($c_j$)**:
  $$\Delta c_j = \frac{\Delta Z}{x_j^*}$$
  If $x_j^* = 0$ (nonbasic variable), altering its objective coefficient has zero marginal impact on $z^*$.
- **Basis Reachability**:
  The required parameter value $p_i + \Delta p_i$ is tested against the allowable range $[L_i, U_i]$ from 1D sensitivity analysis. If within $[L_i, U_i]$, `is_reachable_within_basis = True`. If a `direction` constraint ("increase" or "decrease") is provided, conflicting direction signs are flagged.

---

## 6. Verification and Test Results

The test suite `tests/test_stage_8_46_multi_parameter_sensitivity.py` covers all 30 required verification categories:

| Test ID | Category | Status | Details |
|---|---|---|---|
| `test_01` | Single RHS Scenario | PASSED | Analytical evaluation matches shadow price delta |
| `test_02` | Multiple RHS Changes | PASSED | Simultaneous +5/-5 changes verify linear superposition |
| `test_03` | 100% Rule Within Region | PASSED | Sum of ratios = 0.8 <= 1.0 verified guaranteed optimal |
| `test_04` | 100% Rule Outside Region | PASSED | Sum of ratios = 1.4 > 1.0 flags outside_region |
| `test_05` | Infinite Allowable Direction | PASSED | Slack constraints with inf range yield 0.0 ratio |
| `test_06` | Zero Allowable Direction | PASSED | Degenerate constraints handle 0 range without ZeroDivisionError |
| `test_07` | Exact RHS Feasibility | PASSED | Infeasible basic solution detected via $B^{-1}(b + \Delta b) < 0$ |
| `test_08` | Single Objective Change | PASSED | Verified basic variable cost modification |
| `test_09` | Multiple Nonbasic Objective | PASSED | Verified dual feasibility and unchanged objective |
| `test_10` | Multiple Basic Objective | PASSED | Verified $y' = B^{-T} c_B'$ and reduced costs |
| `test_11` | Mixed Basic/Nonbasic Objective | PASSED | Combined objective perturbations verified |
| `test_12` | Exact Reduced Cost Verification | PASSED | Large objective perturbation violating optimality condition |
| `test_13` | Combined RHS + Objective | PASSED | Simultaneous $b$ and $c$ modifications verified |
| `test_14` | Degenerate Basis | PASSED | Handled without numerical instability |
| `test_15` | Alternate Optimum | PASSED | Handled zero reduced costs |
| `test_16` | Presolve-Fixed Variable | PASSED | Handled all-fixed and partially fixed presolve models |
| `test_17` | Presolve-Redundant Constraint | PASSED | Evaluated activity against postsolve values |
| `test_18` | Negative RHS Sign-Flipped Row | PASSED | Inversion correctly maps perturbation sign |
| `test_19` | Variable Bound Shift | PASSED | Non-zero lower bounds mapped back to original space |
| `test_20` | Scenario Object Validation | PASSED | Unknown variables and constraints raise KeyError |
| `test_21` | Scenario Comparison | PASSED | Multi-scenario evaluation and formatted summary |
| `test_22` | Break-Even Inside Interval | PASSED | Computed exact required delta and validated reachability |
| `test_23` | Break-Even Outside Interval | PASSED | Correctly marked target unreachable within basis |
| `test_24` | MILP Rejection | PASSED | Integer variables strictly rejected with clear error |
| `test_25` | Infeasible Model Rejection | PASSED | Infeasible LP rejected cleanly |
| `test_26` | Unbounded Model Rejection | PASSED | Unbounded LP rejected cleanly |
| `test_27` | Missing Basis Rejection | PASSED | Unsolved model rejected cleanly |
| `test_28` | Model Immutability | PASSED | Model constraints and objective unmutated after re-solve |
| `test_29` | Explicit Re-Solve Behavior | PASSED | Verified `resolve_if_needed=False` vs `True` |
| `test_30` | Backward Compatibility | PASSED | All Stage 8.45 1D sensitivity APIs intact |

### Full Test Suite Execution:
```text
1181 passed, 21 warnings in 20.88s
```
- Total test count: 1181
- Failures: 0
- Regressions: 0

---

## 7. Mathematical Limitations & Assumptions

1. **Conservative Nature of 100% Rule**:
   The 100% Rule is a sufficient condition based on the $L_1$-norm geometry of the parameter bounds. Points outside the 100% Rule boundary may still preserve the optimal basis. Users should rely on `exact_basis_preserved` for definitive verification.
2. **Current-Basis Horizon**:
   When `basis_preserved = False`, analytical evaluation does not guess the new optimal basis. Finding the new optimum outside the basis-preserving cone requires a full simplex re-solve (`resolve_if_needed=True`).
3. **Linearity of Break-Even**:
   OptiForge's break-even analysis is strictly linear within the optimal basis region ($\Delta z = \lambda_i \Delta b_i$ or $\Delta z = x_j^* \Delta c_j$). Non-linear break-even across multiple basis changes is deliberately out of scope.

---

## 8. Recommendation for Stage 8.47

With Stage 8.46 complete:
- Numerical Core is verified and frozen (dense $m \le 1000$).
- Unified pipeline, BasisInfo, and re-optimization hooks are in place.
- 1D sensitivity, 100% Rule, multi-parameter scenarios, and break-even analysis are complete and verified.

**Recommended Next Focus (Stage 8.47)**:
- **Dual Simplex Warm-Start Re-optimization**:
  Leverage the verified basis preservation and perturbation machinery to warm-start the Dual Simplex solver directly from the perturbed basis matrix when primal feasibility is violated, avoiding cold Phase-I simplex re-solves.
