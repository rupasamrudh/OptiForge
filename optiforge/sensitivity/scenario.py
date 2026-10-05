"""What-if scenario analysis, exact basis-preservation checking, and break-even analysis."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from optiforge.core.basis import BasisInfo
from optiforge.core.solution import Solution, SolutionStatus
from optiforge.model.constraint import Constraint, ConstraintSense
from optiforge.model.model import Model
from optiforge.model.objective import Objective, ObjectiveSense
from optiforge.model.variable import Variable
from optiforge.sensitivity.calculator import compute_sensitivity
from optiforge.sensitivity.multi import RuleStatus, evaluate_100_percent_rule
from optiforge.sensitivity.range import SensitivityRange
from optiforge.sensitivity.report import SensitivityReport


@dataclass(frozen=True)
class Scenario:
    """Specification of a what-if scenario with simultaneous parameter perturbations.

    Attributes
    ----------
    name : str
        Human-readable scenario identifier.
    rhs_changes : dict[str, float]
        Constraint RHS perturbations {constraint_name: delta_b}.
    objective_changes : dict[str, float]
        Objective coefficient perturbations {variable_name: delta_c}.
    metadata : dict[str, Any]
        Additional user metadata.
    """

    name: str = "scenario"
    rhs_changes: dict[str, float] = field(default_factory=dict)
    objective_changes: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioResult:
    """Result of evaluating a Scenario against an optimal LP model and basis.

    Attributes
    ----------
    scenario_name : str
        Name of the evaluated scenario.
    basis_preserved : bool
        True if the current optimal basis remains optimal under the perturbations.
    primal_feasible : bool
        True if the current basis remains primal feasible (x_B >= 0).
    dual_feasible : bool
        True if the current basis remains dual feasible (reduced costs optimal).
    hundred_percent_rule_status : str
        "within_region", "outside_region", or "indeterminate".
    hundred_percent_rule_rhs_sum : float
        Sum of RHS percentage ratios under 100% Rule.
    hundred_percent_rule_obj_sum : float
        Sum of objective percentage ratios under 100% Rule.
    objective_value : float | None
        Optimal objective value under the scenario (analytical if basis preserved, or re-solved).
    objective_delta : float | None
        Change in objective value relative to the baseline solve.
    variable_values : dict[str, float]
        Variable assignments under the scenario.
    changed_parameters : dict[str, float]
        Combined dictionary of all parameter perturbations applied.
    limiting_parameters : list[str]
        Parameters that constrain basis preservation or drive 100% Rule sum.
    solver_resolved : bool
        True if a full solver re-solve was executed because basis was not preserved.
    re_solve_solution : Solution | None
        Full Solution object if re-solve was performed.
    status : str
        Status message ("basis_optimal", "re_solved", "infeasible_basis", etc.).
    message : str
        Human-readable description.
    metadata : dict[str, Any]
        Additional diagnostics.
    """

    scenario_name: str
    basis_preserved: bool
    primal_feasible: bool
    dual_feasible: bool
    hundred_percent_rule_status: str
    hundred_percent_rule_rhs_sum: float
    hundred_percent_rule_obj_sum: float
    objective_value: float | None
    objective_delta: float | None
    variable_values: dict[str, float]
    changed_parameters: dict[str, float]
    limiting_parameters: list[str] = field(default_factory=list)
    solver_resolved: bool = False
    re_solve_solution: Any | None = None
    status: str = "evaluated"
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    solve_method: str = "analytical"
    warm_started: bool = False
    initial_basis_preserved: bool = False
    initial_primal_feasible: bool = False
    initial_dual_feasible: bool = False
    reoptimization_iterations: int | None = None


@dataclass
class ScenarioComparison:
    """Comparison table and metrics across multiple evaluated scenarios."""

    baseline_objective: float | None
    scenarios: dict[str, ScenarioResult] = field(default_factory=dict)

    def summary(self) -> str:
        """Return formatted summary table comparing all scenarios."""
        lines = ["=== SCENARIO COMPARISON REPORT ==="]
        base_str = f"{self.baseline_objective:.4f}" if self.baseline_objective is not None else "None"
        lines.append(f"Baseline Objective: {base_str}\n")
        lines.append(
            f"{'Scenario':<18} {'Basis Preserved':<16} {'Obj Value':<12} {'Obj Delta':<12} {'Method':<18} {'Warm-Started'}"
        )
        lines.append("-" * 90)
        for name, res in self.scenarios.items():
            pres_str = "YES" if res.basis_preserved else "NO"
            obj_str = f"{res.objective_value:.4f}" if res.objective_value is not None else "N/A"
            delta_str = f"{res.objective_delta:+.4f}" if res.objective_delta is not None else "N/A"
            ws_str = "YES" if res.warm_started else "NO"
            lines.append(
                f"{name:<18} {pres_str:<16} {obj_str:<12} {delta_str:<12} {res.solve_method:<18} {ws_str}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class BreakEvenResult:
    """Outcome of 1D break-even analysis for a specific target objective value."""

    parameter_name: str
    parameter_type: str
    current_parameter_value: float
    target_objective_value: float
    current_objective_value: float
    required_delta: float | None
    required_parameter_value: float | None
    is_reachable_within_basis: bool
    sensitivity_range: SensitivityRange
    message: str = ""


def evaluate_scenario(
    model: Model,
    scenario: Scenario,
    basis_info: BasisInfo | None = None,
    sensitivity_report: SensitivityReport | None = None,
    resolve_if_needed: bool = False,
    feasibility_tol: float = 1e-7,
    reduced_cost_tol: float = 1e-7,
) -> ScenarioResult:
    """Analytically evaluate a what-if scenario against an optimal continuous LP basis.

    Parameters
    ----------
    model : Model
        The solved continuous optimization model.
    scenario : Scenario
        What-if scenario with RHS and/or objective perturbations.
    basis_info : BasisInfo | None, optional
        Optimal basis info (default model.last_basis).
    sensitivity_report : SensitivityReport | None, optional
        Precomputed SensitivityReport (default model.last_sensitivity).
    resolve_if_needed : bool, optional
        If True, re-solve the LP using the full solver pipeline if the current
        basis does not remain optimal.
    feasibility_tol : float, optional
        Primal feasibility tolerance (default 1e-7).
    reduced_cost_tol : float, optional
        Reduced cost optimality tolerance (default 1e-7).

    Returns
    -------
    ScenarioResult
        Detailed evaluation of basis preservation, 100% Rule status, and objective delta.
    """
    if any(v.is_integer() for v in model.variables.values()):
        raise ValueError("Sensitivity analysis is only supported for continuous LP models.")

    binfo = basis_info or getattr(model, "last_basis", None)
    if binfo is None:
        raise ValueError("No optimal basis available. Model must be solved before evaluating scenarios.")
    if not binfo.is_optimal:
        raise ValueError("Scenario evaluation requires an optimal basis.")

    # Validate parameter names in scenario
    for cname in scenario.rhs_changes:
        if cname not in model.constraints:
            raise KeyError(f"Constraint '{cname}' in rhs_changes does not exist in model.")
    for vname in scenario.objective_changes:
        if vname not in model.variables:
            raise KeyError(f"Variable '{vname}' in objective_changes does not exist in model.")

    # 1. 1D Sensitivity & 100% Rule
    report = sensitivity_report or getattr(model, "last_sensitivity", None)
    if report is None:
        report = compute_sensitivity(model=model, basis_info=binfo)

    rule_results = evaluate_100_percent_rule(
        report=report,
        rhs_changes=scenario.rhs_changes,
        objective_changes=scenario.objective_changes,
        tol=feasibility_tol,
    )

    rhs_sum = rule_results["rhs"].total_percentage_sum if "rhs" in rule_results else 0.0
    obj_sum = rule_results["objective"].total_percentage_sum if "objective" in rule_results else 0.0

    rhs_within = ("rhs" not in rule_results) or (rule_results["rhs"].status == RuleStatus.WITHIN_REGION)
    obj_within = ("objective" not in rule_results) or (rule_results["objective"].status == RuleStatus.WITHIN_REGION)

    if rhs_within and obj_within:
        hundred_pct_status = "within_region"
    else:
        hundred_pct_status = "outside_region"

    # 2. Exact Basis Feasibility Check
    was_presolved = bool(binfo.metadata.get("presolved", False)) or bool(binfo.metadata.get("presolved_all_fixed", False))
    presolve_res = None
    if was_presolved:
        from optiforge.presolve import presolve_model

        presolve_res = presolve_model(model)
        target_model = presolve_res.reduced_model
    else:
        target_model = model

    # Edge case: All variables fixed in presolve
    if binfo.metadata.get("presolved_all_fixed") or target_model is None or target_model.num_variables == 0:
        var_vals = binfo.metadata.get("variable_values")
        if not var_vals:
            from optiforge.presolve import postsolve_solution

            if presolve_res is not None:
                var_vals = postsolve_solution({}, presolve_res)
            else:
                var_vals = {v: float(model.variables[v].lower_bound) for v in model.variables}

        primal_feasible = True
        limiting_params: list[str] = []
        for cname, delta in scenario.rhs_changes.items():
            con = model.constraints[cname]
            act = sum(float(con.coefficients.get(v, 0.0)) * var_vals.get(v, 0.0) for v in con.coefficients)
            new_rhs = float(con.rhs) + delta
            if con.sense is ConstraintSense.LESS_EQUAL and act > new_rhs + feasibility_tol:
                primal_feasible = False
                limiting_params.append(cname)
            elif con.sense is ConstraintSense.GREATER_EQUAL and act < new_rhs - feasibility_tol:
                primal_feasible = False
                limiting_params.append(cname)
            elif con.sense is ConstraintSense.EQUAL and abs(act - new_rhs) > feasibility_tol:
                primal_feasible = False
                limiting_params.append(cname)

        basis_preserved = primal_feasible
        base_obj = binfo.objective_value
        changed_params = {}
        for cname, d in scenario.rhs_changes.items():
            changed_params[f"rhs:{cname}"] = d
        for vname, d in scenario.objective_changes.items():
            changed_params[f"obj:{vname}"] = d

        if basis_preserved:
            new_obj = 0.0
            for vname, val in var_vals.items():
                base_c = float(model.objective.coefficients.get(vname, 0.0))
                delta_c = scenario.objective_changes.get(vname, 0.0)
                new_obj += (base_c + delta_c) * val
            obj_delta = (new_obj - base_obj) if base_obj is not None else None
            return ScenarioResult(
                scenario_name=scenario.name,
                basis_preserved=True,
                primal_feasible=True,
                dual_feasible=True,
                hundred_percent_rule_status=hundred_pct_status,
                hundred_percent_rule_rhs_sum=rhs_sum,
                hundred_percent_rule_obj_sum=obj_sum,
                objective_value=new_obj,
                objective_delta=obj_delta,
                variable_values=var_vals,
                changed_parameters=changed_params,
                limiting_parameters=limiting_params,
                solver_resolved=False,
                re_solve_solution=None,
                status="basis_optimal",
                message="Optimal basis preserved analytically (presolved fixed variables).",
                solve_method="analytical",
                warm_started=False,
                initial_basis_preserved=True,
                initial_primal_feasible=True,
                initial_dual_feasible=True,
                reoptimization_iterations=0,
            )
        elif resolve_if_needed:
            cloned = _clone_model_with_scenario(model, scenario)
            new_sol = cloned.solve()
            new_obj = new_sol.objective_value
            obj_delta = (new_obj - base_obj) if (new_obj is not None and base_obj is not None) else None
            return ScenarioResult(
                scenario_name=scenario.name,
                basis_preserved=False,
                primal_feasible=False,
                dual_feasible=True,
                hundred_percent_rule_status=hundred_pct_status,
                hundred_percent_rule_rhs_sum=rhs_sum,
                hundred_percent_rule_obj_sum=obj_sum,
                objective_value=new_obj,
                objective_delta=obj_delta,
                variable_values=new_sol.variable_values,
                changed_parameters=changed_params,
                limiting_parameters=limiting_params,
                solver_resolved=True,
                re_solve_solution=new_sol,
                status="re_solved",
                message="Basis not preserved; LP successfully re-solved with modified parameters.",
                solve_method="cold_resolve",
                warm_started=False,
                initial_basis_preserved=False,
                initial_primal_feasible=False,
                initial_dual_feasible=True,
                reoptimization_iterations=None,
            )
        else:
            return ScenarioResult(
                scenario_name=scenario.name,
                basis_preserved=False,
                primal_feasible=False,
                dual_feasible=True,
                hundred_percent_rule_status=hundred_pct_status,
                hundred_percent_rule_rhs_sum=rhs_sum,
                hundred_percent_rule_obj_sum=obj_sum,
                objective_value=None,
                objective_delta=None,
                variable_values={},
                changed_parameters=changed_params,
                limiting_parameters=limiting_params,
                solver_resolved=False,
                re_solve_solution=None,
                status="basis_infeasible",
                message="Basis does not remain optimal. Re-solve required (use resolve_if_needed=True).",
                solve_method="analytical",
                warm_started=False,
                initial_basis_preserved=False,
                initial_primal_feasible=False,
                initial_dual_feasible=True,
                reoptimization_iterations=None,
            )

    from optiforge.lp.standard_form import to_standard_form

    msf = to_standard_form(target_model)
    A = msf.A
    b = msf.b
    c = msf.c
    m_sf, n_sf = A.shape
    basis_indices = binfo.get_basis_indices()

    B = A[:, basis_indices]
    try:
        B_inv = np.linalg.inv(B)
    except np.linalg.LinAlgError as err:
        raise ValueError(f"Basis matrix is singular: {err}") from err

    limiting_params: list[str] = []
    primal_feasible = True
    dual_feasible = True

    # -- RHS Exact Check --
    delta_b_sf = np.zeros(m_sf, dtype=np.float64)
    target_constrs = list(target_model.constraints.values())
    target_constrs_by_name = {con.name: i for i, con in enumerate(target_constrs)}
    full_var_vals = binfo.metadata.get("variable_values") or binfo.primal_basic_values

    for cname, delta in scenario.rhs_changes.items():
        if presolve_res is not None and cname not in target_constrs_by_name:
            # Removed constraint in presolve: check activity condition
            con = model.constraints[cname]
            act = sum(float(con.coefficients.get(v, 0.0)) * full_var_vals.get(v, 0.0) for v in con.coefficients)
            new_rhs = float(con.rhs) + delta
            if con.sense is ConstraintSense.LESS_EQUAL and act > new_rhs + feasibility_tol:
                primal_feasible = False
                limiting_params.append(cname)
            elif con.sense is ConstraintSense.GREATER_EQUAL and act < new_rhs - feasibility_tol:
                primal_feasible = False
                limiting_params.append(cname)
        elif cname in target_constrs_by_name:
            row = target_constrs_by_name[cname]
            con = target_constrs[row]
            shift_adj = sum(
                float(con.coefficients.get(vname, 0.0))
                * msf.variable_shifts.get(msf.original_variable_mapping.get(vname, -1), 0.0)
                for vname in con.coefficients
            )
            raw_rhs = float(con.rhs) - shift_adj
            flip_sign = -1.0 if raw_rhs < 0.0 else 1.0
            delta_b_sf[row] += flip_sign * delta

    if primal_feasible and len(scenario.rhs_changes) > 0:
        b_prime = b + delta_b_sf
        x_B_prime = np.dot(B_inv, b_prime)
        for i in range(m_sf):
            if x_B_prime[i] < -feasibility_tol:
                primal_feasible = False
                col = basis_indices[i]
                vname = msf.variable_names[col]
                limiting_params.append(f"basic_{vname}")
                break
    else:
        x_B_prime = np.dot(B_inv, b)

    # -- Objective Exact Check --
    delta_c_sf = np.zeros(n_sf, dtype=np.float64)
    for vname, delta in scenario.objective_changes.items():
        if vname in msf.original_variable_mapping:
            col = msf.original_variable_mapping[vname]
            delta_c_sf[col] += delta

    if len(scenario.objective_changes) > 0:
        c_prime = c + delta_c_sf
        c_B_prime = c_prime[basis_indices]
        y_prime = np.dot(B_inv.T, c_B_prime)
        r_prime = c_prime - np.dot(A.T, y_prime)

        basis_set = set(basis_indices)
        art_set = set(getattr(msf, "artificial_indices", []))
        nonbasic = [j for j in range(n_sf) if j not in basis_set and j not in art_set]

        is_min = (target_model.objective.sense is ObjectiveSense.MINIMIZE)
        for j in nonbasic:
            if is_min and r_prime[j] < -reduced_cost_tol:
                dual_feasible = False
                limiting_params.append(msf.variable_names[j])
                break
            elif not is_min and r_prime[j] > reduced_cost_tol:
                dual_feasible = False
                limiting_params.append(msf.variable_names[j])
                break

    basis_preserved = primal_feasible and dual_feasible

    # Combine changed parameters
    changed_params = {}
    for cname, d in scenario.rhs_changes.items():
        changed_params[f"rhs:{cname}"] = d
    for vname, d in scenario.objective_changes.items():
        changed_params[f"obj:{vname}"] = d

    # 3. Objective & Solution Construction
    base_obj = binfo.objective_value
    if basis_preserved:
        # Construct full standard-form variable values x'
        x_sf_prime = np.zeros(n_sf, dtype=np.float64)
        for i, col in enumerate(basis_indices):
            x_sf_prime[col] = max(0.0, float(x_B_prime[i]))

        # Map back to target_model variable values
        target_vals = {}
        for vname in target_model.variables:
            col = msf.original_variable_mapping.get(vname)
            if col is not None:
                shift = msf.variable_shifts.get(col, 0.0)
                target_vals[vname] = float(x_sf_prime[col]) + shift

        # Map through presolve if needed
        if presolve_res is not None:
            from optiforge.presolve import postsolve_solution

            var_vals = postsolve_solution(target_vals, presolve_res)
        else:
            var_vals = target_vals

        # Compute new objective value
        new_obj = 0.0
        for vname, val in var_vals.items():
            base_coeff = float(model.objective.coefficients.get(vname, 0.0))
            delta_coeff = scenario.objective_changes.get(vname, 0.0)
            new_obj += (base_coeff + delta_coeff) * val

        obj_delta = new_obj - base_obj if base_obj is not None else None

        return ScenarioResult(
            scenario_name=scenario.name,
            basis_preserved=True,
            primal_feasible=True,
            dual_feasible=True,
            hundred_percent_rule_status=hundred_pct_status,
            hundred_percent_rule_rhs_sum=rhs_sum,
            hundred_percent_rule_obj_sum=obj_sum,
            objective_value=new_obj,
            objective_delta=obj_delta,
            variable_values=var_vals,
            changed_parameters=changed_params,
            limiting_parameters=limiting_params,
            solver_resolved=False,
            re_solve_solution=None,
            status="basis_optimal",
            message="Optimal basis preserved analytically without re-solving.",
            solve_method="analytical",
            warm_started=False,
            initial_basis_preserved=True,
            initial_primal_feasible=True,
            initial_dual_feasible=True,
            reoptimization_iterations=0,
        )

    # Basis is NOT preserved
    if resolve_if_needed:
        # Build a safe clone of model with scenario modifications
        cloned_model = _clone_model_with_scenario(model, scenario)

        # Attempt warm-start solve using model.last_basis
        binfo = getattr(model, "last_basis", None)
        new_sol = None
        if binfo is not None and getattr(binfo, "is_lp", False) and getattr(binfo, "is_optimal", False):
            try:
                new_sol = cloned_model.solve(basis=binfo)
            except Exception:
                new_sol = None

        if new_sol is None:
            new_sol = cloned_model.solve()

        new_obj = new_sol.objective_value
        obj_delta = (new_obj - base_obj) if (new_obj is not None and base_obj is not None) else None

        meta = getattr(new_sol, "solver_metadata", {}) or {}
        reason = getattr(new_sol, "solver_reason", "") or ""
        solver_name = getattr(new_sol, "solver_selected", "") or ""
        is_warm = meta.get("warm_started", False) or (reason in ("warm_start_primal_feasible", "warm_start_dual_feasible"))

        reopt_iters = meta.get("reoptimization_iterations")
        if reopt_iters is None and is_warm:
            reopt_iters = new_sol.iterations

        if is_warm and (solver_name == "dual_simplex" or reason == "warm_start_dual_feasible"):
            solve_method = "dual_warm_start"
        elif is_warm and (solver_name == "revised_simplex" or reason == "warm_start_primal_feasible"):
            solve_method = "primal_warm_start"
        else:
            solve_method = "cold_resolve"
            is_warm = False
            reopt_iters = None

        return ScenarioResult(
            scenario_name=scenario.name,
            basis_preserved=False,
            primal_feasible=primal_feasible,
            dual_feasible=dual_feasible,
            hundred_percent_rule_status=hundred_pct_status,
            hundred_percent_rule_rhs_sum=rhs_sum,
            hundred_percent_rule_obj_sum=obj_sum,
            objective_value=new_obj,
            objective_delta=obj_delta,
            variable_values=new_sol.variable_values,
            changed_parameters=changed_params,
            limiting_parameters=limiting_params,
            solver_resolved=True,
            re_solve_solution=new_sol,
            status="re_solved",
            message="Basis not preserved; LP successfully re-solved with modified parameters.",
            solve_method=solve_method,
            warm_started=is_warm,
            initial_basis_preserved=False,
            initial_primal_feasible=primal_feasible,
            initial_dual_feasible=dual_feasible,
            reoptimization_iterations=reopt_iters,
        )

    # Basis not preserved and re-solve not requested
    return ScenarioResult(
        scenario_name=scenario.name,
        basis_preserved=False,
        primal_feasible=primal_feasible,
        dual_feasible=dual_feasible,
        hundred_percent_rule_status=hundred_pct_status,
        hundred_percent_rule_rhs_sum=rhs_sum,
        hundred_percent_rule_obj_sum=obj_sum,
        objective_value=None,
        objective_delta=None,
        variable_values={},
        changed_parameters=changed_params,
        limiting_parameters=limiting_params,
        solver_resolved=False,
        re_solve_solution=None,
        status="basis_infeasible",
        message="Basis does not remain optimal. Re-solve required (use resolve_if_needed=True).",
        solve_method="analytical",
        warm_started=False,
        initial_basis_preserved=False,
        initial_primal_feasible=primal_feasible,
        initial_dual_feasible=dual_feasible,
        reoptimization_iterations=None,
    )


def compare_scenarios(
    model: Model,
    scenarios: list[Scenario],
    resolve_if_needed: bool = False,
) -> ScenarioComparison:
    """Evaluate and compare multiple scenarios against an optimal LP model."""
    baseline_obj = model.last_basis.objective_value if model.last_basis else None
    results: dict[str, ScenarioResult] = {}
    for sc in scenarios:
        res = evaluate_scenario(model=model, scenario=sc, resolve_if_needed=resolve_if_needed)
        results[sc.name] = res

    return ScenarioComparison(baseline_objective=baseline_obj, scenarios=results)


def compute_break_even(
    model: Model,
    parameter: str,
    target_value: float,
    target: str = "objective_value",
    parameter_type: str = "auto",
    direction: str | None = None,
) -> BreakEvenResult:
    """Calculate 1D break-even parameter delta required to achieve a target objective value.

    Parameters
    ----------
    model : Model
        Solved continuous LP model.
    parameter : str
        Constraint name or variable name.
    target_value : float
        Desired target value.
    target : str, optional
        Target metric (default "objective_value"). Unsupported targets are rejected.
    parameter_type : str, optional
        "rhs", "objective", or "auto" (default "auto" inspects model).
    direction : str | None, optional
        Optional expected direction ("increase", "decrease").

    Returns
    -------
    BreakEvenResult
        Result containing required delta, feasibility within current basis, and sensitivity range.
    """
    if target != "objective_value":
        raise ValueError(
            f"Unsupported break-even target: '{target}'. Only 'objective_value' is supported."
        )
    if any(v.is_integer() for v in model.variables.values()):
        raise ValueError("Sensitivity analysis is only supported for continuous LP models.")

    binfo = getattr(model, "last_basis", None)
    if binfo is None or not binfo.is_optimal:
        raise ValueError("Model must be solved to optimality before performing break-even analysis.")

    report = getattr(model, "last_sensitivity", None)
    if report is None:
        report = compute_sensitivity(model=model, basis_info=binfo)

    base_obj = float(binfo.objective_value) if binfo.objective_value is not None else 0.0
    delta_obj_needed = target_value - base_obj

    # Auto-detect parameter type
    p_type = parameter_type
    if p_type == "auto":
        if parameter in model.constraints:
            p_type = "rhs"
        elif parameter in model.variables:
            p_type = "objective"
        else:
            raise KeyError(
                f"Parameter '{parameter}' not found as a constraint or variable in model."
            )

    if p_type == "rhs":
        if parameter not in report.rhs:
            raise KeyError(f"Constraint '{parameter}' not found in model.")
        rng = report.rhs[parameter]
        lambda_val = binfo.dual_multipliers.get(parameter, 0.0)
        curr_param = rng.current_value

        if abs(lambda_val) < 1e-9:
            # Non-binding constraint: shadow price is zero
            return BreakEvenResult(
                parameter_name=parameter,
                parameter_type="rhs",
                current_parameter_value=curr_param,
                target_objective_value=target_value,
                current_objective_value=base_obj,
                required_delta=None,
                required_parameter_value=None,
                is_reachable_within_basis=False,
                sensitivity_range=rng,
                message=f"Constraint '{parameter}' is non-binding (shadow price = 0); target cannot be reached.",
            )

        req_delta = delta_obj_needed / lambda_val
        req_param = curr_param + req_delta
        is_reachable = rng.contains(req_param)
        if direction is not None:
            dir_lower = direction.lower()
            if dir_lower in ("increase", "inc", "+", "up") and req_delta < -1e-9:
                is_reachable = False
            elif dir_lower in ("decrease", "dec", "-", "down") and req_delta > 1e-9:
                is_reachable = False

        msg = "Target reachable within optimal basis." if is_reachable else "Target exceeds allowable basis range or violates specified direction."

        return BreakEvenResult(
            parameter_name=parameter,
            parameter_type="rhs",
            current_parameter_value=curr_param,
            target_objective_value=target_value,
            current_objective_value=base_obj,
            required_delta=req_delta,
            required_parameter_value=req_param,
            is_reachable_within_basis=is_reachable,
            sensitivity_range=rng,
            message=msg,
        )

    elif p_type == "objective":
        if parameter not in report.objective:
            raise KeyError(f"Variable '{parameter}' not found in model.")
        rng = report.objective[parameter]
        curr_param = rng.current_value
        x_val = binfo.primal_basic_values.get(parameter, 0.0)

        if abs(x_val) < 1e-9:
            # Nonbasic variable: changing its cost does not change current objective
            return BreakEvenResult(
                parameter_name=parameter,
                parameter_type="objective",
                current_parameter_value=curr_param,
                target_objective_value=target_value,
                current_objective_value=base_obj,
                required_delta=None,
                required_parameter_value=None,
                is_reachable_within_basis=False,
                sensitivity_range=rng,
                message=f"Variable '{parameter}' is nonbasic (value = 0); cost change does not affect objective.",
            )

        req_delta = delta_obj_needed / x_val
        req_param = curr_param + req_delta
        is_reachable = rng.contains(req_param)
        if direction is not None:
            dir_lower = direction.lower()
            if dir_lower in ("increase", "inc", "+", "up") and req_delta < -1e-9:
                is_reachable = False
            elif dir_lower in ("decrease", "dec", "-", "down") and req_delta > 1e-9:
                is_reachable = False

        msg = "Target reachable within optimal basis." if is_reachable else "Target exceeds allowable basis range or violates specified direction."

        return BreakEvenResult(
            parameter_name=parameter,
            parameter_type="objective",
            current_parameter_value=curr_param,
            target_objective_value=target_value,
            current_objective_value=base_obj,
            required_delta=req_delta,
            required_parameter_value=req_param,
            is_reachable_within_basis=is_reachable,
            sensitivity_range=rng,
            message=msg,
        )
    else:
        raise ValueError(f"Unknown parameter_type: '{p_type}'. Must be 'rhs', 'objective', or 'auto'.")


def _clone_model_with_scenario(model: Model, scenario: Scenario) -> Model:
    """Clone a Model applying scenario modifications without mutating the original."""
    cloned = Model(name=f"{model.name}_{scenario.name}")
    for v in model.variables.values():
        cloned.add_variable(
            Variable(
                name=v.name,
                lower_bound=float(v.lower_bound),
                upper_bound=float(v.upper_bound),
                variable_type=v.variable_type,
            )
        )

    # Cloned objective
    new_obj_coeffs = dict(model.objective.coefficients)
    for vname, delta in scenario.objective_changes.items():
        new_obj_coeffs[vname] = new_obj_coeffs.get(vname, 0.0) + delta
    cloned.set_objective(Objective(coefficients=new_obj_coeffs, sense=model.objective.sense))

    # Cloned constraints
    for con in model.constraints.values():
        rhs_mod = scenario.rhs_changes.get(con.name, 0.0)
        cloned.add_constraint(
            Constraint(
                name=con.name,
                coefficients=dict(con.coefficients),
                sense=con.sense,
                rhs=float(con.rhs) + rhs_mod,
            )
        )

    return cloned
