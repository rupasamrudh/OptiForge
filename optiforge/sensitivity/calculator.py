"""LP post-optimality sensitivity analysis calculator."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from optiforge.core.basis import BasisInfo
from optiforge.core.solution import Solution, SolutionStatus
from optiforge.model.constraint import ConstraintSense
from optiforge.model.model import Model
from optiforge.model.objective import ObjectiveSense
from optiforge.sensitivity.range import SensitivityRange
from optiforge.sensitivity.report import SensitivityReport


def compute_sensitivity(
    model: Model,
    solution: Solution | None = None,
    basis_info: BasisInfo | None = None,
    feasibility_tol: float = 1e-7,
) -> SensitivityReport:
    """Compute exact LP sensitivity analysis for RHS values and objective coefficients.

    Parameters
    ----------
    model : Model
        The user's original optimization model.
    solution : Solution | None, optional
        The solved Solution instance. If supplied, status must be OPTIMAL.
    basis_info : BasisInfo | None, optional
        The optimal BasisInfo. If None, retrieved from solution or model.
    feasibility_tol : float, optional
        Numerical tolerance for feasibility and bound clamping (default 1e-7).

    Returns
    -------
    SensitivityReport
        Report containing allowable variation ranges for each original constraint and variable.

    Raises
    ------
    ValueError
        If model contains integer variables, is not solved to optimality, or lacks a valid basis.
    """
    # 1. Continuous LP & Status Validations
    if any(v.is_integer() for v in model.variables.values()):
        raise ValueError("Sensitivity analysis is supported for continuous LPs only.")

    if solution is not None and solution.status is not SolutionStatus.OPTIMAL:
        raise ValueError(
            f"Sensitivity analysis requires an optimal solution (got {solution.status.value})."
        )

    # 2. Extract and validate BasisInfo
    binfo = basis_info
    if binfo is None and solution is not None:
        binfo = solution.basis_info
    if binfo is None:
        binfo = getattr(model, "last_basis", None)

    if binfo is None:
        raise ValueError("No optimal basis information available for sensitivity analysis.")
    if not binfo.is_lp:
        raise ValueError("Sensitivity analysis is supported for continuous LPs only.")
    if not binfo.is_optimal:
        raise ValueError("Sensitivity analysis requires an optimal basis.")

    # 3. Presolve space reconciliation
    was_presolved = bool(binfo.metadata.get("presolved", False)) or bool(binfo.metadata.get("presolved_all_fixed", False))
    presolve_res = None
    if was_presolved:
        from optiforge.presolve import presolve_model

        presolve_res = presolve_model(model)
        target_model = presolve_res.reduced_model
    else:
        target_model = model

    # Edge case: All variables fixed in presolve
    if target_model is None or target_model.num_variables == 0:
        rhs_ranges: dict[str, SensitivityRange] = {}
        for cname, con in model.constraints.items():
            curr_rhs = float(con.rhs)
            rhs_ranges[cname] = SensitivityRange(
                name=cname,
                current_value=curr_rhs,
                allowable_decrease=0.0,
                allowable_increase=0.0,
                lower_bound=curr_rhs,
                upper_bound=curr_rhs,
                metadata={"presolved_all_fixed": True},
            )
        obj_ranges: dict[str, SensitivityRange] = {}
        for vname in model.variables:
            curr_c = float(model.objective.coefficients.get(vname, 0.0))
            obj_ranges[vname] = SensitivityRange(
                name=vname,
                current_value=curr_c,
                allowable_decrease=math.inf,
                allowable_increase=math.inf,
                lower_bound=-math.inf,
                upper_bound=math.inf,
                is_unbounded_below=True,
                is_unbounded_above=True,
                metadata={"presolved_all_fixed": True},
            )
        return SensitivityReport(
            rhs=rhs_ranges,
            objective=obj_ranges,
            basis_info=binfo,
            solver_selected=binfo.solver_selected,
            is_optimal=True,
            metadata={"presolved_all_fixed": True},
        )

    # 4. Standard-form conversion of target_model
    from optiforge.lp.standard_form import to_standard_form

    msf = to_standard_form(target_model)
    A = msf.A
    b = msf.b
    c = msf.c
    m_sf, n_sf = A.shape

    basis_indices = binfo.get_basis_indices()
    if len(basis_indices) != m_sf:
        raise ValueError(
            f"Basis dimension mismatch: basis length {len(basis_indices)} != standard-form rows {m_sf}."
        )

    # 5. Factorize basis matrix and compute B^{-1}
    B = A[:, basis_indices]
    try:
        B_inv = np.linalg.inv(B)
    except np.linalg.LinAlgError as err:
        raise ValueError(f"Basis matrix is numerically singular: {err}") from err

    # Primal basic values x_B = B^{-1} b
    x_B = np.dot(B_inv, b)
    for i in range(m_sf):
        if -feasibility_tol <= x_B[i] < 0.0:
            x_B[i] = 0.0

    is_degenerate = bool(np.any(np.abs(x_B) <= feasibility_tol))

    # Dual multipliers y = B^{-T} c_B
    c_B = c[basis_indices]
    y = np.dot(B_inv.T, c_B)

    # Reduced costs r = c - A^T y
    r = c - np.dot(A.T, y)
    for col in basis_indices:
        r[col] = 0.0

    basis_set = set(basis_indices)
    art_set = set(getattr(msf, "artificial_indices", []))
    nonbasic_indices = [
        j for j in range(n_sf) if j not in basis_set and j not in art_set
    ]


    # 6. RHS Sensitivity in target_model
    target_rhs_ranges: dict[str, SensitivityRange] = {}
    target_constrs = list(target_model.constraints.values())
    m_target = len(target_constrs)

    for k in range(min(m_target, m_sf)):
        con = target_constrs[k]
        shift_adj = sum(
            float(con.coefficients.get(vname, 0.0))
            * msf.variable_shifts.get(msf.original_variable_mapping.get(vname, -1), 0.0)
            for vname in con.coefficients
        )
        raw_rhs = float(con.rhs) - shift_adj
        flip_sign = -1.0 if raw_rhs < 0.0 else 1.0

        beta_k = B_inv[:, k]
        delta_min = -math.inf
        delta_max = math.inf

        for i in range(m_sf):
            b_ki = float(beta_k[i])
            x_Bi = max(0.0, float(x_B[i]))
            if b_ki > 1e-9:
                bound = -x_Bi / b_ki
                if bound > delta_min:
                    delta_min = bound
            elif b_ki < -1e-9:
                bound = -x_Bi / b_ki
                if bound < delta_max:
                    delta_max = bound

        # Account for negative RHS normalization flip
        if flip_sign > 0.0:
            allowable_dec = -delta_min if delta_min > -math.inf else math.inf
            allowable_inc = delta_max if delta_max < math.inf else math.inf
        else:
            allowable_dec = delta_max if delta_max < math.inf else math.inf
            allowable_inc = -delta_min if delta_min > -math.inf else math.inf

        curr_rhs = float(con.rhs)
        lb = -math.inf if allowable_dec == math.inf else curr_rhs - allowable_dec
        ub = math.inf if allowable_inc == math.inf else curr_rhs + allowable_inc

        # Determine binding status from dual multiplier
        dual_val = binfo.dual_multipliers.get(con.name, 0.0)
        is_binding = abs(dual_val) > 1e-7

        target_rhs_ranges[con.name] = SensitivityRange(
            name=con.name,
            current_value=curr_rhs,
            allowable_decrease=allowable_dec,
            allowable_increase=allowable_inc,
            lower_bound=lb,
            upper_bound=ub,
            is_unbounded_below=(allowable_dec == math.inf),
            is_unbounded_above=(allowable_inc == math.inf),
            basis_remains_optimal=True,
            metadata={
                "binding": is_binding,
                "degenerate_basis": is_degenerate,
                "flip_sign": flip_sign,
                "dual_multiplier": dual_val,
            },
        )

    # 7. Presolve Inversion for RHS
    final_rhs_ranges: dict[str, SensitivityRange] = {}
    if presolve_res is None:
        final_rhs_ranges = target_rhs_ranges
    else:
        # Reconstruct primal solution in original coordinates for activity checks
        x_orig = {}
        for vname in model.variables:
            red_name = presolve_res.original_to_reduced.get(vname, vname)
            if red_name in binfo.primal_basic_values:
                x_orig[vname] = float(binfo.primal_basic_values[red_name])
            elif vname in presolve_res.fixed_variables:
                x_orig[vname] = float(presolve_res.fixed_variables[vname])
            else:
                x_orig[vname] = 0.0

        for cname, con in model.constraints.items():
            if cname in target_rhs_ranges:
                final_rhs_ranges[cname] = target_rhs_ranges[cname]
            else:
                # Removed constraint: evaluate activity
                act = sum(float(con.coefficients.get(v, 0.0)) * x_orig.get(v, 0.0) for v in con.coefficients)
                curr_rhs = float(con.rhs)
                if con.sense is ConstraintSense.LESS_EQUAL:
                    allowable_inc = math.inf
                    allowable_dec = max(0.0, curr_rhs - act)
                    lb = act
                    ub = math.inf
                    final_rhs_ranges[cname] = SensitivityRange(
                        name=cname,
                        current_value=curr_rhs,
                        allowable_decrease=allowable_dec,
                        allowable_increase=allowable_inc,
                        lower_bound=lb,
                        upper_bound=ub,
                        is_unbounded_below=False,
                        is_unbounded_above=True,
                        basis_remains_optimal=True,
                        metadata={"presolved_redundant": True, "slack": curr_rhs - act},
                    )
                elif con.sense is ConstraintSense.GREATER_EQUAL:
                    allowable_inc = max(0.0, act - curr_rhs)
                    allowable_dec = math.inf
                    lb = -math.inf
                    ub = act
                    final_rhs_ranges[cname] = SensitivityRange(
                        name=cname,
                        current_value=curr_rhs,
                        allowable_decrease=allowable_dec,
                        allowable_increase=allowable_inc,
                        lower_bound=lb,
                        upper_bound=ub,
                        is_unbounded_below=True,
                        is_unbounded_above=False,
                        basis_remains_optimal=True,
                        metadata={"presolved_redundant": True, "surplus": act - curr_rhs},
                    )
                else:
                    final_rhs_ranges[cname] = SensitivityRange(
                        name=cname,
                        current_value=curr_rhs,
                        allowable_decrease=0.0,
                        allowable_increase=0.0,
                        lower_bound=curr_rhs,
                        upper_bound=curr_rhs,
                        is_unbounded_below=False,
                        is_unbounded_above=False,
                        basis_remains_optimal=True,
                        metadata={"presolved_equality_fixed": True},
                    )

    # 8. Objective Coefficient Sensitivity in target_model
    target_obj_ranges: dict[str, SensitivityRange] = {}
    is_minimize = (target_model.objective.sense is ObjectiveSense.MINIMIZE)
    basis_col_to_pos = {col: pos for pos, col in enumerate(basis_indices)}

    for vname, var in target_model.variables.items():
        col = msf.original_variable_mapping.get(vname)
        if col is None:
            continue

        curr_c = float(model.objective.coefficients.get(vname, 0.0))

        if col in basis_col_to_pos:
            # BASIC VARIABLE: position p in basis
            p = basis_col_to_pos[col]
            gamma_p = B_inv[p, :]  # p-th row of B^{-1}
            delta_min = -math.inf
            delta_max = math.inf

            for j in nonbasic_indices:
                alpha_pj = float(np.dot(gamma_p, A[:, j]))
                rj = float(r[j])
                if is_minimize:
                    # Condition: r_j - delta * alpha_pj >= -tol
                    if -1e-7 <= rj < 0.0:
                        rj = 0.0
                    if alpha_pj > 1e-9:
                        bound = rj / alpha_pj
                        if bound < delta_max:
                            delta_max = bound
                    elif alpha_pj < -1e-9:
                        bound = rj / alpha_pj
                        if bound > delta_min:
                            delta_min = bound
                else:
                    # MAXIMIZE: Condition: r_j - delta * alpha_pj <= tol
                    if 0.0 < rj <= 1e-7:
                        rj = 0.0
                    if alpha_pj < -1e-9:
                        bound = rj / alpha_pj
                        if bound < delta_max:
                            delta_max = bound
                    elif alpha_pj > 1e-9:
                        bound = rj / alpha_pj
                        if bound > delta_min:
                            delta_min = bound

            allowable_dec = -delta_min if delta_min > -math.inf else math.inf
            allowable_inc = delta_max if delta_max < math.inf else math.inf

        else:
            # NONBASIC VARIABLE
            rj = float(r[col])
            if is_minimize:
                # Need r_j + delta >= 0 => delta >= -r_j
                if -1e-7 <= rj < 0.0:
                    rj = 0.0
                allowable_dec = max(0.0, rj)
                allowable_inc = math.inf
            else:
                # MAXIMIZE: Need r_j + delta <= 0 => delta <= -r_j
                if 0.0 < rj <= 1e-7:
                    rj = 0.0
                allowable_dec = math.inf
                allowable_inc = max(0.0, -rj)

        lb = -math.inf if allowable_dec == math.inf else curr_c - allowable_dec
        ub = math.inf if allowable_inc == math.inf else curr_c + allowable_inc

        target_obj_ranges[vname] = SensitivityRange(
            name=vname,
            current_value=curr_c,
            allowable_decrease=allowable_dec,
            allowable_increase=allowable_inc,
            lower_bound=lb,
            upper_bound=ub,
            is_unbounded_below=(allowable_dec == math.inf),
            is_unbounded_above=(allowable_inc == math.inf),
            basis_remains_optimal=True,
            metadata={
                "is_basic": col in basis_col_to_pos,
                "reduced_cost": float(r[col]),
            },
        )

    # 9. Presolve Inversion for Objective
    final_obj_ranges: dict[str, SensitivityRange] = {}
    if presolve_res is None:
        final_obj_ranges = target_obj_ranges
    else:
        for vname in model.variables:
            red_name = presolve_res.original_to_reduced.get(vname, vname)
            if red_name in target_obj_ranges:
                final_obj_ranges[vname] = target_obj_ranges[red_name]
            else:
                # Fixed variable in presolve
                curr_c = float(model.objective.coefficients.get(vname, 0.0))
                final_obj_ranges[vname] = SensitivityRange(
                    name=vname,
                    current_value=curr_c,
                    allowable_decrease=math.inf,
                    allowable_increase=math.inf,
                    lower_bound=-math.inf,
                    upper_bound=math.inf,
                    is_unbounded_below=True,
                    is_unbounded_above=True,
                    basis_remains_optimal=True,
                    metadata={"presolved_fixed": True, "fixed_value": presolve_res.fixed_variables.get(vname, 0.0)},
                )

    return SensitivityReport(
        rhs=final_rhs_ranges,
        objective=final_obj_ranges,
        basis_info=binfo,
        solver_selected=binfo.solver_selected,
        is_optimal=True,
        metadata={
            "presolved": was_presolved,
            "degenerate_basis": is_degenerate,
        },
    )
