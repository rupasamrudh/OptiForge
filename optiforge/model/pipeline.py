"""Unified orchestration pipeline, solver selection, and basis extraction for OptiForge models.

This module provides the Stage 8.43 & 8.44 deterministic routing pipeline:
    Model -> Presolve -> Problem Fingerprint -> Solver Selection -> Postsolve -> BasisInfo -> Solution

Solvers managed:
- RevisedSimplexSolver (Primal Two-Phase Revised Simplex)
- DualRevisedSimplexSolver (Dual Revised Simplex)
- MILPSolver (Branch-and-Bound for Mixed-Integer Linear Programs)
"""

from __future__ import annotations

import inspect
import math
from typing import Any

import numpy as np

from optiforge.core.basis import BasisInfo, BasisStatus
from optiforge.core.solution import Solution, SolutionStatus
from optiforge.model.objective import ObjectiveSense



def select_solver(
    model: Any,
    fingerprint: Any,
    solver_preference: str | None = None,
    initial_basis: list[int] | None = None,
    tolerances: Any = None,
) -> tuple[str, str]:
    """Select the appropriate solver deterministically based on model structure, fingerprint, and basis.

    Parameters
    ----------
    model : Model
        The model to solve (original or presolved).
    fingerprint : ProblemFingerprint
        Deterministic structural fingerprint of the model.
    solver_preference : str | None, optional
        Optional user preference override ('auto', 'primal', 'dual', 'milp').
    initial_basis : list[int] | None, optional
        Optional warm-start basis indices.
    tolerances : Any | None, optional
        Numerical tolerances.

    Returns
    -------
    tuple[str, str]
        (solver_name, selection_reason), where solver_name is one of:
        - 'milp'
        - 'dual_simplex'
        - 'revised_simplex'

    Raises
    ------
    ValueError
        If an unknown solver is requested or an LP solver is requested for a MILP model.
    """
    pref = (solver_preference or "auto").strip().lower()

    has_integers = (
        getattr(fingerprint, "integer_variable_count", 0) + getattr(fingerprint, "binary_variable_count", 0) > 0
        or any(v.is_integer() for v in model.variables.values())
    )

    if pref != "auto":
        if pref in ("milp", "bnb", "branch_and_bound"):
            return "milp", "explicit_user_preference"
        elif pref in ("dual", "dual_simplex", "dual_revised_simplex"):
            if has_integers:
                raise ValueError(
                    "Dual Revised Simplex cannot solve MILP problems containing integer or binary variables."
                )
            return "dual_simplex", "explicit_user_preference"
        elif pref in ("primal", "simplex", "revised_simplex", "primal_revised_simplex"):
            if has_integers:
                raise ValueError(
                    "Revised Simplex cannot solve MILP problems containing integer or binary variables."
                )
            return "revised_simplex", "explicit_user_preference"
        else:
            raise ValueError(
                f"Unknown solver preference: '{solver_preference}'. "
                f"Supported options: 'auto', 'primal', 'dual', 'milp'."
            )

    # Automatic selection based on model classification and fingerprint
    if has_integers:
        return "milp", "integer_variables_detected"

    # Continuous LP:
    # If initial warm-start basis is provided, inspect primal and dual feasibility
    if initial_basis is not None:
        try:
            from optiforge.lp.standard_form import to_standard_form
            from optiforge.numerics.basis import BasisFactorization
            from optiforge.numerics.basis_quality import evaluate_basis_quality
            from optiforge.numerics.policy import NumericalPolicy

            pol = NumericalPolicy.from_tolerances(tolerances)
            feas_tol = pol.feasibility_tolerance
            piv_tol = pol.pivot_tolerance

            msf = to_standard_form(model)
            m_sf, n_sf = msf.A.shape
            if len(initial_basis) == m_sf and len(set(initial_basis)) == m_sf and all(0 <= idx < n_sf for idx in initial_basis):
                # Stage 8.48: Evaluate basis quality and numerical safety before accepting warm-start
                bq = evaluate_basis_quality(
                    A=msf.A,
                    b=msf.b,
                    c=msf.c,
                    basis_indices=initial_basis,
                    sense=msf.objective_sense,
                    policy=pol,
                )
                if not bq.is_numerically_safe:
                    # Safely reject warm start for numerically unsafe basis and fall back to cold solve
                    return "revised_simplex", "warm_start_fallback_cold_start"

                B = msf.A[:, initial_basis]
                bf = BasisFactorization(
                    B,
                    pivot_tolerance=piv_tol,
                    use_scipy_triangular_solve=getattr(tolerances, "use_scipy_triangular_solve", False) if tolerances else False,
                    scipy_triangular_min_dimension=getattr(tolerances, "scipy_triangular_min_dimension", 10) if tolerances else 10,
                    triangular_solve_provider=getattr(tolerances, "triangular_solve_provider", "python") if tolerances else "python",
                    use_scipy_lu_factor=getattr(tolerances, "use_scipy_lu_factor", False) if tolerances else False,
                    scipy_lu_min_dimension=getattr(tolerances, "scipy_lu_min_dimension", 10) if tolerances else 10,
                    lu_factor_provider=getattr(tolerances, "lu_factor_provider", "python") if tolerances else "python",
                )
                xB = bf.solve(msf.b)
                primal_feasible = bool(np.all(xB >= -feas_tol))

                cB = msf.c[initial_basis]
                y = bf.solve_transpose(cB)
                r = msf.c - np.dot(msf.A.T, y)

                art_set = set(getattr(msf, "artificial_indices", []))
                basis_set = set(initial_basis)
                nonbasic_check = [j for j in range(n_sf) if j not in basis_set and j not in art_set]

                is_min = (getattr(fingerprint, "objective_sense", None) == ObjectiveSense.MINIMIZE)
                if is_min:
                    dual_feasible = all(r[j] >= -piv_tol for j in nonbasic_check) if nonbasic_check else True
                else:
                    dual_feasible = all(r[j] <= piv_tol for j in nonbasic_check) if nonbasic_check else True

                if primal_feasible:
                    return "revised_simplex", "warm_start_primal_feasible"
                elif dual_feasible:
                    return "dual_simplex", "warm_start_dual_feasible"
                else:
                    return "revised_simplex", "warm_start_fallback_cold_start"
        except Exception:
            pass

    # Check conservative dual-feasible condition for initial slack basis:
    # 1. No >= or == constraints (only <= constraints, no artificial variables needed)
    # 2. All variable lower bounds == 0.0 (no non-zero lower bound shifts)
    # 3. All constraint RHS >= 0.0
    # 4. Objective: For MINIMIZE, all c_j >= 0.0; for MAXIMIZE, all c_j <= 0.0
    is_all_le = (
        getattr(fingerprint, "constraint_sense_ge_count", 0) == 0
        and getattr(fingerprint, "constraint_sense_eq_count", 0) == 0
    )
    all_lb_zero = all(
        v.lower_bound == 0.0 for v in model.variables.values()
    )
    all_rhs_nonneg = all(
        con.rhs >= 0.0 for con in model.constraints.values()
    )
    if model.objective is not None:
        if getattr(fingerprint, "objective_sense", None) == ObjectiveSense.MINIMIZE:
            dual_obj_feasible = all(
                c >= 0.0 for c in model.objective.coefficients.values()
            )
        else:
            dual_obj_feasible = all(
                c <= 0.0 for c in model.objective.coefficients.values()
            )
    else:
        dual_obj_feasible = False

    if is_all_le and all_lb_zero and all_rhs_nonneg and dual_obj_feasible and model.num_constraints > 0:
        return "dual_simplex", "dual_feasible_slack_basis"

    # Conservative default for general continuous LPs
    return "revised_simplex", "primal_simplex_default"


def _instantiate_solver(cls: type, kwargs_dict: dict[str, Any]) -> Any:
    """Instantiate a solver class filtering kwargs to its constructor signature."""
    sig = inspect.signature(cls.__init__)
    valid_keys = {
        k
        for k, p in sig.parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    filtered = {k: v for k, v in kwargs_dict.items() if k in valid_keys and v is not None}
    return cls(**filtered)


def extract_basis_info(
    model: Any,
    target_model: Any,
    presolve_res: Any | None,
    solver_instance: Any,
    raw_solution: Solution,
    solver_name: str,
) -> BasisInfo | None:
    """Extract and reconstruct structured optimal BasisInfo.

    Parameters
    ----------
    model : Model
        The user's original Model before presolve.
    target_model : Model
        The model that was solved (reduced_model or model).
    presolve_res : PresolveResult | None
        Presolve reduction result if presolve was executed.
    solver_instance : Any
        Instance of RevisedSimplexSolver or DualRevisedSimplexSolver.
    raw_solution : Solution
        The solution container returned by the solver.
    solver_name : str
        Solver name ('revised_simplex', 'dual_simplex', etc.).

    Returns
    -------
    BasisInfo | None
        Structured immutable basis information, or None if unavailable/infeasible/MILP.
    """
    if raw_solution.status is not SolutionStatus.OPTIMAL:
        return None

    # Never fabricate sensitivity basis for MILP branch-and-bound solutions
    if solver_name == "milp" or any(v.is_integer() for v in model.variables.values()):
        return None

    # Case A: Model was fully reduced during presolve (all variables fixed)
    if presolve_res is not None and (
        presolve_res.reduced_model is None or presolve_res.reduced_model.num_variables == 0
    ):
        final_status = {v: BasisStatus.FIXED for v in model.variables}
        final_duals = {c: 0.0 for c in model.constraints}
        final_rc = {
            v: float(model.objective.coefficients.get(v, 0.0)) for v in model.variables
        }
        from optiforge.presolve import postsolve_solution

        restored_vals = postsolve_solution({}, presolve_res)
        return BasisInfo(
            basic_variable_indices=[],
            nonbasic_variable_indices=[],
            basic_variables=[],
            nonbasic_variables=list(model.variables.keys()),
            variable_status=final_status,
            primal_basic_values=restored_vals,
            reduced_costs=final_rc,
            dual_multipliers=final_duals,
            objective_value=float(presolve_res.objective_constant),
            standard_form_dimension=(0, 0),
            solver_selected="presolve",
            is_lp=True,
            is_optimal=True,
            metadata={
                "presolved_all_fixed": True,
                "presolved": True,
                "variable_values": restored_vals,
            },
        )

    # Case B: LP solver produced a basis on target_model
    from optiforge.lp.standard_form import VariableRole, to_standard_form

    msf = getattr(solver_instance, "last_standard_form_model", None)
    if msf is None:
        try:
            msf = to_standard_form(target_model)
        except Exception:
            return None

    basis_indices = getattr(solver_instance, "last_standard_form_basis_indices", None)
    last_res = getattr(solver_instance, "last_iteration_result", None)
    if basis_indices is None and last_res is not None:
        basis_indices = list(getattr(last_res, "basis_indices", []))

    if not basis_indices:
        return None

    m_sf, n_sf = msf.A.shape
    basis_set = set(basis_indices)
    nonbasic_indices = [j for j in range(n_sf) if j not in basis_set]

    # Primal basic values
    x_sf = getattr(last_res, "x", None)
    if x_sf is None or len(x_sf) != n_sf:
        try:
            B = msf.A[:, basis_indices]
            x_B = np.linalg.solve(B, msf.b)
            x_sf = np.zeros(n_sf, dtype=np.float64)
            for k, col in enumerate(basis_indices):
                x_sf[col] = x_B[k]
        except Exception:
            x_sf = np.zeros(n_sf, dtype=np.float64)

    primal_basic_values: dict[str, float] = {}
    basic_variable_names: list[str] = []
    for col in basis_indices:
        name = msf.variable_names[col]
        basic_variable_names.append(name)
        primal_basic_values[name] = float(x_sf[col])

    nonbasic_variable_names = [msf.variable_names[col] for col in nonbasic_indices]

    # Dual multipliers (y)
    y_sf = getattr(last_res, "y", None)
    if y_sf is None or len(y_sf) != m_sf:
        try:
            B = msf.A[:, basis_indices]
            cB = msf.c[basis_indices]
            y_sf = np.linalg.solve(B.T, cB)
        except Exception:
            y_sf = np.zeros(m_sf, dtype=np.float64)

    # Reduced costs (r = c - A^T y)
    r_sf = getattr(last_res, "reduced_costs", None)
    if r_sf is None or len(r_sf) != n_sf:
        r_sf = msf.c - np.dot(msf.A.T, y_sf)

    # Constraint shadow prices in target_model
    # Normalization flip check: if b_i < 0, row was flipped in standard form
    target_constrs = list(target_model.constraints.values())
    target_duals: dict[str, float] = {}
    for i, con in enumerate(target_constrs):
        if i < m_sf:
            # Check if row was flipped during standard form normalization
            shift_adj = sum(
                float(con.coefficients.get(vname, 0.0)) * msf.variable_shifts.get(msf.original_variable_mapping.get(vname, -1), 0.0)
                for vname in con.coefficients
            )
            raw_rhs = float(con.rhs) - shift_adj
            flip_sign = -1.0 if raw_rhs < 0.0 else 1.0
            target_duals[con.name] = float(y_sf[i] * flip_sign)

    # Variable status & reduced costs in target_model
    target_status: dict[str, BasisStatus] = {}
    target_rc: dict[str, float] = {}
    for vname, var in target_model.variables.items():
        col = msf.original_variable_mapping.get(vname)
        if col is not None:
            target_rc[vname] = float(r_sf[col])
            if col in basis_set:
                target_status[vname] = BasisStatus.BASIC
            else:
                val = float(x_sf[col])
                if math.isfinite(var.upper_bound) and abs(val - (var.upper_bound - msf.variable_shifts.get(col, 0.0))) <= 1e-6:
                    target_status[vname] = BasisStatus.NONBASIC_UPPER
                elif abs(val) <= 1e-6:
                    target_status[vname] = BasisStatus.NONBASIC_ZERO if var.lower_bound == 0.0 else BasisStatus.NONBASIC_LOWER
                else:
                    target_status[vname] = BasisStatus.NONBASIC_LOWER

    # Reconstruct original model space across presolve transformations
    if presolve_res is None:
        final_duals = target_duals
        final_rc = target_rc
        final_status = target_status
    else:
        # Constraint duals: surviving constraints carry target_duals, removed redundant carry 0.0
        final_duals = {}
        for cname in model.constraints:
            if cname in target_duals:
                final_duals[cname] = target_duals[cname]
            else:
                final_duals[cname] = 0.0

        # Variable reduced costs and status
        final_rc = {}
        final_status = {}
        for vname in model.variables:
            red_name = presolve_res.original_to_reduced.get(vname, vname)
            if red_name in target_rc:
                final_rc[vname] = target_rc[red_name]
                final_status[vname] = target_status[red_name]
            else:
                # Variable was fixed in presolve
                final_status[vname] = BasisStatus.FIXED
                c_v = float(model.objective.coefficients.get(vname, 0.0))
                # r_k = c_k - A_{:, k}^T lambda
                a_dot_lambda = sum(
                    float(con.coefficients.get(vname, 0.0)) * final_duals.get(con.name, 0.0)
                    for con in model.constraints.values()
                )
                final_rc[vname] = float(c_v - a_dot_lambda)

    # Include slack and surplus reduced costs
    for vinfo in msf.variable_info:
        if vinfo.role in (VariableRole.SLACK, VariableRole.SURPLUS):
            final_rc[vinfo.name] = float(r_sf[vinfo.column_index])

    final_obj = raw_solution.objective_value
    if presolve_res is not None and final_obj is not None:
        final_obj += float(presolve_res.objective_constant)

    var_vals_b = dict(raw_solution.variable_values)
    if presolve_res is not None:
        from optiforge.presolve import postsolve_solution

        var_vals_b = postsolve_solution(var_vals_b, presolve_res)

    return BasisInfo(
        basic_variable_indices=list(basis_indices),
        nonbasic_variable_indices=list(nonbasic_indices),
        basic_variables=basic_variable_names,
        nonbasic_variables=nonbasic_variable_names,
        variable_status=final_status,
        primal_basic_values=primal_basic_values,
        reduced_costs=final_rc,
        dual_multipliers=final_duals,
        objective_value=final_obj,
        standard_form_dimension=(m_sf, n_sf),
        solver_selected=solver_name,
        is_lp=True,
        is_optimal=True,
        metadata={
            "presolved": presolve_res is not None,
            "fixed_variable_count": len(presolve_res.fixed_variables) if presolve_res else 0,
            "removed_constraint_count": len(presolve_res.removed_constraints) if presolve_res else 0,
            "variable_values": var_vals_b,
        },
    )


def execute_pipeline(
    model: Any,
    tolerances: Any = None,
    stop_event: Any = None,
    solver: str | None = None,
    presolve: bool = True,
    pricing: str | None = None,
    numerical_monitor: Any = None,
    numerical_thresholds: Any = None,
    basis: Any = None,
    numerical_verification: str | None = None,
    scaling: bool | None = None,
    **kwargs: Any,
) -> Solution:
    """Execute the unified model optimization pipeline.

    Parameters
    ----------
    model : Model
        The optimization model to solve.
    tolerances : SimplexTolerances | MILPTolerances | None, optional
        Numerical tolerances.
    stop_event : threading.Event | None, optional
        Optional cancellation event.
    solver : str | None, optional
        Explicit solver preference ('auto', 'primal', 'dual', 'milp').
    presolve : bool, optional
        Whether to apply Stage 5A presolve (default True for continuous LP).
    pricing : str | None, optional
        Simplex pricing strategy ('bland', 'dantzig') if supported.
    numerical_monitor : Any | None, optional
        Optional numerical stability monitor.
    numerical_thresholds : Any | None, optional
        Optional numerical stability thresholds.
    basis : BasisInfo | list[int] | None, optional
        Optional basis for warm-start re-optimization.
    **kwargs : Any
        Additional solver-specific options forwarded to the selected solver.

    Returns
    -------
    Solution
        Standard OptiForge Solution container.
    """
    from optiforge.lp.simplex import SimplexTolerances
    from optiforge.milp.fingerprint import fingerprint_model

    # 1. Validation & Edge Cases
    if model.objective is None:
        return Solution(
            status=SolutionStatus.UNKNOWN,
            variable_values={},
            objective_value=None,
            message="Model has no objective.",
        )

    if model.num_variables == 0:
        return Solution(
            status=SolutionStatus.OPTIMAL if model.num_constraints == 0 else SolutionStatus.INFEASIBLE,
            variable_values={},
            objective_value=0.0 if model.num_constraints == 0 else None,
            message="Model has no variables.",
        )

    # 2. Model Classification
    is_milp = any(v.is_integer() for v in model.variables.values())

    # 3. Warm-Start Basis Extraction & Validation
    initial_basis_indices: list[int] | None = None
    if basis is not None:
        if isinstance(basis, BasisInfo):
            if not basis.is_lp:
                raise ValueError("Cannot warm-start with a non-LP basis.")
            initial_basis_indices = basis.get_basis_indices()
        elif isinstance(basis, (list, tuple)):
            initial_basis_indices = list(basis)
        else:
            raise TypeError(
                f"Expected BasisInfo or list[int] for basis, got {type(basis).__name__}."
            )

    # 4. Presolve
    presolve_res = None
    should_presolve = not is_milp and presolve
    if should_presolve and basis is not None:
        if isinstance(basis, BasisInfo):
            should_presolve = bool(basis.metadata.get("presolved", False))
        else:
            from optiforge.lp.standard_form import to_standard_form

            try:
                msf_raw = to_standard_form(model)
                if len(initial_basis_indices) == msf_raw.A.shape[0]:
                    should_presolve = False
            except Exception:
                pass

    if should_presolve:
        from optiforge.presolve import postsolve_solution, presolve_model

        presolve_res = presolve_model(model)
        if presolve_res.status in {SolutionStatus.INFEASIBLE, SolutionStatus.UNBOUNDED}:
            fp = fingerprint_model(model)
            model.last_fingerprint = fp
            model.last_solver_selected = "presolve"
            model.last_solver_reason = f"presolve_detected_{presolve_res.status.value}"
            return Solution(
                status=presolve_res.status,
                variable_values={},
                objective_value=None,
                message=f"Presolve detected {presolve_res.status.value}.",
                iterations=0,
                problem_fingerprint=fp,
                solver_selected="presolve",
                basis_info=None,
            )

        if presolve_res.reduced_model is None or presolve_res.reduced_model.num_variables == 0:
            restored = postsolve_solution({}, presolve_res)
            fp = fingerprint_model(model)
            model.last_fingerprint = fp
            model.last_solver_selected = "presolve"
            model.last_solver_reason = "presolve_fixed_all_variables"
            basis_info = extract_basis_info(
                model=model,
                target_model=model,
                presolve_res=presolve_res,
                solver_instance=None,
                raw_solution=Solution(status=SolutionStatus.OPTIMAL),
                solver_name="presolve",
            )
            model.last_basis = basis_info
            sensitivity_report = None
            if basis_info is not None and not is_milp:
                try:
                    from optiforge.sensitivity import compute_sensitivity

                    sensitivity_report = compute_sensitivity(model=model, basis_info=basis_info)
                except Exception:
                    sensitivity_report = None
            model.last_sensitivity = sensitivity_report
            sol_presolve = Solution(
                status=SolutionStatus.OPTIMAL,
                variable_values=restored,
                objective_value=float(presolve_res.objective_constant),
                message="All variables fixed during presolve.",
                iterations=0,
                problem_fingerprint=fp,
                solver_selected="presolve",
                basis_info=basis_info,
                sensitivity=sensitivity_report,
            )
            sol_presolve._model = model
            eff_verif = numerical_verification
            if eff_verif is None:
                eff_verif = getattr(tolerances, "numerical_verification", "none") if tolerances else "none"
            if eff_verif and str(eff_verif).lower() != "none":
                from optiforge.numerics.policy import NumericalPolicy
                from optiforge.numerics.verification import verify_solution

                pol = NumericalPolicy.from_tolerances(tolerances, verification_mode=str(eff_verif).lower())
                diag = verify_solution(
                    model=model,
                    solution=sol_presolve,
                    target_model=model,
                    presolve_res=presolve_res,
                    policy=pol,
                    mode=str(eff_verif).lower(),
                )
                sol_presolve.diagnostics = diag
            return sol_presolve

        target_model = presolve_res.reduced_model
    else:
        target_model = model

    # Validate basis dimensions against target_model standard-form
    if initial_basis_indices is not None:
        from optiforge.lp.standard_form import to_standard_form
        from optiforge.numerics.basis import BasisFactorization
        from optiforge.numerics.exceptions import SingularBasisError

        msf = to_standard_form(target_model)
        m_sf, n_sf = msf.A.shape
        if len(initial_basis_indices) != m_sf:
            raise ValueError(
                f"Incompatible basis: basis length {len(initial_basis_indices)} does not match row count {m_sf}."
            )
        if len(set(initial_basis_indices)) != m_sf:
            raise ValueError("Incompatible basis: duplicate column indices in basis.")
        if any(idx < 0 or idx >= n_sf for idx in initial_basis_indices):
            raise ValueError(
                f"Incompatible basis: column indices must be within [0, {n_sf - 1}]."
            )
        try:
            B = msf.A[:, initial_basis_indices]
            piv_tol = getattr(tolerances, "basis_singularity_tolerance", 1e-12) if tolerances else 1e-12
            _ = BasisFactorization(B, pivot_tolerance=piv_tol)
        except (SingularBasisError, ArithmeticError, Exception) as err:
            raise ValueError(f"Incompatible basis: basis matrix is singular ({err}).") from err

    # 5. Problem Fingerprint
    fp = fingerprint_model(target_model)
    model.last_fingerprint = fp

    # 6. Solver Selection
    solver_name, reason = select_solver(
        target_model,
        fp,
        solver_preference=solver,
        initial_basis=initial_basis_indices,
        tolerances=tolerances,
    )
    model.last_solver_selected = solver_name
    model.last_solver_reason = reason

    # 6.5 Dynamic Ruiz Scaling (if enabled and continuous LP)
    is_scaling_active = (
        scaling
        if scaling is not None
        else (getattr(tolerances, "scaling_enabled", False) if tolerances else False)
    )
    model_scaling_res = None
    if is_scaling_active and not is_milp:
        from optiforge.numerics.scaling import scale_model
        solve_model, model_scaling_res = scale_model(target_model, policy=tolerances)
    else:
        solve_model = target_model

    # 7. Solver Execution
    if solver_name == "milp":
        from optiforge.milp.solver import MILPSolver, MILPTolerances

        milp_kwargs = {
            "tolerances": tolerances if isinstance(tolerances, MILPTolerances) else None,
            "simplex_tolerances": tolerances if isinstance(tolerances, SimplexTolerances) else None,
            "stop_event": stop_event,
            "numerical_monitor": numerical_monitor,
            "numerical_thresholds": numerical_thresholds,
            **kwargs,
        }
        solver_instance = _instantiate_solver(MILPSolver, milp_kwargs)
        raw_solution = solver_instance.solve(solve_model)

    elif solver_name == "dual_simplex":
        from optiforge.lp.dual_simplex import DualRevisedSimplexSolver

        dual_kwargs = {
            "tolerances": tolerances if isinstance(tolerances, SimplexTolerances) else None,
            "stop_event": stop_event,
            "numerical_monitor": numerical_monitor,
            "initial_basis": initial_basis_indices,
            **kwargs,
        }
        solver_instance = _instantiate_solver(DualRevisedSimplexSolver, dual_kwargs)
        raw_solution = solver_instance.solve(solve_model, initial_basis=initial_basis_indices)

    else:  # 'revised_simplex'
        from optiforge.lp.simplex import RevisedSimplexSolver

        effective_basis = (
            initial_basis_indices
            if reason != "warm_start_fallback_cold_start"
            else None
        )
        primal_kwargs = {
            "tolerances": tolerances if isinstance(tolerances, SimplexTolerances) else None,
            "stop_event": stop_event,
            "pricing": pricing,
            "numerical_monitor": numerical_monitor,
            "numerical_thresholds": numerical_thresholds,
            "initial_basis": effective_basis,
            **kwargs,
        }
        solver_instance = _instantiate_solver(RevisedSimplexSolver, primal_kwargs)
        raw_solution = solver_instance.solve(solve_model, initial_basis=effective_basis)

    # 8. Basis & Sensitivity Extraction
    basis_info = extract_basis_info(
        model=model,
        target_model=solve_model,
        presolve_res=presolve_res,
        solver_instance=solver_instance,
        raw_solution=raw_solution,
        solver_name=solver_name,
    )

    if model_scaling_res is not None:
        from optiforge.numerics.scaling import unscale_model_solution
        raw_solution.basis_info = basis_info
        raw_solution = unscale_model_solution(raw_solution, model_scaling_res)
        basis_info = raw_solution.basis_info

    model.last_basis = basis_info

    sensitivity_report = None
    if basis_info is not None and basis_info.is_optimal and basis_info.is_lp and not is_milp:
        try:
            from optiforge.sensitivity import compute_sensitivity

            sensitivity_report = compute_sensitivity(model=model, basis_info=basis_info)
        except Exception:
            sensitivity_report = None
    model.last_sensitivity = sensitivity_report

    # 9. Postsolve & Solution Assembly
    meta = dict(getattr(raw_solution, "solver_metadata", {}) or {})
    if solver_name == "revised_simplex" and reason == "warm_start_primal_feasible":
        meta = {
            "warm_started": True,
            "initial_basis": tuple(initial_basis_indices) if initial_basis_indices is not None else None,
            "initial_primal_feasible": True,
            "initial_dual_feasible": True,
            "reoptimization_iterations": getattr(raw_solution, "iterations", 0),
        }

    if presolve_res is not None and raw_solution.status is SolutionStatus.OPTIMAL:
        from optiforge.presolve import postsolve_solution

        restored = postsolve_solution(raw_solution.variable_values, presolve_res)
        objective_value = float(raw_solution.objective_value) + float(presolve_res.objective_constant)
        final_sol = Solution(
            status=SolutionStatus.OPTIMAL,
            variable_values=restored,
            objective_value=objective_value,
            message=raw_solution.message or "Optimal solution found after presolve/postsolve reconstruction.",
            iterations=raw_solution.iterations,
            node_count=raw_solution.node_count,
            numerical_snapshots=raw_solution.numerical_snapshots,
            incumbent_objective=raw_solution.incumbent_objective,
            best_bound=raw_solution.best_bound,
            optimality_gap=raw_solution.optimality_gap,
            absolute_gap=raw_solution.absolute_gap,
            nodes_remaining=raw_solution.nodes_remaining,
            max_depth=raw_solution.max_depth,
            feasible_solutions_found=raw_solution.feasible_solutions_found,
            profile_stats=raw_solution.profile_stats,
            problem_fingerprint=fp,
            solver_selected=solver_name,
            solver_reason=reason,
            solver_metadata=meta,
            basis_info=basis_info,
            sensitivity=sensitivity_report,
        )
    else:
        # Pass-through or un-presolved solution
        raw_solution.problem_fingerprint = fp
        raw_solution.solver_selected = solver_name
        raw_solution.solver_reason = reason
        raw_solution.solver_metadata = meta
        raw_solution.basis_info = basis_info
        raw_solution.sensitivity = sensitivity_report
        final_sol = raw_solution

    final_sol._model = model

    # 10. Post-Solve Verification & Diagnostics
    eff_verif = numerical_verification
    if eff_verif is None:
        eff_verif = getattr(tolerances, "numerical_verification", "none") if tolerances else "none"

    if eff_verif and str(eff_verif).lower() != "none":
        from optiforge.numerics.policy import NumericalPolicy
        from optiforge.numerics.verification import verify_solution

        pol = NumericalPolicy.from_tolerances(tolerances, verification_mode=str(eff_verif).lower())
        diag = verify_solution(
            model=model,
            solution=final_sol,
            target_model=target_model,
            presolve_res=presolve_res,
            policy=pol,
            mode=str(eff_verif).lower(),
        )
        final_sol.diagnostics = diag

    return final_sol

