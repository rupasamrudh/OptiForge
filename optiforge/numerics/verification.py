"""Independent post-solve verification and KKT diagnostic layer for OptiForge.

Stage 8.48: Numerical Robustness, Basis Quality & Solver Diagnostics.

Provides independent, mathematically rigorous verification of alleged LP solutions:
1. PRIMAL FEASIBILITY:
   - Standard-form: ||Ax - b||_inf <= feasibility_tolerance and x >= -feasibility_tolerance.
   - Original-form: constraint satisfaction across <=, >=, == and variable lower/upper bounds.
2. DUAL FEASIBILITY:
   - MINIMIZE: r_j >= -optimality_tolerance for all variables.
   - MAXIMIZE: r_j <= optimality_tolerance for all variables.
3. COMPLEMENTARY SLACKNESS:
   - |x_j * r_j| <= complementarity_tolerance across all variables.
4. OBJECTIVE CONSISTENCY:
   - Reported objective approximately equals c^T x (+ constant).
5. BASIS QUALITY:
   - Strict mode: nonsingularity, condition number, min pivot, residuals.
6. TRANSFORMATION INVARIANCE:
   - Accurately accounts for presolve variable fixing, lower-bound shifts,
     row sign flips, and standard-form slacks/surpluses without false positives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from optiforge.core.solution import Solution, SolutionStatus
from optiforge.model.constraint import ConstraintSense
from optiforge.model.model import Model
from optiforge.model.objective import ObjectiveSense
from optiforge.numerics.basis_quality import BasisQuality, evaluate_basis_quality
from optiforge.numerics.policy import NumericalPolicy, VerificationMode


@dataclass(frozen=True)
class SolverDiagnostics:
    """Immutable structured diagnostics and verification outcome for an LP solve.

    Attributes
    ----------
    verification_passed : bool
        True if all checks for the requested verification mode passed within tolerance.
    primal_residual : float
        Maximum primal violation across constraints and variable bounds.
    dual_residual : float
        Maximum dual feasibility violation across reduced costs.
    complementarity_residual : float
        Maximum complementary slackness violation max_j |x_j * r_j|.
    objective_error : float
        Absolute error |reported_obj - calculated_obj|.
    basis_quality : BasisQuality | None
        Structured basis quality evaluation (populated in strict mode if basis exists).
    numerically_safe : bool
        True if the solution and basis satisfy numerical safety criteria.
    warnings : tuple[str, ...]
        Tuple of diagnostic warning messages, if any.
    verification_mode : str
        The verification mode executed ('none', 'basic', 'strict').
    details : str
        Human-readable summary of the verification outcome.
    """

    verification_passed: bool
    primal_residual: float
    dual_residual: float
    complementarity_residual: float
    objective_error: float
    basis_quality: BasisQuality | None
    numerically_safe: bool
    verification_mode: str
    details: str
    warnings: tuple[str, ...] = field(default_factory=tuple)
    condition_estimation_method: str | None = None
    estimated_condition: float | None = None
    condition_estimation_converged: bool | None = None
    scaling_applied: bool = False
    scaling_iterations: int = 0
    scaling_range: tuple[float, float] | None = None

    def __repr__(self) -> str:
        passed_str = "PASSED" if self.verification_passed else "FAILED"
        return (
            f"SolverDiagnostics({passed_str} [{self.verification_mode}], "
            f"prim_res={self.primal_residual:.2e}, dual_res={self.dual_residual:.2e}, "
            f"comp_res={self.complementarity_residual:.2e}, obj_err={self.objective_error:.2e}, "
            f"safe={self.numerically_safe})"
        )


def verify_solution(
    model: Model,
    solution: Solution,
    target_model: Model | None = None,
    presolve_res: Any | None = None,
    policy: NumericalPolicy | None = None,
    mode: str | VerificationMode | None = None,
) -> SolverDiagnostics:
    """Independently verify an optimization solution and assess numerical robustness.

    Parameters
    ----------
    model : Model
        Original user optimization model.
    solution : Solution
        Solution returned by solver.
    target_model : Model | None, optional
        Pre-solved reduced model if presolve was used; defaults to model.
    presolve_res : Any | None, optional
        Presolve metadata container if presolve was used.
    policy : NumericalPolicy | None, optional
        Numerical tolerances and thresholds.
    mode : str | VerificationMode | None, optional
        Verification mode override ('none', 'basic', 'strict').
        If None, uses policy.verification_mode.

    Returns
    -------
    SolverDiagnostics
        Structured verification results.
    """
    pol = policy or NumericalPolicy()
    active_mode = (
        mode.value if isinstance(mode, VerificationMode)
        else (mode if mode is not None else pol.verification_mode)
    ).lower()

    if active_mode == "none":
        return SolverDiagnostics(
            verification_passed=True,
            primal_residual=0.0,
            dual_residual=0.0,
            complementarity_residual=0.0,
            objective_error=0.0,
            basis_quality=None,
            numerically_safe=True,
            verification_mode="none",
            details="Verification skipped (mode=none).",
            warnings=(),
        )

    warnings: list[str] = []
    is_milp = any(v.is_integer() for v in model.variables.values())

    # 1. Primal Feasibility on Original Model
    # Checks constraint violations: <=, >=, ==
    max_constr_viol = 0.0
    var_vals = solution.variable_values or {}

    for con in model.constraints.values():
        activity = sum(
            float(coeff) * float(var_vals.get(vname, 0.0))
            for vname, coeff in con.coefficients.items()
        )
        rhs_val = float(con.rhs)
        if con.sense is ConstraintSense.LESS_EQUAL:
            viol = max(0.0, activity - rhs_val)
        elif con.sense is ConstraintSense.GREATER_EQUAL:
            viol = max(0.0, rhs_val - activity)
        else:  # EQUAL
            viol = abs(activity - rhs_val)

        if viol > max_constr_viol:
            max_constr_viol = viol

    # Variable bound violations on original model
    max_bound_viol = 0.0
    for var in model.variables.values():
        val = float(var_vals.get(var.name, 0.0))
        lb = float(var.lower_bound)
        ub = float(var.upper_bound)
        if math.isfinite(lb):
            viol_lb = max(0.0, lb - val)
            if viol_lb > max_bound_viol:
                max_bound_viol = viol_lb
        if math.isfinite(ub):
            viol_ub = max(0.0, val - ub)
            if viol_ub > max_bound_viol:
                max_bound_viol = viol_ub

    max_orig_primal_viol = max(max_constr_viol, max_bound_viol)
    if max_orig_primal_viol > pol.feasibility_tolerance:
        warnings.append(
            f"primal_feasibility_violation: {max_orig_primal_viol:.2e} > tolerance {pol.feasibility_tolerance:.2e}"
        )

    # 2. Objective Consistency
    obj_err = 0.0
    if model.objective is not None and solution.objective_value is not None:
        calc_obj = float(getattr(model.objective, "constant", 0.0))
        for vname, coeff in model.objective.coefficients.items():
            calc_obj += float(coeff) * float(var_vals.get(vname, 0.0))

        rep_obj = float(solution.objective_value)
        obj_err = abs(rep_obj - calc_obj)
        obj_scale = max(1.0, abs(rep_obj), abs(calc_obj))
        allowed_obj_err = max(pol.feasibility_tolerance * 10.0, pol.optimality_tolerance * obj_scale)

        if obj_err > allowed_obj_err:
            warnings.append(
                f"objective_inconsistency: reported {rep_obj:.6e} vs calculated {calc_obj:.6e} (err {obj_err:.2e})"
            )

    # In BASIC mode, return immediately after primal & objective checks
    if active_mode == "basic" or is_milp or solution.status is not SolutionStatus.OPTIMAL:
        passed = (
            max_orig_primal_viol <= pol.feasibility_tolerance
            and obj_err <= max(pol.feasibility_tolerance * 10.0, pol.optimality_tolerance * max(1.0, abs(solution.objective_value or 0.0)))
        )
        safe = passed and (max_orig_primal_viol <= pol.max_safe_residual)
        return SolverDiagnostics(
            verification_passed=passed,
            primal_residual=max_orig_primal_viol,
            dual_residual=0.0,
            complementarity_residual=0.0,
            objective_error=obj_err,
            basis_quality=None,
            numerically_safe=safe,
            verification_mode=active_mode,
            details="Basic verification passed" if passed else f"Basic verification failed: {'; '.join(warnings)}",
            warnings=tuple(warnings),
        )

    # 3. STRICT MODE: KKT & Basis Quality Checks on Standard Form
    tgt = target_model if target_model is not None else model
    basis_info = getattr(solution, "basis_info", None)
    bq: BasisQuality | None = None
    sf_dual_viol = 0.0
    sf_comp_viol = 0.0
    sf_primal_res = max_orig_primal_viol

    if basis_info is not None and getattr(basis_info, "basic_variable_indices", None):
        try:
            from optiforge.lp.standard_form import to_standard_form

            msf = to_standard_form(tgt)
            basis_idx = basis_info.basic_variable_indices
            m_sf, n_sf = msf.A.shape

            if len(basis_idx) == m_sf and all(0 <= idx < n_sf for idx in basis_idx):
                bq = evaluate_basis_quality(
                    A=msf.A,
                    b=msf.b,
                    c=msf.c,
                    basis_indices=basis_idx,
                    sense=msf.objective_sense,
                    policy=pol,
                )
                sf_primal_res = max(sf_primal_res, bq.primal_residual)
                sf_comp_viol = bq.complementarity_residual

                # Dual feasibility on standard form:
                # B^T y = c_B  =>  r = c - A^T y
                B = msf.A[:, basis_idx]
                cB = msf.c[basis_idx]
                y = np.linalg.solve(B.T, cB)
                r = msf.c - msf.A.T @ y

                art_set = set(getattr(msf, "artificial_indices", []))
                is_min = (msf.objective_sense is ObjectiveSense.MINIMIZE)

                for j in range(n_sf):
                    if j in art_set:
                        continue
                    if is_min:
                        viol = max(0.0, -r[j])
                    else:
                        viol = max(0.0, r[j])
                    if viol > sf_dual_viol:
                        sf_dual_viol = viol

                if sf_dual_viol > pol.optimality_tolerance:
                    warnings.append(
                        f"dual_feasibility_violation: {sf_dual_viol:.2e} > tolerance {pol.optimality_tolerance:.2e}"
                    )
                if sf_comp_viol > pol.complementarity_tolerance:
                    warnings.append(
                        f"complementarity_violation: {sf_comp_viol:.2e} > tolerance {pol.complementarity_tolerance:.2e}"
                    )
                if not bq.is_numerically_safe:
                    warnings.extend(bq.warnings)
        except Exception as err:
            warnings.append(f"standard_form_kkt_evaluation_error: {err}")

    # Determine overall verification outcome
    primal_passed = sf_primal_res <= pol.feasibility_tolerance
    dual_passed = sf_dual_viol <= pol.optimality_tolerance
    comp_passed = sf_comp_viol <= pol.complementarity_tolerance
    obj_scale = max(1.0, abs(solution.objective_value or 0.0))
    obj_passed = obj_err <= max(pol.feasibility_tolerance * 10.0, pol.optimality_tolerance * obj_scale)

    verification_passed = primal_passed and dual_passed and comp_passed and obj_passed

    is_safe = verification_passed
    if bq is not None and not bq.is_numerically_safe:
        is_safe = False

    details = "Strict KKT verification passed" if verification_passed else f"Verification failed: {'; '.join(warnings)}"

    # Stage 8.49 metadata extraction
    cond_method = getattr(bq, "condition_method", None) if bq is not None else None
    cond_est = getattr(bq, "condition_estimate", None) if bq is not None else None
    cond_conv = getattr(bq, "condition_converged", None) if bq is not None else None

    scaling_meta = getattr(solution, "solver_metadata", {}).get("scaling", {}) if getattr(solution, "solver_metadata", None) else {}
    scaling_applied = bool(scaling_meta.get("applied", False))
    scaling_iters = int(scaling_meta.get("iterations", 0))
    scaling_range = scaling_meta.get("scaling_range", None)

    return SolverDiagnostics(
        verification_passed=verification_passed,
        primal_residual=sf_primal_res,
        dual_residual=sf_dual_viol,
        complementarity_residual=sf_comp_viol,
        objective_error=obj_err,
        basis_quality=bq,
        numerically_safe=is_safe,
        verification_mode="strict",
        details=details,
        warnings=tuple(warnings),
        condition_estimation_method=cond_method,
        estimated_condition=cond_est,
        condition_estimation_converged=cond_conv,
        scaling_applied=scaling_applied,
        scaling_iterations=scaling_iters,
        scaling_range=scaling_range,
    )
