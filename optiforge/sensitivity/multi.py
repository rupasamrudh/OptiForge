"""Multi-parameter sensitivity analysis and 100% Rule implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from optiforge.sensitivity.report import SensitivityReport


class RuleStatus(Enum):
    """Evaluation outcome of the 100% Rule."""

    WITHIN_REGION = "within_region"
    OUTSIDE_REGION = "outside_region"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class MultiParameterSensitivityResult:
    """Outcome of 100% Rule evaluation for simultaneous parameter perturbations.

    Attributes
    ----------
    parameter_type : str
        "rhs" or "objective".
    total_percentage_sum : float
        Sum of |Δp_i| / allowable_direction_i across all perturbed parameters.
    status : RuleStatus
        WITHIN_REGION if total_percentage_sum <= 1.0 + tol, OUTSIDE_REGION otherwise.
    is_guaranteed_optimal : bool
        True if the 100% Rule guarantees optimal basis preservation.
    individual_ratios : dict[str, float]
        Per-parameter ratio |Δp_i| / allowable_direction_i.
    limiting_parameters : list[str]
        Parameters that contribute the most to the percentage sum.
    metadata : dict[str, Any]
        Additional context (e.g. infinite ranges, zero ranges encountered).
    """

    parameter_type: str
    total_percentage_sum: float
    status: RuleStatus
    is_guaranteed_optimal: bool
    individual_ratios: dict[str, float] = field(default_factory=dict)
    limiting_parameters: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def evaluate_100_percent_rule(
    report: SensitivityReport,
    rhs_changes: dict[str, float] | None = None,
    objective_changes: dict[str, float] | None = None,
    tol: float = 1e-7,
) -> dict[str, MultiParameterSensitivityResult]:
    """Evaluate the 100% Rule for simultaneous RHS and/or objective coefficient changes.

    The 100% Rule is a conservative sufficient condition:
        sum_i (|Δp_i| / allowable_direction_i) <= 1.0
    guarantees that the current basis remains optimal.

    Parameters
    ----------
    report : SensitivityReport
        The baseline 1D sensitivity report from an optimal continuous LP solve.
    rhs_changes : dict[str, float] | None, optional
        Simultaneous RHS modifications {constraint_name: delta_b}.
    objective_changes : dict[str, float] | None, optional
        Simultaneous objective coefficient modifications {var_name: delta_c}.
    tol : float, optional
        Numerical tolerance for boundary evaluation (default 1e-7).

    Returns
    -------
    dict[str, MultiParameterSensitivityResult]
        Results keyed by "rhs" and/or "objective".
    """
    results: dict[str, MultiParameterSensitivityResult] = {}

    # 1. RHS 100% Rule
    if rhs_changes is not None and len(rhs_changes) > 0:
        ratios: dict[str, float] = {}
        has_zero_direction = False
        has_infinite_direction = False

        for cname, delta in rhs_changes.items():
            if cname not in report.rhs:
                raise KeyError(
                    f"Constraint '{cname}' in rhs_changes not found in sensitivity report. "
                    f"Available constraints: {list(report.rhs.keys())}"
                )
            rng = report.rhs[cname]

            if abs(delta) <= tol:
                ratios[cname] = 0.0
            elif delta > 0.0:
                inc = rng.allowable_increase
                if inc == math.inf:
                    ratios[cname] = 0.0
                    has_infinite_direction = True
                elif inc <= tol:
                    ratios[cname] = math.inf
                    has_zero_direction = True
                else:
                    ratios[cname] = delta / inc
            else:  # delta < 0.0
                dec = rng.allowable_decrease
                if dec == math.inf:
                    ratios[cname] = 0.0
                    has_infinite_direction = True
                elif dec <= tol:
                    ratios[cname] = math.inf
                    has_zero_direction = True
                else:
                    ratios[cname] = abs(delta) / dec

        total_sum = sum(ratios.values())
        if math.isinf(total_sum) or has_zero_direction:
            status = RuleStatus.OUTSIDE_REGION
            guaranteed = False
        elif total_sum <= 1.0 + tol:
            status = RuleStatus.WITHIN_REGION
            guaranteed = True
        else:
            status = RuleStatus.OUTSIDE_REGION
            guaranteed = False

        # Identify limiting parameters sorted by ratio descending
        sorted_params = sorted(ratios.keys(), key=lambda k: ratios[k], reverse=True)
        results["rhs"] = MultiParameterSensitivityResult(
            parameter_type="rhs",
            total_percentage_sum=total_sum,
            status=status,
            is_guaranteed_optimal=guaranteed,
            individual_ratios=ratios,
            limiting_parameters=sorted_params[:3],
            metadata={
                "has_zero_direction": has_zero_direction,
                "has_infinite_direction": has_infinite_direction,
            },
        )

    # 2. Objective 100% Rule
    if objective_changes is not None and len(objective_changes) > 0:
        ratios = {}
        has_zero_direction = False
        has_infinite_direction = False

        for vname, delta in objective_changes.items():
            if vname not in report.objective:
                raise KeyError(
                    f"Variable '{vname}' in objective_changes not found in sensitivity report. "
                    f"Available variables: {list(report.objective.keys())}"
                )
            rng = report.objective[vname]

            if abs(delta) <= tol:
                ratios[vname] = 0.0
            elif delta > 0.0:
                inc = rng.allowable_increase
                if inc == math.inf:
                    ratios[vname] = 0.0
                    has_infinite_direction = True
                elif inc <= tol:
                    ratios[vname] = math.inf
                    has_zero_direction = True
                else:
                    ratios[vname] = delta / inc
            else:  # delta < 0.0
                dec = rng.allowable_decrease
                if dec == math.inf:
                    ratios[vname] = 0.0
                    has_infinite_direction = True
                elif dec <= tol:
                    ratios[vname] = math.inf
                    has_zero_direction = True
                else:
                    ratios[vname] = abs(delta) / dec

        total_sum = sum(ratios.values())
        if math.isinf(total_sum) or has_zero_direction:
            status = RuleStatus.OUTSIDE_REGION
            guaranteed = False
        elif total_sum <= 1.0 + tol:
            status = RuleStatus.WITHIN_REGION
            guaranteed = True
        else:
            status = RuleStatus.OUTSIDE_REGION
            guaranteed = False

        sorted_params = sorted(ratios.keys(), key=lambda k: ratios[k], reverse=True)
        results["objective"] = MultiParameterSensitivityResult(
            parameter_type="objective",
            total_percentage_sum=total_sum,
            status=status,
            is_guaranteed_optimal=guaranteed,
            individual_ratios=ratios,
            limiting_parameters=sorted_params[:3],
            metadata={
                "has_zero_direction": has_zero_direction,
                "has_infinite_direction": has_infinite_direction,
            },
        )

    return results
