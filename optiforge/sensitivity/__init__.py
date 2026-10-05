"""LP Sensitivity Analysis, Multi-Parameter & Scenario Module for OptiForge."""

from __future__ import annotations

from optiforge.sensitivity.calculator import compute_sensitivity
from optiforge.sensitivity.multi import (
    MultiParameterSensitivityResult,
    RuleStatus,
    evaluate_100_percent_rule,
)
from optiforge.sensitivity.range import SensitivityRange
from optiforge.sensitivity.report import SensitivityReport
from optiforge.sensitivity.scenario import (
    BreakEvenResult,
    Scenario,
    ScenarioComparison,
    ScenarioResult,
    compare_scenarios,
    compute_break_even,
    evaluate_scenario,
)

__all__ = [
    "BreakEvenResult",
    "MultiParameterSensitivityResult",
    "RuleStatus",
    "Scenario",
    "ScenarioComparison",
    "ScenarioResult",
    "SensitivityRange",
    "SensitivityReport",
    "compare_scenarios",
    "compute_break_even",
    "compute_sensitivity",
    "evaluate_100_percent_rule",
    "evaluate_scenario",
]
