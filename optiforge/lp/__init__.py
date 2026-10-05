"""LP solver package.

Provides the baseline Revised Simplex solver, standard-form converter,
and Phase-I preparation for linear programming.
"""

from optiforge.lp.dual_simplex import (
    DualRevisedSimplexSolver,
    DualSimplexResult,
)
from optiforge.lp.profiling import (
    SimplexProfileStats,
    SimplexProfiler,
)
from optiforge.lp.simplex import (
    RevisedSimplexSolver,
    SimplexTolerances,
)
from optiforge.lp.standard_form import (
    SingularBasisError,
    StandardFormModel,
    VariableInfo,
    VariableRole,
    to_standard_form,
)
from optiforge.numerics.numerical_monitor import (
    NumericalEvent,
    NumericalEventType,
    NumericalSnapshot,
    NumericalStabilityMonitor,
    NumericalThresholds,
    NumericalWarningLevel,
)

__all__ = [
    "DualRevisedSimplexSolver",
    "DualSimplexResult",
    "RevisedSimplexSolver",
    "SimplexTolerances",
    "SimplexProfileStats",
    "SimplexProfiler",
    "SingularBasisError",
    "StandardFormModel",
    "VariableInfo",
    "VariableRole",
    "to_standard_form",
    "NumericalEvent",
    "NumericalEventType",
    "NumericalSnapshot",
    "NumericalStabilityMonitor",
    "NumericalThresholds",
    "NumericalWarningLevel",
]
