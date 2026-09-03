"""Round orchestration: the ten-step co-evolution loop."""

from crucible.loop.collapse import RoundSignals, detect
from crucible.loop.graph import CoEvolutionLoop, LoopComponents
from crucible.loop.regression import find_regressions
from crucible.loop.reports import (
    BareRate,
    HaltReason,
    Regression,
    RoundReport,
    RunReport,
    RunStatus,
)
from crucible.loop.runner import (
    LoopFactories,
    LoopRunner,
    LoopSettings,
    build_components,
    load_run_report,
)
from crucible.loop.state import STEP_NAMES, LoopEvent, LoopState
from crucible.loop.statistics import (
    Proportion,
    ProportionTest,
    improved,
    two_proportion_test,
    wilson_interval,
)

__all__ = [
    "STEP_NAMES",
    "BareRate",
    "CoEvolutionLoop",
    "HaltReason",
    "LoopComponents",
    "LoopEvent",
    "LoopFactories",
    "LoopRunner",
    "LoopSettings",
    "LoopState",
    "Proportion",
    "ProportionTest",
    "Regression",
    "RoundReport",
    "RoundSignals",
    "RunReport",
    "RunStatus",
    "build_components",
    "detect",
    "find_regressions",
    "improved",
    "load_run_report",
    "two_proportion_test",
    "wilson_interval",
]
