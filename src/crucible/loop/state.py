"""The loop's graph state and its instrumented event log.

The event log exists so that the ten-step ordering in docs/spec.md section 13 is
*assertable* rather than asserted in a comment. Every step appends one event,
and the Phase 6 verification test reads them back in order.
"""

from __future__ import annotations

import operator
import uuid
from datetime import datetime
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from crucible.defenses.config import DefenseConfig
from crucible.loop.collapse import RoundSignals
from crucible.loop.reports import Regression, RoundReport


class LoopEvent(BaseModel):
    """One step of one round, as it happened."""

    model_config = ConfigDict(frozen=True)

    round_number: int
    step: int = Field(ge=1, le=10)
    name: str
    detail: dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        return f"round {self.round_number} step {self.step}: {self.name}"


#: The ten steps, in the only order they may run.
STEP_NAMES: tuple[str, ...] = (
    "attacker_generates",
    "novelty_filter_and_execute",
    "oracle_and_archive_update",
    "defender_sees_breaches",
    "candidates_evaluated",
    "select_defense",
    "regression_check",
    "generalization_check",
    "record_round_report",
    "collapse_detection",
)


def event(round_number: int, step: int, **detail: Any) -> LoopEvent:
    return LoopEvent(round_number=round_number, step=step, name=STEP_NAMES[step - 1], detail=detail)


class LoopState(TypedDict, total=False):
    """LangGraph state for a whole run.

    Everything here has to survive a checkpoint round-trip: a nine-round run
    takes hours and will meet a rate limit, a network blip, or a closed laptop.
    """

    run_id: str
    round_number: int
    rounds_planned: int
    attacker_mode: str
    seed: int

    current_config: DefenseConfig
    previous_config: DefenseConfig | None
    baseline_utility: float
    #: The denominator behind `baseline_utility`, so a decline can be tested
    #: for significance rather than compared as a bare point estimate.
    baseline_utility_trials: int
    #: Per-attack outcomes under D(n-1), for the regression check.
    before_outcomes: dict[str, str]

    events: Annotated[list[LoopEvent], operator.add]
    signals: list[RoundSignals]
    reports: list[RoundReport]

    # --- per-round working state -------------------------------------
    round_started_at: datetime | None
    round_id: str
    cells_before: int
    attacks_generated: int
    attacks_rejected_novelty: int
    mean_novelty: float
    breaches_found: int
    selected_config: DefenseConfig | None
    regressions: list[Regression]
    config_promoted: bool
    archive_successes: int
    archive_trials: int
    holdout_successes: int
    holdout_trials: int
    utility_successes: int
    utility_trials: int
    round_cost: str

    halt_reason: str | None
    status: str


def round_id_for(run_id: str, round_number: int) -> uuid.UUID:
    """A stable round id, so a resumed round meters against the same bucket."""
    return uuid.uuid5(uuid.UUID(run_id), f"round-{round_number}")
