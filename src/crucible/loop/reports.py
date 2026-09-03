"""What a round records.

Every rate here is a `Proportion`, which carries its Wilson interval. That is
not a convention: a bare float cannot be assigned to these fields, so a report
with an uninterval'd rate fails validation rather than being published
(docs/spec.md section 15).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from crucible.loop.statistics import Proportion
from crucible.schemas.provenance import RunProvenance


class BareRate(ValueError):
    """A rate reached a report without its interval."""


class HaltReason(StrEnum):
    """Why a run stopped early. Each maps to one signal in spec section 14."""

    ATTACKER_EXHAUSTED = "attacker_exhausted"
    SEARCH_STALLED = "search_stalled"
    REDISCOVERY_ONLY = "rediscovery_only"
    UTILITY_COLLAPSE = "utility_collapse"
    OVERFITTING = "overfitting"
    BUDGET_EXCEEDED = "budget_exceeded"


class RunStatus(StrEnum):
    """A halted run is a valid experiment, not a failure."""

    RUNNING = "running"
    COMPLETED = "completed"
    HALTED = "halted"
    FAILED = "failed"


class Regression(BaseModel):
    """An archived attack that the new config reopened."""

    model_config = ConfigDict(frozen=True)

    attack_id: uuid.UUID
    cell_key: str | None = None
    previous_outcome: str = "blocked"
    new_outcome: str = "breached"

    def __str__(self) -> str:
        cell = self.cell_key or "unclassified"
        return f"{self.attack_id} ({cell}): {self.previous_outcome} -> {self.new_outcome}"


class RoundReport(BaseModel):
    """One round's result. Every rate carries an interval."""

    model_config = ConfigDict(frozen=True)

    run_id: uuid.UUID
    round_number: int = Field(ge=0)
    attacker_mode: str
    defense_before: str
    defense_after: str

    attacks_generated: int = 0
    attacks_rejected_novelty: int = 0
    breaches_found: int = 0

    #: Over the full non-holdout archive. Never a screening number.
    archive_block: Proportion
    #: The only honest generalization number in the project.
    holdout_block: Proportion
    utility_pass: Proportion

    mean_novelty: float = 0.0
    cells_occupied: int = 0
    new_cells: int = 0
    #: Attacks the newly selected config reopened. Never auto-accepted.
    regressions: tuple[Regression, ...] = ()
    config_promoted: bool = True
    cost_usd: Decimal = Decimal(0)
    halt_reason: HaltReason | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None

    @property
    def overfit_gap(self) -> float:
        """Archive block rate minus holdout block rate. The overfitting measure."""
        return self.archive_block.rate - self.holdout_block.rate

    @property
    def has_regressions(self) -> bool:
        return bool(self.regressions)

    def rates(self) -> dict[str, Proportion]:
        return {
            "archive_block": self.archive_block,
            "holdout_block": self.holdout_block,
            "utility_pass": self.utility_pass,
        }

    def validate_intervals(self) -> None:
        """Every reported rate must carry a usable interval."""
        for name, proportion in self.rates().items():
            if not isinstance(proportion, Proportion):
                raise BareRate(f"{name} is a bare rate: every reported rate needs an interval")
            low, high = proportion.interval
            if not 0.0 <= low <= high <= 1.0:
                raise BareRate(f"{name} has an invalid interval [{low}, {high}]")

    def summary(self) -> str:
        lines = [
            f"round {self.round_number}  [{self.defense_before[:8]} -> {self.defense_after[:8]}]",
            f"  archive block rate  {self.archive_block}",
            f"  holdout block rate  {self.holdout_block}",
            f"  overfit gap         {self.overfit_gap:+.3f} (archive - holdout)",
            f"  utility pass rate   {self.utility_pass}",
            f"  attacks generated   {self.attacks_generated} "
            f"({self.attacks_rejected_novelty} rejected on novelty, "
            f"{self.breaches_found} breached)",
            f"  coverage            {self.cells_occupied}/96 (+{self.new_cells} this round)",
            f"  mean novelty        {self.mean_novelty:.3f}",
            f"  cost                ${self.cost_usd:.4f}",
        ]
        if self.regressions:
            lines.append(f"  REGRESSIONS         {len(self.regressions)} — config NOT promoted")
            lines.extend(f"    {regression}" for regression in self.regressions)
        if self.halt_reason is not None:
            lines.append(f"  HALT                {self.halt_reason.value}")
        return "\n".join(lines)


class RunReport(BaseModel):
    """A whole run: its rounds and how it ended."""

    model_config = ConfigDict(frozen=True)

    run_id: uuid.UUID
    status: RunStatus
    attacker_mode: str
    starting_config_id: str
    final_config_id: str
    rounds: tuple[RoundReport, ...] = ()
    halt_reason: HaltReason | None = None
    #: Which provider and model each agent used. A run whose provenance is empty
    #: cannot prove it was live, which is why `stubbed` defaults to True.
    provenance: RunProvenance = RunProvenance()
    stubbed: bool = True

    @property
    def banner(self) -> str | None:
        """The warning a report leads with, or None when the run is live."""
        if not self.stubbed:
            return None
        return self.provenance.banner() or (
            "STUBBED RUN — NOT A MEASUREMENT. This run did not record which "
            "models produced it, so it cannot be read as a measurement. "
            "Re-run with `--provider groq`."
        )

    @property
    def rounds_completed(self) -> int:
        return len(self.rounds)

    @property
    def total_regressions(self) -> int:
        return sum(len(report.regressions) for report in self.rounds)
