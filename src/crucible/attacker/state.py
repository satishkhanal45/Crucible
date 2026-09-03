"""The attacker's graph state and its operating modes."""

from __future__ import annotations

import operator
import uuid
from enum import StrEnum
from typing import Annotated, Self, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator

from crucible.archive.survey import CoverageReport, ParentAttack
from crucible.schemas.attack import Attack
from crucible.target.adapter import BehaviorSpec, TargetCapabilities


class AttackerMode(StrEnum):
    """How much the attacker is allowed to know about the defense.

    `black_box` is the default and the mode the main experiment runs in: the
    attacker sees only the outcomes of its own past attempts. The resulting
    attacks generalize better, and the number they produce is the honest one.
    `white_box` exists for a single upper-bound run in Phase 8.
    """

    BLACK_BOX = "black_box"
    WHITE_BOX = "white_box"
    #: Cut B2. Kept in the enum so the validator can name it; never selectable.
    GREY_BOX = "grey_box"


#: Cut B2 in docs/spec.md section 3, restorable as deferred item D2.
GREY_BOX_MESSAGE = (
    "attacker mode 'grey_box' is not implemented in this build: it is cut B2 in "
    "docs/spec.md section 3, restorable as deferred item D2. To restore it, add one "
    "branch in attacker/prompts.py that emits defense category names without their "
    "parameter values. Use 'black_box' or 'white_box'."
)


class AttackerSettings(BaseModel):
    """Per-round limits. The attacker is the expensive component."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: AttackerMode = AttackerMode.BLACK_BOX
    #: Cells targeted per round, and hence parallel generation branches.
    cells_per_round: int = Field(default=4, ge=1, le=16)
    parents_per_cell: int = Field(default=2, ge=1, le=8)
    #: Hard cap on candidates generated in one round, budget or no budget.
    max_candidates: int = Field(default=8, ge=1, le=64)
    #: How many times the generate cycle may be re-entered.
    max_regenerate_rounds: int = Field(default=2, ge=0, le=5)

    @field_validator("mode")
    @classmethod
    def _grey_box_is_cut(cls, value: AttackerMode) -> AttackerMode:
        if value is AttackerMode.GREY_BOX:
            raise ValueError(GREY_BOX_MESSAGE)
        return value

    @classmethod
    def black_box(cls) -> Self:
        return cls(mode=AttackerMode.BLACK_BOX)


class RejectionRecord(BaseModel):
    """Why a candidate did not survive. Carries ids and reasons, not payloads."""

    model_config = ConfigDict(frozen=True)

    stage: str
    cell_key: str | None = None
    reason: str = ""
    attack_id: uuid.UUID | None = None
    nearest_neighbour_id: uuid.UUID | None = None
    novelty: float | None = None


class OutcomeSummary(BaseModel):
    """What black-box mode is allowed to tell the attacker: its own results."""

    model_config = ConfigDict(frozen=True)

    attempts: int = 0
    breached: int = 0
    blocked: int = 0
    refused: int = 0
    errors: int = 0

    @property
    def breach_rate(self) -> float:
        decided = self.breached + self.blocked + self.refused
        return self.breached / decided if decided else 0.0


STATUS_OK = "ok"
STATUS_BUDGET_EXCEEDED = "budget_exceeded"
STATUS_INSUFFICIENT = "insufficient"


class AttackerState(TypedDict, total=False):
    """LangGraph state for one attacker round."""

    round: int
    target_capabilities: TargetCapabilities
    behavior_spec: BehaviorSpec
    #: In black-box mode this is a summary of the attacker's own outcomes and
    #: contains no field of the DefenseConfig. See `build_defense_summary`.
    current_defense_summary: str
    coverage_report: CoverageReport
    selected_cells: list[str]
    parents: list[ParentAttack]
    strategies: dict[str, str]
    candidates: Annotated[list[Attack], operator.add]
    rejected: Annotated[list[RejectionRecord], operator.add]
    accepted: list[Attack]
    budget_remaining: float
    #: Written from the parallel generation branches, so it needs a reducer: any
    #: branch that runs out of budget exhausts the round.
    budget_exhausted: Annotated[bool, operator.or_]
    regenerate_rounds: int
    cells_needing_retry: list[str]
    status: str
    #: Set on the parallel generation branches only.
    target_cell: str
