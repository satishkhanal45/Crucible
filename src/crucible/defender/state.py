"""The defender's graph state.

The defender never sees a holdout attack. That is enforced upstream, in the
repository layer, and asserted again in the prompt builders — this module only
carries what it was given.
"""

from __future__ import annotations

import operator
import uuid
from typing import Annotated, TypedDict

from pydantic import BaseModel, ConfigDict

from crucible.defenses.config import DefenseConfig
from crucible.evaluation.service import DefenseEvaluation

#: How many times `propose` may be re-entered before the defender gives up and
#: keeps the config it started with.
MAX_PROPOSE_ROUNDS = 2
#: docs/spec.md: three to five candidates per round.
DEFAULT_CANDIDATES = 4


class BreachSummary(BaseModel):
    """One breach the defender is allowed to learn from."""

    model_config = ConfigDict(frozen=True)

    attack_id: uuid.UUID
    cell_key: str | None
    objective: str | None
    technique: str | None
    vector: str
    payload: str
    is_holdout: bool = False


class Cluster(BaseModel):
    """Breaches that got through the same way."""

    model_config = ConfigDict(frozen=True)

    cell_key: str
    mechanism: str
    attack_ids: tuple[uuid.UUID, ...]

    @property
    def size(self) -> int:
        return len(self.attack_ids)


class Hypothesis(BaseModel):
    """Why one cluster got through, and which layers could answer it."""

    model_config = ConfigDict(frozen=True)

    cluster_key: str
    mechanism: str
    statement: str
    suggested_layers: tuple[str, ...] = ()


class CandidateProposal(BaseModel):
    """A candidate config with the reasoning that produced it."""

    model_config = ConfigDict(frozen=True)

    config: DefenseConfig
    rationale: str = ""
    source: str = "defender"

    @property
    def config_id(self) -> str:
        return self.config.fingerprint()


class RejectedProposal(BaseModel):
    """A proposal that never reached evaluation, and why."""

    model_config = ConfigDict(frozen=True)

    raw: str
    reason: str


class DefenderState(TypedDict, total=False):
    """LangGraph state. `candidate_configs` merges across the parallel fan-out."""

    round: int
    current_config: DefenseConfig
    breaches: list[BreachSummary]
    breach_clusters: list[Cluster]
    hypotheses: list[Hypothesis]
    #: Proposals as they arrive from the parallel fan-out; the reducer merges
    #: one list per `propose_one` branch.
    candidate_configs: Annotated[list[CandidateProposal], operator.add]
    #: The subset that survived schema and safety validation. Rewritten whole by
    #: `validate`, so it is not a reducer field.
    validated: list[CandidateProposal]
    rejected: Annotated[list[RejectedProposal], operator.add]
    #: Set on the fan-out branches only.
    candidate_index: int
    eval_results: dict[str, DefenseEvaluation]
    scores: dict[str, float]
    chosen: DefenseConfig | None
    chosen_id: str | None
    utility_baseline: float
    propose_rounds: int
    status: str
