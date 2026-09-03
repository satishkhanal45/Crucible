"""What each oracle tier returns, and what their combination decides."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from crucible.schemas.outcome import Outcome, Tier, Verdict
from crucible.target.canary import CanaryClass


class Tier1Result(BaseModel):
    """Deterministic evidence. This is the headline metric."""

    model_config = ConfigDict(frozen=True)

    breached: bool = False
    #: Which canary classes were found. Never the values.
    canaries_hit: tuple[CanaryClass, ...] = ()
    #: Privileged tools that ran without authorization.
    unauthorized_tools: tuple[str, ...] = ()
    #: Privileged tools the defense stack stopped before they ran.
    blocked_tools: tuple[str, ...] = ()
    contract_violated: bool = False
    confidential_doc_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


class Tier2Result(BaseModel):
    """Rule-based behavioural checks against the target's `BehaviorSpec`."""

    model_config = ConfigDict(frozen=True)

    verdict: Verdict = Verdict.INCONCLUSIVE
    #: True when the target declined. A refusal is a held behaviour, not a block.
    refused: bool = False
    reasons: tuple[str, ...] = ()


class Tier3Result(BaseModel):
    """LLM judge. Not implemented in this build (cut B4 / deferred item D4)."""

    model_config = ConfigDict(frozen=True)

    verdict: Verdict = Verdict.INCONCLUSIVE
    score: float | None = None
    rationale: str | None = None


class OracleVerdict(BaseModel):
    """The oracle's decision about one attempt."""

    model_config = ConfigDict(frozen=True)

    outcome: Outcome
    tier: Tier
    tier1: Tier1Result = Tier1Result()
    tier2: Tier2Result = Tier2Result()
    tier3: Tier3Result = Tier3Result()
    reasons: tuple[str, ...] = ()

    @property
    def canaries_hit(self) -> tuple[CanaryClass, ...]:
        return self.tier1.canaries_hit

    @property
    def unauthorized_tools(self) -> tuple[str, ...]:
        return self.tier1.unauthorized_tools

    @property
    def breached(self) -> bool:
        return self.outcome is Outcome.BREACHED
