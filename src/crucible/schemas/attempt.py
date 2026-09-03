"""Boundary schemas for attempts: one execution of one attack against one config."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from crucible.schemas.outcome import Outcome, Tier
from crucible.schemas.taxonomy import DeliveryVector


class NewAttempt(BaseModel):
    """An attempt about to be persisted."""

    model_config = ConfigDict(frozen=True)

    attack_id: uuid.UUID
    #: The `DefenseConfig` fingerprint. Order-independent, so two semantically
    #: identical configs share a cache entry (docs/spec.md section 9).
    defense_config_id: str = Field(min_length=1, max_length=64)
    round_id: uuid.UUID | None = None
    vector: DeliveryVector
    outcome: Outcome
    tier: Tier = Tier.NONE
    canaries_hit: tuple[str, ...] = ()
    unauthorized_tools: tuple[str, ...] = ()
    #: Privileged calls the structural layer stopped. The attempt is still
    #: `blocked`; this records that a hijack was attempted and caught.
    blocked_tools: tuple[str, ...] = ()
    judge_score: Decimal | None = None
    judge_rationale: str | None = None
    response_text: str | None = None
    #: Sufficient to replay the attempt. Contains this attempt's canary values,
    #: so it is never logged and never written to a report unredacted.
    trace: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int | None = None
    cost_usd: Decimal | None = None


class AttemptRecord(NewAttempt):
    """A persisted attempt."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    created_at: datetime

    @property
    def hijack_attempted_but_blocked(self) -> bool:
        """The model emitted a privileged call and Layer 5 stopped it.

        Distinct from "the model was never hijacked", which is the same
        `blocked` outcome with no blocked tools.
        """
        return bool(self.blocked_tools)


class AttemptResult(BaseModel):
    """What the executor returns for one attack/config pair."""

    model_config = ConfigDict(frozen=True)

    attempt: AttemptRecord
    #: True when the stored outcome was reused instead of re-running the target.
    cache_hit: bool = False

    @property
    def outcome(self) -> Outcome:
        return self.attempt.outcome


class ExecutionMetrics(BaseModel):
    """Counters for one executor run. Cheap to log, and never contains payloads."""

    model_config = ConfigDict(frozen=True)

    requested: int = 0
    executed: int = 0
    cache_hits: int = 0
    errors: int = 0
    timeouts: int = 0
    egress_violations: int = 0

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / self.requested if self.requested else 0.0
