"""The tiered success oracle.

Tier 1 is deterministic and is the headline metric. Tier 2 is rule-based. Tier 3
is not implemented in this build (cut B4); its seam is kept intact.
"""

from __future__ import annotations

from collections.abc import Iterable

from crucible.oracle import tier1 as _tier1
from crucible.oracle import tier2 as _tier2
from crucible.oracle.combine import (
    INCONCLUSIVE_WARNING_THRESHOLD,
    OutcomeAggregate,
    aggregate,
    combine,
)
from crucible.oracle.results import OracleVerdict, Tier1Result, Tier2Result, Tier3Result
from crucible.oracle.tier3 import StubTier3Judge, Tier3Judge
from crucible.schemas.attack import Attack
from crucible.schemas.outcome import Verdict
from crucible.target.adapter import BehaviorSpec, TargetResponse
from crucible.target.canary import CanarySet

__all__ = [
    "INCONCLUSIVE_WARNING_THRESHOLD",
    "OracleVerdict",
    "OutcomeAggregate",
    "StubTier3Judge",
    "Tier1Result",
    "Tier2Result",
    "Tier3Judge",
    "Tier3Result",
    "aggregate",
    "combine",
]


class Oracle:
    """Runs the tiers over one attempt and combines their verdicts."""

    def __init__(self, judge: Tier3Judge | None = None) -> None:
        # TODO(D4): pass a real judge here; it must never see the attack payload.
        self._judge = judge or StubTier3Judge()

    async def judge_attempt(
        self,
        response: TargetResponse,
        canaries: CanarySet,
        attack: Attack,
        behavior: BehaviorSpec,
        *,
        confidential_doc_ids: Iterable[str] = (),
        role_spec: str = "",
    ) -> OracleVerdict:
        first = _tier1.evaluate(
            response, canaries, attack, confidential_doc_ids=confidential_doc_ids
        )
        second = _tier2.evaluate(response, behavior)
        third = Tier3Result()
        if not first.breached and second.verdict is Verdict.INCONCLUSIVE:
            third = await self._judge.judge(response.text, role_spec)
        return combine(first, second, third)
