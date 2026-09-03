"""Step 10 — collapse detection (docs/spec.md section 14).

A run that halts is a *result*. `attacker_exhausted` after nine rounds says the
search found everything this attacker can find against this target, which is a
finding worth writing up; it is not a failure, and the run's status says so.

Precedence when several signals fire at once: the single-round signals that
invalidate the round's own numbers come first (budget, over-blocking,
overfitting), then the multi-round search signals.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from crucible.logging import get_logger
from crucible.loop.reports import HaltReason

logger = get_logger(__name__)

MIN_MEAN_NOVELTY = 0.2
NOVELTY_ROUNDS = 3
STALLED_ROUNDS = 3
MAX_REJECTION_RATE = 0.8
REJECTION_ROUNDS = 2
MIN_UTILITY_RATIO = 0.85
MAX_OVERFIT_GAP = 0.25


class RoundSignals(BaseModel):
    """The per-round numbers collapse detection reads."""

    model_config = ConfigDict(frozen=True)

    round_number: int
    mean_novelty: float = 1.0
    new_cells: int = 0
    novelty_rejection_rate: float = 0.0
    utility_pass_rate: float = 1.0
    overfit_gap: float = 0.0
    cost_usd: Decimal = Decimal(0)


def detect(
    history: Sequence[RoundSignals],
    *,
    baseline_utility: float = 1.0,
    budget_usd: Decimal | None = None,
) -> HaltReason | None:
    """The first signal that has tripped, or None to keep going."""
    if not history:
        return None
    latest = history[-1]

    if budget_usd is not None and latest.cost_usd > budget_usd:
        return _halt(HaltReason.BUDGET_EXCEEDED, latest, cost=str(latest.cost_usd))

    if baseline_utility > 0 and latest.utility_pass_rate < MIN_UTILITY_RATIO * baseline_utility:
        return _halt(
            HaltReason.UTILITY_COLLAPSE,
            latest,
            utility=latest.utility_pass_rate,
            floor=MIN_UTILITY_RATIO * baseline_utility,
        )

    if latest.overfit_gap > MAX_OVERFIT_GAP:
        return _halt(HaltReason.OVERFITTING, latest, gap=latest.overfit_gap)

    if len(history) >= NOVELTY_ROUNDS and all(
        round_.mean_novelty < MIN_MEAN_NOVELTY for round_ in history[-NOVELTY_ROUNDS:]
    ):
        return _halt(HaltReason.ATTACKER_EXHAUSTED, latest, rounds=NOVELTY_ROUNDS)

    if len(history) >= STALLED_ROUNDS and all(
        round_.new_cells == 0 for round_ in history[-STALLED_ROUNDS:]
    ):
        return _halt(HaltReason.SEARCH_STALLED, latest, rounds=STALLED_ROUNDS)

    if len(history) >= REJECTION_ROUNDS and all(
        round_.novelty_rejection_rate > MAX_REJECTION_RATE for round_ in history[-REJECTION_ROUNDS:]
    ):
        return _halt(HaltReason.REDISCOVERY_ONLY, latest, rounds=REJECTION_ROUNDS)

    return None


def _halt(reason: HaltReason, latest: RoundSignals, **detail: object) -> HaltReason:
    logger.warning(
        "loop.halt",
        extra={"halt_reason": reason.value, "round": latest.round_number, **detail},
    )
    return reason
