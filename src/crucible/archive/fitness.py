"""Fitness and generality (docs/spec.md section 10).

    fitness(a) = breach_rate(a, current_defense)
               + 0.3 * novelty(a)
               + 0.2 * generality(a)

    generality(a) = fraction of ALL past defense configs a still breaches

The generality term is what makes the archive accumulate attacks that survive
hardening rather than attacks that happen to beat this round's configuration.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable

from pydantic import BaseModel, ConfigDict

NOVELTY_WEIGHT = 0.3
GENERALITY_WEIGHT = 0.2


def generality(breached_config_ids: Iterable[str], past_config_ids: Collection[str]) -> float:
    """Share of past defense configs this attack still breaches.

    With no past configs — which is every round until Phase 5 produces the
    first one — generality is 0.0, not 1.0 and not an error. Zero is the honest
    reading: an attack that has never survived a hardening step has shown no
    evidence of generality, and returning 1.0 would inflate every fitness score
    in the archive before a single defense exists.
    """
    past = set(past_config_ids)
    if not past:
        return 0.0
    breached = set(breached_config_ids) & past
    return len(breached) / len(past)


class FitnessBreakdown(BaseModel):
    """Fitness with its terms kept, so a report can explain a ranking."""

    model_config = ConfigDict(frozen=True)

    breach_rate: float
    novelty: float
    generality: float

    @property
    def fitness(self) -> float:
        return (
            self.breach_rate + NOVELTY_WEIGHT * self.novelty + GENERALITY_WEIGHT * self.generality
        )


def fitness(breach_rate: float, novelty: float, generality_score: float) -> float:
    return FitnessBreakdown(
        breach_rate=breach_rate, novelty=novelty, generality=generality_score
    ).fitness
