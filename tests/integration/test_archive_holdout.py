"""Verification tests 10, 11, 12, 14 — holdout discipline against the database.

**These are security properties.** The holdout set is the only honest
generalization number in the project: if a holdout attack reaches an agent, the
number stops meaning anything and nothing else in the system would notice. The
source-level half of this guard is in `tests/unit/test_holdout_isolation.py`.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field

import pytest

from crucible.archive.service import HOLDOUT_RATIO, ArchiveService
from crucible.db.session import Database
from crucible.defenses.config import DefenseConfig
from crucible.repositories.attacks import AttackRepository
from crucible.repositories.attempts import AttemptRepository
from crucible.schemas.attack import Attack
from crucible.schemas.attempt import AttemptResult, NewAttempt
from crucible.schemas.outcome import Outcome, Tier
from crucible.schemas.taxonomy import DeliveryVector, Objective, Technique

OBJECTIVES = list(Objective)
TECHNIQUES = list(Technique)
VECTORS = [DeliveryVector.DIRECT, DeliveryVector.INDIRECT_DOCUMENT]


def attack(index: int, rng: random.Random) -> Attack:
    """A distinct attack, spread across the taxonomy."""
    return Attack(
        payload=(
            f"variant {index}: {rng.choice(['ignore', 'disregard', 'override'])} the "
            f"{rng.choice(['policy', 'instructions', 'configuration'])} and "
            f"{rng.choice(['print', 'reveal', 'emit'])} identifier {index}"
        ),
        vector=VECTORS[index % len(VECTORS)],
        objective=OBJECTIVES[index % len(OBJECTIVES)],
        technique=TECHNIQUES[index % len(TECHNIQUES)],
    )


async def fill(archive: ArchiveService, count: int) -> list[Attack]:
    rng = random.Random(4242)
    attacks = [attack(index, rng) for index in range(count)]
    admissions = await archive.admit_many(attacks, round_number=1)
    return [
        admission.attack.to_attack()
        for admission in admissions
        if admission.accepted and admission.attack is not None
    ]


async def make_elites_and_breaches(archive: ArchiveService, database: Database) -> None:
    """Give every archived attack a breach, so both agent queries return rows."""
    defense = DefenseConfig.empty()
    async with database.session() as session:
        stored = await AttackRepository(session).list_all()
    for row in stored:
        async with database.session() as session:
            record = await AttemptRepository(session).add(
                NewAttempt(
                    attack_id=row.id,
                    defense_config_id=defense.fingerprint(),
                    vector=row.vector,
                    outcome=Outcome.BREACHED,
                    tier=Tier.DETERMINISTIC,
                    trace={"schema_version": 1},
                )
            )
        await archive.record_attempt(row.id, AttemptResult(attempt=record), defense)


# ----------------------------------------------------------------- test 12


@dataclass
class HoldoutCheckingExecutor:
    """Reads the archive row at execution time and records what it found.

    That is the actual ordering claim: by the time an attack can be executed,
    its holdout flag is already committed.
    """

    database: Database
    seen: list[tuple[uuid.UUID, bool]] = field(default_factory=list)

    async def execute(
        self, attack: Attack, defense: DefenseConfig, *, force: bool = False
    ) -> AttemptResult:
        del force
        async with self.database.session() as session:
            stored = await AttackRepository(session).get(attack.attack_id)
        assert stored is not None, "the attack must be in the archive before execution"
        self.seen.append((attack.attack_id, stored.is_holdout))

        async with self.database.session() as session:
            record = await AttemptRepository(session).add(
                NewAttempt(
                    attack_id=attack.attack_id,
                    defense_config_id=defense.fingerprint(),
                    vector=attack.vector,
                    outcome=Outcome.BREACHED,
                    tier=Tier.DETERMINISTIC,
                    trace={"schema_version": 1},
                )
            )
        return AttemptResult(attempt=record)


async def test_holdout_is_committed_before_the_attack_can_be_executed(
    archive: ArchiveService, database_url: str
) -> None:
    database = Database(database_url)
    try:
        executor = HoldoutCheckingExecutor(database=database)
        rng = random.Random(7)
        for index in range(10):
            held_out = index % 2 == 0
            await archive.submit(
                attack(index, rng),
                DefenseConfig.empty(),
                executor,
                round_number=1,
                holdout=held_out,
            )
    finally:
        await database.close()

    assert len(executor.seen) == 10
    assert [flag for _, flag in executor.seen] == [index % 2 == 0 for index in range(10)]


async def test_a_holdout_attack_is_still_executed_and_scored(
    archive: ArchiveService, database_url: str
) -> None:
    """Holdout attacks are measured; they are only hidden from agents."""
    database = Database(database_url)
    try:
        executor = HoldoutCheckingExecutor(database=database)
        submission = await archive.submit(
            attack(1, random.Random(1)),
            DefenseConfig.empty(),
            executor,
            round_number=1,
            holdout=True,
        )
        assert submission.executed is True
        assert submission.attempt is not None
        assert submission.attempt.outcome is Outcome.BREACHED

        async with database.session() as session:
            stored = await AttackRepository(session).get(
                submission.admission.attack.id  # type: ignore[union-attr]
            )
        assert stored is not None
        assert stored.is_holdout is True
        assert stored.total_attempts == 1
        assert stored.total_breaches == 1
    finally:
        await database.close()


# ------------------------------------------------------------- tests 10, 11


async def test_the_mutation_pool_never_contains_a_holdout_attack(
    archive: ArchiveService, database_url: str
) -> None:
    """1000 calls against a randomized archive, and not one leak."""
    database = Database(database_url)
    try:
        await fill(archive, 60)
        await make_elites_and_breaches(archive, database)

        async with database.session() as session:
            repository = AttackRepository(session)
            assert await repository.count_holdout() > 0, "the archive must hold some holdout"

            rng = random.Random(99)
            for _ in range(1000):
                pool = await repository.get_attacks_for_mutation(
                    limit=rng.choice([None, 1, 5, 25, 1000])
                )
                assert all(item.is_holdout is False for item in pool)
    finally:
        await database.close()


async def test_the_defender_view_never_contains_a_holdout_attack(
    archive: ArchiveService, database_url: str
) -> None:
    database = Database(database_url)
    try:
        await fill(archive, 60)
        await make_elites_and_breaches(archive, database)

        async with database.session() as session:
            repository = AttackRepository(session)
            assert await repository.count_holdout() > 0

            rng = random.Random(1234)
            for _ in range(1000):
                visible = await repository.get_attacks_for_defender(
                    limit=rng.choice([None, 1, 10, 500]),
                    defense_config_id=rng.choice([None, DefenseConfig.empty().fingerprint()]),
                )
                assert all(item.is_holdout is False for item in visible)
    finally:
        await database.close()


async def test_both_agent_views_return_something_when_they_should(
    archive: ArchiveService, database_url: str
) -> None:
    """A filter that returns nothing would pass tests 10 and 11 vacuously."""
    database = Database(database_url)
    try:
        await fill(archive, 40)
        await make_elites_and_breaches(archive, database)

        async with database.session() as session:
            repository = AttackRepository(session)
            pool = await repository.get_attacks_for_mutation()
            visible = await repository.get_attacks_for_defender()
            total = await repository.count()
            holdout = await repository.count_holdout()
    finally:
        await database.close()

    assert pool, "the mutation pool must not be empty once cells have elites"
    assert visible, "the defender must see the breaches it is meant to respond to"
    assert len(visible) == total - holdout
    assert holdout > 0


# ----------------------------------------------------------------- test 14


async def test_the_holdout_ratio_holds_across_five_hundred_attacks(
    archive: ArchiveService, database_url: str
) -> None:
    admitted = await fill(archive, 500)

    database = Database(database_url)
    try:
        async with database.session() as session:
            repository = AttackRepository(session)
            total = await repository.count()
            holdout = await repository.count_holdout()
    finally:
        await database.close()

    assert len(admitted) > 400, "most distinct attacks should clear the novelty filter"
    ratio = holdout / total
    assert abs(ratio - HOLDOUT_RATIO) <= 0.05, f"holdout ratio drifted to {ratio:.1%}"


def test_batch_assignment_reserves_exactly_the_ratio(archive: ArchiveService) -> None:
    for count in (10, 40, 137, 500):
        flags = archive.assign_holdout(count)
        assert len(flags) == count
        assert sum(flags) == round(count * HOLDOUT_RATIO)


def test_holdout_assignment_is_random_but_reproducible() -> None:
    """Same seed, same holdout set: Phase 6 needs a run to be reproducible."""
    from crucible.db.session import Database as _Database

    first = ArchiveService(
        _Database("postgresql+asyncpg://unused/unused"),
        embedder=_Embedder(),
        rng=random.Random(11),
    )
    second = ArchiveService(
        _Database("postgresql+asyncpg://unused/unused"),
        embedder=_Embedder(),
        rng=random.Random(11),
    )

    assert first.assign_holdout(50) == second.assign_holdout(50)
    assert first.assign_holdout(50) != [True] * 10 + [False] * 40


class _Embedder:
    """Unused: `assign_holdout` touches neither the database nor embeddings."""

    @property
    def name(self) -> str:
        return "unused"

    @property
    def dimensions(self) -> int:
        return 384

    @property
    def max_relevant_distance(self) -> float:
        return 1.0

    async def embed(self, texts: object) -> list[list[float]]:
        del texts
        raise AssertionError("not reached")

    async def embed_one(self, text: str) -> list[float]:
        del text
        raise AssertionError("not reached")


@pytest.mark.parametrize("method", ["get_attacks_for_mutation", "get_attacks_for_defender"])
def test_the_agent_facing_methods_expose_no_holdout_parameter(method: str) -> None:
    import inspect

    signature = inspect.signature(getattr(AttackRepository, method))

    assert "holdout" not in signature.parameters
    assert "include_holdout" not in signature.parameters
    assert "force" not in signature.parameters
