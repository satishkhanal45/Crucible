"""The archive: admission, novelty gating, elites, and holdout assignment.

Three of the project's never-cut properties live here.

* **Novelty pressure.** `submit()` is the only path from a generated attack to
  the executor, and it applies the novelty filter first. An attack below
  `MIN_NOVELTY` is rejected *before execution*: the executor is never called,
  the rejection is persisted with its nearest neighbour's id, and the budget is
  not spent on a rediscovery.
* **Holdout discipline.** Holdout is assigned when an attack is admitted, which
  is strictly before it can be executed, and the flag is written to the row in
  the same transaction.
* **The archive itself.** Every attack ever generated, with its embedding,
  lineage, cell and outcome history, which is what makes full-archive
  re-evaluation meaningful.
"""

from __future__ import annotations

import random
import uuid
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from crucible.archive.fitness import fitness as compute_fitness
from crucible.archive.fitness import generality
from crucible.archive.grid import Coverage, coverage
from crucible.archive.novelty import (
    MIN_NOVELTY,
    NoveltyRejection,
    NoveltyScore,
    payload_fingerprint,
    score,
)
from crucible.db.session import Database
from crucible.defenses.config import DefenseConfig
from crucible.logging import get_logger
from crucible.repositories.attacks import AttackRepository
from crucible.repositories.cells import CellRepository, RejectionRepository
from crucible.schemas.archive import (
    ArchivedAttack,
    ArchiveStats,
    CellRecord,
    NewArchivedAttack,
    NoveltyDistribution,
)
from crucible.schemas.attack import Attack
from crucible.schemas.attempt import AttemptResult
from crucible.schemas.outcome import Outcome
from crucible.services.embeddings import Embedder

logger = get_logger(__name__)

#: docs/spec.md section 10: 20% of seeds and 20% of each round's new attacks.
HOLDOUT_RATIO = 0.2


class AttemptRunner(Protocol):
    """What `submit()` needs from the executor. Kept narrow so the novelty gate
    can be tested with a spy that fails loudly if it is ever called."""

    async def execute(
        self, attack: Attack, defense: DefenseConfig, *, force: bool = False
    ) -> AttemptResult: ...


class Admission(BaseModel):
    """Whether an attack entered the archive, and why not if it did not."""

    model_config = ConfigDict(frozen=True)

    accepted: bool
    novelty: NoveltyScore
    attack: ArchivedAttack | None = None
    rejection: NoveltyRejection | None = None


class Submission(BaseModel):
    """One attack's journey: novelty gate, then execution if it passed."""

    model_config = ConfigDict(frozen=True)

    admission: Admission
    attempt: AttemptResult | None = None

    @property
    def executed(self) -> bool:
        return self.attempt is not None


class ArchiveService:
    def __init__(
        self,
        database: Database,
        embedder: Embedder,
        *,
        min_novelty: float = MIN_NOVELTY,
        holdout_ratio: float = HOLDOUT_RATIO,
        rng: random.Random | None = None,
    ) -> None:
        self._database = database
        self._embedder = embedder
        self._min_novelty = min_novelty
        self._holdout_ratio = holdout_ratio
        # Seeded so that a run is reproducible: Phase 6 requires two runs with
        # the same seed to produce identical archives.
        self._rng = rng or random.Random(20260903)

    @property
    def min_novelty(self) -> float:
        return self._min_novelty

    @property
    def holdout_ratio(self) -> float:
        return self._holdout_ratio

    # ------------------------------------------------------------ admission

    def assign_holdout(self, count: int) -> list[bool]:
        """Randomly reserve exactly `ratio` of a batch, before any execution."""
        if count <= 0:
            return []
        reserved = round(count * self._holdout_ratio)
        flags = [True] * reserved + [False] * (count - reserved)
        self._rng.shuffle(flags)
        return flags

    async def novelty_of(
        self, embedding: Sequence[float], *, exclude_id: uuid.UUID | None = None
    ) -> NoveltyScore:
        async with self._database.session() as session:
            repository = AttackRepository(session)
            size = await repository.count()
            neighbours = await repository.nearest_neighbours(embedding, exclude_id=exclude_id)
        return score(neighbours, archive_size=size)

    async def admit(
        self, attack: Attack, *, round_number: int = 0, holdout: bool | None = None
    ) -> Admission:
        """Score novelty and, if it clears the bar, write the attack."""
        (admission,) = await self.admit_many([attack], round_number=round_number, holdout=holdout)
        return admission

    async def admit_many(
        self,
        attacks: Sequence[Attack],
        *,
        round_number: int = 0,
        holdout: bool | None = None,
    ) -> list[Admission]:
        """Admit a batch, reserving 20% of it for the holdout set.

        Holdout is decided here, before any of these attacks can be executed.
        """
        if not attacks:
            return []
        embeddings = await self._embedder.embed([attack.payload for attack in attacks])
        flags = (
            [holdout] * len(attacks) if holdout is not None else self.assign_holdout(len(attacks))
        )

        admissions: list[Admission] = []
        for attack, embedding, is_holdout in zip(attacks, embeddings, flags, strict=True):
            admissions.append(
                await self._admit_one(
                    attack, embedding, is_holdout=is_holdout, round_number=round_number
                )
            )
        return admissions

    async def _admit_one(
        self,
        attack: Attack,
        embedding: Sequence[float],
        *,
        is_holdout: bool,
        round_number: int,
    ) -> Admission:
        novelty = await self.novelty_of(embedding, exclude_id=attack.attack_id)

        if not novelty.is_novel(self._min_novelty):
            nearest = novelty.nearest
            rejection = NoveltyRejection(
                novelty=novelty.value,
                threshold=self._min_novelty,
                nearest_neighbour_id=nearest.attack_id if nearest else None,
                nearest_distance=nearest.distance if nearest else None,
                payload_hash=payload_fingerprint(attack.payload),
                round_number=round_number,
            )
            async with self._database.session() as session:
                await RejectionRepository(session).add(rejection)
            logger.info(
                "novelty.rejected",
                extra={
                    "novelty": round(novelty.value, 4),
                    "threshold": self._min_novelty,
                    "nearest_neighbour_id": str(rejection.nearest_neighbour_id),
                    "nearest_distance": rejection.nearest_distance,
                    "round": round_number,
                },
            )
            return Admission(accepted=False, novelty=novelty, rejection=rejection)

        stored = attack.model_copy(update={"is_holdout": is_holdout})
        async with self._database.session() as session:
            archived = await AttackRepository(session).add(
                NewArchivedAttack(
                    attack=stored,
                    embedding=tuple(embedding),
                    novelty_score=novelty.value,
                    round_generated=round_number,
                )
            )
        logger.info(
            "archive.admitted",
            extra={
                "attack_id": str(archived.id),
                "cell_key": archived.cell_key,
                "novelty": round(novelty.value, 4),
                "is_holdout": archived.is_holdout,
                "round": round_number,
            },
        )
        return Admission(accepted=True, novelty=novelty, attack=archived)

    # ----------------------------------------------------- novelty gate + run

    async def submit(
        self,
        attack: Attack,
        defense: DefenseConfig,
        executor: AttemptRunner,
        *,
        round_number: int = 0,
        holdout: bool | None = None,
    ) -> Submission:
        """The only path from a generated attack to the executor.

        A rejected attack never reaches `executor.execute`. That ordering is the
        anti-collapse mechanism and the budget guard at once.
        """
        admission = await self.admit(attack, round_number=round_number, holdout=holdout)
        if not admission.accepted or admission.attack is None:
            return Submission(admission=admission)

        result = await executor.execute(admission.attack.to_attack(), defense)
        await self.record_attempt(admission.attack.id, result, defense, round_number=round_number)
        return Submission(admission=admission, attempt=result)

    async def record_attempt(
        self,
        attack_id: uuid.UUID,
        result: AttemptResult,
        defense: DefenseConfig,
        *,
        round_number: int = 0,
    ) -> CellRecord | None:
        """Fold an attempt into the archive and refresh the attack's cell."""
        breached = result.outcome is Outcome.BREACHED
        async with self._database.session() as session:
            await AttackRepository(session).mark_attempt(
                attack_id, breached=breached, round_number=round_number
            )
        return await self.refresh_cell(attack_id, defense.fingerprint(), round_number=round_number)

    async def refresh_cell(
        self, attack_id: uuid.UUID, defense_config_id: str, *, round_number: int = 0
    ) -> CellRecord | None:
        """Recompute fitness and update the cell's elite if this attack beats it."""
        async with self._database.session() as session:
            attacks = AttackRepository(session)
            archived = await attacks.get(attack_id)
            if archived is None or archived.cell_key is None:
                # An unclassified attack occupies no cell and holds no elite.
                return None

            breach_rate = await attacks.breach_rate_against(attack_id, defense_config_id)
            breached_configs = await attacks.breached_config_ids(attack_id)
            all_configs = await attacks.all_config_ids()
            novelty = float(archived.novelty_score or 0.0)
            fitness = compute_fitness(
                breach_rate, novelty, generality(breached_configs, all_configs)
            )
            return await CellRepository(session).record_occupant(
                archived.cell_key,
                attack_id,
                fitness,
                round_number=round_number,
                # Elites are the mutation pool, so a holdout attack may occupy a
                # cell for coverage but may never hold it.
                eligible_for_elite=not archived.is_holdout,
            )

    # ---------------------------------------------------------------- seeds

    async def admit_seeds(self, seeds: Sequence[Attack]) -> list[Admission]:
        """Admit the seed corpus, reserving 20% of it as holdout.

        Idempotent: a seed already in the archive is skipped rather than
        re-admitted, because seed ids are stable and re-running `make seed`
        should not fill the archive with duplicates.
        """
        async with self._database.session() as session:
            repository = AttackRepository(session)
            fresh = [seed for seed in seeds if await repository.get(seed.attack_id) is None]
        if not fresh:
            logger.info("archive.seeds_already_loaded", extra={"seeds": len(seeds)})
            return []
        return await self.admit_many(fresh, round_number=0)

    # ---------------------------------------------------------------- stats

    async def coverage(self) -> Coverage:
        async with self._database.session() as session:
            return coverage(await AttackRepository(session).occupied_cell_keys())

    async def stats(self) -> ArchiveStats:
        async with self._database.session() as session:
            attacks = AttackRepository(session)
            cells = CellRepository(session)
            rejections = RejectionRepository(session)

            size = await attacks.count()
            holdout = await attacks.count_holdout()
            unclassified = await attacks.count_unclassified()
            occupied = await attacks.occupied_cell_keys()
            scores = await attacks.novelty_scores()
            rejected = await rejections.count()
            elites = await cells.occupied()

        seen = size + rejected
        return ArchiveStats(
            archive_size=size,
            holdout_count=holdout,
            unclassified_count=unclassified,
            coverage=coverage(occupied),
            novelty=_distribution(scores, self._min_novelty),
            rejections=rejected,
            rejection_rate=rejected / seen if seen else 0.0,
            elites=tuple(elites),
        )


def _distribution(scores: Sequence[float], threshold: float) -> NoveltyDistribution:
    if not scores:
        return NoveltyDistribution(count=0)
    ordered = sorted(scores)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return NoveltyDistribution(
        count=len(ordered),
        minimum=ordered[0],
        median=median,
        mean=sum(ordered) / len(ordered),
        maximum=ordered[-1],
        below_threshold=sum(1 for value in ordered if value < threshold),
    )


class BoundNoveltyGate:
    """The Phase 3 admission gate with its defense and executor already bound.

    The attacker calls `submit(attack)` and never holds a `DefenseConfig`: in
    black-box mode it must not be able to reach one even by accident.
    """

    def __init__(
        self, archive: ArchiveService, defense: DefenseConfig, executor: AttemptRunner
    ) -> None:
        self._archive = archive
        self._defense = defense
        self._executor = executor

    async def submit(self, attack: Attack, *, round_number: int = 0) -> Submission:
        return await self._archive.submit(
            attack, self._defense, self._executor, round_number=round_number
        )
