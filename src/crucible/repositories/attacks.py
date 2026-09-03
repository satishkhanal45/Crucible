"""The only module that reads or writes `attacks`.

**Holdout isolation is enforced here, and nowhere else is trusted to do it.**

docs/spec.md section 10 requires that a holdout attack never enters the
mutation pool and never appears in a prompt shown to an agent. Two methods on
this class may hand attacks to an agent — `get_attacks_for_mutation()` and
`get_attacks_for_defender()` — and both filter `is_holdout = false` in SQL.
Neither takes a parameter that could relax the filter: there is no
`include_holdout`, no `force`, and no debug bypass, because a flag that exists
will eventually be passed.

Every other method here is archive maintenance: statistics, novelty, elite
bookkeeping. Those must never be called from a module that builds a prompt, and
`tests/unit/test_holdout_isolation.py` fails the suite if one ever is.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from decimal import Decimal

from sqlalchemy import Select, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from crucible.archive.novelty import K_NEIGHBOURS, Neighbour
from crucible.db.models import AttackRow, Attempt, CellRow
from crucible.schemas.archive import ArchivedAttack, NewArchivedAttack
from crucible.schemas.outcome import Outcome

#: The only methods that may return attacks to an agent. Both filter holdout.
AGENT_SAFE_METHODS = frozenset({"get_attacks_for_mutation", "get_attacks_for_defender"})

#: HNSW is approximate. A wide search list makes recall exact at the scales this
#: project reaches (see `tests/integration/test_archive_pgvector.py`, which
#: asserts k-NN equals brute force), and ordering by `(distance, id)` makes the
#: result deterministic, which Phase 6's "same seed, same archive" test needs.
HNSW_EF_SEARCH = 1000

#: Extra candidates fetched so that a tie at the k-th place is broken on id
#: rather than on whatever order the index happened to return.
TIE_MARGIN = 5


class AttackRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------- agent-facing

    async def get_attacks_for_mutation(
        self, *, limit: int | None = None, cell_keys: Sequence[str] | None = None
    ) -> list[ArchivedAttack]:
        """The mutation pool: cell elites, never holdout, never retired."""
        statement = (
            select(AttackRow)
            .join(CellRow, CellRow.elite_attack_id == AttackRow.id)
            .where(AttackRow.is_holdout.is_(False))
            .where(AttackRow.retired.is_(False))
            .order_by(CellRow.elite_fitness.desc(), AttackRow.id)
        )
        if cell_keys:
            statement = statement.where(CellRow.cell_key.in_(list(cell_keys)))
        return await self._fetch(statement, limit)

    async def get_attacks_for_defender(
        self,
        *,
        round_number: int | None = None,
        defense_config_id: str | None = None,
        limit: int | None = None,
    ) -> list[ArchivedAttack]:
        """Breaching attacks the defender is allowed to see. Never holdout."""
        breaches = (
            select(Attempt.attack_id).where(Attempt.outcome == Outcome.BREACHED.value).distinct()
        )
        if defense_config_id is not None:
            breaches = breaches.where(Attempt.defense_config_id == defense_config_id)

        statement = (
            select(AttackRow)
            .where(AttackRow.is_holdout.is_(False))
            .where(AttackRow.retired.is_(False))
            .where(AttackRow.id.in_(breaches))
            .order_by(AttackRow.round_generated, AttackRow.id)
        )
        if round_number is not None:
            statement = statement.where(AttackRow.round_generated == round_number)
        return await self._fetch(statement, limit)

    async def _fetch(
        self, statement: Select[tuple[AttackRow]], limit: int | None
    ) -> list[ArchivedAttack]:
        if limit is not None:
            statement = statement.limit(limit)
        rows = (await self._session.execute(statement)).scalars().all()
        return [ArchivedAttack.model_validate(row) for row in rows]

    # -------------------------------------------------- archive maintenance

    async def add(self, entry: NewArchivedAttack) -> ArchivedAttack:
        attack = entry.attack
        row = AttackRow(
            id=attack.attack_id,
            round_generated=entry.round_generated,
            parent_id=attack.parent_id,
            payload=attack.payload,
            vector=attack.vector.value,
            objective=attack.objective.value if attack.objective else None,
            technique=attack.technique.value if attack.technique else None,
            cell_key=attack.cell_key,
            embedding=list(entry.embedding),
            novelty_score=(
                Decimal(str(round(entry.novelty_score, 5)))
                if entry.novelty_score is not None
                else None
            ),
            is_holdout=attack.is_holdout,
            benign_user_input=attack.benign_user_input,
            carrier_title=attack.carrier_title,
            carrier_doc_id=attack.carrier_doc_id,
        )
        self._session.add(row)
        await self._session.flush()
        await self._session.refresh(row)
        return ArchivedAttack.model_validate(row)

    async def get(self, attack_id: uuid.UUID) -> ArchivedAttack | None:
        row = await self._session.get(AttackRow, attack_id)
        return None if row is None else ArchivedAttack.model_validate(row)

    async def embedding_of(self, attack_id: uuid.UUID) -> tuple[float, ...] | None:
        row = await self._session.get(AttackRow, attack_id)
        return None if row is None else tuple(float(value) for value in row.embedding)

    async def count(self) -> int:
        total = await self._session.scalar(select(func.count()).select_from(AttackRow))
        return int(total or 0)

    async def count_holdout(self) -> int:
        total = await self._session.scalar(
            select(func.count()).select_from(AttackRow).where(AttackRow.is_holdout.is_(True))
        )
        return int(total or 0)

    async def count_unclassified(self) -> int:
        total = await self._session.scalar(
            select(func.count()).select_from(AttackRow).where(AttackRow.cell_key.is_(None))
        )
        return int(total or 0)

    async def occupied_cell_keys(self) -> list[str]:
        """Distinct cells with at least one attack in them."""
        rows = await self._session.execute(
            select(AttackRow.cell_key)
            .where(AttackRow.cell_key.is_not(None))
            .where(AttackRow.retired.is_(False))
            .distinct()
        )
        return [key for key in rows.scalars().all() if key]

    async def novelty_scores(self) -> list[float]:
        rows = await self._session.execute(
            select(AttackRow.novelty_score).where(AttackRow.novelty_score.is_not(None))
        )
        return [float(value) for value in rows.scalars().all() if value is not None]

    async def nearest_neighbours(
        self,
        embedding: Sequence[float],
        *,
        k: int = K_NEIGHBOURS,
        exclude_id: uuid.UUID | None = None,
    ) -> list[Neighbour]:
        """The k nearest archived attacks, by cosine distance.

        Three details matter here.

        * The SQL orders by distance alone. An HNSW index can only answer
          `ORDER BY embedding <=> q`; adding a second sort key would force the
          planner into a sequential scan and sort, and the index would never be
          used.
        * Ties are broken on the attack id **after** the rows come back, so an
          identical archive always yields an identical neighbour list. Phase 6
          requires two runs with the same seed to build the same archive.
        * `exclude_id` is dropped in Python rather than filtered in SQL:
          filtering inside an approximate index scan is what lets HNSW
          over-filter and silently return the wrong neighbours.
        """
        await self._session.execute(text(f"SET LOCAL hnsw.ef_search = {HNSW_EF_SEARCH}"))
        distance = AttackRow.embedding.cosine_distance(list(embedding)).label("distance")
        wanted = k + (1 if exclude_id is not None else 0) + TIE_MARGIN
        rows = (
            await self._session.execute(
                select(AttackRow.id, distance).order_by(distance).limit(wanted)
            )
        ).all()
        neighbours = sorted(
            (
                Neighbour(attack_id=row[0], distance=float(row.distance))
                for row in rows
                if row[0] != exclude_id
            ),
            key=lambda neighbour: (neighbour.distance, str(neighbour.attack_id)),
        )
        return neighbours[:k]

    async def mark_attempt(
        self, attack_id: uuid.UUID, *, breached: bool, round_number: int
    ) -> None:
        """Fold one attempt's result into the attack's running totals."""
        row = await self._session.get(AttackRow, attack_id)
        if row is None:
            return
        row.total_attempts += 1
        if breached:
            row.total_breaches += 1
            if row.first_breach_round is None:
                row.first_breach_round = round_number

    async def breached_config_ids(self, attack_id: uuid.UUID) -> set[str]:
        """Every defense config this attack has ever breached."""
        rows = await self._session.execute(
            select(Attempt.defense_config_id)
            .where(Attempt.attack_id == attack_id)
            .where(Attempt.outcome == Outcome.BREACHED.value)
            .distinct()
        )
        return set(rows.scalars().all())

    async def all_config_ids(self) -> set[str]:
        """Every defense config that has ever been evaluated."""
        rows = await self._session.execute(select(Attempt.defense_config_id).distinct())
        return set(rows.scalars().all())

    async def breach_rate_against(self, attack_id: uuid.UUID, defense_config_id: str) -> float:
        rows = (
            (
                await self._session.execute(
                    select(Attempt.outcome)
                    .where(Attempt.attack_id == attack_id)
                    .where(Attempt.defense_config_id == defense_config_id)
                )
            )
            .scalars()
            .all()
        )
        if not rows:
            return 0.0
        breaches = sum(1 for outcome in rows if outcome == Outcome.BREACHED.value)
        return breaches / len(rows)

    async def list_all(self, *, limit: int | None = None) -> list[ArchivedAttack]:
        """Every attack, holdout included. Archive maintenance only."""
        statement = select(AttackRow).order_by(AttackRow.created_at, AttackRow.id)
        return await self._fetch(statement, limit)

    async def list_non_holdout(self, *, limit: int | None = None) -> list[ArchivedAttack]:
        """The full non-holdout archive, for **evaluation**, not for an agent.

        docs/spec.md non-negotiable 2: a round's recorded block rate comes from
        running the selected config against every non-holdout attack, not just
        the ones that breached. `get_attacks_for_defender` deliberately returns
        far less than this, and must keep doing so — this method exists for the
        measurement layer and is off-limits to prompt builders, which
        `tests/unit/test_holdout_isolation.py` enforces.
        """
        statement = (
            select(AttackRow)
            .where(AttackRow.is_holdout.is_(False))
            .where(AttackRow.retired.is_(False))
            .order_by(AttackRow.created_at, AttackRow.id)
        )
        return await self._fetch(statement, limit)

    async def list_holdout(self, *, limit: int | None = None) -> list[ArchivedAttack]:
        """The holdout set, for the generalization measurement only.

        No agent may see these, ever. Only the evaluation service calls this,
        and only to compute `holdout_block_rate`.
        """
        statement = (
            select(AttackRow)
            .where(AttackRow.is_holdout.is_(True))
            .where(AttackRow.retired.is_(False))
            .order_by(AttackRow.created_at, AttackRow.id)
        )
        return await self._fetch(statement, limit)

    async def ever_breached_ids(self) -> set[uuid.UUID]:
        """Attacks that have breached at least one config, ever.

        The stratified screening sample in docs/spec.md section 12 always
        includes these.
        """
        rows = await self._session.execute(
            select(Attempt.attack_id).where(Attempt.outcome == Outcome.BREACHED.value).distinct()
        )
        return set(rows.scalars().all())
