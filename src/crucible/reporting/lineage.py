"""Attack lineage as an indented text tree.

The lineage is what makes the co-evolution visible: an attack that breaches in
round 6 is a named mutation of something that breached in round 2, which is a
mutation of a hand-written seed. A seed has no parent and renders as a root.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from crucible.reporting.redaction import present_payload
from crucible.schemas.archive import ArchivedAttack

#: A lineage deeper than this is truncated rather than allowed to run away.
MAX_DEPTH = 12


class LineageNode(BaseModel):
    """One ancestor in a lineage chain."""

    model_config = ConfigDict(frozen=True)

    attack_id: uuid.UUID
    depth: int
    round_generated: int
    cell_key: str | None
    mutation_operator: str | None
    summary: str

    @property
    def is_seed(self) -> bool:
        return self.mutation_operator is None


def build_lineage(
    attack: ArchivedAttack,
    by_id: Mapping[uuid.UUID, ArchivedAttack],
    *,
    include_payloads: bool = False,
) -> tuple[LineageNode, ...]:
    """From the seed ancestor down to this attack."""
    chain: list[ArchivedAttack] = []
    current: ArchivedAttack | None = attack
    seen: set[uuid.UUID] = set()

    while current is not None and len(chain) < MAX_DEPTH:
        if current.id in seen:
            break
        seen.add(current.id)
        chain.append(current)
        current = by_id.get(current.parent_id) if current.parent_id else None

    chain.reverse()
    return tuple(
        LineageNode(
            attack_id=item.id,
            depth=depth,
            round_generated=item.round_generated,
            cell_key=item.cell_key,
            mutation_operator=item.mutation_operator,
            summary=present_payload(
                item.payload,
                objective=item.objective.value if item.objective else None,
                vector=item.vector.value,
                technique=item.technique.value if item.technique else None,
                include_payloads=include_payloads,
            ),
        )
        for depth, item in enumerate(chain)
    )


def render_lineage(nodes: Sequence[LineageNode]) -> str:
    """An indented tree. The root is the seed; each level is one mutation."""
    if not nodes:
        return "(no lineage recorded)"
    lines: list[str] = []
    for node in nodes:
        indent = "  " * node.depth
        marker = "seed" if node.is_seed else f"{node.mutation_operator}"
        prefix = "" if node.depth == 0 else "└─ "
        lines.append(
            f"{indent}{prefix}[{marker}] round {node.round_generated} "
            f"{node.cell_key or 'unclassified'} ({node.attack_id})"
        )
        lines.append(f"{indent}   {node.summary}")
    return "\n".join(lines)
