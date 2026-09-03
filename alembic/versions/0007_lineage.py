"""Attack lineage: the second parent and the mutation operator

Revision ID: 0007_lineage
Revises: 0006_archive
Create Date: Phase 4

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007_lineage"
down_revision: str | None = "0006_archive"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Named mutation operators are what make the attacker's search legible in
    # the writeup, so the lineage is stored rather than inferred.
    op.add_column("attacks", sa.Column("recombined_with", sa.UUID(), nullable=True))
    op.add_column("attacks", sa.Column("mutation_operator", sa.String(length=32), nullable=True))
    op.create_foreign_key(
        op.f("fk_attacks_recombined_with_attacks"),
        "attacks",
        "attacks",
        ["recombined_with"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(op.f("fk_attacks_recombined_with_attacks"), "attacks", type_="foreignkey")
    op.drop_column("attacks", "mutation_operator")
    op.drop_column("attacks", "recombined_with")
