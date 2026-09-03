"""`crucible archive stats` — what the archive looks like right now.

Coverage is always printed with its denominator: "37/96 (38.5%)". A bare
occupancy count invites the reader to compare it against the wrong grid.

TODO(phase-6): fold this into the `crucible` typer CLI as `crucible archive
stats`, alongside `loop status`.
"""

from __future__ import annotations

import asyncio
import sys

from crucible.archive.novelty import MIN_NOVELTY
from crucible.archive.service import ArchiveService
from crucible.config import Settings, get_settings
from crucible.db.session import Database
from crucible.logging import configure_logging
from crucible.repositories.agreement import ClassifierAgreementRepository
from crucible.schemas.agreement import UNMEASURED_CAPTION, ClassifierAgreement
from crucible.schemas.archive import ArchiveStats
from crucible.services.embeddings import embedder_from_settings


async def gather(settings: Settings) -> tuple[ArchiveStats, ClassifierAgreement | None]:
    database = Database(settings.DATABASE_URL)
    try:
        service = ArchiveService(database, embedder_from_settings(settings))
        stats = await service.stats()
        async with database.session() as session:
            agreement = await ClassifierAgreementRepository(session).latest()
        return stats, agreement
    finally:
        await database.close()


def render(stats: ArchiveStats, agreement: ClassifierAgreement | None = None) -> str:
    lines = [
        "archive",
        f"  attacks          {stats.archive_size}",
        f"  holdout          {stats.holdout_count} ({stats.holdout_ratio:.1%})",
        f"  unclassified     {stats.unclassified_count} (occupy no cell)",
        f"  coverage         {stats.coverage}",
        # No coverage figure is printed without the label agreement behind it:
        # a cell count means nothing if the cells are wrong.
        f"  label agreement  {agreement.caption() if agreement else UNMEASURED_CAPTION}",
        "",
        "novelty",
    ]
    novelty = stats.novelty
    if novelty.count == 0:
        lines.append("  (no scored attacks yet)")
    else:
        lines.extend(
            [
                f"  scored           {novelty.count}",
                f"  min / median     {novelty.minimum:.3f} / {novelty.median:.3f}",
                f"  mean / max       {novelty.mean:.3f} / {novelty.maximum:.3f}",
                f"  below {MIN_NOVELTY:.2f}       {novelty.below_threshold}",
            ]
        )
    lines.extend(
        [
            "",
            "novelty filter",
            f"  rejected         {stats.rejections}",
            f"  rejection rate   {stats.rejection_rate:.1%}",
            "",
            # Cells appear once an attack in them has been executed and scored:
            # fitness needs a defense config to be measured against.
            f"elites (cells scored: {len(stats.elites)}; coverage {stats.coverage})",
        ]
    )
    if not stats.elites:
        lines.append("  (no cell has been scored yet: run the loop, or `make loop`)")
    for cell in stats.elites:
        fitness = "-" if cell.elite_fitness is None else f"{float(cell.elite_fitness):.4f}"
        elite = cell.elite_attack_id or "-"
        lines.append(f"  {cell.cell_key:<62} {fitness:>8}  {elite}")
    return "\n".join(lines)


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level_number)
    stats, agreement = asyncio.run(gather(settings))
    print(render(stats, agreement))
    return 0


if __name__ == "__main__":
    sys.exit(main())
