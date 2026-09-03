"""`crucible archive reclassify --dry-run` — the classifier smoke check.

The taxonomy classifier has only ever run against a scripted stub. If it is
inaccurate, the coverage metric and the signature visual are both meaningless,
and Phase 8 needs to know that *before* the main run rather than after it. This
command runs the real classifier over the 40 committed seed attacks, whose cells
were assigned by hand, and prints declared against predicted with an agreement
rate.

It is a manual command, not a test: it makes real API calls, and no test in this
repository does.
"""

from __future__ import annotations

from decimal import Decimal

import httpx
from rich.console import Console
from rich.table import Table

from crucible.archive.classifier import GroqClassifierClient, TaxonomyClassifier
from crucible.archive.seeds import load_seed_attacks
from crucible.config import Settings
from crucible.db.session import Database
from crucible.execution.egress import EgressGuard, guarded_client
from crucible.repositories.spend import DatabaseSpendRepository
from crucible.services.cost_meter import CostMeter, pricing_from
from crucible.services.pacing import ProviderPacer


async def reclassify(
    settings: Settings, console: Console, *, dry_run: bool = True, model: str | None = None
) -> float:
    """Classify every seed attack and report agreement with its declared cell.

    `model` defaults to `CLASSIFIER_MODEL` from settings. It is not defaulted at
    the typer option, because a literal there makes editing `.env` a no-op.
    """
    model = model or settings.CLASSIFIER_MODEL
    seeds = load_seed_attacks()
    database = Database(settings.DATABASE_URL)
    guard = EgressGuard.from_settings(settings)
    client: httpx.AsyncClient = guarded_client(guard)

    agreed = 0
    unclassified = 0
    table = Table(title=f"seed classification ({model})")
    for column in ("seed", "declared", "predicted", "match"):
        table.add_column(column)

    try:
        meter = CostMeter(
            DatabaseSpendRepository(database),
            settings.ROUND_BUDGET_USD,
            pricing=pricing_from(settings.model_pricing),
            pacer=ProviderPacer.from_settings(
                settings.PROVIDER_MIN_INTERVAL_SECONDS, settings.PROVIDER_MAX_CONCURRENCY
            ),
        )
        classifier = TaxonomyClassifier(
            GroqClassifierClient(settings.GROQ_API_KEY, client, model=model), meter
        )
        for attack in seeds:
            result = await classifier.classify(attack.payload)
            predicted = (
                f"{result.objective.value}|{result.technique.value}"
                if result.classified and result.objective and result.technique
                else "unclassified"
            )
            declared = (
                f"{attack.objective.value}|{attack.technique.value}"
                if attack.objective and attack.technique
                else "unclassified"
            )
            match = predicted == declared
            agreed += int(match)
            unclassified += int(predicted == "unclassified")
            table.add_row(
                str(attack.attack_id)[:8],
                declared,
                predicted,
                "[green]yes[/green]" if match else "[yellow]no[/yellow]",
            )
    finally:
        await client.aclose()
        await database.close()

    rate = agreed / len(seeds) if seeds else 0.0
    console.print(table)
    console.print(
        f"agreement {agreed}/{len(seeds)} ({rate:.0%}); {unclassified} unclassified after one retry"
    )
    console.print(
        "Coverage and the coverage grid are only as trustworthy as this number: "
        "a classifier that disagrees with hand labels is bucketing attacks into "
        "the wrong cells."
    )
    if dry_run:
        console.print("[cyan]dry run: nothing was written to the archive[/cyan]")
    else:
        console.print(
            "[yellow]writing predictions back is not implemented: re-running the "
            "classifier over a live archive would silently move attacks between "
            "cells mid-experiment[/yellow]"
        )
    return rate


def spend_note(spent: Decimal) -> str:
    return f"spent ${spent:.4f}"
