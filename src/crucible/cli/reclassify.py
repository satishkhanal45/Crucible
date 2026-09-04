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

from crucible.archive.classifier import ChatClassifierClient, TaxonomyClassifier
from crucible.archive.seeds import load_seed_attacks
from crucible.config import Settings
from crucible.db.session import Database
from crucible.execution.egress import EgressGuard, guarded_client
from crucible.repositories.agreement import ClassifierAgreementRepository
from crucible.repositories.spend import DatabaseSpendRepository
from crucible.schemas.agreement import ClassifierAgreement
from crucible.services.cost_meter import CostMeter, pricing_from
from crucible.services.pacing import ProviderPacer


async def reclassify(
    settings: Settings, console: Console, *, dry_run: bool = True, model: str | None = None
) -> ClassifierAgreement:
    """Classify every seed attack and report agreement with its declared cell.

    Agreement is reported **per axis**. A live run agreed 39/40 on objective and
    roughly 22/40 on technique, and a single combined percentage hides exactly
    the thing a reader of a coverage figure needs to know: which axis of the
    grid is measurement and which is a guess.

    `model` defaults to `CLASSIFIER_MODEL` from settings. It is not defaulted at
    the typer option, because a literal there makes editing `.env` a no-op.
    """
    model = model or settings.CLASSIFIER_MODEL
    seeds = load_seed_attacks()
    database = Database(settings.DATABASE_URL)
    guard = EgressGuard.from_settings(settings)
    client: httpx.AsyncClient = guarded_client(guard)

    objective_agreed = 0
    technique_agreed = 0
    combined_agreed = 0
    unclassified = 0
    table = Table(title=f"seed classification ({settings.CLASSIFIER_PROVIDER.value}/{model})")
    for column in ("seed", "declared", "predicted", "objective", "technique"):
        table.add_column(column)

    try:
        meter = CostMeter(
            DatabaseSpendRepository(database),
            settings.ROUND_BUDGET_USD,
            pricing=pricing_from(settings.model_pricing),
            pacer=ProviderPacer.from_settings(
                settings.PROVIDER_MIN_INTERVAL_SECONDS,
                settings.PROVIDER_MAX_CONCURRENCY,
                tokens_per_minute=settings.PROVIDER_TOKENS_PER_MINUTE,
                requests_per_minute=settings.PROVIDER_REQUESTS_PER_MINUTE,
                limits=settings.provider_rate_limits,
            ),
        )
        provider = settings.CLASSIFIER_PROVIDER
        classifier = TaxonomyClassifier(
            ChatClassifierClient(
                settings.api_key_for(provider), client, model=model, provider=provider
            ),
            meter,
        )
        for attack in seeds:
            result = await classifier.classify(attack.payload)
            objective_match = result.objective is not None and result.objective == attack.objective
            technique_match = result.technique is not None and result.technique == attack.technique
            objective_agreed += int(objective_match)
            technique_agreed += int(technique_match)
            combined_agreed += int(objective_match and technique_match)
            unclassified += int(not result.classified)
            table.add_row(
                str(attack.attack_id)[:8],
                _cell(attack.objective, attack.technique),
                _cell(result.objective, result.technique),
                _tick(objective_match),
                _tick(technique_match),
            )
    finally:
        await client.aclose()

    agreement = ClassifierAgreement(
        model_name=model,
        total=len(seeds),
        objective_agreed=objective_agreed,
        technique_agreed=technique_agreed,
        combined_agreed=combined_agreed,
        unclassified=unclassified,
    )

    try:
        # The measurement is stored even on a dry run: `--dry-run` promises not
        # to relabel the archive, and recording what was measured is the whole
        # point of running it. Every report reads this row rather than a number
        # someone typed.
        async with database.session() as session:
            await ClassifierAgreementRepository(session).record(agreement)
    finally:
        await database.close()

    console.print(table)
    console.print(f"objective  {agreement.objective.render()}   [dim]the trustworthy axis[/dim]")
    console.print(
        f"technique  {agreement.technique.render()}"
        + ("   [yellow]SOFT: below 0.70[/yellow]" if agreement.technique_is_soft else "")
    )
    console.print(f"combined   {agreement.combined.render()}")
    console.print(f"{unclassified} unclassified after one retry")
    console.print(
        "Coverage and the coverage grid are only as trustworthy as these numbers: "
        "a classifier that disagrees with hand labels is bucketing attacks into "
        "the wrong cells. The two axes are reported apart because they fail apart."
    )
    if agreement.technique_is_soft:
        console.print(
            "[yellow]technique-axis coverage is a SOFT metric and every report will "
            "label it as one until this rises above 0.70[/yellow]"
        )
    if dry_run:
        console.print(
            "[cyan]dry run: no attack was relabelled; the agreement measurement was "
            "recorded so reports can quote it[/cyan]"
        )
    else:
        console.print(
            "[yellow]writing predictions back is not implemented: re-running the "
            "classifier over a live archive would silently move attacks between "
            "cells mid-experiment[/yellow]"
        )
    return agreement


def _cell(objective: object | None, technique: object | None) -> str:
    if objective is None or technique is None:
        return "unclassified"
    return f"{getattr(objective, 'value', objective)}|{getattr(technique, 'value', technique)}"


def _tick(matched: bool) -> str:
    return "[green]yes[/green]" if matched else "[yellow]no[/yellow]"


def spend_note(spent: Decimal) -> str:
    return f"spent ${spent:.4f}"
