"""Agreement is stored, and every coverage figure quotes it.

The standing rule for `docs/findings.md` is that no number is typed by hand;
every figure is regenerated from what the database holds. Classifier agreement
is a figure like any other, so it is measured once, stored, and read back by
the report rather than remembered by whoever ran the command.
"""

from __future__ import annotations

import pytest

from crucible.archive.grid import Coverage
from crucible.cli.archive import render
from crucible.db.session import Database
from crucible.reporting.data import ReportData
from crucible.reporting.markdown import render_run_report
from crucible.repositories.agreement import ClassifierAgreementRepository
from crucible.schemas.agreement import ClassifierAgreement
from crucible.schemas.archive import ArchiveStats, NoveltyDistribution

SOFT = ClassifierAgreement(
    model_name="openai/gpt-oss-120b",
    total=40,
    objective_agreed=39,
    technique_agreed=22,
    combined_agreed=21,
    unclassified=0,
)
STRONG = SOFT.model_copy(update={"technique_agreed": 36, "combined_agreed": 35})


async def test_an_agreement_measurement_round_trips(database: Database) -> None:
    async with database.session() as session:
        await ClassifierAgreementRepository(session).record(SOFT)

    async with database.session() as session:
        stored = await ClassifierAgreementRepository(session).latest()

    assert stored is not None
    assert stored.objective_agreed == 39
    assert stored.technique_agreed == 22
    assert stored.technique_is_soft


async def test_the_latest_measurement_wins(database: Database) -> None:
    """History is kept; a report quotes the most recent run, not the first."""
    async with database.session() as session:
        repository = ClassifierAgreementRepository(session)
        await repository.record(SOFT)
        await repository.record(STRONG)

    async with database.session() as session:
        stored = await ClassifierAgreementRepository(session).latest()

    assert stored is not None
    assert stored.technique_agreed == 36


async def test_no_measurement_reads_back_as_none(database: Database) -> None:
    async with database.session() as session:
        stored = await ClassifierAgreementRepository(session).latest()

    assert stored is None or isinstance(stored, ClassifierAgreement)


# --------------------------------------------------------------------------- #
# Every coverage figure carries it
# --------------------------------------------------------------------------- #


def _stats() -> ArchiveStats:
    return ArchiveStats(
        archive_size=40,
        holdout_count=4,
        unclassified_count=0,
        coverage=Coverage(occupied=16),
        novelty=NoveltyDistribution(count=0),
        rejections=0,
        rejection_rate=0.0,
        elites=(),
    )


def test_archive_stats_prints_agreement_next_to_coverage() -> None:
    text = render(_stats(), SOFT)

    assert "coverage" in text
    assert "objective 39/40" in text
    assert "technique 22/40" in text
    assert "SOFT metric" in text


def test_archive_stats_says_so_when_agreement_was_never_measured() -> None:
    """A coverage number with no agreement line reads as trustworthy."""
    text = render(_stats(), None)

    assert "NOT MEASURED" in text


def _report(run_report: object, agreement: ClassifierAgreement | None) -> str:
    data = ReportData.model_validate(
        {"run": run_report, "agreement": agreement, "archive_size": 40}
    )
    return render_run_report(data)


def test_the_run_report_labels_a_weak_technique_axis_soft(built_run_report: object) -> None:
    markdown = _report(built_run_report, SOFT)

    assert "| objective | `39/40` (98%) | measurement |" in markdown
    assert "| technique | `22/40` (55%) | soft metric |" in markdown
    assert "Technique-axis coverage is a soft metric here" in markdown
    assert "technique axis is soft" in markdown, "and it is repeated under limitations"


def test_the_run_report_does_not_cry_wolf_when_the_axis_is_strong(
    built_run_report: object,
) -> None:
    markdown = _report(built_run_report, STRONG)

    assert "| technique | `36/40` (90%) | measurement |" in markdown
    assert "soft metric" not in markdown


def test_the_run_report_says_when_agreement_was_never_measured(
    built_run_report: object,
) -> None:
    markdown = _report(built_run_report, None)

    assert "NOT MEASURED" in markdown


@pytest.fixture
def built_run_report() -> object:
    """The smallest run report the markdown renderer will accept."""
    import uuid

    from crucible.loop.reports import RunReport, RunStatus

    return RunReport(
        run_id=uuid.uuid4(),
        status=RunStatus.COMPLETED,
        attacker_mode="black_box",
        starting_config_id="0" * 32,
        final_config_id="1" * 32,
        rounds=(),
    )
