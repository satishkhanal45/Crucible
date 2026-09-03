"""A stubbed run must be impossible to mistake for a live one.

A run using scripted clients produces results of exactly the same shape as a
real one: the same tables, the same intervals, the same charts. Without the
mechanisms tested here, `docs/findings.md` could be filled with the output of
test doubles and nothing in the system would say so.

Four defences, one test group each:
  1. Live providers are the default; scripted is reachable only by asking.
  2. The run row records provider and model per agent, and `stubbed`.
  3. Every report leads with a STUBBED RUN banner.
  4. `crucible findings regenerate` refuses stubbed numbers unless forced.
"""

from __future__ import annotations

import inspect
import uuid
from contextlib import AsyncExitStack

import pytest

from crucible.cli import main as cli_main
from crucible.cli.providers import Provider, build_factories
from crucible.config import Settings
from crucible.loop.reports import RunReport, RunStatus
from crucible.reporting import findings
from crucible.reporting.data import ReportData
from crucible.reporting.markdown import render_round_report, render_run_report
from crucible.schemas.provenance import STUB_PROVIDER, AgentModel, RunProvenance

LIVE = RunProvenance(
    agents=(
        AgentModel(role="target", provider="groq", model="llama-3.1-8b-instant"),
        AgentModel(role="attacker", provider="groq", model="openai/gpt-oss-120b"),
        AgentModel(role="defender", provider="groq", model="openai/gpt-oss-120b"),
        AgentModel(role="classifier", provider="groq", model="openai/gpt-oss-120b"),
    )
)
PARTLY_STUBBED = RunProvenance(
    agents=(
        AgentModel(role="target", provider="groq", model="llama-3.1-8b-instant"),
        AgentModel(role="attacker", provider=STUB_PROVIDER, model="scripted-attacker"),
        AgentModel(role="defender", provider="groq", model="openai/gpt-oss-120b"),
        AgentModel(role="classifier", provider="groq", model="openai/gpt-oss-120b"),
    )
)
ALL_STUBBED = RunProvenance(
    agents=tuple(
        AgentModel(role=role, provider=STUB_PROVIDER, model=f"scripted-{role}")
        for role in ("target", "attacker", "defender", "classifier")
    )
)


def _report(provenance: RunProvenance, *, stubbed: bool | None = None) -> RunReport:
    return RunReport(
        run_id=uuid.uuid4(),
        status=RunStatus.COMPLETED,
        attacker_mode="black_box",
        starting_config_id="0" * 32,
        final_config_id="1" * 32,
        rounds=(),
        provenance=provenance,
        stubbed=provenance.stubbed if stubbed is None else stubbed,
    )


# ------------------------------------------- 1. live is the default


def test_the_provider_option_defaults_to_live_everywhere() -> None:
    """Scripted clients must never be what you get by not choosing."""
    for command in (cli_main.loop_start, cli_main.loop_resume, cli_main.experiment_run):
        signature = inspect.signature(command.__wrapped__)
        default = signature.parameters["provider"].default
        assert default is Provider.GROQ, f"{command.__name__} does not default to a live provider"


def test_scripted_is_only_reachable_by_naming_it(settings: Settings) -> None:
    async def _build(provider: Provider) -> object:
        async with AsyncExitStack() as stack:
            return build_factories(settings, provider, stack)
        raise AssertionError("unreachable")

    import asyncio

    scripted = asyncio.run(_build(Provider.SCRIPTED))
    live = asyncio.run(_build(Provider.GROQ))

    assert scripted is not live
    assert Provider("scripted") is Provider.SCRIPTED


def test_live_factories_take_their_models_from_settings(settings: Settings) -> None:
    """A literal here would be the model-id defect all over again."""
    import asyncio

    async def _models() -> dict[str, str]:
        async with AsyncExitStack() as stack:
            factories = build_factories(settings, Provider.GROQ, stack)
            return {
                "target": factories.target_llm().model,
                "attacker": factories.attacker_llm().model,
                "defender": factories.defender_llm().model,
                "classifier": factories.classifier_client().model,
            }
        raise AssertionError("unreachable")

    models = asyncio.run(_models())

    assert models["target"] == settings.TARGET_MODEL
    assert models["attacker"] == settings.ATTACKER_MODEL
    assert models["defender"] == settings.DEFENDER_MODEL
    assert models["classifier"] == settings.CLASSIFIER_MODEL


def test_live_factories_report_a_real_provider(settings: Settings) -> None:
    import asyncio

    async def _provider() -> str:
        async with AsyncExitStack() as stack:
            return str(build_factories(settings, Provider.GROQ, stack).target_llm().provider)
        raise AssertionError("unreachable")

    assert asyncio.run(_provider()) != STUB_PROVIDER


def test_scripted_factories_report_the_stub_provider(settings: Settings) -> None:
    import asyncio

    async def _provider() -> str:
        async with AsyncExitStack() as stack:
            return str(build_factories(settings, Provider.SCRIPTED, stack).target_llm().provider)
        raise AssertionError("unreachable")

    assert asyncio.run(_provider()) == STUB_PROVIDER


# ------------------------------------------- 2. the run records what ran


def test_any_stubbed_agent_makes_the_whole_run_stubbed() -> None:
    """A real target with a scripted attacker is not a measurement either."""
    assert PARTLY_STUBBED.stubbed
    assert ALL_STUBBED.stubbed
    assert not LIVE.stubbed


def test_provenance_names_the_model_per_agent() -> None:
    assert LIVE.for_role("attacker") is not None
    assert LIVE.for_role("attacker").model == "openai/gpt-oss-120b"  # type: ignore[union-attr]
    assert "target: groq/llama-3.1-8b-instant" in LIVE.render_lines()


def test_a_run_with_no_provenance_cannot_claim_to_be_live() -> None:
    """The safe default: unprovable is treated as stubbed."""
    report = RunReport(
        run_id=uuid.uuid4(),
        status=RunStatus.COMPLETED,
        attacker_mode="black_box",
        starting_config_id="0" * 32,
        final_config_id="1" * 32,
    )

    assert report.stubbed
    assert report.banner is not None
    assert "cannot be read as a measurement" in report.banner


def test_the_run_repository_defaults_to_stubbed() -> None:
    """A caller that forgets to pass provenance gets the cautious answer."""
    from crucible.repositories.rounds import RunRepository

    signature = inspect.signature(RunRepository.create)

    assert signature.parameters["stubbed"].default is True


# ------------------------------------------- 3. every report banners it


@pytest.mark.parametrize("provenance", [ALL_STUBBED, PARTLY_STUBBED])
def test_the_run_report_leads_with_the_banner(provenance: RunProvenance) -> None:
    data = ReportData(run=_report(provenance))

    markdown = render_run_report(data)
    first_lines = markdown.splitlines()[:3]

    assert "STUBBED RUN" in markdown
    assert any("STUBBED RUN" in line for line in first_lines), (
        "the banner must be above the title: a reader who scrolls past it is already misled"
    )
    assert "NOT A MEASUREMENT" in markdown


def test_the_banner_names_which_agents_were_stubbed() -> None:
    data = ReportData(run=_report(PARTLY_STUBBED))

    markdown = render_run_report(data)

    assert "attacker" in markdown.split("STUBBED RUN")[1][:200]


def test_a_live_run_carries_no_banner_but_still_names_its_models() -> None:
    data = ReportData(run=_report(LIVE))

    markdown = render_run_report(data)

    assert "STUBBED RUN" not in markdown
    assert "target: groq/llama-3.1-8b-instant" in markdown
    assert "attacker: groq/openai/gpt-oss-120b" in markdown


def test_the_round_report_banners_too() -> None:
    """A round report is quoted as often as a run report."""
    data = ReportData(run=_report(ALL_STUBBED))

    markdown = render_round_report(data, 1)

    assert "STUBBED RUN" in markdown


def test_a_run_that_recorded_nothing_says_its_models_are_unknown() -> None:
    data = ReportData(run=_report(RunProvenance(), stubbed=True))

    markdown = render_run_report(data)

    assert "not recorded" in markdown


# ------------------------------------------- 4. findings refuses stubbed data


def _findings(stubbed: bool, provenance: RunProvenance) -> findings.FindingsData:
    return findings.FindingsData(
        main_run_id=uuid.uuid4(),
        rounds=(),
        agreement=None,
        experiments={},
        stubbed=stubbed,
        provenance=provenance,
    )


def test_regeneration_refuses_a_stubbed_run() -> None:
    with pytest.raises(findings.StubbedRunRefused) as raised:
        findings.guard_stubbed(_findings(True, ALL_STUBBED))

    assert "STUBBED" in str(raised.value)
    assert "--provider groq" in str(raised.value)
    assert "--force-stubbed" in str(raised.value)


def test_the_refusal_names_the_models_that_produced_the_numbers() -> None:
    with pytest.raises(findings.StubbedRunRefused) as raised:
        findings.guard_stubbed(_findings(True, PARTLY_STUBBED))

    assert "scripted-attacker" in str(raised.value)


def test_regeneration_allows_a_live_run() -> None:
    findings.guard_stubbed(_findings(False, LIVE))


def test_the_refusal_can_be_forced_but_only_explicitly() -> None:
    findings.guard_stubbed(_findings(True, ALL_STUBBED), force=True)


def test_the_force_flag_exists_on_the_command() -> None:
    signature = inspect.signature(cli_main.findings_regenerate.__wrapped__)

    assert "force_stubbed" in signature.parameters
    assert signature.parameters["force_stubbed"].default is False
