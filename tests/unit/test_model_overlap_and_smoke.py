"""The overlap experiment fails loudly, and the smoke run is small and live.

`model_overlap` exists to bound a known-hard problem: an attacker agent explores
one model's priors. An overlap computed against a single family, or against a
stub, would look exactly like a real overlap measurement and would mean nothing,
so the experiment refuses rather than falls back.

The smoke run is the two-minute check before committing five hours: one round,
live providers, small caps.
"""

from __future__ import annotations

import inspect
import io

import pytest
from rich.console import Console

from crucible.cli import main as cli_main
from crucible.config import LLMProvider, Settings
from crucible.experiments.config import (
    SMOKE_BUDGET,
    SMOKE_CANDIDATES,
    SMOKE_CELLS,
    SMOKE_ROUNDS,
    load_experiment,
    smoke_experiment,
)
from crucible.experiments.runner import ProviderMismatch, mean_nearest_distance, overlap_by_cell


def _announced(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    *,
    provider: cli_main.Provider = cli_main.Provider.GROQ,
) -> str:
    """What `_announce_provider` writes to the terminal, as text."""
    buffer = io.StringIO()
    monkeypatch.setattr(cli_main, "console", Console(file=buffer, width=200, no_color=True))
    cli_main._announce_provider(settings, provider)
    return buffer.getvalue()


# --------------------------------------------------- failing loudly


def test_a_missing_model_raises_and_names_it() -> None:
    error = ProviderMismatch("qwen/qwen3.6-27b", "openai/gpt-oss-120b")

    assert "qwen/qwen3.6-27b" in str(error)
    assert "openai/gpt-oss-120b" in str(error)
    assert "will not" in str(error) and "fall back" in str(error)


def test_the_mismatch_error_points_at_the_deprecation_list() -> None:
    """A model that vanished is the likeliest cause, and it has a public list."""
    assert "console.groq.com/docs/deprecations" in str(ProviderMismatch("a", "b"))


def test_a_stub_answering_is_a_mismatch_not_a_result() -> None:
    error = ProviderMismatch("qwen/qwen3.6-27b", "stub")

    assert "stub" in str(error)


async def test_one_family_cannot_overlap_with_itself() -> None:
    """Two factories, or the experiment is not an overlap measurement."""
    from crucible.experiments.runner import run_model_overlap

    with pytest.raises(ValueError, match="two model families"):
        await run_model_overlap(
            load_experiment("model_overlap"),
            context=None,  # type: ignore[arg-type]
            attacker_factories={"openai/gpt-oss-120b": lambda: None},  # type: ignore[dict-item]
        )


def test_the_committed_config_names_two_families() -> None:
    models = load_experiment("model_overlap").models

    assert len(models) == 2
    assert len(set(models)) == 2, "two entries of the same model is not an overlap"
    assert "openai/gpt-oss-120b" in models
    assert "qwen/qwen3.6-27b" in models


def test_the_overlap_metrics_are_reported_both_ways() -> None:
    """By cell and by embedding distance: they answer different questions."""
    assert overlap_by_cell(["a"], ["a"]) == 1.0
    assert mean_nearest_distance([[1.0, 0.0]], [[1.0, 0.0]]) == pytest.approx(0.0, abs=1e-9)


def test_scripted_providers_are_refused_for_model_overlap() -> None:
    """Asserted in the CLI, where the choice is made."""
    source = inspect.getsource(cli_main._run_experiment)

    assert "model_overlap compares two real model families" in source
    assert "Provider.SCRIPTED" in source


# --------------------------------------------------- the smoke run


def test_smoke_shrinks_a_run_to_one_round() -> None:
    smoke = smoke_experiment(load_experiment("main"))

    assert smoke.rounds == SMOKE_ROUNDS == 1
    assert smoke.candidates_per_round == SMOKE_CANDIDATES
    assert smoke.cells_per_round == SMOKE_CELLS
    assert smoke.budget_usd == SMOKE_BUDGET


def test_smoke_changes_nothing_that_would_make_it_a_different_experiment() -> None:
    """It is the same run, smaller. Not a different configuration."""
    main = load_experiment("main")
    smoke = smoke_experiment(main)

    assert smoke.name == main.name
    assert smoke.seed == main.seed
    assert smoke.mode == main.mode
    assert smoke.persona == main.persona
    assert smoke.utility_weight == main.utility_weight
    assert smoke.min_novelty == main.min_novelty
    assert smoke.defender_scope == main.defender_scope


def test_smoke_never_unlocks_a_guarded_knob() -> None:
    """Shrinking a run must not become a way past the ablation guards."""
    for name in ("main", "ablation_utility", "ablation_archive", "ablation_novelty"):
        original = load_experiment(name)
        smoke = smoke_experiment(original)
        assert smoke.ablation is original.ablation
        assert smoke.utility_weight == original.utility_weight
        assert smoke.defender_scope == original.defender_scope


def test_the_smoke_flag_exists_and_is_off_by_default() -> None:
    signature = inspect.signature(cli_main.experiment_run.__wrapped__)

    assert signature.parameters["smoke"].default is False
    assert signature.parameters["provider"].default.value == "groq", (
        "a smoke run is a check that LIVE providers answer"
    )


def test_the_smoke_run_prints_provider_models_and_cost(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the operator reads before committing five hours.

    Asserted against the rendered output rather than the source: the announcement
    reads the four agents out of settings, so there is no role name in the code
    to grep for, and what matters is what the operator sees anyway.
    """
    printed = _announced(settings, monkeypatch)
    dispatch = inspect.getsource(cli_main._run_experiment)
    cost = inspect.getsource(cli_main._print_cost)

    for role, provider, model in settings.agents:
        assert role in printed, f"{role} is not named in the announcement"
        assert provider.value in printed, f"{role}'s provider is not named"
        assert model.split("/")[-1] in printed, f"{role}'s model is not named"
    assert "_print_cost" in dispatch
    assert "cost is $0.000000" in cost, "a zero cost on a live run must be called out"


def test_the_announcement_names_the_provider_of_every_agent_in_a_mixed_run(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run split across two providers says so before it spends anything."""
    mixed = settings.model_copy(
        update={
            "ATTACKER_PROVIDER": LLMProvider.DEEPSEEK,
            "ATTACKER_MODEL": "deepseek-chat",
        }
    )

    printed = _announced(mixed, monkeypatch)

    assert "deepseek" in printed and "groq" in printed
    assert "2 providers in this run" in printed


def test_a_scripted_run_is_announced_as_stubbed(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    printed = _announced(settings, monkeypatch, provider=cli_main.Provider.SCRIPTED)

    assert "stubbed" in printed
    assert "STUBBED RUN" in printed


def test_a_zero_cost_on_a_live_run_is_flagged() -> None:
    source = inspect.getsource(cli_main._print_cost)

    assert "not report.stubbed" in source
    assert "MODEL_PRICING" in source
