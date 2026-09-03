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

import pytest

from crucible.cli import main as cli_main
from crucible.experiments.config import (
    SMOKE_BUDGET,
    SMOKE_CANDIDATES,
    SMOKE_CELLS,
    SMOKE_ROUNDS,
    load_experiment,
    smoke_experiment,
)
from crucible.experiments.runner import ProviderMismatch, mean_nearest_distance, overlap_by_cell

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


def test_the_smoke_run_prints_provider_models_and_cost() -> None:
    """What the operator reads before committing five hours."""
    announce = inspect.getsource(cli_main._announce_provider)
    dispatch = inspect.getsource(cli_main._run_experiment)
    cost = inspect.getsource(cli_main._print_cost)

    assert "target" in announce and "attacker" in announce
    assert "TARGET_MODEL" in announce or "settings.TARGET_MODEL" in announce
    assert "_print_cost" in dispatch
    assert "cost is $0.000000" in cost, "a zero cost on a live run must be called out"


def test_a_zero_cost_on_a_live_run_is_flagged() -> None:
    source = inspect.getsource(cli_main._print_cost)

    assert "not report.stubbed" in source
    assert "MODEL_PRICING" in source
