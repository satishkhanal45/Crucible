"""Phase 8 verification tests 2 and 7: configs load, and the guards hold.

Test 2 — every experiment is reproducible from its config plus a seed.
Test 7 — the two never-cut properties this phase violates on purpose each fail
         validation outside their own named ablation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from crucible.archive.novelty import MIN_NOVELTY
from crucible.defenses.config import (
    ContextLayer,
    DefenseConfig,
    HeuristicRule,
    InputLayer,
    OutputLayer,
    PromptLayer,
    StructuralLayer,
)
from crucible.evaluation.objective import UTILITY_WEIGHT
from crucible.experiments.config import (
    EXPERIMENTS_DIR,
    AblationLabel,
    DefenderScope,
    ExperimentConfig,
    ExperimentKind,
    NeverCutViolation,
    experiment_names,
    load_experiment,
)
from crucible.experiments.runner import (
    LAYERS,
    cost_estimate_minutes,
    disable_layer,
    loop_settings_for,
    mean_nearest_distance,
    overlap_by_cell,
    seeded_rng,
)

#: The experiments docs/spec.md section 16 asks for, plus the model-overlap
#: mitigation. Every one is a committed file.
REQUIRED = (
    "main",
    "ablation_novelty",
    "ablation_utility",
    "ablation_archive",
    "layer_ablation",
    "transfer",
    "model_overlap",
)


# ------------------------------------------------------- test 2: they exist


@pytest.mark.parametrize("name", REQUIRED)
def test_every_experiment_config_exists_and_loads(name: str) -> None:
    path = EXPERIMENTS_DIR / f"{name}.yaml"

    assert path.exists(), f"{path} is missing: the experiment is not reproducible without it"
    config = load_experiment(name)
    assert config.name == name


@pytest.mark.parametrize("name", REQUIRED)
def test_every_experiment_states_an_explicit_seed(name: str) -> None:
    """Reproducible from the config plus a seed means the seed is in the file."""
    raw = yaml.safe_load((EXPERIMENTS_DIR / f"{name}.yaml").read_text())

    assert "seed" in raw, f"{name} has no explicit seed"
    assert isinstance(raw["seed"], int)


@pytest.mark.parametrize("name", REQUIRED)
def test_the_same_config_produces_the_same_settings(name: str) -> None:
    """Loading twice gives identical settings: nothing is drawn at load time."""
    first = loop_settings_for(load_experiment(name))
    second = loop_settings_for(load_experiment(name))

    assert first == second
    assert seeded_rng(load_experiment(name)).random() == pytest.approx(
        seeded_rng(load_experiment(name)).random()
    )


def test_no_experiment_file_is_undocumented() -> None:
    """A config nobody asked for is as bad as a missing one."""
    assert set(experiment_names()) == set(REQUIRED)


@pytest.mark.parametrize("name", REQUIRED)
def test_every_experiment_records_why_its_round_count_is_what_it_is(name: str) -> None:
    """Round counts are set by wall clock; findings records that as a limitation."""
    config = load_experiment(name)

    assert config.rounds_rationale.strip(), f"{name} does not say why it runs that many rounds"


def test_the_main_run_changes_nothing() -> None:
    config = load_experiment("main")

    assert config.ablation is AblationLabel.NONE
    assert config.min_novelty == MIN_NOVELTY
    assert config.utility_weight == UTILITY_WEIGHT
    assert config.defender_scope is DefenderScope.ARCHIVE
    assert config.rounds == 6


def test_the_transfer_experiment_names_the_second_target() -> None:
    config = load_experiment("transfer")

    assert config.kind is ExperimentKind.TRANSFER
    assert config.persona == "meridian"


def test_model_overlap_compares_two_families() -> None:
    config = load_experiment("model_overlap")

    assert config.kind is ExperimentKind.MODEL_OVERLAP
    assert len(config.models) == 2, "an overlap needs exactly two sets to overlap"


# --------------------------------------------- test 7: the ablation guards


def _config(**overrides: object) -> ExperimentConfig:
    values: dict[str, object] = {"name": "main", "seed": 1}
    values.update(overrides)
    return ExperimentConfig.model_validate(values)


def test_utility_weight_zero_is_refused_outside_its_ablation() -> None:
    """The never-cut property: weight 2.0 on utility_loss (spec section 12)."""
    with pytest.raises(NeverCutViolation) as raised:
        _config(name="main", utility_weight=0.0)

    assert "never-cut" in str(raised.value)
    assert "ablation_utility" in str(raised.value)


def test_utility_weight_zero_is_allowed_in_its_own_ablation() -> None:
    config = _config(
        name="ablation_utility", utility_weight=0.0, ablation=AblationLabel.UTILITY.value
    )

    assert config.utility_weight == 0.0
    assert config.violates_never_cut


@pytest.mark.parametrize("name", ["main", "transfer", "layer_ablation", "ablation_novelty"])
def test_no_other_experiment_may_claim_the_utility_label(name: str) -> None:
    """The label belongs to the experiment of the same name and nowhere else."""
    with pytest.raises(NeverCutViolation):
        _config(name=name, utility_weight=0.0, ablation=AblationLabel.UTILITY.value)


def test_the_archive_blind_defender_is_refused_outside_its_ablation() -> None:
    with pytest.raises(NeverCutViolation) as raised:
        _config(name="main", defender_scope=DefenderScope.CURRENT_ROUND.value)

    assert "never-cut" in str(raised.value)
    assert "ablation_archive" in str(raised.value)


def test_the_archive_blind_defender_is_allowed_in_its_own_ablation() -> None:
    config = _config(
        name="ablation_archive",
        defender_scope=DefenderScope.CURRENT_ROUND.value,
        ablation=AblationLabel.ARCHIVE.value,
    )

    assert config.defender_scope is DefenderScope.CURRENT_ROUND


def test_moving_min_novelty_is_refused_outside_its_ablation() -> None:
    with pytest.raises(NeverCutViolation):
        _config(name="main", min_novelty=0.0)


def test_the_committed_configs_obey_their_own_guards() -> None:
    """The guards are worthless if a committed file already violates one."""
    for name in REQUIRED:
        config = load_experiment(name)
        if config.violates_never_cut:
            assert config.name == config.ablation.value
        else:
            assert config.utility_weight == UTILITY_WEIGHT
            assert config.min_novelty == MIN_NOVELTY
            assert config.defender_scope is DefenderScope.ARCHIVE


def test_exactly_one_experiment_violates_each_never_cut_property() -> None:
    """Once each, under a label. Not zero, and not twice."""
    configs = [load_experiment(name) for name in REQUIRED]

    assert sum(1 for c in configs if c.utility_weight != UTILITY_WEIGHT) == 1
    assert sum(1 for c in configs if c.defender_scope is DefenderScope.CURRENT_ROUND) == 1
    assert sum(1 for c in configs if c.min_novelty != MIN_NOVELTY) == 1


def test_the_guarded_knobs_reach_the_loop_settings() -> None:
    """A guard that does not change the run would be theatre."""
    settings = loop_settings_for(load_experiment("ablation_utility"))

    assert settings.utility_weight == 0.0
    assert loop_settings_for(load_experiment("ablation_archive")).defender_scope == "current_round"
    assert loop_settings_for(load_experiment("ablation_novelty")).min_novelty == 0.0


# ------------------------------------------------------- the layer ablation


def _all_layers_on() -> DefenseConfig:
    """A config with every one of the five layers doing something.

    The Phase 6 harness config leaves some layers at their defaults, which would
    make "disabling" them a no-op and the test below vacuous.
    """
    return DefenseConfig(
        input=InputLayer(
            heuristic_rules=(
                HeuristicRule(
                    name="instructions_in_retrieved",
                    pattern_class="instruction_like",
                    applies_to=("retrieved_context",),
                    action="strip",
                ),
            )
        ),
        context=ContextLayer(strip_instructions_from_retrieved=True, provenance_tags=True),
        prompt=PromptLayer(role_reassertion="both", precedence_statement=True),
        output=OutputLayer(citation_verification=True),
        structural=StructuralLayer(
            tool_allowlist=("send_email",), require_user_origin_for_privileged=True
        ),
    )


@pytest.mark.parametrize("layer", LAYERS)
def test_disabling_a_layer_changes_only_that_layer(layer: str) -> None:
    config = _all_layers_on()
    without = disable_layer(config, layer)

    assert getattr(without, layer) != getattr(config, layer), f"{layer} was not disabled"
    for other in LAYERS:
        if other != layer:
            assert getattr(without, other) == getattr(config, other)


def test_disabling_a_layer_produces_a_different_config_id() -> None:
    """Each ablated config is stored and resolvable by its own fingerprint."""
    config = _all_layers_on()
    ids = {disable_layer(config, layer).fingerprint() for layer in LAYERS}

    assert len(ids) == len(LAYERS)
    assert config.fingerprint() not in ids


def test_disabling_every_layer_reaches_the_empty_config() -> None:
    config = _all_layers_on()
    for layer in LAYERS:
        config = disable_layer(config, layer)

    assert config.fingerprint() == DefenseConfig.empty().fingerprint()


def test_an_unknown_layer_is_refused() -> None:
    with pytest.raises(KeyError):
        disable_layer(DefenseConfig.empty(), "not_a_layer")


# ------------------------------------------------------- the overlap metrics


def test_cell_overlap_is_jaccard() -> None:
    assert overlap_by_cell(["a", "b"], ["a", "b"]) == 1.0
    assert overlap_by_cell(["a", "b"], ["c", "d"]) == 0.0
    assert overlap_by_cell(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)


def test_cell_overlap_of_nothing_is_zero_not_one() -> None:
    """Two empty runs agree about nothing; they have not agreed perfectly."""
    assert overlap_by_cell([], []) == 0.0


def test_identical_attacks_have_zero_nearest_distance() -> None:
    vectors = [[1.0, 0.0], [0.0, 1.0]]

    assert mean_nearest_distance(vectors, vectors) == pytest.approx(0.0, abs=1e-9)


def test_orthogonal_attacks_are_a_full_distance_apart() -> None:
    assert mean_nearest_distance([[1.0, 0.0]], [[0.0, 1.0]]) == pytest.approx(1.0)


def test_a_wall_clock_estimate_is_offered_for_every_experiment() -> None:
    """So a run can be planned against an overnight window."""
    for name in REQUIRED:
        assert cost_estimate_minutes(load_experiment(name)) > 0


def test_the_experiments_directory_is_the_committed_one() -> None:
    assert Path("experiments") == EXPERIMENTS_DIR
