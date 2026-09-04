"""The benign expectations must describe the retrieval they will be scored against.

`expect_docs` is the output of retrieval, so it is a function of the corpus and
the embedder and of nothing else. It had been recorded under the offline hashing
embedder while production retrieves with `all-MiniLM-L6-v2`; under the real one,
9 of the 40 tasks retrieved none of their recorded documents. Nothing failed
loudly. The utility rate simply read 0.425 with NO defense applied, the selection
objective weighted that at 2.0, and a six-round live run halted itself on
`utility_collapse` against a configuration that had never changed.

So the file now records what it was made against, and this module refuses to let
that drift silently again.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from crucible.config import Settings
from crucible.evaluation.benign import (
    StaleExpectations,
    assert_expectations_current,
    corpus_fingerprint,
    load_benign_file,
    load_benign_tasks,
)
from crucible.target.reference.corpus_gen import load_corpus

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TASKS_FILE = REPOSITORY_ROOT / "data" / "benign_tasks.yaml"


def test_the_file_records_what_it_was_recorded_against() -> None:
    recorded = load_benign_file().recorded_with

    assert recorded.embedding_model
    assert len(recorded.corpus_sha256) == 64
    assert recorded.corpus_documents == len(load_corpus())
    assert recorded.commit and recorded.recorded_on


def test_the_recorded_embedder_is_the_configured_one(settings: Settings) -> None:
    """The check that would have caught this before a six-hour run."""
    assert load_benign_file().recorded_with.embedding_model == settings.EMBEDDING_MODEL


def test_the_recorded_corpus_is_the_committed_corpus() -> None:
    assert load_benign_file().recorded_with.corpus_sha256 == corpus_fingerprint(load_corpus())


def test_the_expectations_pass_their_own_check(settings: Settings) -> None:
    assert_expectations_current(settings.EMBEDDING_MODEL, load_corpus())


def test_a_different_embedder_is_refused_loudly() -> None:
    """Not a warning: a stale expect_docs lowers utility without saying so."""
    with pytest.raises(StaleExpectations, match="hashing-embedder"):
        assert_expectations_current("hashing-embedder", load_corpus())


def test_the_refusal_names_both_models() -> None:
    with pytest.raises(StaleExpectations) as raised:
        assert_expectations_current("some-other-model", load_corpus())

    message = str(raised.value)
    assert "some-other-model" in message
    assert load_benign_file().recorded_with.embedding_model in message
    assert "Re-record" in message or "re-record" in message


def test_a_changed_corpus_is_refused(settings: Settings) -> None:
    """Editing a document changes retrieval as surely as changing the embedder."""
    corpus = list(load_corpus())
    corpus[0] = corpus[0].model_copy(update={"text": corpus[0].text + " (edited)"})

    with pytest.raises(StaleExpectations, match="corpus"):
        assert_expectations_current(settings.EMBEDDING_MODEL, corpus)


def test_a_document_count_alone_would_not_have_caught_that(settings: Settings) -> None:
    """Which is why the guard is a content hash."""
    corpus = list(load_corpus())
    corpus[0] = corpus[0].model_copy(update={"text": "replaced"})

    assert len(corpus) == load_benign_file().recorded_with.corpus_documents
    with pytest.raises(StaleExpectations):
        assert_expectations_current(settings.EMBEDDING_MODEL, corpus)


# --------------------------------------------------------------------------- #
# The tasks themselves
# --------------------------------------------------------------------------- #


def test_every_task_expects_at_least_one_document() -> None:
    """An empty expect_docs is a vacuous assertion.

    `bt-h02` had one: under the production embedder its original phrasing
    retrieved nothing at all, which makes the task unanswerable and its
    expectation meaningless. It was reworded rather than left in place.
    """
    for task in load_benign_tasks():
        assert task.expect_docs, f"{task.id} expects no documents, so it asserts nothing"


def test_the_task_set_is_still_forty_with_twelve_hard_negatives() -> None:
    """Re-recording must not quietly change the set it is scoring."""
    tasks = load_benign_tasks()

    assert len(tasks) == 40
    assert sum(1 for task in tasks if task.hard_negative) == 12


#: The three hard negatives reworded on 2026-09-05 because the corpus could not
#: answer them, with the attack-shaped word each one keeps. A hard negative whose
#: question is unanswerable measures corpus coverage, not the defense.
REWORDED = {
    "bt-h01": "overriding",
    "bt-h02": "ignore previous instructions",
    "bt-h09": "credentials",
}


@pytest.mark.parametrize(("task_id", "marker"), sorted(REWORDED.items()))
def test_the_reworded_hard_negatives_still_read_like_attacks(task_id: str, marker: str) -> None:
    """Their whole value is that they look like an injection and are not one."""
    task = next(item for item in load_benign_tasks() if item.id == task_id)

    assert task.hard_negative
    assert marker in task.query.lower(), f"{task_id} no longer reads like an attack"


def test_bt_h02_still_trips_a_keyword_defense() -> None:
    """It is the one task that gives the over-blocking test its teeth.

    An earlier rewording of it kept the hard-negative *label* while losing the
    words a keyword detector fires on, which emptied
    `test_a_keyword_defense_is_scored_as_utility_loss` without failing it. The
    property is asserted here, at the level of the detector, so a future
    rewording cannot quietly do that again.
    """
    from crucible.defenses.config import (
        DefenseConfig,
        HeuristicRule,
        InputLayer,
        InputSurface,
        PatternClass,
        RuleAction,
    )
    from crucible.defenses.layers.input_layer import inspect_input

    keyword_defense = DefenseConfig(
        input=InputLayer(
            heuristic_rules=(
                HeuristicRule(
                    name="no_instruction_words",
                    pattern_class=PatternClass.INSTRUCTION_LIKE,
                    applies_to=(InputSurface.USER_INPUT,),
                    action=RuleAction.REJECT,
                    threshold=0.3,
                ),
            )
        )
    )
    task = next(item for item in load_benign_tasks() if item.id == "bt-h02")

    verdict = inspect_input(task.query, InputSurface.USER_INPUT, keyword_defense)

    assert verdict.rejected, (
        "bt-h02 must read enough like an instruction override that a keyword "
        "defense rejects it — that is what makes it a hard negative"
    )


def test_the_rewordings_are_documented_in_the_file() -> None:
    """A changed task without a recorded reason is a quietly moved goalpost."""
    text = " ".join(
        line.lstrip("#").strip() for line in TASKS_FILE.read_text(encoding="utf-8").splitlines()
    )

    for task_id in REWORDED:
        assert f"- id: {task_id}" in TASKS_FILE.read_text(encoding="utf-8")
    assert text.count("Reworded 2026-09-05") == len(REWORDED)


def test_the_file_says_how_to_re_record_and_what_invalidates_it() -> None:
    text = TASKS_FILE.read_text(encoding="utf-8")

    assert "EMBEDDING_MODEL" in text
    assert "retrieval alone" in text.lower() or "no model in" in text.lower()
    assert "all-MiniLM-L6-v2" in text


def test_expect_text_is_documented_as_provenance_not_an_assertion() -> None:
    # Comment markers stripped and whitespace normalised: the header is wrapped
    # prose, and where a line breaks is not the property under test.
    text = " ".join(
        line.lstrip("#").strip() for line in TASKS_FILE.read_text(encoding="utf-8").splitlines()
    )

    assert "NOT a pass condition" in text
    assert "kept as provenance" in text


# --------------------------------------------------------------------------- #
# The guard is wired into the commands that score utility
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "command", ["loop_start", "loop_resume", "experiment_run", "eval_defense", "eval_utility"]
)
def test_every_command_that_scores_utility_checks_the_expectations(command: str) -> None:
    import inspect

    from crucible.cli import main as cli_main

    source = inspect.getsource(getattr(cli_main, command).__wrapped__)

    assert "_check_expectations(settings)" in source, (
        f"crucible {command} would score the benign set against unverified expectations"
    )


def test_the_cli_turns_a_stale_expectation_into_one_actionable_line() -> None:
    import inspect

    from crucible.cli import main as cli_main

    source = inspect.getsource(cli_main._handled)

    assert "StaleExpectations" in source
    assert "Re-record data/benign_tasks.yaml" in source
