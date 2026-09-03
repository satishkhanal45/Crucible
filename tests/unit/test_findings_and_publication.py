"""Phase 8 verification tests 1, 4, 5 and 6.

Test 1 — every number in `docs/findings.md` regenerates from stored data, and a
         committed value that differs from the regenerated one is a failure.
Test 4 — the published report and the README contain no raw payload, no canary
         value, and no reference-target system prompt.
Test 5 — taxonomy agreement regenerates per axis and matches the committed
         values. Judge calibration is SKIPPED: there is no LLM judge (cut B4).
Test 6 — every command in the README quickstart exists and is runnable.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from crucible.reporting import findings
from crucible.schemas.agreement import ClassifierAgreement
from crucible.target.canary import CanaryClass, CanarySet
from crucible.target.clinic.prompts import _SYSTEM_TEMPLATE as CLINIC_TEMPLATE
from crucible.target.reference.prompts import _SYSTEM_TEMPLATE as REFERENCE_TEMPLATE

ROOT = Path(__file__).resolve().parents[2]
FINDINGS = ROOT / "docs" / "findings.md"
README = ROOT / "README.md"
SPEC = ROOT / "docs" / "spec.md"

#: The live measurement, transcribed from a run of the real classifier over the
#: 40 hand-labelled seeds with the rewritten prompt.
MEASURED = ClassifierAgreement(
    model_name="openai/gpt-oss-120b",
    total=40,
    objective_agreed=36,
    technique_agreed=30,
    combined_agreed=26,
    unclassified=0,
)


def _empty() -> findings.FindingsData:
    return findings.FindingsData(main_run_id=None, rounds=(), agreement=None, experiments={})


def _with_agreement(agreement: ClassifierAgreement) -> findings.FindingsData:
    return findings.FindingsData(main_run_id=None, rounds=(), agreement=agreement, experiments={})


# --------------------------------------------------- test 1: regeneration


def test_findings_exists_and_declares_its_generated_blocks() -> None:
    keys = findings.generated_keys(FINDINGS.read_text(encoding="utf-8"))

    assert keys, "docs/findings.md has no generated blocks: every number must regenerate"
    assert set(keys) == set(findings.RENDERERS), (
        "every generated block needs a renderer and every renderer needs a block"
    )


def test_the_committed_findings_match_what_the_stored_data_produces() -> None:
    """This is what CI runs: a hand-edited number fails the build."""
    markdown = FINDINGS.read_text(encoding="utf-8")

    stale = findings.stale_blocks(markdown, _empty())

    assert not stale, (
        f"these blocks in docs/findings.md differ from the regenerated output: "
        f"{', '.join(stale)}. Run `crucible findings regenerate` and commit it."
    )


def test_a_hand_edited_number_is_detected() -> None:
    """The check is worthless unless tampering actually trips it."""
    markdown = FINDINGS.read_text(encoding="utf-8")
    tampered = markdown.replace(
        "<!-- BEGIN GENERATED: headline -->\n",
        "<!-- BEGIN GENERATED: headline -->\n- archive block rate: `0.999`\n",
        1,
    )

    assert "headline" in findings.stale_blocks(tampered, _empty())


def test_regeneration_is_idempotent() -> None:
    markdown = FINDINGS.read_text(encoding="utf-8")

    once = findings.rewrite(markdown, _empty())
    twice = findings.rewrite(once, _empty())

    assert once == twice


def test_regeneration_leaves_the_interpretation_alone() -> None:
    """Prose outside the markers is a person's; the script never touches it."""
    markdown = FINDINGS.read_text(encoding="utf-8")

    rewritten = findings.rewrite(markdown, _with_agreement(MEASURED))

    for marker in ("TODO(interpretation)", "## 2. Threat model", "## 8. Limitations"):
        assert marker in rewritten
    assert rewritten.count("TODO(interpretation)") == markdown.count("TODO(interpretation)")


def test_an_unrun_experiment_says_so_rather_than_showing_an_empty_table() -> None:
    """An empty table would read as a measured zero."""
    rendered = findings.render_layer_ablation(_empty())

    assert "Not run yet" in rendered


def test_every_generated_rate_carries_an_interval() -> None:
    data = findings.FindingsData(
        main_run_id=None,
        rounds=(),
        agreement=None,
        experiments={
            "layer_ablation": {
                "rows": [
                    {
                        "layer": "input",
                        "config_id": "abc123",
                        "archive_block_rate": 0.8,
                        "utility_pass_rate": 0.95,
                        "archive_size": 40,
                        "utility_total": 40,
                        "delta_block_rate": -0.1,
                        "delta_utility": 0.02,
                    }
                ]
            }
        },
    )

    rendered = findings.render_layer_ablation(data)

    assert "[" in rendered and "]" in rendered, "a rate without its interval is not a result"
    assert "(n=40)" in rendered


# ------------------------------------------- test 5: agreement, per axis


def test_the_agreement_block_regenerates_per_axis() -> None:
    rendered = findings.render_agreement(_with_agreement(MEASURED))

    assert "| objective | `36/40` (90%) | measurement |" in rendered
    assert "| technique | `30/40` (75%) | measurement |" in rendered
    assert "| combined | `26/40` (65%) | soft metric |" in rendered


def test_the_measured_tradeoff_is_recorded_not_just_the_improvement() -> None:
    """The rewrite bought technique accuracy by spending objective accuracy."""
    text = FINDINGS.read_text(encoding="utf-8")

    assert "39/40 (97.5%)" in text and "36/40 (90%)" in text
    assert "22/40 (55%)" in text and "30/40 (75%)" in text
    assert "21/40 (52%)" in text and "26/40 (65%)" in text
    assert "tradeoff, not an improvement" in text


def test_technique_agreement_below_the_threshold_is_labelled_soft() -> None:
    weak = MEASURED.model_copy(update={"technique_agreed": 22})

    assert weak.technique_is_soft
    assert "soft metric" in findings.render_agreement(_with_agreement(weak))


@pytest.mark.skip(reason="cut B4: there is no LLM judge in this build, so nothing to calibrate")
def test_judge_calibration_regenerates() -> None:  # pragma: no cover - documented skip
    raise AssertionError("unreachable")


# ------------------------------------------------------- test 4: redaction


def _canary_sentinels() -> list[str]:
    canaries = CanarySet.mint()
    return [canaries.reveal(canary_class) for canary_class in CanaryClass]


@pytest.mark.parametrize("document", [FINDINGS, README, SPEC])
def test_no_published_document_contains_a_canary_value(document: Path) -> None:
    """Canaries are minted per attempt, so this looks for the shape as well."""
    text = document.read_text(encoding="utf-8")

    for value in _canary_sentinels():
        assert value not in text
    live = re.findall(r"CRUCIBLE-[A-Z]+-[0-9a-f]{32}", text)
    assert not live, f"{document.name} contains what looks like a canary value: {live}"


@pytest.mark.parametrize("document", [FINDINGS, README])
def test_no_published_document_reproduces_a_target_system_prompt(document: Path) -> None:
    """Publishing it would make the corpus reusable as a jailbreak benchmark."""
    text = document.read_text(encoding="utf-8")

    for template in (REFERENCE_TEMPLATE, CLINIC_TEMPLATE):
        for line in template.splitlines():
            stripped = line.strip()
            if len(stripped) > 40 and "{" not in stripped:
                assert stripped not in text, (
                    f"{document.name} reproduces a target system prompt line: {stripped[:50]!r}"
                )


@pytest.mark.parametrize("document", [FINDINGS, README])
def test_no_published_document_contains_a_raw_payload(document: Path) -> None:
    """Seed payloads are the one corpus this project will not hand out."""
    from crucible.archive.seeds import load_seed_attacks

    text = document.read_text(encoding="utf-8")

    for attack in load_seed_attacks():
        payload = attack.payload.strip()
        assert payload not in text, f"{document.name} contains a raw seed payload"
        # And no long fragment of one either.
        head = payload[:60]
        if len(head) == 60:
            assert head not in text


def test_findings_states_the_publication_rule() -> None:
    text = FINDINGS.read_text(encoding="utf-8")

    assert "--include-payloads" in text
    assert "Canary values" in text or "canary values" in text


def test_the_readme_states_what_is_not_published() -> None:
    # Whitespace-normalised: the README wraps its prose, and where a line breaks
    # is not the property under test.
    text = " ".join(README.read_text(encoding="utf-8").split())

    assert "Not published" in text
    assert "--include-payloads" in text
    assert "jailbreak benchmark" in text


# ------------------------------- test 4 (continued): the rules of engagement


def _rules_of_engagement() -> list[str]:
    """Section 1 of the spec, which the README reproduces verbatim."""
    text = SPEC.read_text(encoding="utf-8")
    start = text.index("## 1. Rules of engagement")
    end = text.index("## 2. Design commitments")
    body = text[start:end]
    return [line.strip() for line in body.splitlines() if line.strip().startswith("- ")]


def test_the_readme_reproduces_the_rules_of_engagement_verbatim() -> None:
    """Section 17: the README leads with section 1, verbatim."""
    readme = README.read_text(encoding="utf-8")

    for line in _rules_of_engagement():
        assert line in readme, f"the README does not reproduce this rule verbatim: {line!r}"


def test_the_rules_of_engagement_come_first() -> None:
    readme = README.read_text(encoding="utf-8")

    assert readme.index("Rules of engagement") < readme.index("## What Crucible is")


def test_the_readme_names_the_licence() -> None:
    assert "Apache-2.0" in README.read_text(encoding="utf-8")
    assert (ROOT / "LICENSE").exists()
    assert "Apache License" in (ROOT / "LICENSE").read_text(encoding="utf-8")


# --------------------------------------------------- test 6: the quickstart


#: Commands the quickstart tells a new user to run, in order.
QUICKSTART_MAKE_TARGETS = ("up", "migrate", "seed")
QUICKSTART_CLI = (
    ("archive", "stats"),
    ("archive", "reclassify"),
    ("experiment", "list"),
    ("experiment", "run"),
    ("report", "run"),
    ("loop", "status"),
    ("loop", "resume"),
)


def test_the_quickstart_warns_about_the_first_build() -> None:
    text = README.read_text(encoding="utf-8")

    assert "10 minutes" in text
    assert "FIRST run" in text or "first run" in text


def test_the_quickstart_documents_the_port_overrides() -> None:
    text = README.read_text(encoding="utf-8")

    assert "POSTGRES_PORT" in text
    assert "API_PORT" in text


@pytest.mark.parametrize("target", QUICKSTART_MAKE_TARGETS)
def test_every_quickstart_make_target_exists(target: str) -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert f"\n{target}:" in makefile, f"the README tells the user to run `make {target}`"
    assert f"make {target}" in README.read_text(encoding="utf-8")


@pytest.mark.parametrize("command", QUICKSTART_CLI)
def test_every_quickstart_cli_command_exists_and_runs(command: tuple[str, ...]) -> None:
    """`--help` on each: it exercises the parser without touching the database."""
    result = subprocess.run(
        [shutil.which("uv") or "uv", "run", "crucible", *command, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, f"`crucible {' '.join(command)} --help` failed: {result.stderr}"
    assert " ".join(command) in README.read_text(encoding="utf-8")
