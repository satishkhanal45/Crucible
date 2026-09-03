"""Hotfix verification for BUG 3: model ids live in configuration, nowhere else.

A live run was fixed by editing `.env`, and the edit had no effect, because
`crucible archive reclassify` took its model from a typer option default. A
hardcoded id anywhere is the same defect waiting to happen: the operator changes
configuration, the process keeps calling the decommissioned model, and nothing
says so.

This is a source scan, in the same shape as the holdout scan in
`tests/unit/test_holdout_isolation.py`. It walks the AST rather than grepping
text, so a mention in prose does not fail it and a real literal cannot hide
behind formatting.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from crucible.config import DEFAULT_MODEL_PRICING, Settings

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src" / "crucible"
ENV_EXAMPLE = REPOSITORY_ROOT / ".env.example"

#: The one module allowed to name a model: it is the configuration.
CONFIG_MODULE = SOURCE_ROOT / "config.py"

#: What a provider's model id looks like, across the families this project can
#: reach. Deliberately broad: a new id from an unknown family should trip this
#: and be moved into configuration, not quietly accepted.
MODEL_ID = re.compile(
    r"^(?:[a-z0-9-]+:)?"  # an optional `provider:` prefix, as pricing keys carry
    r"(?:[a-z0-9][a-z0-9-]*/)?"
    r"(?:llama|gpt|gpt-oss|gemini|gemma|mixtral|mistral|claude|qwen|deepseek|whisper)"
    r"[-a-z0-9._/]*$",
    re.IGNORECASE,
)

#: Settings that must carry every model id the process can call.
MODEL_SETTINGS = (
    "TARGET_MODEL",
    "ATTACKER_MODEL",
    "DEFENDER_MODEL",
    "CLASSIFIER_MODEL",
    "DEFAULT_JUDGE_MODEL",
    "EMBEDDING_MODEL",
)


def source_modules() -> list[Path]:
    return sorted(path for path in SOURCE_ROOT.rglob("*.py") if path != CONFIG_MODULE)


def docstring_nodes(tree: ast.AST) -> set[int]:
    """Every string node that is a docstring, by identity.

    Prose is allowed to name a model; code is not.
    """
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                found.add(id(first.value))
    return found


def model_literals(path: Path) -> list[tuple[int, str]]:
    """Model-id-shaped string literals in one module, excluding docstrings."""
    tree = ast.parse(path.read_text())
    docstrings = docstring_nodes(tree)
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in docstrings:
            continue
        if MODEL_ID.match(node.value.strip()):
            hits.append((node.lineno, node.value))
    return hits


@pytest.mark.parametrize("module", source_modules(), ids=lambda path: str(path.name))
def test_no_module_outside_config_hardcodes_a_model_id(module: Path) -> None:
    """The never-again property: an id in code makes `.env` a no-op."""
    hits = model_literals(module)

    assert not hits, (
        f"{module.relative_to(REPOSITORY_ROOT)} hardcodes a model id "
        f"{[value for _, value in hits]} at lines {[line for line, _ in hits]}. "
        f"Model ids belong in crucible/config.py and .env.example: a literal here "
        f"means editing .env has no effect."
    )


def test_the_config_module_is_where_the_ids_actually_are() -> None:
    """A scan that passes because nothing names a model anywhere is worthless."""
    hits = [value for _, value in model_literals(CONFIG_MODULE)]

    assert hits, "config.py should carry the pricing table's model ids"
    assert any("gpt-oss" in value for value in hits)


@pytest.mark.parametrize("name", MODEL_SETTINGS)
def test_every_model_setting_is_documented_in_env_example(name: str) -> None:
    lines = [line for line in ENV_EXAMPLE.read_text().splitlines() if line.startswith(f"{name}=")]

    assert len(lines) == 1, f"{name} must appear exactly once in .env.example"
    assert lines[0].split("=", 1)[1].strip(), f"{name} in .env.example has no value"


def test_env_example_points_at_the_deprecation_list() -> None:
    """Groq retires ids on a schedule; the next operator needs to know where."""
    text = ENV_EXAMPLE.read_text()

    assert "console.groq.com/docs/deprecations" in text
    assert "2026" in text, "the model ids carry the date they were checked"


def test_every_model_setting_exists_on_settings(settings: Settings) -> None:
    for name in MODEL_SETTINGS:
        assert getattr(settings, name), f"{name} is not readable from Settings"


def test_the_pricing_table_keys_are_provider_qualified() -> None:
    for key in DEFAULT_MODEL_PRICING:
        provider, separator, model = key.partition(":")
        assert separator, f"{key} must read provider:model"
        assert provider and model
        assert key == key.lower(), "keys are lower-cased so .env casing cannot miss"


def test_the_reclassify_command_takes_its_model_from_settings() -> None:
    """The exact defect: a typer option default that shadowed `.env`."""
    from crucible.cli import main as cli_main

    parameters = cli_main.archive_reclassify.__wrapped__.__defaults__ or ()

    assert None in parameters, (
        "--model must default to None so the command falls back to CLASSIFIER_MODEL"
    )
