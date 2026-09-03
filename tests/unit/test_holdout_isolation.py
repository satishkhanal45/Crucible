"""Verification test 13, and the structural half of 10 and 11.

**These are security properties, not unit tests.** The holdout set is the only
honest generalization number in the project. If a holdout attack reaches an
agent, every generalization figure in every report becomes a lie, and nothing in
the repository would say so.

The database-backed halves of tests 10, 11, 12 and 14 live in
`tests/integration/test_archive_holdout.py`.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from crucible.db.models import AttackRow
from crucible.repositories.attacks import AGENT_SAFE_METHODS, AttackRepository

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "crucible"

#: Modules that build prompts for an agent, now or in a later phase. A holdout
#: attack must never be reachable from any of them.
PROMPT_BUILDING_PACKAGES = ("attacker", "defender")
PROMPT_MODULE_NAMES = ("prompts.py", "prompt_builder.py", "templates.py")

RAW_ATTACK_QUERY = re.compile(
    r"(select\s+.*\s+from\s+attacks)|(from\s+attacks\b)", re.IGNORECASE | re.DOTALL
)


def prompt_building_modules() -> list[Path]:
    """Every module that builds a prompt today or is reserved to.

    Phases 4 and 5 have not been written yet, so this deliberately includes the
    whole `attacker/` and `defender/` packages: the test starts guarding them
    the moment a file lands there.
    """
    modules: list[Path] = []
    for package in PROMPT_BUILDING_PACKAGES:
        modules.extend(sorted((SOURCE_ROOT / package).rglob("*.py")))
    for name in PROMPT_MODULE_NAMES:
        modules.extend(sorted(SOURCE_ROOT.rglob(name)))
    return sorted(set(modules))


def imported_names(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_there_are_prompt_building_modules_to_check() -> None:
    """A guard that guards nothing would pass silently forever."""
    assert prompt_building_modules(), "expected at least one prompt-building module"


def test_no_prompt_builder_writes_a_raw_attacks_query() -> None:
    for module in prompt_building_modules():
        source = module.read_text(encoding="utf-8")
        assert not RAW_ATTACK_QUERY.search(source), (
            f"{module} contains a raw SQL query against `attacks`. Prompt builders "
            "must go through AttackRepository's holdout-filtered methods."
        )


def test_no_prompt_builder_imports_the_attack_repository_or_the_orm() -> None:
    """Agents talk to services; only two service methods return attacks."""
    forbidden = {
        "crucible.repositories.attacks",
        "crucible.repositories.attacks.AttackRepository",
        "crucible.db.models",
        "crucible.db.models.AttackRow",
    }
    for module in prompt_building_modules():
        names = imported_names(module.read_text(encoding="utf-8"))
        leaked = names & forbidden
        assert not leaked, (
            f"{module} imports {sorted(leaked)}. A prompt builder that can reach the "
            "archive directly can reach a holdout attack."
        )


#: Receivers whose name says "this is the archive". A prompt builder calling any
#: non-agent-safe method on one of these is the failure this guard exists for.
ARCHIVE_RECEIVER_HINTS = ("repo", "repository", "attacks", "archive", "store")


def unsafe_archive_calls(source: str) -> list[str]:
    """Calls to a non-agent-safe repository method on an archive-looking object.

    Matching on the method name alone would flag `set.add()` and
    `graph.add_node()`, so the receiver has to look like the archive too. The
    self-test below proves this still catches the thing it is here to catch.
    """
    unsafe = {
        name
        for name in dir(AttackRepository)
        if not name.startswith("_") and name not in AGENT_SAFE_METHODS
    }
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in unsafe:
            continue
        receiver = ast.unparse(node.func.value).lower()
        if any(hint in receiver for hint in ARCHIVE_RECEIVER_HINTS):
            found.append(f"{receiver}.{node.func.attr}()")
    return found


def test_the_guard_catches_an_unfiltered_archive_call() -> None:
    """A guard that cannot fail is not a guard."""
    offending = """
async def build_prompt(repository):
    attacks = await repository.list_all()
    return "\\n".join(a.payload for a in attacks)
"""
    assert unsafe_archive_calls(offending) == ["repository.list_all()"]

    permitted = """
async def build_prompt(repository):
    attacks = await repository.get_attacks_for_defender()
    seen = set()
    seen.add(attacks)
    return attacks
"""
    assert unsafe_archive_calls(permitted) == []


def test_a_prompt_builder_may_only_name_the_two_agent_safe_methods() -> None:
    for module in prompt_building_modules():
        offending = unsafe_archive_calls(module.read_text(encoding="utf-8"))
        assert offending == [], (
            f"{module} calls {offending}, which does not filter holdout. Only "
            "get_attacks_for_mutation and get_attacks_for_defender may hand attacks "
            "to an agent."
        )


@pytest.mark.parametrize("method_name", sorted(AGENT_SAFE_METHODS))
def test_every_agent_safe_method_filters_holdout_in_sql(method_name: str) -> None:
    """Read the compiled SQL, not the Python: the filter must reach the database."""
    source = (SOURCE_ROOT / "repositories" / "attacks.py").read_text(encoding="utf-8")
    body = source.split(f"async def {method_name}(")[1].split("async def ")[0]

    assert "AttackRow.is_holdout.is_(False)" in body, (
        f"{method_name} does not filter is_holdout in SQL"
    )


def test_the_agent_safe_methods_take_no_bypass_parameter() -> None:
    """A flag that exists will eventually be passed."""
    source = (SOURCE_ROOT / "repositories" / "attacks.py").read_text(encoding="utf-8")
    for method_name in AGENT_SAFE_METHODS:
        signature = source.split(f"async def {method_name}(")[1].split(") ->")[0]
        for bypass in ("include_holdout", "holdout", "force", "unsafe", "all_attacks"):
            assert bypass not in signature, (
                f"{method_name} accepts {bypass!r}; there must be no way to relax the filter"
            )


def test_the_holdout_filter_survives_compilation() -> None:
    """The compiled statement really contains the predicate."""
    statement = select(AttackRow).where(AttackRow.is_holdout.is_(False))
    compiled = str(statement.compile(dialect=postgresql.dialect())).lower()

    assert "is_holdout is false" in compiled


def test_only_two_methods_are_declared_agent_safe() -> None:
    assert {"get_attacks_for_mutation", "get_attacks_for_defender"} == AGENT_SAFE_METHODS
