"""Verification test 1: configuration fails loudly and parses the allowlist."""

from __future__ import annotations

from decimal import Decimal

import pytest

from crucible.config import ConfigurationError, Settings, load_settings
from tests.conftest import ENV_VALUES


def _env_without(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    for key in ENV_VALUES:
        monkeypatch.delenv(key, raising=False)
    for key, value in ENV_VALUES.items():
        if key != missing:
            monkeypatch.setenv(key, value)


def test_all_settings_present_loads(env: dict[str, str]) -> None:
    settings = load_settings(_env_file=None)
    assert env["DATABASE_URL"] == settings.DATABASE_URL
    expected_budget = Decimal("5.00")
    assert settings.ENV == "test"
    assert expected_budget == settings.ROUND_BUDGET_USD
    assert settings.DEFAULT_JUDGE_MODEL == "gemini-2.0-flash"


@pytest.mark.parametrize("missing", sorted(ENV_VALUES))
def test_missing_variable_is_named(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    _env_without(monkeypatch, missing)

    with pytest.raises(ConfigurationError) as raised:
        load_settings(_env_file=None)

    message = str(raised.value)
    assert missing in message
    others = [key for key in ENV_VALUES if key != missing]
    assert not [key for key in others if key in message], message


def test_target_allowlist_parses_comma_separated(env: dict[str, str]) -> None:
    del env
    settings = load_settings(_env_file=None, TARGET_ALLOWLIST="localhost,127.0.0.1")
    assert settings.TARGET_ALLOWLIST == ["localhost", "127.0.0.1"]


def test_target_allowlist_strips_and_drops_blanks(env: dict[str, str]) -> None:
    del env
    settings = load_settings(_env_file=None, TARGET_ALLOWLIST=" localhost , 127.0.0.1 ,")
    assert settings.TARGET_ALLOWLIST == ["localhost", "127.0.0.1"]


@pytest.mark.parametrize("value", ["", "   ", ",", " , "])
def test_target_allowlist_rejects_empty(env: dict[str, str], value: str) -> None:
    del env
    with pytest.raises(ConfigurationError) as raised:
        load_settings(_env_file=None, TARGET_ALLOWLIST=value)
    assert "TARGET_ALLOWLIST" in str(raised.value)


def test_log_level_is_normalised_and_validated(env: dict[str, str]) -> None:
    del env
    assert load_settings(_env_file=None, LOG_LEVEL="debug").LOG_LEVEL == "DEBUG"
    with pytest.raises(ConfigurationError) as raised:
        load_settings(_env_file=None, LOG_LEVEL="chatty")
    assert "LOG_LEVEL" in str(raised.value)


def test_round_budget_must_be_positive(env: dict[str, str]) -> None:
    del env
    with pytest.raises(ConfigurationError) as raised:
        load_settings(_env_file=None, ROUND_BUDGET_USD="0")
    assert "ROUND_BUDGET_USD" in str(raised.value)


def test_settings_reads_env_vars_case_sensitively(env: dict[str, str]) -> None:
    del env
    settings = Settings(_env_file=None)
    assert settings.TARGET_ALLOWLIST == ["localhost", "127.0.0.1"]
