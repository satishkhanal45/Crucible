"""Shared fixtures.

No test in this suite may make a live API call; provider calls are always
stubbed with scripted responses.
"""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal

import pytest

from crucible.config import Settings, load_settings

ENV_VALUES = {
    "DATABASE_URL": "postgresql+asyncpg://crucible:crucible@localhost:5432/crucible",
    "ENV": "test",
    "LOG_LEVEL": "INFO",
    "GROQ_API_KEY": "test-groq-key",
    "GEMINI_API_KEY": "test-gemini-key",
    "TARGET_ALLOWLIST": "localhost,127.0.0.1",
    "ROUND_BUDGET_USD": "5.00",
    "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "LLM_PROVIDER": "groq",
    "TARGET_MODEL": "llama-3.1-8b-instant",
    "ATTACKER_MODEL": "openai/gpt-oss-120b",
    "DEFENDER_MODEL": "openai/gpt-oss-120b",
    "CLASSIFIER_MODEL": "openai/gpt-oss-120b",
    "JUDGE_PROVIDER": "gemini",
    "DEFAULT_JUDGE_MODEL": "gemini-2.0-flash",
    "PROVIDER_MAX_CONCURRENCY": "1",
    "PROVIDER_MIN_INTERVAL_SECONDS": "0",
}


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, str]]:
    """A complete, valid environment, isolated from the developer's own .env."""
    for key, value in ENV_VALUES.items():
        monkeypatch.setenv(key, value)
    yield dict(ENV_VALUES)


@pytest.fixture
def settings(env: dict[str, str]) -> Settings:
    del env  # the fixture's effect is on the environment
    return load_settings(_env_file=None)


@pytest.fixture
def budget() -> Decimal:
    return Decimal("1.00")
