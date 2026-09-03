"""Application settings.

Every setting is required: there are no silent defaults for anything that
changes behaviour. A missing variable fails at startup with an error that names
the variable, rather than surfacing as a confusing failure later in a run.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from typing import Annotated, Any, Literal

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from crucible.logging import get_logger

logger = get_logger(__name__)

Environment = Literal["dev", "test", "prod"]

_LOG_LEVELS = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"})

# --------------------------------------------------------------------------- #
# Model pricing
# --------------------------------------------------------------------------- #
# USD per one million tokens, as (prompt, completion). Free tiers cost nothing
# in cash, but the meter still needs a number: with no entry the cost is NULL,
# ROUND_BUDGET_USD is inert, and `BudgetExceeded` can never fire, which removes
# the only cost control in the system.
#
# These are list prices transcribed on 2026-09-03 from the providers' published
# pricing pages (https://groq.com/pricing, https://ai.google.dev/pricing).
# Providers change rates and retire model ids without notice: override with the
# MODEL_PRICING environment variable rather than editing this table, and see
# https://console.groq.com/docs/deprecations for retirements.
DEFAULT_MODEL_PRICING: Mapping[str, tuple[str, str]] = {
    "groq:openai/gpt-oss-120b": ("0.15", "0.75"),
    "groq:openai/gpt-oss-20b": ("0.10", "0.50"),
    "groq:llama-3.1-8b-instant": ("0.05", "0.08"),
    "groq:llama-3.3-70b-versatile": ("0.59", "0.79"),
    "gemini:gemini-2.0-flash": ("0.10", "0.40"),
    "gemini:gemini-2.0-flash-lite": ("0.075", "0.30"),
    "gemini:gemini-1.5-flash": ("0.075", "0.30"),
}


def price_key(provider: str, model: str) -> str:
    """The pricing table's key. Lower-cased so `.env` casing cannot miss."""
    return f"{provider.strip().lower()}:{model.strip().lower()}"


class ConfigurationError(RuntimeError):
    """Raised when the process is not configured well enough to start."""


class Settings(BaseSettings):
    """Runtime configuration, loaded from the environment or a `.env` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    DATABASE_URL: str
    ENV: Environment
    LOG_LEVEL: str
    GROQ_API_KEY: str
    GEMINI_API_KEY: str
    # Comma-separated in the environment; a list everywhere else. NoDecode stops
    # pydantic-settings from trying to read the raw value as JSON.
    TARGET_ALLOWLIST: Annotated[list[str], NoDecode]
    ROUND_BUDGET_USD: Decimal = Field(gt=Decimal(0))
    EMBEDDING_MODEL: str

    # Model ids. Every one is read from here, never from a CLI default: an id
    # hardcoded in a typer option makes editing `.env` a no-op, which is how a
    # decommissioned model survived a configuration change.
    LLM_PROVIDER: str
    TARGET_MODEL: str
    ATTACKER_MODEL: str
    DEFENDER_MODEL: str
    CLASSIFIER_MODEL: str
    JUDGE_PROVIDER: str
    DEFAULT_JUDGE_MODEL: str

    # Free-tier pacing. Required like everything else here: a silent default
    # for a rate limit is how a run dies 23 calls in. `.env.example` ships the
    # free-tier-safe values, concurrency 1 with a small delay.
    PROVIDER_MAX_CONCURRENCY: int = Field(ge=1, le=16)
    PROVIDER_MIN_INTERVAL_SECONDS: float = Field(ge=0.0, le=60.0)
    # The limit that actually bites on Groq's free tier is tokens per minute,
    # not requests: 8000 TPM, which one long prompt and its reply can consume
    # several times over inside a quiet minute. Ship a margin below the real
    # ceiling, because the estimate that books the window is approximate.
    PROVIDER_TOKENS_PER_MINUTE: int = Field(ge=0, le=10_000_000)
    PROVIDER_REQUESTS_PER_MINUTE: int = Field(ge=0, le=10_000)

    # `provider:model=prompt/completion` entries, comma separated. Merged over
    # DEFAULT_MODEL_PRICING, so it only needs the rates that have changed.
    MODEL_PRICING: Annotated[dict[str, tuple[Decimal, Decimal]], NoDecode] = {}

    @field_validator("MODEL_PRICING", mode="before")
    @classmethod
    def _parse_pricing(cls, value: object) -> dict[str, tuple[Decimal, Decimal]]:
        if isinstance(value, dict):
            return {
                str(key): (Decimal(str(rates[0])), Decimal(str(rates[1])))
                for key, rates in value.items()
            }
        if not isinstance(value, str):
            raise ValueError("MODEL_PRICING must be a comma-separated string or a mapping")
        parsed: dict[str, tuple[Decimal, Decimal]] = {}
        for entry in (item.strip() for item in value.split(",")):
            if not entry:
                continue
            key, separator, rates = entry.rpartition("=")
            if not separator or "/" not in rates:
                raise ValueError(
                    f"MODEL_PRICING entry {entry!r} must read provider:model=prompt/completion"
                )
            prompt, _, completion = rates.partition("/")
            try:
                parsed[key.strip().lower()] = (Decimal(prompt.strip()), Decimal(completion.strip()))
            except InvalidOperation as error:
                raise ValueError(f"MODEL_PRICING entry {entry!r} has a non-numeric rate") from error
        return parsed

    @field_validator("TARGET_ALLOWLIST", mode="before")
    @classmethod
    def _split_allowlist(cls, value: object) -> list[str]:
        if isinstance(value, str):
            hosts = [host.strip() for host in value.split(",")]
        elif isinstance(value, list):
            hosts = [str(host).strip() for host in value]
        else:
            raise ValueError("TARGET_ALLOWLIST must be a comma-separated string or a list")
        hosts = [host for host in hosts if host]
        if not hosts:
            raise ValueError("TARGET_ALLOWLIST must not be empty: egress is deny-by-default")
        return hosts

    @field_validator("LOG_LEVEL")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        level = value.strip().upper()
        if level not in _LOG_LEVELS:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(_LOG_LEVELS)}, got {value!r}")
        return level

    @property
    def log_level_number(self) -> int:
        return logging.getLevelNamesMapping()[self.LOG_LEVEL]

    @property
    def is_production(self) -> bool:
        return self.ENV == "prod"

    @property
    def model_pricing(self) -> dict[str, tuple[Decimal, Decimal]]:
        """The effective pricing table: the dated defaults, then any override."""
        merged = {
            key: (Decimal(prompt), Decimal(completion))
            for key, (prompt, completion) in DEFAULT_MODEL_PRICING.items()
        }
        merged.update(self.MODEL_PRICING)
        return merged

    @property
    def configured_models(self) -> tuple[str, ...]:
        """Every `provider:model` this process can actually call, in order."""
        agents = (
            self.TARGET_MODEL,
            self.ATTACKER_MODEL,
            self.DEFENDER_MODEL,
            self.CLASSIFIER_MODEL,
        )
        keys = [price_key(self.LLM_PROVIDER, model) for model in agents]
        keys.append(price_key(self.JUDGE_PROVIDER, self.DEFAULT_JUDGE_MODEL))
        seen: dict[str, None] = {}
        for key in keys:
            seen.setdefault(key, None)
        return tuple(seen)

    def unpriced_models(self) -> tuple[str, ...]:
        """Configured models the meter has no rate for, in configuration order."""
        pricing = self.model_pricing
        return tuple(key for key in self.configured_models if key not in pricing)


def validate_model_pricing(settings: Settings) -> tuple[str, ...]:
    """Warn once per configured model that has no price, and never fail.

    An unpriced model still runs: the point of the warning is that its spend
    records as NULL, so `ROUND_BUDGET_USD` does not constrain it and the
    `budget_exceeded` halt reason is unreachable for that model. Refusing to
    start would be worse — it would block a run over a missing price.
    """
    unpriced = settings.unpriced_models()
    for key in unpriced:
        logger.warning(
            "config.model_not_priced",
            extra={
                "model": key,
                "detail": (
                    f"no price for {key}: spend records as NULL and ROUND_BUDGET_USD "
                    f"cannot be enforced for it. Add it to MODEL_PRICING in .env."
                ),
            },
        )
    return unpriced


def _describe(error: ValidationError) -> str:
    lines: list[str] = []
    for detail in error.errors():
        name = ".".join(str(part) for part in detail["loc"]) or "<settings>"
        if detail["type"] == "missing":
            lines.append(f"{name}: required environment variable is not set")
        else:
            lines.append(f"{name}: {detail['msg']}")
    return "; ".join(lines)


def load_settings(**overrides: Any) -> Settings:
    """Build `Settings`, converting validation failures into a named error."""
    try:
        return Settings(**overrides)
    except ValidationError as error:
        raise ConfigurationError(f"Invalid configuration -> {_describe(error)}") from error


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Cached, so import order cannot produce two views.

    The pricing check runs here, which is once per process because of the cache.
    """
    settings = load_settings()
    validate_model_pricing(settings)
    return settings
