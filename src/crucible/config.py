"""Application settings.

Every setting is required: there are no silent defaults for anything that
changes behaviour. A missing variable fails at startup with an error that names
the variable, rather than surfacing as a confusing failure later in a run.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from functools import lru_cache
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from crucible.logging import get_logger

logger = get_logger(__name__)

Environment = Literal["dev", "test", "prod"]

_LOG_LEVELS = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"})


class LLMProvider(StrEnum):
    """A provider an agent can be pointed at.

    Two providers exist because free-tier rate limits are the binding constraint
    on this project: Groq gives one token-per-minute pool per model that all
    four agents contend for, and a second host is a second pool. Which agent
    runs where is chosen per agent, in settings, never globally.
    """

    GROQ = "groq"
    DEEPSEEK = "deepseek"


#: The four agents whose provider and model are configured independently.
AGENT_ROLES: tuple[str, ...] = ("target", "attacker", "defender", "classifier")

# --------------------------------------------------------------------------- #
# Provider endpoints
# --------------------------------------------------------------------------- #
# Both providers speak the OpenAI chat-completions shape, which is why one
# client serves both: `ChatCompletionsLLM` is parameterised by the base URL
# below, so there is exactly one place where a provider's HTTP status maps to
# our typed errors. A base URL is configuration, like a model id, and belongs
# here rather than in a module that calls it.
#
# Checked on 2026-09-04. DeepSeek documents the OpenAI-compatible endpoint at
# https://api-docs.deepseek.com; Groq's is https://console.groq.com/docs.
PROVIDER_BASE_URLS: Mapping[str, str] = {
    LLMProvider.GROQ.value: "https://api.groq.com/openai/v1",
    LLMProvider.DEEPSEEK.value: "https://api.deepseek.com/v1",
}

# --------------------------------------------------------------------------- #
# HTTP timeouts
# --------------------------------------------------------------------------- #
# A client-side timeout is not a model failure, and it killed a run: the
# defender's proposal prompt is long, and DeepSeek took longer to answer it than
# a flat 60s read timeout allowed. Read timeouts are therefore per role, because
# the roles ask for very different amounts of work, and overridable per provider,
# because the same prompt takes different times on different hosts.
#
# The connect timeout stays short everywhere. A host that has not completed a TCP
# handshake in ten seconds is not slow, it is unreachable, and waiting three
# minutes to discover that helps nobody.
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10.0

#: Read timeout per agent role, in seconds. The defender's is generous because
#: `propose_one` sends the longest prompt in the system and asks for a whole
#: `DefenseConfig` back; the attacker's is raised for the same reason, to a
#: lesser degree. Override per provider with PROVIDER_READ_TIMEOUTS.
ROLE_READ_TIMEOUT_SECONDS: Mapping[str, float] = {
    "target": 60.0,
    "attacker": 120.0,
    "defender": 180.0,
    "classifier": 60.0,
}

#: For a caller with no role of its own.
DEFAULT_READ_TIMEOUT_SECONDS = 60.0

# --------------------------------------------------------------------------- #
# Per-provider rate limits
# --------------------------------------------------------------------------- #
# The pacer stays: it is what keeps a burst from earning a 429, and it is keyed
# per provider so two accounts never share a window. What has to be right is the
# SIZE of each window, and that is a property of the account's tier, not of the
# code. `PROVIDER_RATE_LIMITS` in `.env` is the normal way to set it.
#
# These defaults assume the tiers this project was developed against, on
# 2026-09-05: Groq's FREE tier, whose binding limit is 8000 tokens per minute,
# and a PAID DeepSeek key, which is far less constrained. A limit sized for the
# wrong tier is not a safety margin — a free-tier number on a paid key spent most
# of a smoke round sitting in 40-second token-budget waits.
DEFAULT_PROVIDER_RATE_LIMITS: Mapping[str, tuple[int, int]] = {
    LLMProvider.GROQ.value: (6500, 25),
    LLMProvider.DEEPSEEK.value: (120_000, 300),
}

#: Concurrent in-flight calls per provider. One is right for a free tier, where
#: a second call in flight is a 429; a paid tier can overlap. Overridable with
#: PROVIDER_CONCURRENCY.
DEFAULT_PROVIDER_CONCURRENCY: Mapping[str, int] = {
    LLMProvider.GROQ.value: 1,
    LLMProvider.DEEPSEEK.value: 4,
}

#: The Tier 3 judge's host. The judge is not built in this cut (B4), but the
#: egress guard still has to know the host the setting names.
JUDGE_HOST = "generativelanguage.googleapis.com"


def provider_hosts() -> tuple[str, ...]:
    """Every host the process may reach to call a model.

    These are **provider** hosts, kept deliberately separate from
    `TARGET_ALLOWLIST`, which is the list of attack targets. A provider is not
    a target and must never be reachable as one.
    """
    hosts = [urlsplit(url).hostname or "" for url in PROVIDER_BASE_URLS.values()]
    hosts.append(JUDGE_HOST)
    return tuple(dict.fromkeys(host for host in hosts if host))


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
    # DeepSeek v4, 2026-09-05. These are the three ids the models endpoint
    # returns for this account; `deepseek-chat` and `deepseek-reasoner` were
    # removed because they no longer exist.
    #
    # !! THE RATES BELOW ARE UNCONFIRMED PLACEHOLDERS. !! They are carried over
    # from the previous generation's published prices, because an entry with an
    # approximate rate still enforces ROUND_BUDGET_USD while a missing entry
    # silently disables it. They have NOT been read off a page listing v4.
    # **Confirm every one against https://deepseek.com/pricing before any number
    # produced with them reaches docs/findings.md** — a cost figure derived from
    # a guessed rate is not a measurement. Override with MODEL_PRICING rather
    # than editing this table. Cache-hit and off-peak rates are not modelled, so
    # a recorded cost is an upper bound.
    "deepseek:deepseek-v4-flash": ("0.27", "1.10"),
    "deepseek:deepseek-v4-pro": ("0.55", "2.19"),
    "deepseek:deepseek-v4-flash-vision-exp": ("0.27", "1.10"),
}


def price_key(provider: str, model: str) -> str:
    """The pricing table's key. Lower-cased so `.env` casing cannot miss."""
    return f"{provider.strip().lower()}:{model.strip().lower()}"


def _parse_pairs(value: object, name: str, shape: str) -> dict[str, float]:
    """`key=number` entries, comma separated, for the per-provider overrides."""
    if isinstance(value, dict):
        return {str(key).strip().lower(): float(number) for key, number in value.items()}
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a comma-separated string or a mapping")
    parsed: dict[str, float] = {}
    for entry in (item.strip() for item in value.split(",")):
        if not entry:
            continue
        key, separator, number = entry.partition("=")
        if not separator:
            raise ValueError(f"{name} entry {entry!r} must read {shape}")
        try:
            parsed[key.strip().lower()] = float(number.strip())
        except ValueError as error:
            raise ValueError(f"{name} entry {entry!r} has a non-numeric value") from error
    return parsed


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
    DEEPSEEK_API_KEY: str
    GEMINI_API_KEY: str
    # Comma-separated in the environment; a list everywhere else. NoDecode stops
    # pydantic-settings from trying to read the raw value as JSON.
    TARGET_ALLOWLIST: Annotated[list[str], NoDecode]
    ROUND_BUDGET_USD: Decimal = Field(gt=Decimal(0))
    EMBEDDING_MODEL: str

    # Model ids. Every one is read from here, never from a CLI default: an id
    # hardcoded in a typer option makes editing `.env` a no-op, which is how a
    # decommissioned model survived a configuration change.
    #
    # Provider is chosen **per agent**, paired with that agent's model. There is
    # deliberately no global provider setting: the point of the second provider
    # is that the four agents can be spread across two rate-limit pools, which a
    # single `LLM_PROVIDER` could not express. (That setting was replaced by
    # these four; `.env.example` records the change.)
    TARGET_PROVIDER: LLMProvider
    TARGET_MODEL: str
    ATTACKER_PROVIDER: LLMProvider
    ATTACKER_MODEL: str
    DEFENDER_PROVIDER: LLMProvider
    DEFENDER_MODEL: str
    CLASSIFIER_PROVIDER: LLMProvider
    CLASSIFIER_MODEL: str
    # The Tier 3 judge is a separate concern on a separate host, and is not one
    # of the four agents: it is not built in this cut (B4) and its provider is
    # not an `LLMProvider`. It stays a plain string so that naming a judge host
    # cannot be mistaken for pointing an agent at one.
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
    # Per-provider overrides, `provider=tokens/requests` comma separated. This
    # is the normal way to set a limit: it must match the account's actual tier,
    # which the code cannot know. Falls back to DEFAULT_PROVIDER_RATE_LIMITS,
    # then to the global pair above.
    PROVIDER_RATE_LIMITS: Annotated[dict[str, tuple[int, int]], NoDecode] = {}
    # Per-provider concurrent calls, `provider=n` comma separated. A free tier
    # wants 1; a paid one can overlap. Falls back to DEFAULT_PROVIDER_CONCURRENCY
    # and then to PROVIDER_MAX_CONCURRENCY.
    PROVIDER_CONCURRENCY: Annotated[dict[str, int], NoDecode] = {}
    # Per-provider, per-role read timeouts, `provider:role=seconds` or
    # `role=seconds`, comma separated. Falls back to ROLE_READ_TIMEOUT_SECONDS.
    PROVIDER_READ_TIMEOUTS: Annotated[dict[str, float], NoDecode] = {}

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

    @field_validator("PROVIDER_RATE_LIMITS", mode="before")
    @classmethod
    def _parse_rate_limits(cls, value: object) -> dict[str, tuple[int, int]]:
        if isinstance(value, dict):
            return {
                str(key).strip().lower(): (int(limits[0]), int(limits[1]))
                for key, limits in value.items()
            }
        if not isinstance(value, str):
            raise ValueError("PROVIDER_RATE_LIMITS must be a comma-separated string or a mapping")
        parsed: dict[str, tuple[int, int]] = {}
        for entry in (item.strip() for item in value.split(",")):
            if not entry:
                continue
            provider, separator, limits = entry.partition("=")
            if not separator or "/" not in limits:
                raise ValueError(
                    f"PROVIDER_RATE_LIMITS entry {entry!r} must read "
                    f"provider=tokens_per_minute/requests_per_minute"
                )
            tokens, _, requests = limits.partition("/")
            try:
                parsed[provider.strip().lower()] = (int(tokens.strip()), int(requests.strip()))
            except ValueError as error:
                raise ValueError(
                    f"PROVIDER_RATE_LIMITS entry {entry!r} has a non-integer limit"
                ) from error
        return parsed

    @field_validator("PROVIDER_CONCURRENCY", mode="before")
    @classmethod
    def _parse_concurrency(cls, value: object) -> dict[str, int]:
        return {
            key: int(number)
            for key, number in _parse_pairs(value, "PROVIDER_CONCURRENCY", "provider=n").items()
        }

    @field_validator("PROVIDER_READ_TIMEOUTS", mode="before")
    @classmethod
    def _parse_read_timeouts(cls, value: object) -> dict[str, float]:
        return {
            key: float(number)
            for key, number in _parse_pairs(
                value, "PROVIDER_READ_TIMEOUTS", "provider:role=seconds"
            ).items()
        }

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
    def agents(self) -> tuple[tuple[str, LLMProvider, str], ...]:
        """`(role, provider, model)` for each agent, in a fixed order.

        One place resolves the pairing, so nothing else has to know that
        `ATTACKER_PROVIDER` goes with `ATTACKER_MODEL` and not with another.
        """
        return (
            ("target", self.TARGET_PROVIDER, self.TARGET_MODEL),
            ("attacker", self.ATTACKER_PROVIDER, self.ATTACKER_MODEL),
            ("defender", self.DEFENDER_PROVIDER, self.DEFENDER_MODEL),
            ("classifier", self.CLASSIFIER_PROVIDER, self.CLASSIFIER_MODEL),
        )

    def provider_for(self, role: str) -> LLMProvider:
        """The provider configured for one agent role."""
        for name, provider, _ in self.agents:
            if name == role:
                return provider
        raise KeyError(f"{role!r} is not an agent role: expected one of {AGENT_ROLES}")

    def model_for(self, role: str) -> str:
        """The model configured for one agent role."""
        for name, _, model in self.agents:
            if name == role:
                return model
        raise KeyError(f"{role!r} is not an agent role: expected one of {AGENT_ROLES}")

    def api_key_for(self, provider: LLMProvider) -> str:
        """The credential for one provider. Each has its own; they never share."""
        keys = {
            LLMProvider.GROQ: self.GROQ_API_KEY,
            LLMProvider.DEEPSEEK: self.DEEPSEEK_API_KEY,
        }
        return keys[provider]

    def base_url_for(self, provider: LLMProvider) -> str:
        return PROVIDER_BASE_URLS[provider.value]

    def rate_limits_for(self, provider: LLMProvider | str) -> tuple[int, int]:
        """`(tokens_per_minute, requests_per_minute)` for one provider.

        The global pair is the default; `PROVIDER_RATE_LIMITS` overrides it per
        provider, because the two hosts publish different limits and a window
        sized for one would either throttle or overrun the other.
        """
        name = str(provider).strip().lower()
        configured = self.PROVIDER_RATE_LIMITS.get(name)
        if configured is not None:
            return configured
        return DEFAULT_PROVIDER_RATE_LIMITS.get(
            name, (self.PROVIDER_TOKENS_PER_MINUTE, self.PROVIDER_REQUESTS_PER_MINUTE)
        )

    def concurrency_for(self, provider: LLMProvider | str) -> int:
        """How many calls to this provider may be in flight at once."""
        name = str(provider).strip().lower()
        configured = self.PROVIDER_CONCURRENCY.get(name)
        if configured is not None:
            return max(1, configured)
        return max(1, DEFAULT_PROVIDER_CONCURRENCY.get(name, self.PROVIDER_MAX_CONCURRENCY))

    def read_timeout_for(self, provider: LLMProvider | str, role: str) -> float:
        """Seconds to wait for a reply, for one provider and one agent role.

        `provider:role` wins over `role`, so one slow host can be given more
        time without loosening the timeout everywhere.
        """
        name = str(provider).strip().lower()
        for key in (f"{name}:{role}", role):
            configured = self.PROVIDER_READ_TIMEOUTS.get(key)
            if configured is not None:
                return configured
        return ROLE_READ_TIMEOUT_SECONDS.get(role, DEFAULT_READ_TIMEOUT_SECONDS)

    def timeout_for(self, provider: LLMProvider | str, role: str) -> tuple[float, float]:
        """`(connect, read)` seconds. Connect stays short; read is per role."""
        return (DEFAULT_CONNECT_TIMEOUT_SECONDS, self.read_timeout_for(provider, role))

    @property
    def provider_rate_limits(self) -> dict[str, tuple[int, int]]:
        """Every configured provider's limits, resolved. Passed to the pacer."""
        return {provider.value: self.rate_limits_for(provider) for provider in LLMProvider}

    @property
    def provider_concurrency(self) -> dict[str, int]:
        """Every configured provider's concurrency, resolved. For the pacer."""
        return {provider.value: self.concurrency_for(provider) for provider in LLMProvider}

    @property
    def configured_models(self) -> tuple[str, ...]:
        """Every `provider:model` this process can actually call, in order."""
        keys = [price_key(provider.value, model) for _, provider, model in self.agents]
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
