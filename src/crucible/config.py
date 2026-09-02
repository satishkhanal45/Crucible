"""Application settings.

Every setting is required: there are no silent defaults for anything that
changes behaviour. A missing variable fails at startup with an error that names
the variable, rather than surfacing as a confusing failure later in a run.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from functools import lru_cache
from typing import Annotated, Any, Literal

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

Environment = Literal["dev", "test", "prod"]

_LOG_LEVELS = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"})


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
    DEFAULT_JUDGE_MODEL: str

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
    """Process-wide settings. Cached, so import order cannot produce two views."""
    return load_settings()
