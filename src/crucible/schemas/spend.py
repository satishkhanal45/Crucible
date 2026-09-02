"""Boundary schemas for metered LLM spend."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class TokenUsage(BaseModel):
    """Token counts reported by a provider for one call."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class SpendRecord(BaseModel):
    """One persisted row of the spend ledger."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    round_id: uuid.UUID | None
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    # None means the model was not in the price table.
    estimated_cost_usd: Decimal | None
    created_at: datetime


class NewSpend(BaseModel):
    """A spend row about to be written."""

    model_config = ConfigDict(frozen=True)

    round_id: uuid.UUID | None
    provider: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=128)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    estimated_cost_usd: Decimal | None = None
