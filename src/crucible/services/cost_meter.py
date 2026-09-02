"""The cost meter: every LLM call in Crucible routes through here.

It is both the ledger (what did this round spend?) and the circuit breaker
(stop the round before it blows the budget). Retry and backoff live next door in
`crucible.services.retry` and are applied by `CostMeter.call`.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal

from crucible.logging import get_logger
from crucible.repositories.spend import SpendRepositoryProtocol
from crucible.schemas.spend import NewSpend, SpendRecord, TokenUsage
from crucible.services.retry import RetryPolicy, retry_async

logger = get_logger(__name__)

CENT = Decimal("0.00000001")
PER_MILLION = Decimal(1_000_000)


class BudgetExceeded(RuntimeError):
    """A round spent more than `ROUND_BUDGET_USD`.

    Callers end the round cleanly with partial results; they never crash.
    """

    def __init__(self, round_id: uuid.UUID | None, spent: Decimal, budget: Decimal) -> None:
        self.round_id = round_id
        self.spent = spent
        self.budget = budget
        label = str(round_id) if round_id is not None else "unassigned"
        super().__init__(
            f"Round {label} exceeded ROUND_BUDGET_USD: spent ${spent:.6f} of ${budget:.6f}"
        )


@dataclass(frozen=True)
class ModelPrice:
    """USD per one million tokens."""

    prompt_usd_per_1m: Decimal
    completion_usd_per_1m: Decimal


@dataclass(frozen=True)
class MeteredResult[T]:
    """What a metered provider call returns: a value plus its token usage."""

    value: T
    usage: TokenUsage


def price_key(provider: str, model: str) -> str:
    return f"{provider.strip().lower()}:{model.strip().lower()}"


# Published list prices. Free tiers cost nothing in cash, but the meter still
# needs a number to reason about budgets and to compare configurations.
DEFAULT_PRICING: Mapping[str, ModelPrice] = {
    "groq:llama-3.3-70b-versatile": ModelPrice(Decimal("0.59"), Decimal("0.79")),
    "groq:llama-3.1-8b-instant": ModelPrice(Decimal("0.05"), Decimal("0.08")),
    "groq:openai/gpt-oss-20b": ModelPrice(Decimal("0.10"), Decimal("0.50")),
    "gemini:gemini-2.0-flash": ModelPrice(Decimal("0.10"), Decimal("0.40")),
    "gemini:gemini-2.0-flash-lite": ModelPrice(Decimal("0.075"), Decimal("0.30")),
    "gemini:gemini-1.5-flash": ModelPrice(Decimal("0.075"), Decimal("0.30")),
}


class CostMeter:
    """Records spend per round and refuses to let a round run past its budget."""

    def __init__(
        self,
        repository: SpendRepositoryProtocol,
        budget_usd: Decimal,
        *,
        pricing: Mapping[str, ModelPrice] = DEFAULT_PRICING,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        if budget_usd <= 0:
            raise ValueError("budget_usd must be greater than zero")
        self._repository = repository
        self._budget = budget_usd
        self._pricing = pricing
        self._retry_policy = retry_policy or RetryPolicy()

    @property
    def budget_usd(self) -> Decimal:
        return self._budget

    def estimate(self, provider: str, model: str, usage: TokenUsage) -> Decimal | None:
        """Cost of one call, or None when the model has no published price."""
        price = self._pricing.get(price_key(provider, model))
        if price is None:
            return None
        cost = (
            Decimal(usage.prompt_tokens) * price.prompt_usd_per_1m
            + Decimal(usage.completion_tokens) * price.completion_usd_per_1m
        ) / PER_MILLION
        return cost.quantize(CENT)

    async def spent(self, round_id: uuid.UUID | None) -> Decimal:
        return await self._repository.total_for_round(round_id)

    async def ensure_within_budget(self, round_id: uuid.UUID | None) -> None:
        """Raise if the round has already gone over. Call before spending more."""
        total = await self.spent(round_id)
        if total > self._budget:
            raise BudgetExceeded(round_id, total, self._budget)

    async def record(
        self,
        *,
        round_id: uuid.UUID | None,
        provider: str,
        model: str,
        usage: TokenUsage,
    ) -> SpendRecord:
        """Persist one call's spend, then enforce the round budget."""
        cost = self.estimate(provider, model, usage)
        if cost is None:
            logger.warning(
                "cost_meter.unknown_model",
                extra={
                    "provider": provider,
                    "model": model,
                    "round_id": str(round_id) if round_id else None,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                },
            )

        record = await self._repository.add(
            NewSpend(
                round_id=round_id,
                provider=provider,
                model=model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                estimated_cost_usd=cost,
            )
        )

        total = await self.spent(round_id)
        if total > self._budget:
            raise BudgetExceeded(round_id, total, self._budget)
        return record

    async def call[T](
        self,
        provider_call: Callable[[], Awaitable[MeteredResult[T]]],
        *,
        round_id: uuid.UUID | None,
        provider: str,
        model: str,
        retry_policy: RetryPolicy | None = None,
    ) -> T:
        """Run a provider call with backoff, meter it, and return its value.

        `BudgetExceeded` is raised after the spend is recorded, so the ledger
        always reflects what was actually consumed.
        """
        await self.ensure_within_budget(round_id)
        result = await retry_async(provider_call, retry_policy or self._retry_policy)
        await self.record(round_id=round_id, provider=provider, model=model, usage=result.usage)
        return result.value
