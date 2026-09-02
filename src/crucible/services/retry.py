"""Retry with exponential backoff and jitter.

Free-tier rate limits are the binding constraint on this project, so every
provider call goes through here on its way through `CostMeter`.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from crucible.logging import get_logger

logger = get_logger(__name__)

SleepFn = Callable[[float], Awaitable[None]]
JitterFn = Callable[[], float]


class ProviderError(RuntimeError):
    """A call to an LLM provider failed."""


class RateLimited(ProviderError):
    """The provider refused the call because of a rate limit."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class TransientProviderError(ProviderError):
    """A provider failure worth retrying (5xx, connection reset, timeout)."""


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 5
    base_delay: float = 0.5
    max_delay: float = 30.0
    jitter: float = 0.25
    retry_on: tuple[type[BaseException], ...] = field(default=(RateLimited, TransientProviderError))

    def delay_for(self, attempt: int, jitter_fraction: float) -> float:
        """Backoff for a 1-based attempt number, with `jitter_fraction` applied."""
        exponential = self.base_delay * 2.0 ** (attempt - 1)
        capped = min(exponential, self.max_delay)
        return capped * (1.0 + self.jitter * jitter_fraction)


async def retry_async[T](
    call: Callable[[], Awaitable[T]],
    policy: RetryPolicy | None = None,
    *,
    sleep: SleepFn = asyncio.sleep,
    jitter: JitterFn = random.random,
) -> T:
    """Run `call`, retrying the policy's exception types with backoff."""
    active = policy or RetryPolicy()
    if active.attempts < 1:
        raise ValueError("RetryPolicy.attempts must be at least 1")

    last: BaseException | None = None
    for attempt in range(1, active.attempts + 1):
        try:
            return await call()
        except active.retry_on as error:
            last = error
            if attempt == active.attempts:
                break
            retry_after = getattr(error, "retry_after", None)
            delay = (
                float(retry_after)
                if isinstance(retry_after, int | float)
                else active.delay_for(attempt, jitter())
            )
            logger.warning(
                "provider.retry",
                extra={
                    "attempt": attempt,
                    "max_attempts": active.attempts,
                    "delay_seconds": round(delay, 3),
                    "error": str(error),
                },
            )
            await sleep(delay)

    assert last is not None  # the loop only breaks after an exception was captured
    raise last
