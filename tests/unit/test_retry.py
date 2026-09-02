"""Backoff behaviour for provider calls. Free-tier rate limits are the constraint."""

from __future__ import annotations

import pytest

from crucible.services.retry import (
    ProviderError,
    RateLimited,
    RetryPolicy,
    TransientProviderError,
    retry_async,
)


class Clock:
    """Records requested sleeps instead of performing them."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def sleep(self, delay: float) -> None:
        self.delays.append(delay)


async def test_returns_immediately_on_success() -> None:
    clock = Clock()

    async def call() -> str:
        return "ok"

    assert await retry_async(call, RetryPolicy(), sleep=clock.sleep) == "ok"
    assert clock.delays == []


async def test_backoff_grows_exponentially_and_is_capped() -> None:
    clock = Clock()
    calls = 0

    async def call() -> str:
        nonlocal calls
        calls += 1
        if calls < 5:
            raise TransientProviderError("503")
        return "ok"

    policy = RetryPolicy(attempts=6, base_delay=1.0, max_delay=4.0, jitter=0.0)
    assert await retry_async(call, policy, sleep=clock.sleep, jitter=lambda: 0.0) == "ok"
    assert clock.delays == [1.0, 2.0, 4.0, 4.0]


async def test_retry_after_overrides_the_computed_backoff() -> None:
    clock = Clock()
    calls = 0

    async def call() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RateLimited("429", retry_after=7.5)
        return "ok"

    await retry_async(call, RetryPolicy(base_delay=1.0), sleep=clock.sleep)
    assert clock.delays == [7.5]


async def test_raises_the_last_error_after_exhausting_attempts() -> None:
    clock = Clock()

    async def call() -> str:
        raise RateLimited("429 forever")

    with pytest.raises(RateLimited, match="429 forever"):
        await retry_async(call, RetryPolicy(attempts=3, jitter=0.0), sleep=clock.sleep)
    assert len(clock.delays) == 2


async def test_unlisted_errors_are_not_retried() -> None:
    clock = Clock()

    async def call() -> str:
        raise ProviderError("a bad request is not worth retrying")

    with pytest.raises(ProviderError):
        await retry_async(call, RetryPolicy(), sleep=clock.sleep)
    assert clock.delays == []
