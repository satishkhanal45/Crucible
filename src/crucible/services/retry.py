"""Retry with exponential backoff and jitter.

Free-tier rate limits are the binding constraint on this project, so every
provider call goes through here on its way through `CostMeter`.
"""

from __future__ import annotations

import asyncio
import email.utils
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

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


class ProviderTimeout(TransientProviderError):
    """The provider did not answer inside our own timeout.

    A `TransientProviderError` on purpose: a read timeout is a client-side
    deadline, not a refusal, and a long prompt to a slow model is the ordinary
    reason for one. `httpx.ReadTimeout` is in no retry policy, so letting it
    escape ended a multi-hour run in the defender's proposal node — the same
    class of mistake as the unmapped 429 before it.
    """

    def __init__(self, provider: str, model: str, seconds: float, kind: str = "read") -> None:
        self.provider = provider
        self.model = model
        self.seconds = seconds
        self.kind = kind
        super().__init__(
            f"{provider} did not answer the {kind} within {seconds:.0f}s for {model!r}"
        )


class ProviderConfigurationError(ProviderError):
    """A provider refusal that retrying cannot fix.

    Retrying one of these burns free-tier quota to receive the same answer, so
    these are raised straight through the retry loop and are meant to reach the
    CLI boundary, which prints the one line that names the cause.
    """


class AuthenticationFailed(ProviderConfigurationError):
    """The provider rejected the API key (401, or 403 on a key-scope refusal)."""

    def __init__(self, provider: str, detail: str = "") -> None:
        self.provider = provider
        self.detail = detail
        message = (
            f"{provider} rejected the API key: check {provider.upper()}_API_KEY in .env "
            f"(the value is read at startup, so a stale shell environment can shadow it)"
        )
        super().__init__(f"{message} -- {detail}" if detail else message)


class ModelNotFound(ProviderConfigurationError):
    """The provider does not know this model id (404).

    Providers retire model ids on a schedule, so this is the expected failure
    after a decommissioning, not a transient one.
    """

    def __init__(self, provider: str, model: str, detail: str = "") -> None:
        self.provider = provider
        self.model = model
        self.detail = detail
        message = (
            f"{provider} does not recognise the model id {model!r}: it may have been "
            f"decommissioned. Update the model settings in .env"
        )
        super().__init__(f"{message} -- {detail}" if detail else message)


#: Statuses worth another attempt. 429 is the one that matters on a free tier.
RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


def parse_retry_after(value: str | None) -> float | None:
    """Seconds to wait, from a `Retry-After` header in either permitted form.

    The header is a number of seconds or an HTTP date. A date in the past, a
    negative number, or anything unparseable yields None so the caller falls
    back to its own backoff.
    """
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        try:
            parsed = email.utils.parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            # Neither a number nor an HTTP date: fall back to our own backoff.
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        seconds = (parsed - datetime.now(UTC)).total_seconds()
    return seconds if seconds > 0 else None


def provider_error_for(
    status_code: int,
    *,
    provider: str,
    model: str,
    retry_after: str | None = None,
    detail: str = "",
) -> ProviderError:
    """Map an HTTP status onto the typed error that says what to do about it.

    This is the single place that decides retryable from terminal. `httpx`'s own
    `HTTPStatusError` is deliberately not raised: it is not in any retry policy,
    so letting it escape is how a 429 came to end a run.
    """
    if status_code in {401, 403}:
        return AuthenticationFailed(provider, detail)
    if status_code == 404:
        return ModelNotFound(provider, model, detail)
    if status_code == 429:
        return RateLimited(
            f"{provider} rate limited the call to {model!r}" + (f": {detail}" if detail else ""),
            retry_after=parse_retry_after(retry_after),
        )
    if status_code in RETRYABLE_STATUSES or status_code >= 500:
        return TransientProviderError(
            f"{provider} returned {status_code} for {model!r}" + (f": {detail}" if detail else "")
        )
    return ProviderError(
        f"{provider} returned {status_code} for {model!r}" + (f": {detail}" if detail else "")
    )


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
