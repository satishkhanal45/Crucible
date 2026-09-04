"""Egress guard: deny-by-default outbound networking.

docs/spec.md section 1 makes this a constraint rather than documentation. A
target that is not on `TARGET_ALLOWLIST` cannot be reached, and the failure
names the host that was attempted.

TODO(phase-2): the executor wires this into every attempt and records an
`EgressViolation` as an `error` outcome.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from urllib.parse import urlsplit

import httpx

from crucible.config import Settings, provider_hosts
from crucible.logging import get_logger

logger = get_logger(__name__)

#: Hosts for the LLM providers this process can call, derived from the base URLs
#: in `crucible.config`, so adding a provider there is what adds its host here.
#:
#: These are **not** `TARGET_ALLOWLIST`. That list names attack targets; this one
#: names model providers. Keeping them separate is what stops a provider from
#: becoming reachable as a target, or a target from being called as a provider.
DEFAULT_PROVIDER_HOSTS: tuple[str, ...] = provider_hosts()


class EgressViolation(RuntimeError):
    """An outbound request was attempted to a host outside the allowlist."""

    def __init__(self, host: str, url: str | None = None) -> None:
        self.host = host
        self.url = url
        detail = f" (url: {url})" if url else ""
        super().__init__(
            f"egress to host {host!r} is not permitted: it is on neither "
            f"TARGET_ALLOWLIST nor the provider allowlist{detail}"
        )


def _normalise(host: str) -> str:
    """Lower-case hostname, without scheme, port, credentials, or brackets."""
    candidate = host.strip().lower()
    if "://" in candidate:
        candidate = urlsplit(candidate).hostname or ""
    if "@" in candidate:
        candidate = candidate.rsplit("@", 1)[1]
    if candidate.startswith("["):  # IPv6 literal
        candidate = candidate[1:].split("]", 1)[0]
    elif candidate.count(":") == 1:
        candidate = candidate.split(":", 1)[0]
    return candidate.rstrip(".")


class EgressGuard:
    """Checks hosts against `TARGET_ALLOWLIST` plus the provider hosts."""

    def __init__(
        self,
        allowlist: Iterable[str],
        provider_hosts: Iterable[str] = DEFAULT_PROVIDER_HOSTS,
    ) -> None:
        self.targets = frozenset(_normalise(host) for host in allowlist if host.strip())
        self.providers = frozenset(_normalise(host) for host in provider_hosts if host.strip())
        if not self.targets:
            raise ValueError("an egress allowlist may not be empty: egress is deny-by-default")

    @classmethod
    def from_settings(
        cls, settings: Settings, provider_hosts: Iterable[str] = DEFAULT_PROVIDER_HOSTS
    ) -> EgressGuard:
        return cls(settings.TARGET_ALLOWLIST, provider_hosts)

    @property
    def allowed_hosts(self) -> frozenset[str]:
        return self.targets | self.providers

    def is_allowed(self, host: str) -> bool:
        return _normalise(host) in self.allowed_hosts

    def check_host(self, host: str) -> str:
        normalised = _normalise(host)
        if normalised not in self.allowed_hosts:
            logger.warning("egress.blocked", extra={"attempted_host": normalised})
            raise EgressViolation(normalised)
        return normalised

    def check_url(self, url: str) -> str:
        """Validate a URL and return its host. Raises `EgressViolation`."""
        host = urlsplit(url).hostname
        if not host:
            raise EgressViolation("<no host>", url)
        normalised = _normalise(host)
        if normalised not in self.allowed_hosts:
            logger.warning("egress.blocked", extra={"attempted_host": normalised})
            raise EgressViolation(normalised, url)
        return normalised


class GuardedTransport(httpx.AsyncBaseTransport):
    """An httpx transport that refuses to send anywhere off the allowlist."""

    def __init__(self, guard: EgressGuard, inner: httpx.AsyncBaseTransport | None = None) -> None:
        self._guard = guard
        self._inner = inner or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self._guard.check_url(str(request.url))
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


def guarded_client(guard: EgressGuard, **kwargs: Any) -> httpx.AsyncClient:
    """An `httpx.AsyncClient` that cannot reach a host outside the allowlist."""
    return httpx.AsyncClient(transport=GuardedTransport(guard), **kwargs)
