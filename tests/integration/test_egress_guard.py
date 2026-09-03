"""Verification test 15: the egress guard.

docs/spec.md section 1 makes the allowlist a constraint, not documentation.
"""

from __future__ import annotations

import httpx
import pytest

from crucible.config import Settings, load_settings
from crucible.execution.egress import (
    DEFAULT_PROVIDER_HOSTS,
    EgressGuard,
    EgressViolation,
    GuardedTransport,
    guarded_client,
)


@pytest.fixture
def guard(env: dict[str, str]) -> EgressGuard:
    del env
    return EgressGuard(["localhost", "127.0.0.1"])


def test_allowlisted_target_hosts_pass(guard: EgressGuard) -> None:
    assert guard.check_url("http://localhost:8000/query") == "localhost"
    assert guard.check_url("http://127.0.0.1:9000/") == "127.0.0.1"
    assert guard.is_allowed("LOCALHOST") is True


def test_provider_hosts_pass(guard: EgressGuard) -> None:
    for host in DEFAULT_PROVIDER_HOSTS:
        assert guard.check_url(f"https://{host}/v1/chat/completions") == host


def test_a_host_off_the_allowlist_raises_and_names_it(guard: EgressGuard) -> None:
    url = "https://evil.example.com/collect?data=1"

    with pytest.raises(EgressViolation) as raised:
        guard.check_url(url)

    assert raised.value.host == "evil.example.com"
    assert "evil.example.com" in str(raised.value)
    assert "TARGET_ALLOWLIST" in str(raised.value)


@pytest.mark.parametrize(
    "url",
    [
        "https://attacker.test/exfil",
        "http://169.254.169.254/latest/meta-data/",
        "http://192.168.1.10:8080/",
        "https://user:pass@evil.example.com/",
        "https://api.groq.com.evil.example.com/",
        "https://sub.localhost.evil.test/",
    ],
)
def test_lookalike_and_metadata_hosts_are_all_blocked(guard: EgressGuard, url: str) -> None:
    with pytest.raises(EgressViolation):
        guard.check_url(url)


def test_the_guard_is_built_from_settings(env: dict[str, str]) -> None:
    del env
    settings: Settings = load_settings(_env_file=None, TARGET_ALLOWLIST="localhost,10.0.0.5")
    guard = EgressGuard.from_settings(settings)

    assert guard.check_url("http://10.0.0.5/") == "10.0.0.5"
    with pytest.raises(EgressViolation):
        guard.check_url("http://10.0.0.6/")


def test_an_empty_allowlist_is_refused() -> None:
    with pytest.raises(ValueError, match="deny-by-default"):
        EgressGuard([])


async def test_the_guarded_transport_blocks_before_any_socket_is_opened(
    guard: EgressGuard,
) -> None:
    """The request must fail on policy, not on a connection error."""
    async with guarded_client(guard) as client:
        with pytest.raises(EgressViolation) as raised:
            await client.get("https://evil.example.com/collect")

    assert raised.value.host == "evil.example.com"


async def test_the_guarded_transport_lets_an_allowlisted_host_through() -> None:
    """Delegation is verified with a mock transport, so no packet leaves the box."""
    guard = EgressGuard(["localhost"])
    inner = httpx.MockTransport(lambda _request: httpx.Response(200, text="ok"))
    transport = GuardedTransport(guard, inner)

    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.get("http://localhost:8000/health")

    assert response.status_code == 200
