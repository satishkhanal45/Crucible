"""Redaction. `--include-payloads` is off by default.

docs/spec.md section 17: raw payloads only behind the flag, and canary values
and the reference target's system prompt never published in reusable form. A
redacted payload keeps what a reader needs — the mechanism — and drops what they
do not: the working text.

Traces are the sharp edge here. From Phase 2 every attempt's trace carries that
attempt's canary values so the oracle can be re-run, so `redact_trace` strips
`canaries` and `attempt.payload` before a trace can reach a report.
"""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from typing import Any

#: How much of the payload hash a redacted line shows.
HASH_CHARS = 12


def payload_hash(payload: str) -> str:
    return sha256(payload.encode()).hexdigest()[:HASH_CHARS]


def describe_mechanism(objective: str | None, vector: str, technique: str | None) -> str:
    """The mechanism, which is the part worth publishing."""
    return (
        f"{objective or 'unclassified'} via {vector}"
        f" using {technique or 'an unclassified technique'}"
    )


def redact_payload(
    payload: str, *, objective: str | None, vector: str, technique: str | None
) -> str:
    """Replace the payload with its mechanism and a hash."""
    return (
        f"[payload redacted: {describe_mechanism(objective, vector, technique)}"
        f", sha256:{payload_hash(payload)}]"
    )


def redact_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    """A trace safe to publish: no canary values, no payload text."""
    redacted: dict[str, Any] = {}
    for key, value in trace.items():
        if key == "canaries":
            redacted[key] = dict.fromkeys(value, "[redacted]")
            continue
        if key == "attempt" and isinstance(value, Mapping):
            attempt = dict(value)
            if "payload" in attempt:
                attempt["payload"] = redact_payload(
                    str(attempt["payload"]),
                    objective=attempt.get("objective"),
                    vector=str(attempt.get("vector", "unknown")),
                    technique=attempt.get("technique"),
                )
            redacted[key] = attempt
            continue
        redacted[key] = value
    return redacted


def present_payload(
    payload: str,
    *,
    objective: str | None,
    vector: str,
    technique: str | None,
    include_payloads: bool,
) -> str:
    """The payload as a report should show it."""
    if include_payloads:
        return payload.strip()
    return redact_payload(payload, objective=objective, vector=vector, technique=technique)
