"""Choosing the model clients a run uses.

**Live is the default.** Scripted clients are reachable only through
`--provider scripted`, and a run that uses them is recorded as `stubbed=true`,
bannered in every report, and refused by `crucible findings regenerate`. A
stubbed run produces numbers of exactly the same shape as a real one, so the
only safe arrangement is that reaching for one has to be deliberate and leaves a
mark.

Model ids come from settings — `TARGET_MODEL`, `ATTACKER_MODEL`,
`DEFENDER_MODEL`, `CLASSIFIER_MODEL` — never from a literal here. A literal in
this module would be the same defect the model-id hotfix removed: editing `.env`
would stop changing what runs.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from enum import StrEnum

import httpx

from crucible.archive.classifier import GroqClassifierClient, ScriptedClassifierClient
from crucible.attacker.llm import GroqAttackerLLM, ScriptedAttackerLLM
from crucible.config import Settings
from crucible.defender.llm import GroqDefenderLLM, ScriptedDefenderLLM
from crucible.execution.egress import EgressGuard, guarded_client
from crucible.logging import get_logger
from crucible.loop.runner import LoopFactories
from crucible.target.reference.llm import GroqTargetLLM, ScriptedTargetLLM

logger = get_logger(__name__)


class Provider(StrEnum):
    """Where a run's model calls go."""

    GROQ = "groq"
    #: Deterministic test doubles. Never a default, always recorded as stubbed.
    SCRIPTED = "scripted"


def build_factories(settings: Settings, provider: Provider, stack: AsyncExitStack) -> LoopFactories:
    """The four model clients for one run.

    `stack` owns the HTTP client's lifetime, so a live run closes its connection
    pool when the run ends rather than when the process does.
    """
    if provider is Provider.SCRIPTED:
        logger.warning(
            "run.scripted_clients",
            extra={
                "detail": (
                    "this run uses scripted clients: it will be recorded as stubbed, "
                    "bannered in every report, and refused by findings regeneration"
                )
            },
        )
        return LoopFactories(
            target_llm=ScriptedTargetLLM,
            attacker_llm=ScriptedAttackerLLM,
            defender_llm=ScriptedDefenderLLM,
            classifier_client=ScriptedClassifierClient,
        )

    client = _client(settings, stack)
    key = settings.GROQ_API_KEY
    logger.info(
        "run.live_clients",
        extra={
            "provider": provider.value,
            "target_model": settings.TARGET_MODEL,
            "attacker_model": settings.ATTACKER_MODEL,
            "defender_model": settings.DEFENDER_MODEL,
            "classifier_model": settings.CLASSIFIER_MODEL,
        },
    )
    return LoopFactories(
        target_llm=lambda: GroqTargetLLM(key, client, model=settings.TARGET_MODEL),
        attacker_llm=lambda: GroqAttackerLLM(key, client, model=settings.ATTACKER_MODEL),
        defender_llm=lambda: GroqDefenderLLM(key, client, model=settings.DEFENDER_MODEL),
        classifier_client=lambda: GroqClassifierClient(
            key, client, model=settings.CLASSIFIER_MODEL
        ),
    )


def attacker_on(settings: Settings, model: str, stack: AsyncExitStack) -> GroqAttackerLLM:
    """One attacker client on a named model, for the overlap experiment.

    The model is passed explicitly because that experiment's whole point is to
    run two families against an identical starting state.
    """
    return GroqAttackerLLM(settings.GROQ_API_KEY, _client(settings, stack), model=model)


def _client(settings: Settings, stack: AsyncExitStack) -> httpx.AsyncClient:
    """An HTTP client behind the egress guard, closed when the stack unwinds."""
    guard = EgressGuard.from_settings(settings)
    client: httpx.AsyncClient = guarded_client(guard)
    stack.push_async_callback(client.aclose)
    return client
