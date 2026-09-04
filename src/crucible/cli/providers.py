"""Choosing the model clients a run uses.

**Live is the default.** Scripted clients are reachable only through
`--provider scripted`, and a run that uses them is recorded as `stubbed=true`,
bannered in every report, and refused by `crucible findings regenerate`. A
stubbed run produces numbers of exactly the same shape as a real one, so the
only safe arrangement is that reaching for one has to be deliberate and leaves a
mark.

`--provider` chooses between live models and test doubles. It does **not**
choose which provider a run calls: that is per agent, read from
`TARGET_PROVIDER`, `ATTACKER_PROVIDER`, `DEFENDER_PROVIDER` and
`CLASSIFIER_PROVIDER`. Two providers exist so that the four agents can be spread
across two rate-limit pools, and a single flag could not express that.

Model ids and base URLs come from settings — never from a literal here. A
literal in this module would be the same defect the model-id hotfix removed:
editing `.env` would stop changing what runs.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from enum import StrEnum

import httpx

from crucible.archive.classifier import ChatClassifierClient, ScriptedClassifierClient
from crucible.attacker.llm import ChatAttackerLLM, ScriptedAttackerLLM
from crucible.config import LLMProvider, Settings
from crucible.defender.llm import ChatDefenderLLM, ScriptedDefenderLLM
from crucible.execution.egress import EgressGuard, guarded_client
from crucible.logging import get_logger
from crucible.loop.runner import LoopFactories
from crucible.target.reference.llm import ChatCompletionsLLM, ScriptedTargetLLM

logger = get_logger(__name__)


class Provider(StrEnum):
    """Whether a run calls real models or test doubles.

    `groq` is the live value and the default. It is named for a provider for
    historical reasons — it predates per-agent provider selection — and now
    means "live, with each agent on the provider its settings name", which for
    a default `.env` is Groq throughout.
    """

    GROQ = "groq"
    #: Deterministic test doubles. Never a default, always recorded as stubbed.
    SCRIPTED = "scripted"


def build_factories(settings: Settings, provider: Provider, stack: AsyncExitStack) -> LoopFactories:
    """The four model clients for one run, each on its configured provider.

    `stack` owns the HTTP client's lifetime, so a live run closes its connection
    pool when the run ends rather than when the process does. One client serves
    every provider: the egress guard permits each provider's host, and the base
    URL travels with the call.
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
    logger.info(
        "run.live_clients",
        extra={
            "agents": {
                role: f"{agent_provider.value}/{model}"
                for role, agent_provider, model in settings.agents
            }
        },
    )
    return LoopFactories(
        target_llm=lambda: ChatCompletionsLLM(
            settings.api_key_for(settings.TARGET_PROVIDER),
            client,
            model=settings.TARGET_MODEL,
            provider=settings.TARGET_PROVIDER,
        ),
        attacker_llm=lambda: ChatAttackerLLM(
            settings.api_key_for(settings.ATTACKER_PROVIDER),
            client,
            model=settings.ATTACKER_MODEL,
            provider=settings.ATTACKER_PROVIDER,
        ),
        defender_llm=lambda: ChatDefenderLLM(
            settings.api_key_for(settings.DEFENDER_PROVIDER),
            client,
            model=settings.DEFENDER_MODEL,
            provider=settings.DEFENDER_PROVIDER,
        ),
        classifier_client=lambda: ChatClassifierClient(
            settings.api_key_for(settings.CLASSIFIER_PROVIDER),
            client,
            model=settings.CLASSIFIER_MODEL,
            provider=settings.CLASSIFIER_PROVIDER,
        ),
    )


def attacker_on(
    settings: Settings,
    model: str,
    stack: AsyncExitStack,
    *,
    provider: LLMProvider | None = None,
) -> ChatAttackerLLM:
    """One attacker client on a named model, for the overlap experiment.

    The model is passed explicitly because that experiment's whole point is to
    run two families against an identical starting state. The provider defaults
    to `ATTACKER_PROVIDER`, so which host those families are fetched from is the
    operator's configuration choice, not this module's.
    """
    chosen = provider or settings.ATTACKER_PROVIDER
    return ChatAttackerLLM(
        settings.api_key_for(chosen), _client(settings, stack), model=model, provider=chosen
    )


def _client(settings: Settings, stack: AsyncExitStack) -> httpx.AsyncClient:
    """An HTTP client behind the egress guard, closed when the stack unwinds."""
    guard = EgressGuard.from_settings(settings)
    client: httpx.AsyncClient = guarded_client(guard)
    stack.push_async_callback(client.aclose)
    return client
