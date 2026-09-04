"""A second provider: DeepSeek, alongside Groq.

The reason for a second provider is arithmetic, not preference. A free tier
gives one token-per-minute pool per model, and all four agents contend for it;
a second host is a second pool. It also makes `model_overlap` capable of
comparing two genuinely different model families rather than two models from
one host.

What that requires, and what these tests hold to:
  1. The second provider is configured like the first, per agent, and every
     credential is required at startup rather than discovered mid-run.
  2. One OpenAI-shaped client serves both, so the status-to-typed-error mapping
     that a hotfix installed exists exactly once. Only the base URL differs.
  3. Rate-limit windows are keyed by provider AND model. Two pools that shared a
     window would be one pool, which is the whole point lost.
  4. A mixed run records which agent ran where, or it is not reproducible.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

import httpx
import pytest

from crucible.archive.classifier import ChatClassifierClient
from crucible.attacker.llm import ChatAttackerLLM
from crucible.cli.providers import Provider, build_factories
from crucible.config import (
    DEFAULT_MODEL_PRICING,
    PROVIDER_BASE_URLS,
    ConfigurationError,
    LLMProvider,
    Settings,
    load_settings,
    price_key,
    provider_hosts,
    validate_model_pricing,
)
from crucible.defender.llm import ChatDefenderLLM
from crucible.execution.egress import DEFAULT_PROVIDER_HOSTS, EgressGuard
from crucible.loop.runner import provenance_for
from crucible.services.pacing import ProviderPacer
from crucible.services.retry import AuthenticationFailed, ModelNotFound, RateLimited
from crucible.target.reference.llm import ChatCompletionsLLM, LLMMessage
from crucible.target.reference.tools import TOOL_SPECS

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ENV_EXAMPLE = REPOSITORY_ROOT / ".env.example"

#: Stand-in ids. Real model ids live in config and .env.example, nowhere else.
MODEL = "test-model"

OK_BODY = {
    "choices": [{"message": {"content": '{"answer": "hi", "citations": []}'}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
}


@asynccontextmanager
async def _client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        yield client


def _messages() -> list[LLMMessage]:
    return [LLMMessage(role="user", content="hello")]


# --------------------------------------------------------------------------- #
# 1. Configuration
# --------------------------------------------------------------------------- #


def test_the_deepseek_key_is_required_like_every_other_setting(
    env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing credential fails at startup naming itself, not mid-run."""
    del env  # the fixture's effect is on the environment
    monkeypatch.delenv("DEEPSEEK_API_KEY")

    with pytest.raises(ConfigurationError, match="DEEPSEEK_API_KEY") as raised:
        load_settings(_env_file=None)

    assert "required environment variable is not set" in str(raised.value)


def test_each_provider_has_its_own_credential(settings: Settings) -> None:
    """Two hosts, two keys. Sending one provider's key to the other is a 401."""
    assert settings.api_key_for(LLMProvider.GROQ) == settings.GROQ_API_KEY
    assert settings.api_key_for(LLMProvider.DEEPSEEK) == settings.DEEPSEEK_API_KEY
    assert settings.GROQ_API_KEY != settings.DEEPSEEK_API_KEY


@pytest.mark.parametrize(
    "name",
    ["TARGET_PROVIDER", "ATTACKER_PROVIDER", "DEFENDER_PROVIDER", "CLASSIFIER_PROVIDER"],
)
def test_provider_is_configured_per_agent(settings: Settings, name: str) -> None:
    """Per agent, not globally: the four may be spread across the two pools."""
    assert isinstance(getattr(settings, name), LLMProvider)


def test_the_per_agent_settings_are_documented_in_env_example() -> None:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")

    for name in ("TARGET_PROVIDER", "ATTACKER_PROVIDER", "DEFENDER_PROVIDER"):
        assert f"\n{name}=" in text, f"{name} is not in .env.example"
    assert "DEEPSEEK_API_KEY=" in text
    assert "2026" in text, "the second provider carries the date it was checked"


def test_there_is_no_global_provider_setting_left_to_drift(settings: Settings) -> None:
    """`LLM_PROVIDER` was replaced, not shadowed: two sources would disagree."""
    assert not hasattr(settings, "LLM_PROVIDER")


def test_the_judge_provider_is_not_an_agent_provider(settings: Settings) -> None:
    """The Tier 3 judge is a separate concern on a separate host (cut B4)."""
    assert settings.JUDGE_PROVIDER not in {p.value for p in LLMProvider}
    assert all(provider is not settings.JUDGE_PROVIDER for _, provider, _ in settings.agents)


def test_a_provider_outside_the_enum_is_refused(env: dict[str, str]) -> None:
    del env  # the fixture's effect is on the environment
    with pytest.raises(ConfigurationError, match="TARGET_PROVIDER"):
        load_settings(_env_file=None, TARGET_PROVIDER="openai")


def test_every_agents_provider_and_model_are_paired(settings: Settings) -> None:
    """One place resolves the pairing, so nothing else can mismatch them."""
    assert settings.agents == (
        ("target", settings.TARGET_PROVIDER, settings.TARGET_MODEL),
        ("attacker", settings.ATTACKER_PROVIDER, settings.ATTACKER_MODEL),
        ("defender", settings.DEFENDER_PROVIDER, settings.DEFENDER_MODEL),
        ("classifier", settings.CLASSIFIER_PROVIDER, settings.CLASSIFIER_MODEL),
    )
    assert settings.provider_for("attacker") is settings.ATTACKER_PROVIDER
    assert settings.model_for("attacker") == settings.ATTACKER_MODEL


# --------------------------------------------------------------------------- #
# 2. Egress: a provider host is not a target host
# --------------------------------------------------------------------------- #


def test_the_deepseek_host_is_a_provider_host(settings: Settings) -> None:
    guard = EgressGuard.from_settings(settings)

    assert "api.deepseek.com" in DEFAULT_PROVIDER_HOSTS
    assert "api.deepseek.com" in guard.providers
    assert guard.is_allowed("api.deepseek.com")


def test_the_deepseek_host_is_not_a_target(settings: Settings) -> None:
    """TARGET_ALLOWLIST is for attack targets. A provider is never one."""
    guard = EgressGuard.from_settings(settings)

    assert "api.deepseek.com" not in guard.targets
    assert "api.deepseek.com" not in settings.TARGET_ALLOWLIST


def test_the_provider_hosts_come_from_the_configured_base_urls() -> None:
    """Adding a provider in config is what adds its host, not a second list."""
    for url in PROVIDER_BASE_URLS.values():
        assert httpx.URL(url).host in provider_hosts()


# --------------------------------------------------------------------------- #
# 3. One client, two base URLs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("provider", list(LLMProvider))
async def test_the_client_calls_the_base_url_of_its_provider(provider: LLMProvider) -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, json=OK_BODY)

    async with _client(handler) as client:
        llm = ChatCompletionsLLM("key-1", client, model=MODEL, provider=provider)
        await llm.complete(_messages(), TOOL_SPECS)

    assert seen["url"] == f"{PROVIDER_BASE_URLS[provider.value]}/chat/completions"
    assert seen["auth"] == "Bearer key-1"
    assert llm.provider == provider.value


def test_the_client_cannot_be_built_without_naming_its_provider() -> None:
    """Otherwise one provider's key could be sent to another's endpoint."""
    with pytest.raises(TypeError):
        ChatCompletionsLLM("key", httpx.AsyncClient(), model=MODEL)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("status", "expected"),
    [(429, RateLimited), (401, AuthenticationFailed), (404, ModelNotFound)],
)
async def test_deepseek_errors_map_to_the_same_typed_errors_as_groq(
    status: int, expected: type[Exception]
) -> None:
    """One mapping, in one place. A second copy is a second thing to get wrong.

    This is the hotfix property: an unmapped status escapes as
    `httpx.HTTPStatusError`, which no retry policy lists, so a 429 ends a run
    instead of backing off.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            status, json={"error": {"message": "nope"}}, headers={"Retry-After": "7"}
        )

    for provider in LLMProvider:
        async with _client(handler) as client:
            llm = ChatCompletionsLLM("key", client, model=MODEL, provider=provider)
            with pytest.raises(expected) as raised:
                await llm.complete(_messages(), TOOL_SPECS)
        assert not isinstance(raised.value, httpx.HTTPStatusError)
        if expected is RateLimited:
            assert raised.value.retry_after == 7.0  # type: ignore[attr-defined]


async def test_a_deepseek_404_names_the_model_that_was_not_found() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(404, json={"error": {"message": "no such model"}})

    async with _client(handler) as client:
        with pytest.raises(ModelNotFound) as raised:
            await ChatCompletionsLLM(
                "key", client, model="deepseek-nonexistent", provider=LLMProvider.DEEPSEEK
            ).complete(_messages(), TOOL_SPECS)

    assert "deepseek-nonexistent" in str(raised.value)


# --------------------------------------------------------------------------- #
# 4. Pricing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model", ["deepseek-chat", "deepseek-reasoner"])
def test_the_deepseek_models_are_priced(model: str) -> None:
    """An unpriced model records NULL cost, which makes ROUND_BUDGET_USD inert."""
    key = price_key(LLMProvider.DEEPSEEK.value, model)

    assert key in DEFAULT_MODEL_PRICING
    prompt, completion = DEFAULT_MODEL_PRICING[key]
    assert float(prompt) > 0 and float(completion) > 0


def test_a_run_on_deepseek_is_fully_priced(settings: Settings) -> None:
    """Every configured model is priced, so the budget is enforceable."""
    mixed = settings.model_copy(
        update={
            "ATTACKER_PROVIDER": LLMProvider.DEEPSEEK,
            "ATTACKER_MODEL": "deepseek-chat",
        }
    )

    assert price_key("deepseek", "deepseek-chat") in mixed.configured_models
    assert mixed.unpriced_models() == ()


def test_an_unpriced_deepseek_model_only_warns(settings: Settings) -> None:
    """The existing rule: an unpriced model still runs, loudly."""
    mixed = settings.model_copy(
        update={
            "ATTACKER_PROVIDER": LLMProvider.DEEPSEEK,
            "ATTACKER_MODEL": "deepseek-not-yet-released",
        }
    )

    unpriced = validate_model_pricing(mixed)

    assert unpriced == ("deepseek:deepseek-not-yet-released",)


# --------------------------------------------------------------------------- #
# 5. Pacing: two providers are two pools
# --------------------------------------------------------------------------- #


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


async def test_two_providers_do_not_share_a_window() -> None:
    """The reason for the second provider, asserted."""
    clock = FakeClock()
    pacer = ProviderPacer(
        min_interval_seconds=0.0,
        max_concurrency=1,
        tokens_per_minute=1000,
        requests_per_minute=100,
        sleep=clock.sleep,
        clock=clock,
    )

    async with pacer.slot(model="groq:shared-name", estimated_tokens=900):
        pass
    async with pacer.slot(model="deepseek:shared-name", estimated_tokens=900):
        pass

    assert clock.slept == [], "the same model name on two providers is two pools"
    assert pacer.tokens_used("groq:shared-name") == 900
    assert pacer.tokens_used("deepseek:shared-name") == 900


async def test_one_provider_filling_its_window_still_delays_itself() -> None:
    """Separate pools must not become no pool at all."""
    clock = FakeClock()
    pacer = ProviderPacer(
        min_interval_seconds=0.0,
        max_concurrency=1,
        tokens_per_minute=1000,
        requests_per_minute=100,
        sleep=clock.sleep,
        clock=clock,
    )

    for _ in range(2):
        async with pacer.slot(model="deepseek:deepseek-chat", estimated_tokens=900):
            pass

    assert clock.slept, "the second call on the same provider and model waits"


def test_limits_are_configurable_per_provider() -> None:
    """The two hosts do not publish the same limits."""
    pacer = ProviderPacer(
        tokens_per_minute=6500,
        requests_per_minute=25,
        limits={"deepseek": (100_000, 60)},
    )

    assert pacer.limits_for("groq:openai-model") == (6500, 25)
    assert pacer.limits_for("deepseek:deepseek-chat") == (100_000, 60)
    assert pacer.window_for("deepseek:deepseek-chat").tokens_per_minute == 100_000
    assert pacer.window_for("groq:openai-model").tokens_per_minute == 6500


def test_per_provider_limits_are_read_from_settings(env: dict[str, str]) -> None:
    del env
    settings = load_settings(_env_file=None, PROVIDER_RATE_LIMITS="deepseek=100000/60")

    assert settings.rate_limits_for(LLMProvider.DEEPSEEK) == (100_000, 60)
    assert settings.rate_limits_for(LLMProvider.GROQ) == (
        settings.PROVIDER_TOKENS_PER_MINUTE,
        settings.PROVIDER_REQUESTS_PER_MINUTE,
    )


def test_a_malformed_rate_limit_entry_is_refused(env: dict[str, str]) -> None:
    del env
    with pytest.raises(ConfigurationError, match="PROVIDER_RATE_LIMITS"):
        load_settings(_env_file=None, PROVIDER_RATE_LIMITS="deepseek=lots")


def test_a_provider_with_no_override_is_not_windowed_to_zero() -> None:
    """A zero window would let every call through with a warning, not pace it."""
    pacer = ProviderPacer(tokens_per_minute=0, requests_per_minute=0, limits={"groq": (6500, 25)})

    assert pacer.limits_key("groq:m")
    assert not pacer.limits_key("deepseek:m")


# --------------------------------------------------------------------------- #
# 6. A mixed run records which agent ran where
# --------------------------------------------------------------------------- #


def _mixed(settings: Settings) -> Settings:
    """The attacker on the second provider; everything else on the first."""
    return settings.model_copy(
        update={
            "ATTACKER_PROVIDER": LLMProvider.DEEPSEEK,
            "ATTACKER_MODEL": "deepseek-chat",
        }
    )


async def test_a_mixed_run_records_provenance_for_each_agent(settings: Settings) -> None:
    """Without this the run is not reproducible: nothing else says what ran."""
    mixed = _mixed(settings)

    async with AsyncExitStack() as stack:
        factories = build_factories(mixed, Provider.GROQ, stack)
        provenance = provenance_for(
            target=factories.target_llm(),
            attacker=factories.attacker_llm(),
            defender=factories.defender_llm(),
            classifier=factories.classifier_client(),
        )

    assert not provenance.stubbed
    assert provenance.banner() is None
    lines = provenance.render_lines()
    assert f"target: groq/{mixed.TARGET_MODEL}" in lines
    assert "attacker: deepseek/deepseek-chat" in lines
    assert f"defender: groq/{mixed.DEFENDER_MODEL}" in lines


async def test_each_agents_client_carries_its_own_provider(settings: Settings) -> None:
    mixed = _mixed(settings)

    async with AsyncExitStack() as stack:
        factories = build_factories(mixed, Provider.GROQ, stack)
        target = factories.target_llm()
        attacker = factories.attacker_llm()
        classifier = factories.classifier_client()

    assert isinstance(attacker, ChatAttackerLLM)
    assert isinstance(classifier, ChatClassifierClient)
    assert target.provider == "groq"
    assert attacker.provider == "deepseek"
    assert classifier.provider == "groq"


async def test_the_live_factories_build_one_client_per_role_from_settings(
    settings: Settings,
) -> None:
    """Models and providers both come from settings; neither is a literal."""
    async with AsyncExitStack() as stack:
        factories = build_factories(settings, Provider.GROQ, stack)
        built = {
            "target": factories.target_llm(),
            "attacker": factories.attacker_llm(),
            "defender": factories.defender_llm(),
            "classifier": factories.classifier_client(),
        }

    for role, provider, model in settings.agents:
        assert built[role].provider == provider.value
        assert built[role].model == model
    assert isinstance(built["defender"], ChatDefenderLLM)


async def test_a_scripted_run_is_still_stubbed_whatever_the_providers_say(
    settings: Settings,
) -> None:
    """Two providers do not make a scripted run any more of a measurement."""
    async with AsyncExitStack() as stack:
        factories = build_factories(_mixed(settings), Provider.SCRIPTED, stack)
        provenance = provenance_for(
            target=factories.target_llm(),
            attacker=factories.attacker_llm(),
            defender=factories.defender_llm(),
            classifier=factories.classifier_client(),
        )

    assert provenance.stubbed
    assert provenance.banner() is not None
