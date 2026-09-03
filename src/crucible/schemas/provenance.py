"""Which model actually produced a run's numbers.

A stubbed run and a live run produce results of exactly the same shape. Without
this record, a report from scripted clients is indistinguishable from a real
measurement, and every figure in `docs/findings.md` would be a claim nobody can
check. So every run records the provider and model each agent used, and whether
any of them was a stub.

`stubbed` is the load-bearing field: reports banner it, and
`crucible findings regenerate` refuses to emit numbers from a stubbed run.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

#: The provider name every scripted client reports.
STUB_PROVIDER = "stub"

#: The four agent roles whose model choice can change a number.
ROLES: tuple[str, ...] = ("target", "attacker", "defender", "classifier")


class AgentModel(BaseModel):
    """One agent's provider and model, as the client itself reported them."""

    model_config = ConfigDict(frozen=True)

    role: str
    provider: str
    model: str

    @property
    def stubbed(self) -> bool:
        return self.provider == STUB_PROVIDER

    def render(self) -> str:
        return f"{self.role}: {self.provider}/{self.model}"


class RunProvenance(BaseModel):
    """Every agent's model for one run, and whether the run is real."""

    model_config = ConfigDict(frozen=True)

    agents: tuple[AgentModel, ...] = ()

    @property
    def stubbed(self) -> bool:
        """True when ANY agent was scripted.

        Any is the right test, not all: a run whose target is real but whose
        attacker is scripted is not a measurement of anything either.
        """
        return any(agent.stubbed for agent in self.agents)

    @property
    def providers(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for agent in self.agents:
            seen.setdefault(agent.provider, None)
        return tuple(seen)

    def for_role(self, role: str) -> AgentModel | None:
        return next((agent for agent in self.agents if agent.role == role), None)

    def render_lines(self) -> list[str]:
        return [agent.render() for agent in self.agents]

    def banner(self) -> str | None:
        """The warning a report leads with, or None when the run is live."""
        if not self.stubbed:
            return None
        stubs = ", ".join(agent.role for agent in self.agents if agent.stubbed)
        return (
            "STUBBED RUN — NOT A MEASUREMENT. Scripted clients stood in for: "
            f"{stubs}. Every number below is the output of a deterministic test "
            "double, not of a language model. Do not cite, publish, or compare "
            "these figures. Re-run with `--provider groq`."
        )


#: What a run recorded before provenance existed, or one whose row predates it.
UNKNOWN = RunProvenance()
