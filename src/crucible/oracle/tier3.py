"""Tier 3: the LLM judge. NOT IMPLEMENTED IN THIS BUILD.

Cut B4 in docs/spec.md section 3 removes the judge; deferred item D4 restores
it. The stub returns `inconclusive` so that the seam stays exercised: the
`combine()` signature, the `tier` column, the `judge_score` and
`judge_rationale` columns, and the `inconclusive` outcome all exist and are
written today, which is what makes D4 a drop-in rather than a redesign.

TODO(D4) — the rules that restoring the judge must satisfy:

* **The judge never sees the attack payload.** Note the `judge()` signature
  below: it takes the response and the role spec, and there is nowhere to pass a
  payload. That is deliberate. A judge that reads the payload can be talked into
  a verdict by the payload itself, which is a real vulnerability in naive judge
  setups, and Phase 2 test 6 of project_context.md exists to prove it.
* The judge model family must differ from the attacker's and the target's.
* Structured JSON output validated by pydantic; one retry on a parse failure,
  then `inconclusive`.
* Calibration against 100 hand labels, with Cohen's kappa stored on the judge
  config version. Below 0.6 every report carries an `UNCALIBRATED` banner.
* Every judge call routes through `CostMeter`, like every other LLM call.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from crucible.oracle.results import Tier3Result
from crucible.schemas.outcome import Verdict


@runtime_checkable
class Tier3Judge(Protocol):
    """The seam. `payload` is absent from this signature on purpose."""

    async def judge(self, response_text: str, role_spec: str) -> Tier3Result: ...


class StubTier3Judge:
    """Returns `inconclusive`, always. Cut B4."""

    async def judge(self, response_text: str, role_spec: str) -> Tier3Result:
        del response_text, role_spec
        return Tier3Result(
            verdict=Verdict.INCONCLUSIVE,
            rationale="tier 3 is not implemented in this build (cut B4, deferred item D4)",
        )
