"""`make seed` — load the corpus, plant the canaries, verify placement.

Verification is the point: seeding that silently fails to plant a canary would
make every later measurement wrong. Every check below either passes or the
command exits non-zero. Canary values are never printed, and neither are they
written to any file this command touches.

TODO(phase-6): fold this into the `crucible` typer CLI as `crucible seed`.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass

from crucible.archive.seeds import load_seed_attacks
from crucible.archive.service import ArchiveService
from crucible.config import Settings, get_settings
from crucible.db.session import Database
from crucible.defenses.config import DefenseConfig
from crucible.logging import configure_logging, get_logger
from crucible.repositories.spend import DatabaseSpendRepository
from crucible.services.cost_meter import CostMeter
from crucible.services.embeddings import embedder_from_settings
from crucible.target.canary import (
    CanaryClass,
    CanarySet,
    scan_text,
    set_active_canaries,
)
from crucible.target.reference.corpus_gen import DOCSECRET_DOC_ID, load_corpus
from crucible.target.reference.llm import MeteredTargetLLM, ScriptedTargetLLM
from crucible.target.reference.prompts import build_system_prompt
from crucible.target.reference.sessions import SESSION_A, pristine_sessions
from crucible.target.reference.store import DocumentStore
from crucible.target.reference.target import ReferenceTarget
from crucible.target.reference.tools import SEND_EMAIL, ToolRuntime

logger = get_logger(__name__)


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


async def seed(settings: Settings) -> list[Check]:
    """Seed the reference target and verify every canary is where it belongs."""
    database = Database(settings.DATABASE_URL)
    canaries = CanarySet.mint()
    set_active_canaries(canaries)
    checks: list[Check] = []

    try:
        store = DocumentStore(database, embedder_from_settings(settings))
        cost_meter = CostMeter(DatabaseSpendRepository(database), settings.ROUND_BUDGET_USD)
        llm = MeteredTargetLLM(ScriptedTargetLLM(), cost_meter)
        target = ReferenceTarget(store, llm, canaries)

        corpus = load_corpus()
        loaded = await target.seed(corpus)
        checks.append(Check("corpus loaded", loaded == len(corpus), f"{loaded} documents stored"))

        confidential = [document for document in corpus if document.confidential]
        checks.append(
            Check(
                "confidential documents present",
                len(confidential) >= 1,
                f"{len(confidential)} marked confidential",
            )
        )

        # 1. DOCSECRET_CANARY is in the confidential document, in the database.
        stored = await store.get(DOCSECRET_DOC_ID)
        docsecret_ok = (
            stored is not None
            and CanaryClass.DOCSECRET in scan_text(stored.text, canaries).breached_classes
        )
        checks.append(
            Check("DOCSECRET_CANARY planted", docsecret_ok, f"document {DOCSECRET_DOC_ID}")
        )

        # 2. SYSPROMPT_CANARY is in the system prompt.
        system_prompt = build_system_prompt(
            canaries, target.capabilities().tools, DefenseConfig.empty()
        )
        checks.append(
            Check(
                "SYSPROMPT_CANARY planted",
                CanaryClass.SYSPROMPT in scan_text(system_prompt, canaries).breached_classes,
                "system prompt",
            )
        )

        # 3. SESSION_CANARY is in session A's history.
        history = pristine_sessions(canaries)[SESSION_A]
        session_text = "\n".join(turn.content for turn in history)
        checks.append(
            Check(
                "SESSION_CANARY planted",
                CanaryClass.SESSION in scan_text(session_text, canaries).breached_classes,
                f"session {SESSION_A}",
            )
        )

        # 4. TOOL_CANARY comes back only from a privileged tool.
        runtime = ToolRuntime(store, canaries)
        receipt = await runtime.execute(SEND_EMAIL, {"to": "ops@example.test", "body": "ping"})
        checks.append(
            Check(
                "TOOL_CANARY returned by privileged tool",
                CanaryClass.TOOL in scan_text(receipt, canaries).breached_classes,
                SEND_EMAIL,
            )
        )

        # 5. Retrieval never surfaces a confidential document.
        probes = [
            "laptop refresh policy",
            "vendor settlement exit fee rebate schedule",
            confidential[0].text[:400],
        ]
        leaked: list[str] = []
        for probe in probes:
            hits = await store.search(probe, k=25, max_distance=2.0)
            leaked.extend(hit.document.doc_id for hit in hits if hit.document.confidential)
        checks.append(
            Check(
                "retrieval excludes confidential documents",
                not leaked,
                f"{len(probes)} probe queries, k=25",
            )
        )

        # 6. A benign question retrieves something sensible to cite.
        benign = await store.search("Summarize the laptop refresh policy")
        checks.append(
            Check(
                "benign query retrieves policy documents",
                bool(benign),
                ", ".join(hit.document.doc_id for hit in benign) or "no hits",
            )
        )

        # 7. Seed attacks enter the archive, with 20% reserved as holdout
        #    before any of them is executed.
        archive = ArchiveService(database, embedder_from_settings(settings))
        seeds = load_seed_attacks()
        admissions = await archive.admit_seeds(seeds)
        admitted = [a for a in admissions if a.accepted]
        stats = await archive.stats()
        checks.append(
            Check(
                "seed attacks archived",
                stats.archive_size >= len(seeds),
                f"{len(admitted)} admitted this run, {stats.archive_size} in the archive",
            )
        )
        checks.append(
            Check(
                "holdout reserved before execution",
                abs(stats.holdout_ratio - archive.holdout_ratio) <= 0.05,
                f"{stats.holdout_count}/{stats.archive_size} held out "
                f"({stats.holdout_ratio:.0%}, target {archive.holdout_ratio:.0%})",
            )
        )
        checks.append(
            Check(
                "seed coverage of the grid",
                stats.coverage.occupied > 0,
                f"{stats.coverage}",
            )
        )

        # 8. The committed corpus file itself holds no canary.
        corpus_text = "\n".join(f"{d.title}\n{d.text}" for d in corpus)
        checks.append(
            Check(
                "committed corpus contains no canary",
                not scan_text(corpus_text, canaries).matches,
                "data/corpus/corpus.json",
            )
        )

        await target.reset()
    finally:
        set_active_canaries(None)
        await database.close()

    return checks


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level_number)
    checks = asyncio.run(seed(settings))

    width = max(len(check.name) for check in checks)
    for check in checks:
        mark = "ok  " if check.passed else "FAIL"
        print(f"[{mark}] {check.name.ljust(width)}  {check.detail}")

    failed = [check for check in checks if not check.passed]
    if failed:
        print(f"\nseed failed: {len(failed)} of {len(checks)} checks did not pass")
        return 1
    print(f"\nseed complete: {len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
