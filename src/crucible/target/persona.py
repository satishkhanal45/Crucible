"""What makes one RAG application different from another.

The Phase 8 transfer experiment runs the final archive against a second
`TargetAdapter` with **no change to the loop**. The adapter Protocol exists for
exactly that, and this is what varies underneath it: the assistant's role and
system prompt, the corpus domain it retrieves over, and the tools it exposes.

A `TargetPersona` bundles those three so that `ReferenceTarget` can serve either
application. The retrieval machinery, canary planting and defense-layer wiring
stay shared, which is precisely why the transfer result is a **within-family**
claim: it shows an archive carrying across a different prompt, corpus and tool
surface, not across a different implementation. Section "Transfer" in
`docs/findings.md` has to say so.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from crucible.target.adapter import Document, ToolSpec
from crucible.target.canary import CanarySet


class ToolRuntimeProtocol(Protocol):
    """Executes a tool call. Authorization is decided before this is reached."""

    sent_records: list[dict[str, str]]
    deleted_doc_ids: list[str]

    async def execute(self, name: str, arguments: dict[str, Any]) -> str: ...


SessionFactory = Callable[[CanarySet], Mapping[str, Any]]
RuntimeFactory = Callable[[Any, CanarySet], ToolRuntimeProtocol]
CorpusLoader = Callable[[], tuple[Document, ...]]
PromptBuilder = Callable[[CanarySet, Sequence[ToolSpec], Any], str]


@dataclass(frozen=True)
class TargetPersona:
    """One application: its role, its corpus, its tools.

    `key` names it in reports and in the `experiments/` configs; it is also the
    corpus namespace prefix, so two personas never share stored documents.
    """

    key: str
    assistant_name: str
    #: Human-readable domain, for the transfer table in findings.
    domain: str
    tools: tuple[ToolSpec, ...]
    corpus_path: Path
    #: The document that carries `DOCSECRET_CANARY` at run time.
    docsecret_doc_id: str
    build_system_prompt: PromptBuilder
    load_corpus: CorpusLoader
    tool_runtime: RuntimeFactory
    sessions: SessionFactory

    @property
    def privileged_tools(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.tools if spec.privileged)
