"""The Northwind IT service desk persona: the original target application.

Split out of `target.py` so a second application can be described the same way
and run through the same adapter. See `crucible.target.persona` for why the
transfer claim that rests on this is a *within-family* one.
"""

from __future__ import annotations

from crucible.target.persona import TargetPersona
from crucible.target.reference.corpus_gen import CORPUS_PATH, DOCSECRET_DOC_ID, load_corpus
from crucible.target.reference.prompts import ASSISTANT_NAME, build_system_prompt
from crucible.target.reference.sessions import pristine_sessions
from crucible.target.reference.tools import TOOL_SPECS, ToolRuntime

NORTHWIND = TargetPersona(
    key="northwind",
    assistant_name=ASSISTANT_NAME,
    domain="internal IT policy",
    tools=TOOL_SPECS,
    corpus_path=CORPUS_PATH,
    docsecret_doc_id=DOCSECRET_DOC_ID,
    build_system_prompt=build_system_prompt,
    load_corpus=load_corpus,
    tool_runtime=ToolRuntime,
    sessions=pristine_sessions,
)
