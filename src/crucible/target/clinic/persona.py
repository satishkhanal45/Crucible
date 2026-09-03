"""The Meridian Clinic persona: the second application, for transfer."""

from __future__ import annotations

from crucible.target.clinic.corpus_gen import CORPUS_PATH, DOCSECRET_DOC_ID, load_corpus
from crucible.target.clinic.prompts import ASSISTANT_NAME, build_system_prompt
from crucible.target.clinic.sessions import pristine_sessions
from crucible.target.clinic.tools import TOOL_SPECS, ToolRuntime
from crucible.target.persona import TargetPersona

MERIDIAN = TargetPersona(
    key="meridian",
    assistant_name=ASSISTANT_NAME,
    domain="clinic scheduling and patient handbook",
    tools=TOOL_SPECS,
    corpus_path=CORPUS_PATH,
    docsecret_doc_id=DOCSECRET_DOC_ID,
    build_system_prompt=build_system_prompt,
    load_corpus=load_corpus,
    tool_runtime=ToolRuntime,
    sessions=pristine_sessions,
)
