"""Argument provenance: where did the text in this tool call come from?

This is what makes Layer 5's `require_user_origin_for_privileged` a structural
defense rather than another pattern matcher. A privileged call is permitted only
when its arguments trace back to the user's own message. A call whose arguments
appear in a retrieved document is refused, however persuasive the document is,
and without the defense having to recognise the attack at all.

Precedence is deliberately conservative: if a value appears in untrusted content
*and* in the user's message, it is treated as untrusted. A defense that resolves
ambiguity in the attacker's favour is not a defense.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from crucible.defenses.config import CallOrigin

#: Shorter values match too easily by chance ("me", "1"), so they are treated as
#: model-originated rather than traced to a source.
MIN_TRACE_LENGTH = 4

_WHITESPACE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    return _WHITESPACE.sub(" ", text.strip().lower())


class ProvenanceIndex:
    """Traces a string back to the surface it came from."""

    def __init__(
        self,
        *,
        user_input: str = "",
        retrieved: Mapping[str, str] | None = None,
        tool_results: Sequence[str] = (),
        session_history: Sequence[str] = (),
    ) -> None:
        self._user = _normalise(user_input)
        self._retrieved = _normalise(
            "\n".join(f"{doc_id} {text}" for doc_id, text in (retrieved or {}).items())
        )
        self._tool_results = _normalise("\n".join(tool_results))
        self._history = _normalise("\n".join(session_history))

    def trace(self, value: str) -> CallOrigin:
        """Where a single argument value came from."""
        needle = _normalise(value)
        if len(needle) < MIN_TRACE_LENGTH:
            return CallOrigin.MODEL
        # Untrusted surfaces first: a value that appears in retrieved content is
        # untrusted even if it also appears in the user's message.
        if needle in self._retrieved:
            return CallOrigin.RETRIEVED
        if needle in self._tool_results:
            return CallOrigin.TOOL_RESULT
        if needle in self._user:
            return CallOrigin.USER
        if needle in self._history:
            return CallOrigin.RETRIEVED
        return CallOrigin.MODEL

    def trace_mention(self, token: str) -> CallOrigin:
        """Where a tool was named. A tool named only in a document is untrusted."""
        return self.trace(token)

    def trace_call(self, name: str, arguments: Mapping[str, object]) -> CallOrigin:
        """The origin of a whole tool call.

        Arguments decide. They carry the attacker-controlled part of a call — the
        recipient of an email, the id of a document to delete — so an argument
        drawn from a retrieved document makes the call untrusted even if the
        user also happened to name the tool. The tool name only breaks a tie
        when the arguments trace nowhere: blocking on the *mention* alone would
        refuse every legitimate call as soon as one poisoned document happened
        to name the tool.
        """
        argument_origins = {self.trace(str(value)) for value in arguments.values()}

        if CallOrigin.RETRIEVED in argument_origins:
            return CallOrigin.RETRIEVED
        if CallOrigin.TOOL_RESULT in argument_origins:
            return CallOrigin.TOOL_RESULT
        if CallOrigin.USER in argument_origins:
            return CallOrigin.USER

        mention = self.trace_mention(name)
        if mention in {CallOrigin.USER, CallOrigin.RETRIEVED, CallOrigin.TOOL_RESULT}:
            return mention
        return CallOrigin.MODEL
