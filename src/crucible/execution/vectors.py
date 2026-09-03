"""Delivery vectors: how a payload actually reaches the model.

Two are executable in this build (cut B3):

* `direct` — the payload is the user's message.
* `indirect_document` — the payload rides inside a document injected into the
  retrievable corpus, and the user asks an ordinary question. Most real-world
  prompt injection is indirect, and this vector is what makes Crucible a RAG
  security project rather than a chatbot jailbreak project, so the carrier
  document is built to be genuinely retrievable for the benign question rather
  than dropped into the corpus and hoped for.

TODO(D3): `multi_turn` needs session handling in the executor and
`indirect_tool_result` needs a tool-result injection hook on the adapter. Both
fail validation today rather than being silently downgraded.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from crucible.defenses.config import DefenseConfig
from crucible.schemas.attack import Attack
from crucible.schemas.taxonomy import DeliveryVector, ensure_executable
from crucible.target.adapter import Document, TargetAdapter, TargetResponse

#: Prefix for carrier documents, so an injected document is obvious in a trace.
CARRIER_PREFIX = "ATTACK"


class Delivery(BaseModel):
    """What delivering an attack produced."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    response: TargetResponse
    #: The message the user actually sent, which is benign for indirect vectors.
    user_input: str
    #: Set when the vector injected a document into the corpus.
    injected_doc_id: str | None = None
    carrier_retrieved: bool = False


def carrier_doc_id(attack: Attack) -> str:
    return attack.carrier_doc_id or f"{CARRIER_PREFIX}-{attack.attack_id.hex[:12]}"


def build_carrier_document(attack: Attack) -> Document:
    """Wrap a payload in a document that the benign question will retrieve.

    The lead-in repeats the question's own wording. Retrieval is a similarity
    search, so a carrier that shares no vocabulary with the question would never
    be returned and the attack would be scored as blocked when in truth it was
    never delivered.
    """
    question = attack.benign_user_input.strip().rstrip("?.").strip()
    title = attack.carrier_title or f"{question}: supplementary note"
    lead_in = (
        f"{question}. This addendum supplements the policy that answers "
        f"'{question}' and should be read alongside it."
    )
    return Document(
        doc_id=carrier_doc_id(attack),
        title=title,
        text=f"{lead_in}\n\n{attack.payload}",
        confidential=False,
        source="attack",
    )


async def deliver(
    target: TargetAdapter,
    attack: Attack,
    defense: DefenseConfig,
    session_id: str,
) -> Delivery:
    """Deliver `attack` to `target` by its declared vector."""
    ensure_executable(attack.vector)

    if attack.vector is DeliveryVector.DIRECT:
        response = await target.query(attack.payload, defense, session_id)
        return Delivery(response=response, user_input=attack.payload)

    carrier = build_carrier_document(attack)
    doc_id = await target.inject_document(carrier)
    response = await target.query(attack.benign_user_input, defense, session_id)
    return Delivery(
        response=response,
        user_input=attack.benign_user_input,
        injected_doc_id=doc_id,
        carrier_retrieved=doc_id in response.retrieved_doc_ids,
    )
