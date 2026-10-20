"""Typed conversation events (ADR-002).

Every event is a frozen pydantic model with a `Literal` `type` discriminator. A `NewEvent` is
the same model before the store has assigned `seq` and `event_id`: those fields default to
`0` and `""` respectively, and the store returns copies with both filled in. Keeping one set of
classes avoids a parallel hierarchy that would have to stay in sync with this one.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class Event(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: str
    conversation_id: str
    leg_id: str | None = None
    seq: int = 0
    at: float
    bundle_id: str
    causation_id: str | None = None
    event_id: str = ""


class ConversationStarted(Event):
    type: Literal["conversation.started"] = "conversation.started"
    tenant_id: str
    customer_id: str


class LegOpened(Event):
    type: Literal["leg.opened"] = "leg.opened"
    role: Literal["inbound", "outbound_child"]
    party: str
    parent_leg_id: str | None = None
    goal: str | None = None


class LegClosed(Event):
    type: Literal["leg.closed"] = "leg.closed"
    reason: str


class TurnSpeculationStarted(Event):
    type: Literal["turn.speculation_started"] = "turn.speculation_started"
    turn_id: str
    partial_text: str


class TurnSpeculationCancelled(Event):
    type: Literal["turn.speculation_cancelled"] = "turn.speculation_cancelled"
    turn_id: str


class TurnCommitted(Event):
    type: Literal["turn.committed"] = "turn.committed"
    turn_id: str
    text: str
    endpointing_ms: float


class BrainRouted(Event):
    type: Literal["brain.routed"] = "brain.routed"
    turn_id: str
    model_tier: Literal["talker", "thinker"]
    reason: str


class ToolProposed(Event):
    type: Literal["tool.proposed"] = "tool.proposed"
    call_id: str
    tool: str
    args: dict[str, Any]


class PolicyVerdict(Event):
    type: Literal["policy.verdict"] = "policy.verdict"
    call_id: str
    allowed: bool
    rule_ids: list[str]
    repair_hint: str | None = None


class ToolExecuted(Event):
    type: Literal["tool.executed"] = "tool.executed"
    call_id: str
    tool: str
    result: dict[str, Any]
    idempotency_key: str | None = None
    replayed: bool = False


class ToolCompensated(Event):
    type: Literal["tool.compensated"] = "tool.compensated"
    call_id: str
    compensation_tool: str
    result: dict[str, Any]


class SpeechSentenceGated(Event):
    type: Literal["speech.sentence_gated"] = "speech.sentence_gated"
    utterance_id: str
    sentence: str
    decision: Literal["release", "block", "escalate"]
    tier: int
    # Plain dicts because `core` cannot import grounding's Claim type.
    claims: list[dict[str, Any]]


class SpeechPlayed(Event):
    type: Literal["speech.played"] = "speech.played"
    utterance_id: str
    text_played: str
    interrupted: bool


class ProcedureAdvanced(Event):
    type: Literal["procedure.advanced"] = "procedure.advanced"
    step: str
    state: dict[str, Any]


class ChildResult(Event):
    type: Literal["child.result"] = "child.result"
    child_leg_id: str
    result: dict[str, Any]


class KnowledgeRetrieved(Event):
    type: Literal["knowledge.retrieved"] = "knowledge.retrieved"
    fact_ids: list[str]
    as_of: float
    # The retrieved text, aligned with `fact_ids`, so replay rebuilds the grounding evidence.
    # Empty when the runtime runs with content capture off.
    texts: list[str] = Field(default_factory=list)


class MemoryLoaded(Event):
    """Memory read once at call setup (ADR-011). `texts` is empty with content capture off."""

    type: Literal["memory.loaded"] = "memory.loaded"
    count: int
    texts: list[str] = Field(default_factory=list)


class HandoffRequested(Event):
    type: Literal["handoff.requested"] = "handoff.requested"
    reason: str


class ConversationEnded(Event):
    type: Literal["conversation.ended"] = "conversation.ended"
    outcome: str


AnyEvent = Annotated[
    ConversationStarted
    | LegOpened
    | LegClosed
    | TurnSpeculationStarted
    | TurnSpeculationCancelled
    | TurnCommitted
    | BrainRouted
    | ToolProposed
    | PolicyVerdict
    | ToolExecuted
    | ToolCompensated
    | SpeechSentenceGated
    | SpeechPlayed
    | ProcedureAdvanced
    | ChildResult
    | KnowledgeRetrieved
    | MemoryLoaded
    | HandoffRequested
    | ConversationEnded,
    Field(discriminator="type"),
]

# An event whose seq and event_id are still the unassigned defaults (0 and "").
NewEvent = AnyEvent

_ADAPTER: TypeAdapter[AnyEvent] = TypeAdapter(AnyEvent)

EVENT_TYPES: tuple[str, ...] = tuple(
    get_args(cls.model_fields["type"].annotation)[0] for cls in get_args(get_args(AnyEvent)[0])
)


def parse_event(data: dict[str, Any]) -> AnyEvent:
    """Validate a plain dict (for example decoded JSON) into the matching event class."""
    return _ADAPTER.validate_python(data)


def parse_event_json(raw: str | bytes) -> AnyEvent:
    return _ADAPTER.validate_json(raw)
