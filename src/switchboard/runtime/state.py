"""Conversation state as a deterministic fold over the event log (ADR-002).

`apply_event` is the only code that changes a `ConversationState`. The live conversation applies
each event as the store returns it, and `replay` applies a stored log to an empty state, so the
two agree by construction; a test holds them to it.

Turn scoping. Every event a turn produces carries the turn's anchor event id as `causation_id`:
`turn.speculation_started` for a speculative turn, `turn.committed` for a fresh one, and for an
agent-initiated turn whatever event started it (named by its `brain.routed`). Utterance ids are
`<turn_id>/u<n>`. That lets the fold drop a cancelled speculation from the model's history while
keeping its real effects (a read tool that ran still produced evidence).

Evidence. Policy rules read `evidence.<key>`. Keys come only from an explicit `evidence` mapping
inside a successful tool result or a child call result, so a tool decides which of its fields are
policy facts and the model cannot write one.

Grounding inputs. The gate checks spoken claims against tool results, child results, retrieved
knowledge text and memory read at setup. All four are in the log (`knowledge.retrieved` and
`memory.loaded` carry their text when content capture is on), so `grounding_evidence()` on a
replayed state is exactly what the live gate saw.

Incremental by construction. The grounding evidence, the policy event index and the set of
tools that have succeeded are updated here, once per event, and never rebuilt from the whole
log. A check at turn 500 costs what it cost at turn 5; `replay` rebuilds all three.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from switchboard.brain import Message, canonical_json
from switchboard.core import (
    AnyEvent,
    BrainRouted,
    ChildResult,
    ConversationEnded,
    ConversationStarted,
    HandoffRequested,
    KnowledgeRetrieved,
    LegClosed,
    LegOpened,
    MemoryLoaded,
    PolicyVerdict,
    ProcedureAdvanced,
    SpeechPlayed,
    SpeechSentenceGated,
    ToolCompensated,
    ToolExecuted,
    ToolProposed,
    TurnCommitted,
    TurnSpeculationCancelled,
    TurnSpeculationStarted,
)
from switchboard.grounding import Evidence
from switchboard.policy import EventIndex, ProcedureGraph
from switchboard.policy.procedure import tool_succeeded

# Verdicts with these rule ids are runtime refusals, not policy denials the model must repair.
SPECULATION_RULE = "speculation:uncommitted_turn"

TurnStatus = Literal["speculating", "committed", "cancelled", "agent"]
Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class LegState:
    leg_id: str
    role: str
    party: str
    parent_leg_id: str | None
    goal: str | None
    open: bool = True
    close_reason: str | None = None


@dataclass
class TurnState:
    turn_id: str
    leg_id: str | None
    status: TurnStatus
    text: str
    anchor_event_id: str


@dataclass
class CallState:
    call_id: str
    leg_id: str | None
    tool: str
    args: dict[str, Any]
    proposed_seq: int
    turn_id: str | None
    verdicts: list[tuple[bool, tuple[str, ...]]] = field(default_factory=list)
    executed: bool = False
    result: dict[str, Any] | None = None
    idempotency_key: str | None = None
    replayed: bool = False

    @property
    def allowed(self) -> bool:
        """The latest verdict allowed the call."""
        return bool(self.verdicts) and self.verdicts[-1][0]


@dataclass
class HistoryEntry:
    role: Role
    content: str
    turn_id: str | None = None
    tool_call_id: str | None = None
    name: str | None = None
    utterance_id: str | None = None
    played: bool = False


@dataclass
class ConversationState:
    conversation_id: str = ""
    bundle_id: str = ""
    tenant_id: str = ""
    customer_id: str = ""
    last_seq: int = 0
    legs: dict[str, LegState] = field(default_factory=dict)
    turns: dict[str, TurnState] = field(default_factory=dict)
    anchors: dict[str, str] = field(default_factory=dict)
    history: dict[str, list[HistoryEntry]] = field(default_factory=dict)
    calls: dict[str, CallState] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    completed_steps: list[str] = field(default_factory=list)
    child_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    knowledge_ids: list[str] = field(default_factory=list)
    # Retrieved text by fact or document id, in first-retrieved order.
    knowledge: dict[str, str] = field(default_factory=dict)
    memory_texts: list[str] = field(default_factory=list)
    # A `memory.loaded` event carried its text, so recovery need not read memory again.
    memory_captured: bool = False
    # Kept current by `apply_event`; see the module docstring.
    grounding: Evidence = field(default_factory=Evidence)
    index: EventIndex = field(default_factory=EventIndex)
    succeeded_tools: set[str] = field(default_factory=set)
    handoffs: list[str] = field(default_factory=list)
    compensations: list[str] = field(default_factory=list)
    ended: bool = False
    outcome: str | None = None

    def inbound_leg(self) -> LegState | None:
        return next((leg for leg in self.legs.values() if leg.role == "inbound"), None)

    def executed(self, tool: str) -> list[CallState]:
        return [c for c in self.calls.values() if c.tool == tool and c.executed]

    def grounding_evidence(self) -> Evidence:
        """The grounding gate's evidence: tool and child results, knowledge text, memory."""
        return self.grounding

    def procedure_done(self, graph: ProcedureGraph) -> frozenset[str]:
        """Completed steps, as `ProcedureGraph.state_from_events` would compute them from the
        log: advanced steps plus steps whose completing tool succeeded."""
        done = set(self.completed_steps)
        done.update(graph.completes[t] for t in self.succeeded_tools if t in graph.completes)
        return frozenset(s for s in done if s in graph.steps)

    def pending_calls(self) -> list[CallState]:
        """Calls the policy allowed whose execution was never recorded, in proposal order."""
        waiting = [c for c in self.calls.values() if c.allowed and not c.executed]
        return sorted(waiting, key=lambda c: c.proposed_seq)

    def messages(self, leg_id: str) -> list[Message]:
        """Model history for one leg: played speech only, no cancelled turns, no dangling calls."""
        entries = [
            e
            for e in self.history.get(leg_id, [])
            if e.turn_id is None or self.turns.get(e.turn_id, _LIVE).status != "cancelled"
        ]
        answered = {e.tool_call_id for e in entries if e.role == "tool"}
        out: list[Message] = []
        for e in entries:
            if e.role == "assistant" and e.tool_call_id is not None:
                if e.tool_call_id not in answered:
                    continue
            elif e.role == "assistant" and not e.content.strip():
                continue
            out.append(Message(e.role, e.content, tool_call_id=e.tool_call_id, name=e.name))
        return out


_LIVE = TurnState("", None, "committed", "", "")


def turn_of_utterance(utterance_id: str) -> str:
    return utterance_id.rsplit("/", 1)[0]


def _merge_evidence(state: ConversationState, result: Mapping[str, Any]) -> None:
    patch = result.get("evidence")
    if isinstance(patch, Mapping):
        state.evidence.update({str(k): v for k, v in patch.items()})


def _history(state: ConversationState, leg_id: str | None) -> list[HistoryEntry]:
    return state.history.setdefault(leg_id or "", [])


def _utterance_entry(
    state: ConversationState, leg_id: str | None, utterance_id: str
) -> HistoryEntry:
    entries = _history(state, leg_id)
    for entry in reversed(entries):
        if entry.utterance_id == utterance_id:
            return entry
    entry = HistoryEntry(
        "assistant", "", turn_id=turn_of_utterance(utterance_id), utterance_id=utterance_id
    )
    entries.append(entry)
    return entry


def apply_event(state: ConversationState, event: AnyEvent) -> None:
    state.last_seq = event.seq
    state.index.add(event)
    turn_id = state.anchors.get(event.causation_id or "")
    match event:
        case ConversationStarted():
            state.conversation_id = event.conversation_id
            state.bundle_id = event.bundle_id
            state.tenant_id = event.tenant_id
            state.customer_id = event.customer_id
        case LegOpened():
            leg_id = event.leg_id or ""
            state.legs[leg_id] = LegState(
                leg_id, event.role, event.party, event.parent_leg_id, event.goal
            )
            state.history.setdefault(leg_id, [])
        case LegClosed():
            leg = state.legs.get(event.leg_id or "")
            if leg is not None:
                leg.open = False
                leg.close_reason = event.reason
        case TurnSpeculationStarted():
            state.turns[event.turn_id] = TurnState(
                event.turn_id, event.leg_id, "speculating", event.partial_text, event.event_id
            )
            state.anchors[event.event_id] = event.turn_id
            # Placeholder so speculative tool calls sit after the user's words in history.
            _history(state, event.leg_id).append(
                HistoryEntry("user", event.partial_text, turn_id=event.turn_id)
            )
        case TurnSpeculationCancelled():
            if event.turn_id in state.turns:
                state.turns[event.turn_id].status = "cancelled"
        case TurnCommitted():
            turn = state.turns.get(event.turn_id)
            if turn is not None and turn.status == "speculating":
                turn.status = "committed"
                turn.text = event.text
                for entry in _history(state, event.leg_id):
                    if entry.role == "user" and entry.turn_id == event.turn_id:
                        entry.content = event.text
            else:
                state.turns[event.turn_id] = TurnState(
                    event.turn_id, event.leg_id, "committed", event.text, event.event_id
                )
                _history(state, event.leg_id).append(
                    HistoryEntry("user", event.text, turn_id=event.turn_id)
                )
            state.anchors[event.event_id] = event.turn_id
        case BrainRouted():
            if event.turn_id not in state.turns:
                state.turns[event.turn_id] = TurnState(
                    event.turn_id, event.leg_id, "agent", "", event.causation_id or ""
                )
            if event.causation_id is not None:
                state.anchors.setdefault(event.causation_id, event.turn_id)
        case ToolProposed():
            state.calls[event.call_id] = CallState(
                event.call_id, event.leg_id, event.tool, dict(event.args), event.seq, turn_id
            )
            _history(state, event.leg_id).append(
                HistoryEntry(
                    "assistant",
                    canonical_json(event.args),
                    turn_id=turn_id,
                    tool_call_id=event.call_id,
                    name=event.tool,
                )
            )
        case PolicyVerdict():
            call = state.calls.get(event.call_id)
            if call is not None:
                call.verdicts.append((event.allowed, tuple(event.rule_ids)))
            if not event.allowed and SPECULATION_RULE not in event.rule_ids:
                rules = ", ".join(event.rule_ids)
                _history(state, event.leg_id).append(
                    HistoryEntry(
                        "tool",
                        f"Denied by policy ({rules}). {event.repair_hint or ''}".strip(),
                        turn_id=call.turn_id if call else turn_id,
                        tool_call_id=event.call_id,
                    )
                )
        case ToolExecuted():
            call = state.calls.get(event.call_id)
            if call is not None:
                call.executed = True
                call.result = dict(event.result)
                call.idempotency_key = event.idempotency_key
                call.replayed = event.replayed
            if tool_succeeded(event.result):
                result = {"tool": event.tool, **event.result}
                state.tool_results.append(result)
                state.grounding.add(result)
                state.succeeded_tools.add(event.tool)
                _merge_evidence(state, event.result)
            _history(state, event.leg_id).append(
                HistoryEntry(
                    "tool",
                    canonical_json(event.result),
                    turn_id=call.turn_id if call else turn_id,
                    tool_call_id=event.call_id,
                )
            )
        case ToolCompensated():
            state.compensations.append(event.call_id)
        case SpeechSentenceGated():
            if event.decision == "release":
                entry = _utterance_entry(state, event.leg_id, event.utterance_id)
                if not entry.played:
                    entry.content = f"{entry.content} {event.sentence}".strip()
        case SpeechPlayed():
            entry = _utterance_entry(state, event.leg_id, event.utterance_id)
            entry.content = event.text_played
            entry.played = True
        case ProcedureAdvanced():
            if event.step not in state.completed_steps:
                state.completed_steps.append(event.step)
        case ChildResult():
            state.child_results[event.child_leg_id] = dict(event.result)
            state.grounding.add(dict(event.result))
            _merge_evidence(state, event.result)
            child = state.legs.get(event.child_leg_id)
            party = child.party if child is not None else "party"
            summary = {k: v for k, v in event.result.items() if k != "evidence"}
            _history(state, event.leg_id).append(
                HistoryEntry(
                    "system", f"Result of the call to the {party}: {canonical_json(summary)}"
                )
            )
        case KnowledgeRetrieved():
            for fact_id in event.fact_ids:
                if fact_id not in state.knowledge_ids:
                    state.knowledge_ids.append(fact_id)
            for fact_id, text in zip(event.fact_ids, event.texts, strict=False):
                if state.knowledge.get(fact_id) != text:
                    state.grounding.add(text)
                state.knowledge[fact_id] = text
        case MemoryLoaded():
            if event.texts or event.count == 0:
                state.memory_texts = list(event.texts)
                state.memory_captured = True
                state.grounding.add(list(event.texts))
        case HandoffRequested():
            state.handoffs.append(event.reason)
        case ConversationEnded():
            state.ended = True
            state.outcome = event.outcome


def replay(events: Iterable[AnyEvent]) -> ConversationState:
    state = ConversationState()
    for event in events:
        apply_event(state, event)
    return state
