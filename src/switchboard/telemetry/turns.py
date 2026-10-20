"""Per-turn latency decomposition (ADR-015), derived purely from a conversation's events.

A turn starts at `turn.committed` and owns the events that follow it on the same leg until the
next commit on that leg. Events with no leg attach to the most recent turn. Policy verdicts and
tool results attach by `call_id` to the turn that proposed the call, even if they land after a
later commit. Time zero for a turn is the end of the caller's speech: `committed_at -
endpointing_ms`.

Measured per turn (all milliseconds):

| field                  | from                          | to                                 |
|------------------------|-------------------------------|------------------------------------|
| endpointing_ms         | `turn.committed` payload      |                                    |
| router_ms              | commit                        | `brain.routed` for this turn_id    |
| brain_first_output_ms  | routed (or commit if none)    | first `tool.proposed` or gated sentence |
| policy_ms (per tool)   | `tool.proposed`               | `policy.verdict`                   |
| duration_ms (per tool) | `tool.proposed`               | `tool.executed`                    |
| first_audio_ms         | commit                        | first `speech.played`              |
| release_to_audio_ms    | first released sentence       | first `speech.played` (TTS first audio) |
| voice_to_voice_ms      | end of caller speech          | first `speech.played`              |

Span attribute mapping. OpenTelemetry GenAI semantic convention names are used where one exists;
the rest live under `switchboard.*`:

| attribute                               | meaning                                         |
|-----------------------------------------|-------------------------------------------------|
| gen_ai.operation.name                   | `invoke_agent` (turn), `chat` (brain), `execute_tool` |
| gen_ai.agent.id                         | agent id passed to `from_events`                |
| gen_ai.agent.version                    | release bundle id (ADR-012)                     |
| gen_ai.conversation.id                  | conversation id                                 |
| gen_ai.tool.name, gen_ai.tool.call.id   | tool name and call id                           |
| gen_ai.usage.*                          | not emitted: events carry no token counts yet   |
| error.type                              | tool error string from `result["error"]`        |
| switchboard.turn.id, switchboard.leg.id, switchboard.party | turn identity               |
| switchboard.latency.*_ms                | the measured fields in the table above          |
| switchboard.router.tier, .reason        | talker or thinker, and why                      |
| switchboard.policy.allowed, .rule_ids   | pre-execution verdict (ADR-004)                 |
| switchboard.tool.replayed               | result came from the idempotency ledger         |
| switchboard.gate.decision, .tier, .claims | grounding gate result (ADR-005)               |
| switchboard.turn.text                   | caller text, only with `capture_content=True`   |

Content capture is off by default: neither caller text nor spoken sentences appear in spans
unless asked for.

Spans are plain dicts so there is no OpenTelemetry SDK dependency. Times are clock seconds;
`to_otlp_json` shifts them by an epoch offset into Unix nanoseconds. Trace and span ids are
hashes of the conversation id and span path, so the same log always yields the same trace.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypedDict

from switchboard.core.events import (
    AnyEvent,
    BrainRouted,
    LegOpened,
    PolicyVerdict,
    SpeechPlayed,
    SpeechSentenceGated,
    ToolExecuted,
    ToolProposed,
    TurnCommitted,
)

AttrValue = str | int | float | bool | list[str]


class Span(TypedDict):
    name: str
    trace_id: str
    span_id: str
    parent: str | None
    start: float
    end: float
    attributes: dict[str, AttrValue]
    error: str | None


@dataclass(frozen=True, slots=True)
class ToolTiming:
    call_id: str
    tool: str
    proposed_at: float
    verdict_at: float | None
    allowed: bool | None
    rule_ids: tuple[str, ...]
    executed_at: float | None
    replayed: bool
    error: str | None

    @property
    def policy_ms(self) -> float | None:
        return _ms(self.proposed_at, self.verdict_at)

    @property
    def duration_ms(self) -> float | None:
        return _ms(self.proposed_at, self.executed_at)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["rule_ids"] = list(self.rule_ids)
        d["policy_ms"] = self.policy_ms
        d["duration_ms"] = self.duration_ms
        return d


@dataclass(frozen=True, slots=True)
class GatedSentence:
    at: float
    decision: str
    tier: int
    claims: int
    sentence: str


@dataclass(frozen=True, slots=True)
class TurnLatency:
    turn_id: str
    leg_id: str | None
    party: str | None
    committed_at: float
    endpointing_ms: float
    model_tier: str | None
    routed_at: float | None
    route_reason: str | None
    first_output_at: float | None
    first_output_kind: Literal["tool", "text"] | None
    gated: tuple[GatedSentence, ...]
    first_audio_at: float | None
    interrupted: bool
    tools: tuple[ToolTiming, ...]
    text: str

    @property
    def speech_end_at(self) -> float:
        return self.committed_at - self.endpointing_ms / 1000.0

    @property
    def router_ms(self) -> float | None:
        return _ms(self.committed_at, self.routed_at)

    @property
    def brain_first_output_ms(self) -> float | None:
        start = self.routed_at if self.routed_at is not None else self.committed_at
        return _ms(start, self.first_output_at)

    @property
    def gate_decision(self) -> str | None:
        return self.gated[0].decision if self.gated else None

    @property
    def sentences_released(self) -> int:
        return sum(1 for g in self.gated if g.decision == "release")

    @property
    def sentences_blocked(self) -> int:
        return sum(1 for g in self.gated if g.decision == "block")

    @property
    def first_audio_ms(self) -> float | None:
        return _ms(self.committed_at, self.first_audio_at)

    @property
    def release_to_audio_ms(self) -> float | None:
        released = next((g.at for g in self.gated if g.decision == "release"), None)
        return _ms(released, self.first_audio_at) if released is not None else None

    @property
    def voice_to_voice_ms(self) -> float | None:
        return _ms(self.speech_end_at, self.first_audio_at)

    @property
    def end_at(self) -> float:
        times = [self.committed_at]
        candidates = (self.routed_at, self.first_output_at, self.first_audio_at)
        times += [t for t in candidates if t is not None]
        times += [g.at for g in self.gated]
        for tool in self.tools:
            times += [t for t in (tool.verdict_at, tool.executed_at) if t is not None]
        return max(times)

    def to_dict(self, capture_content: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "turn_id": self.turn_id,
            "leg_id": self.leg_id,
            "party": self.party,
            "committed_at": self.committed_at,
            "speech_end_at": self.speech_end_at,
            "endpointing_ms": self.endpointing_ms,
            "model_tier": self.model_tier,
            "router_ms": self.router_ms,
            "brain_first_output_ms": self.brain_first_output_ms,
            "first_output_kind": self.first_output_kind,
            "gate_decision": self.gate_decision,
            "sentences_released": self.sentences_released,
            "sentences_blocked": self.sentences_blocked,
            "first_audio_ms": self.first_audio_ms,
            "release_to_audio_ms": self.release_to_audio_ms,
            "voice_to_voice_ms": self.voice_to_voice_ms,
            "interrupted": self.interrupted,
            "tools": [t.to_dict() for t in self.tools],
        }
        if capture_content:
            d["text"] = self.text
        return d


@dataclass
class _TurnBuilder:
    commit: TurnCommitted
    party: str | None
    routed: BrainRouted | None = None
    first_output: tuple[float, Literal["tool", "text"]] | None = None
    gated: list[GatedSentence] = field(default_factory=list)
    played: SpeechPlayed | None = None
    tools: dict[str, dict[str, Any]] = field(default_factory=dict)

    def output(self, at: float, kind: Literal["tool", "text"]) -> None:
        if self.first_output is None:
            self.first_output = (at, kind)

    def build(self) -> TurnLatency:
        return TurnLatency(
            turn_id=self.commit.turn_id,
            leg_id=self.commit.leg_id,
            party=self.party,
            committed_at=self.commit.at,
            endpointing_ms=self.commit.endpointing_ms,
            model_tier=self.routed.model_tier if self.routed else None,
            routed_at=self.routed.at if self.routed else None,
            route_reason=self.routed.reason if self.routed else None,
            first_output_at=self.first_output[0] if self.first_output else None,
            first_output_kind=self.first_output[1] if self.first_output else None,
            gated=tuple(self.gated),
            first_audio_at=self.played.at if self.played else None,
            interrupted=self.played.interrupted if self.played else False,
            tools=tuple(ToolTiming(**t) for t in self.tools.values()),
            text=self.commit.text,
        )


@dataclass(frozen=True, slots=True)
class TurnTimeline:
    conversation_id: str
    bundle_id: str
    agent_id: str
    turns: list[TurnLatency]
    capture_content: bool = False

    @classmethod
    def from_events(
        cls,
        events: Iterable[AnyEvent],
        *,
        agent_id: str = "switchboard",
        capture_content: bool = False,
    ) -> TurnTimeline:
        ordered = sorted(events, key=lambda e: (e.conversation_id, e.seq, e.at))
        builders: list[_TurnBuilder] = []
        by_leg: dict[str | None, _TurnBuilder] = {}
        by_turn_id: dict[str, _TurnBuilder] = {}
        by_call: dict[str, _TurnBuilder] = {}
        parties: dict[str, str] = {}
        last: _TurnBuilder | None = None

        for e in ordered:
            if isinstance(e, LegOpened) and e.leg_id is not None:
                parties[e.leg_id] = e.party
                continue
            if isinstance(e, TurnCommitted):
                last = _TurnBuilder(commit=e, party=parties.get(e.leg_id or ""))
                builders.append(last)
                by_leg[e.leg_id] = last
                by_turn_id[e.turn_id] = last
                continue
            if isinstance(e, BrainRouted):
                target = by_turn_id.get(e.turn_id)
                if target is not None and target.routed is None:
                    target.routed = e
                continue
            if isinstance(e, PolicyVerdict | ToolExecuted):
                owner = by_call.get(e.call_id)
                if owner is not None:
                    _record_tool_outcome(owner.tools[e.call_id], e)
                continue
            turn = by_leg.get(e.leg_id) if e.leg_id is not None else last
            if turn is None:
                continue
            if isinstance(e, ToolProposed):
                turn.output(e.at, "tool")
                by_call[e.call_id] = turn
                turn.tools[e.call_id] = {
                    "call_id": e.call_id,
                    "tool": e.tool,
                    "proposed_at": e.at,
                    "verdict_at": None,
                    "allowed": None,
                    "rule_ids": (),
                    "executed_at": None,
                    "replayed": False,
                    "error": None,
                }
            elif isinstance(e, SpeechSentenceGated):
                turn.output(e.at, "text")
                turn.gated.append(
                    GatedSentence(
                        at=e.at,
                        decision=e.decision,
                        tier=e.tier,
                        claims=len(e.claims),
                        sentence=e.sentence,
                    )
                )
            elif isinstance(e, SpeechPlayed) and turn.played is None:
                turn.played = e

        first = ordered[0] if ordered else None
        return cls(
            conversation_id=first.conversation_id if first else "",
            bundle_id=first.bundle_id if first else "",
            agent_id=agent_id,
            turns=[b.build() for b in builders],
            capture_content=capture_content,
        )

    def voice_to_voice_percentiles(self) -> tuple[float | None, float | None]:
        values = [t.voice_to_voice_ms for t in self.turns if t.voice_to_voice_ms is not None]
        return percentile(values, 50), percentile(values, 95)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "bundle_id": self.bundle_id,
            "turns": [t.to_dict(self.capture_content) for t in self.turns],
        }

    def spans(self) -> list[Span]:
        out: list[Span] = []
        trace_id = _hex(self.conversation_id, 32)
        common: dict[str, AttrValue] = {
            "gen_ai.conversation.id": self.conversation_id,
            "gen_ai.agent.id": self.agent_id,
            "gen_ai.agent.version": self.bundle_id,
        }

        def add(
            path: str,
            name: str,
            parent: str | None,
            start: float,
            end: float,
            attrs: dict[str, AttrValue],
            error: str | None = None,
        ) -> str:
            span_id = _hex(f"{self.conversation_id}/{path}", 16)
            out.append(
                Span(
                    name=name,
                    trace_id=trace_id,
                    span_id=span_id,
                    parent=parent,
                    start=start,
                    end=max(start, end),
                    attributes=attrs,
                    error=error,
                )
            )
            return span_id

        for index, t in enumerate(self.turns):
            base = f"{index}/{t.turn_id}"
            root_attrs: dict[str, AttrValue] = {
                **common,
                "gen_ai.operation.name": "invoke_agent",
                "switchboard.turn.id": t.turn_id,
            }
            if t.leg_id is not None:
                root_attrs["switchboard.leg.id"] = t.leg_id
            if t.party is not None:
                root_attrs["switchboard.party"] = t.party
            for key, value in (
                ("endpointing_ms", t.endpointing_ms),
                ("router_ms", t.router_ms),
                ("brain_first_output_ms", t.brain_first_output_ms),
                ("first_audio_ms", t.first_audio_ms),
                ("tts_first_audio_ms", t.release_to_audio_ms),
                ("voice_to_voice_ms", t.voice_to_voice_ms),
            ):
                if value is not None:
                    root_attrs[f"switchboard.latency.{key}"] = float(value)
            if t.gate_decision is not None:
                root_attrs["switchboard.gate.decision"] = t.gate_decision
            root_attrs["switchboard.turn.interrupted"] = t.interrupted
            if self.capture_content:
                root_attrs["switchboard.turn.text"] = t.text
            root = add(
                base,
                f"invoke_agent {t.party or self.agent_id}",
                None,
                t.speech_end_at,
                t.end_at,
                root_attrs,
            )
            add(
                f"{base}/endpointing",
                "switchboard.endpointing",
                root,
                t.speech_end_at,
                t.committed_at,
                {"switchboard.latency.endpointing_ms": float(t.endpointing_ms)},
            )
            if t.routed_at is not None and t.model_tier is not None:
                add(
                    f"{base}/route",
                    "switchboard.route",
                    root,
                    t.committed_at,
                    t.routed_at,
                    {
                        "switchboard.router.tier": t.model_tier,
                        "switchboard.router.reason": t.route_reason or "",
                    },
                )
            if t.first_output_at is not None:
                tier = t.model_tier or "brain"
                add(
                    f"{base}/chat",
                    f"chat {tier}",
                    root,
                    t.routed_at if t.routed_at is not None else t.committed_at,
                    t.first_output_at,
                    {
                        **common,
                        "gen_ai.operation.name": "chat",
                        "switchboard.router.tier": tier,
                        "switchboard.brain.first_output": t.first_output_kind or "",
                    },
                )
            for tool in t.tools:
                attrs: dict[str, AttrValue] = {
                    **common,
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": tool.tool,
                    "gen_ai.tool.call.id": tool.call_id,
                    "switchboard.tool.replayed": tool.replayed,
                }
                if tool.allowed is not None:
                    attrs["switchboard.policy.allowed"] = tool.allowed
                    attrs["switchboard.policy.rule_ids"] = list(tool.rule_ids)
                if tool.error is not None:
                    attrs["error.type"] = tool.error
                end = next(
                    t
                    for t in (tool.executed_at, tool.verdict_at, tool.proposed_at)
                    if t is not None
                )
                tool_span = add(
                    f"{base}/tool/{tool.call_id}",
                    f"execute_tool {tool.tool}",
                    root,
                    tool.proposed_at,
                    end,
                    attrs,
                    error=tool.error,
                )
                if tool.verdict_at is not None:
                    add(
                        f"{base}/tool/{tool.call_id}/policy",
                        "switchboard.policy.check",
                        tool_span,
                        tool.proposed_at,
                        tool.verdict_at,
                        {"switchboard.policy.allowed": bool(tool.allowed)},
                    )
            for n, g in enumerate(t.gated):
                gate_attrs: dict[str, AttrValue] = {
                    "switchboard.gate.decision": g.decision,
                    "switchboard.gate.tier": g.tier,
                    "switchboard.gate.claims": g.claims,
                }
                if self.capture_content:
                    gate_attrs["switchboard.gate.sentence"] = g.sentence
                add(f"{base}/gate/{n}", "switchboard.gate", root, g.at, g.at, gate_attrs)
            if t.first_audio_at is not None:
                released = next((g.at for g in t.gated if g.decision == "release"), None)
                add(
                    f"{base}/first_audio",
                    "switchboard.first_audio",
                    root,
                    released if released is not None else t.committed_at,
                    t.first_audio_at,
                    {"switchboard.turn.interrupted": t.interrupted},
                )
        return out


def _record_tool_outcome(slot: dict[str, Any], e: PolicyVerdict | ToolExecuted) -> None:
    if isinstance(e, PolicyVerdict):
        slot["verdict_at"] = e.at
        slot["allowed"] = e.allowed
        slot["rule_ids"] = tuple(e.rule_ids)
    else:
        slot["executed_at"] = e.at
        slot["replayed"] = e.replayed
        slot["error"] = tool_error(e)


def tool_error(event: ToolExecuted) -> str | None:
    """A tool result signals failure with a truthy `error` field."""
    err = event.result.get("error")
    return str(err) if err else None


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile (the numpy default), or None for no data."""
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def _ms(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return (end - start) * 1000.0


def _hex(text: str, length: int) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:length]


def _otlp_value(value: AttrValue) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, list):
        return {"arrayValue": {"values": [{"stringValue": v} for v in value]}}
    return {"stringValue": value}


def _attrs(attrs: dict[str, AttrValue]) -> list[dict[str, Any]]:
    return [{"key": k, "value": _otlp_value(v)} for k, v in attrs.items()]


def _nanos(seconds: float) -> str:
    return str(round(seconds * 1e9))


_SPAN_KIND_INTERNAL = 1
_STATUS_ERROR = 2


def to_otlp_json(
    spans: Sequence[Span],
    *,
    service_name: str = "switchboard",
    epoch_offset_s: float = 0.0,
) -> dict[str, Any]:
    """Encode spans as an OTLP/JSON `ExportTraceServiceRequest` body.

    Ids are lowercase hex as the OTLP/JSON mapping requires; 64-bit integers are strings.
    """
    encoded: list[dict[str, Any]] = []
    for s in spans:
        span: dict[str, Any] = {
            "traceId": s["trace_id"],
            "spanId": s["span_id"],
            "name": s["name"],
            "kind": _SPAN_KIND_INTERNAL,
            "startTimeUnixNano": _nanos(s["start"] + epoch_offset_s),
            "endTimeUnixNano": _nanos(s["end"] + epoch_offset_s),
            "attributes": _attrs(s["attributes"]),
        }
        if s["parent"] is not None:
            span["parentSpanId"] = s["parent"]
        if s["error"] is not None:
            span["status"] = {"code": _STATUS_ERROR, "message": s["error"]}
        encoded.append(span)
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": _attrs({"service.name": service_name})},
                "scopeSpans": [
                    {
                        "scope": {"name": "switchboard.telemetry", "version": "0.1.0"},
                        "spans": encoded,
                    }
                ],
            }
        ]
    }
