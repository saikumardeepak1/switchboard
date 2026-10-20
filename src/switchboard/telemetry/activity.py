"""Activity stream: a read-side projection of the event log (ADR-002).

Every query reads the store and filters in process. That is fine for the in-memory and SQLite
stores this repo ships; a deployment with a large log would push the same filters into SQL.

Items sort by `(at, conversation_id, seq)`. The key is unique (seq is unique per conversation),
so a cursor that encodes the last key seen gives stable pages even when timestamps tie and when
new events are appended between requests.

Type filters accept exact types (`policy.verdict`) and prefixes ending in `.` or `.*`
(`policy.`, `tool.*`). Patterns are resolved against the event catalog up front, so a typo is
an error instead of a silently empty page. `since` is inclusive, `until` exclusive.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from switchboard.core.events import (
    EVENT_TYPES,
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
    ToolUnresolved,
    TurnBackchannel,
    TurnCommitted,
    TurnSpeculationCancelled,
    TurnSpeculationStarted,
)
from switchboard.core.store import EventStore
from switchboard.telemetry.turns import TurnTimeline, tool_error

Severity = Literal["info", "warn", "error"]
Order = Literal["asc", "desc"]

MAX_LIMIT = 500
MAX_BUCKETS = 10_000
SUMMARY_MAX = 160

_ENVELOPE = frozenset(
    {"type", "conversation_id", "leg_id", "seq", "at", "bundle_id", "causation_id", "event_id"}
)


class InvalidFilter(ValueError):
    """A filter value the stream cannot serve. `field` names the offending parameter."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(message)
        self.field = field
        self.message = message


@dataclass(frozen=True, slots=True)
class ActivityFilter:
    conversation_id: str | None = None
    leg_id: str | None = None
    types: tuple[str, ...] = ()
    since: float | None = None
    until: float | None = None
    text: str | None = None
    limit: int = 50
    cursor: str | None = None
    order: Order = "desc"


@dataclass(frozen=True, slots=True)
class ActivityItem:
    at: float
    conversation_id: str
    leg_id: str | None
    seq: int
    event_id: str
    bundle_id: str
    type: str
    severity: Severity
    summary: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ActivityPage:
    items: list[ActivityItem]
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class Bucket:
    start: float
    end: float
    counts_by_type: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts_by_type.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "total": self.total,
            "counts": dict(self.counts_by_type),
        }


@dataclass(frozen=True, slots=True)
class Party:
    leg_id: str
    party: str
    role: str
    parent_leg_id: str | None
    goal: str | None
    opened_at: float
    closed_at: float | None
    close_reason: str | None


@dataclass(frozen=True, slots=True)
class ConversationSummary:
    conversation_id: str
    bundle_id: str
    tenant_id: str | None
    customer_id: str | None
    started_at: float
    ended_at: float | None
    outcome: str | None
    parties: list[Party]
    events: int
    turns: int
    violations_blocked: int
    sentences_blocked: int
    tools_executed: int
    tool_errors: int
    handoffs: int
    voice_to_voice_p50_ms: float | None
    voice_to_voice_p95_ms: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_types(patterns: Iterable[str]) -> frozenset[str] | None:
    """Expand exact types and prefixes into catalog types. None means no type filter."""
    patterns = list(patterns)
    if not patterns:
        return None
    out: set[str] = set()
    for raw in patterns:
        pattern = raw.strip()
        prefix = pattern[:-1] if pattern.endswith(".*") else pattern
        if prefix.endswith("."):
            matched = {t for t in EVENT_TYPES if t.startswith(prefix)}
        else:
            matched = {pattern} & set(EVENT_TYPES)
        if not matched:
            raise InvalidFilter(
                "type",
                f"unknown event type {raw!r}; use an exact type or a prefix ending in '.'"
                f" (known: {', '.join(EVENT_TYPES)})",
            )
        out |= matched
    return frozenset(out)


def severity(event: AnyEvent) -> Severity:
    if isinstance(event, ToolExecuted) and tool_error(event) is not None:
        return "error"
    if isinstance(event, PolicyVerdict) and not event.allowed:
        return "warn"
    if isinstance(event, SpeechSentenceGated) and event.decision == "block":
        return "warn"
    if isinstance(event, HandoffRequested | ToolCompensated | ToolUnresolved):
        return "warn"
    return "info"


def describe(event: AnyEvent) -> str:
    """One-line human summary of an event."""
    return _clip(_describe(event))


def _describe(e: AnyEvent) -> str:
    match e:
        case ConversationStarted():
            return f"Conversation started for customer {e.customer_id} (tenant {e.tenant_id})"
        case LegOpened():
            goal = f": {e.goal}" if e.goal else ""
            return f"{e.party} leg opened ({e.role}){goal}"
        case LegClosed():
            return f"Leg closed: {e.reason}"
        case TurnSpeculationStarted():
            return f"Speculating on {e.partial_text!r}"
        case TurnSpeculationCancelled():
            return f"Speculation {e.turn_id} cancelled, caller kept talking"
        case TurnBackchannel():
            return f"Backchannel while the agent spoke: {e.text!r}"
        case TurnCommitted():
            return f"Turn committed after {e.endpointing_ms:.0f} ms endpointing: {e.text!r}"
        case BrainRouted():
            return f"Routed to {e.model_tier}: {e.reason}"
        case ToolProposed():
            args = ", ".join(f"{k}={v}" for k, v in e.args.items())
            return f"Proposed {e.tool}({args})"
        case PolicyVerdict():
            if e.allowed:
                return f"Allowed {e.call_id}"
            hint = f": {e.repair_hint}" if e.repair_hint else ""
            return f"Denied {e.call_id} by {', '.join(e.rule_ids) or 'policy'}{hint}"
        case ToolExecuted():
            err = tool_error(e)
            if err is not None:
                return f"{e.tool} failed: {err}"
            return f"Executed {e.tool}" + (" (replayed from ledger)" if e.replayed else "")
        case ToolCompensated():
            return f"Compensated {e.call_id} with {e.compensation_tool}"
        case ToolUnresolved():
            return f"{e.tool} outcome unknown at {e.reason or 'shutdown'}; left to its lease"
        case SpeechSentenceGated():
            verb = {"release": "Released", "block": "Blocked", "escalate": "Escalated"}
            return f"{verb[e.decision]} sentence (tier {e.tier}): {e.sentence!r}"
        case SpeechPlayed():
            cut = " (interrupted)" if e.interrupted else ""
            return f"Played{cut}: {e.text_played!r}"
        case ProcedureAdvanced():
            return f"Procedure advanced to {e.step}"
        case ChildResult():
            return f"Child leg {e.child_leg_id} returned {len(e.result)} fields"
        case KnowledgeRetrieved():
            return f"Retrieved {len(e.fact_ids)} facts"
        case MemoryLoaded():
            return f"Loaded {e.count} memory items"
        case HandoffRequested():
            return f"Handoff requested: {e.reason}"
        case ConversationEnded():
            return f"Conversation ended: {e.outcome}"
    return e.type  # pragma: no cover


def _clip(text: str) -> str:
    line = " ".join(text.split())
    return line if len(line) <= SUMMARY_MAX else line[: SUMMARY_MAX - 3] + "..."


def to_item(event: AnyEvent) -> ActivityItem:
    return ActivityItem(
        at=event.at,
        conversation_id=event.conversation_id,
        leg_id=event.leg_id,
        seq=event.seq,
        event_id=event.event_id,
        bundle_id=event.bundle_id,
        type=event.type,
        severity=severity(event),
        summary=describe(event),
        payload=event.model_dump(mode="json", exclude=set(_ENVELOPE)),
    )


_Key = tuple[float, str, int]


def _key(event: AnyEvent) -> _Key:
    return (event.at, event.conversation_id, event.seq)


def encode_cursor(key: _Key) -> str:
    raw = json.dumps(list(key), separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str) -> _Key:
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        at, conv, seq = json.loads(raw)
    except (binascii.Error, ValueError, TypeError) as exc:
        raise InvalidFilter("cursor", f"invalid cursor {cursor!r}") from exc
    if not (
        isinstance(at, int | float)
        and not isinstance(at, bool)
        and isinstance(conv, str)
        and isinstance(seq, int)
        and not isinstance(seq, bool)
    ):
        raise InvalidFilter("cursor", f"invalid cursor {cursor!r}")
    return (float(at), conv, seq)


class ActivityStream:
    def __init__(self, store: EventStore) -> None:
        self._store = store

    async def query(self, filters: ActivityFilter) -> ActivityPage:
        if not 1 <= filters.limit <= MAX_LIMIT:
            raise InvalidFilter("limit", f"limit must be between 1 and {MAX_LIMIT}")
        if filters.order not in ("asc", "desc"):
            raise InvalidFilter("order", "order must be 'asc' or 'desc'")
        after = decode_cursor(filters.cursor) if filters.cursor else None
        events = await self._select(filters)
        desc = filters.order == "desc"
        events.sort(key=_key, reverse=desc)
        if after is not None:
            events = [e for e in events if (_key(e) < after if desc else _key(e) > after)]
        page = events[: filters.limit]
        more = len(events) > filters.limit
        return ActivityPage(
            items=[to_item(e) for e in page],
            next_cursor=encode_cursor(_key(page[-1])) if more else None,
        )

    async def histogram(self, filters: ActivityFilter, bucket_s: float) -> list[Bucket]:
        """Event counts per type in fixed buckets aligned to multiples of `bucket_s`.

        The range runs from `since` (or the first event) to `until` (or the last event), and
        empty buckets inside it are included so a chart can draw gaps.
        """
        if not (bucket_s > 0 and math.isfinite(bucket_s)):
            raise InvalidFilter("bucket_s", "bucket_s must be a positive number of seconds")
        events = await self._select(filters)
        ats = [e.at for e in events]
        lo = filters.since if filters.since is not None else (min(ats) if ats else None)
        if lo is None:
            return []
        first = math.floor(lo / bucket_s)
        if filters.until is not None:
            last = math.ceil(filters.until / bucket_s)
        elif ats:
            last = math.floor(max(ats) / bucket_s) + 1
        else:
            return []
        count = max(0, last - first)
        if count > MAX_BUCKETS:
            raise InvalidFilter(
                "bucket_s",
                f"range needs {count} buckets, more than the {MAX_BUCKETS} allowed;"
                " use a larger bucket_s or a narrower since/until",
            )
        counters: list[Counter[str]] = [Counter() for _ in range(count)]
        for e in events:
            index = math.floor(e.at / bucket_s) - first
            if 0 <= index < count:
                counters[index][e.type] += 1
        return [
            Bucket(
                start=(first + i) * bucket_s,
                end=(first + i + 1) * bucket_s,
                counts_by_type=dict(sorted(c.items())),
            )
            for i, c in enumerate(counters)
        ]

    async def summary(self, conversation_id: str) -> ConversationSummary | None:
        events = await self._store.read(conversation_id)
        if not events:
            return None
        return summarize(events)

    async def timeline(
        self, conversation_id: str, *, capture_content: bool = False
    ) -> TurnTimeline | None:
        events = await self._store.read(conversation_id)
        if not events:
            return None
        return TurnTimeline.from_events(events, capture_content=capture_content)

    async def _select(self, f: ActivityFilter) -> list[AnyEvent]:
        if f.since is not None and f.until is not None and f.since > f.until:
            raise InvalidFilter("since", "since must not be later than until")
        types = resolve_types(f.types)
        if f.conversation_id is not None:
            source: Iterable[AnyEvent] = await self._store.read(f.conversation_id)
        else:
            source = [e async for e in self._store.read_all(types=_as_set(types), since=f.since)]
        needle = f.text.lower() if f.text else None
        out: list[AnyEvent] = []
        for e in source:
            if types is not None and e.type not in types:
                continue
            if f.since is not None and e.at < f.since:
                continue
            if f.until is not None and e.at >= f.until:
                continue
            if f.leg_id is not None and e.leg_id != f.leg_id:
                continue
            if needle is not None and needle not in _payload_text(e):
                continue
            out.append(e)
        return out


def _as_set(types: frozenset[str] | None) -> set[str] | None:
    return set(types) if types is not None else None


def _payload_text(event: AnyEvent) -> str:
    payload = event.model_dump(mode="json", exclude=set(_ENVELOPE))
    return json.dumps(payload, ensure_ascii=False).lower()


def summarize(events: Sequence[AnyEvent]) -> ConversationSummary:
    first = events[0]
    started = next((e for e in events if isinstance(e, ConversationStarted)), None)
    ended = next((e for e in events if isinstance(e, ConversationEnded)), None)
    opened: dict[str, LegOpened] = {}
    closed: dict[str, LegClosed] = {}
    for e in events:
        if isinstance(e, LegOpened) and e.leg_id is not None:
            opened.setdefault(e.leg_id, e)
        elif isinstance(e, LegClosed) and e.leg_id is not None:
            closed.setdefault(e.leg_id, e)
    parties = [
        Party(
            leg_id=leg,
            party=o.party,
            role=o.role,
            parent_leg_id=o.parent_leg_id,
            goal=o.goal,
            opened_at=o.at,
            closed_at=closed[leg].at if leg in closed else None,
            close_reason=closed[leg].reason if leg in closed else None,
        )
        for leg, o in opened.items()
    ]
    timeline = TurnTimeline.from_events(events)
    p50, p95 = timeline.voice_to_voice_percentiles()
    executed = [e for e in events if isinstance(e, ToolExecuted)]
    return ConversationSummary(
        conversation_id=first.conversation_id,
        bundle_id=first.bundle_id,
        tenant_id=started.tenant_id if started else None,
        customer_id=started.customer_id if started else None,
        started_at=started.at if started else first.at,
        ended_at=ended.at if ended else None,
        outcome=ended.outcome if ended else None,
        parties=parties,
        events=len(events),
        turns=len(timeline.turns),
        violations_blocked=sum(1 for e in events if isinstance(e, PolicyVerdict) and not e.allowed),
        sentences_blocked=sum(
            1 for e in events if isinstance(e, SpeechSentenceGated) and e.decision == "block"
        ),
        tools_executed=len(executed),
        tool_errors=sum(1 for e in executed if tool_error(e) is not None),
        handoffs=sum(1 for e in events if isinstance(e, HandoffRequested)),
        voice_to_voice_p50_ms=p50,
        voice_to_voice_p95_ms=p95,
    )
