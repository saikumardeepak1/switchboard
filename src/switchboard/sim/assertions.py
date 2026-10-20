"""Assertions: named predicates over a finished run, composable with all_of, any_of and not_.

Events are matched by labels: an event's type, plus `type:value` for the field that tells
events of that type apart (the tool for tool events, the party for leg events, the decision for
gated sentences, `confirmed` or `declined` for child results, and so on). So
`never_before("tool.executed:mark_delivered", "child.result:confirmed")` reads as written.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from switchboard.core import AnyEvent
from switchboard.grounding import Evidence, extract_claims
from switchboard.sim.adversarial import Fault
from switchboard.sim.caller import CallerSummary


@dataclass(frozen=True)
class AssertionResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class RunRecord:
    events: Sequence[AnyEvent]
    callers: Mapping[str, CallerSummary] = field(default_factory=dict)
    faults: Sequence[Fault] = ()
    # Text the gate also saw but the log keeps only by id: retrieved knowledge, memory.
    extra_evidence: Sequence[str] = ()


Assertion = Callable[[RunRecord], AssertionResult]


def predicate(name: str) -> Callable[[Callable[[RunRecord], tuple[bool, str]]], Assertion]:
    """Turn `(record) -> (passed, detail)` into a named `Assertion`."""

    def wrap(fn: Callable[[RunRecord], tuple[bool, str]]) -> Assertion:
        def check(record: RunRecord) -> AssertionResult:
            passed, detail = fn(record)
            return AssertionResult(name, passed, detail)

        check.__name__ = name
        check.__qualname__ = name
        return check

    return wrap


def _name(a: Assertion) -> str:
    return getattr(a, "__name__", repr(a))


def event_labels(event: AnyEvent) -> set[str]:
    labels: set[str] = {event.type}
    values: list[Any] = []
    for attr in ("tool", "party", "role", "decision", "outcome", "model_tier", "reason", "step"):
        values.append(getattr(event, attr, None))
    if event.type == "policy.verdict":
        values.append("allowed" if getattr(event, "allowed", False) else "denied")
    if event.type == "tool.executed" and getattr(event, "replayed", False):
        values.append("replayed")
    if event.type == "speech.played" and getattr(event, "interrupted", False):
        values.append("interrupted")
    if event.type == "child.result":
        result = getattr(event, "result", {}) or {}
        values.append("confirmed" if result.get("confirmed") is True else "declined")
    labels.update(f"{event.type}:{v}" for v in values if isinstance(v, str) and v)
    return labels


def _count(record: RunRecord, label: str) -> int:
    return sum(1 for e in record.events if label in event_labels(e))


def all_of(*assertions: Assertion) -> Assertion:
    @predicate(f"all_of({', '.join(_name(a) for a in assertions)})")
    def check(record: RunRecord) -> tuple[bool, str]:
        failed = [r for r in (a(record) for a in assertions) if not r.passed]
        return not failed, "; ".join(f"{r.name}: {r.detail}" for r in failed)

    return check


def any_of(*assertions: Assertion) -> Assertion:
    @predicate(f"any_of({', '.join(_name(a) for a in assertions)})")
    def check(record: RunRecord) -> tuple[bool, str]:
        results = [a(record) for a in assertions]
        if any(r.passed for r in results):
            return True, ""
        return False, "; ".join(f"{r.name}: {r.detail}" for r in results)

    return check


def not_(assertion: Assertion) -> Assertion:
    @predicate(f"not_({_name(assertion)})")
    def check(record: RunRecord) -> tuple[bool, str]:
        result = assertion(record)
        return not result.passed, "" if not result.passed else f"{result.name} held"

    return check


def tool_executed_times(tool: str, n: int) -> Assertion:
    @predicate(f"tool_executed_times({tool}, {n})")
    def check(record: RunRecord) -> tuple[bool, str]:
        got = _count(record, f"tool.executed:{tool}")
        return got == n, f"executed {got} times, expected {n}"

    return check


def tool_executed_once(tool: str) -> Assertion:
    inner = tool_executed_times(tool, 1)

    @predicate(f"tool_executed_once({tool})")
    def check(record: RunRecord) -> tuple[bool, str]:
        result = inner(record)
        return result.passed, result.detail

    return check


def tool_never_executed(tool: str) -> Assertion:
    inner = tool_executed_times(tool, 0)

    @predicate(f"tool_never_executed({tool})")
    def check(record: RunRecord) -> tuple[bool, str]:
        result = inner(record)
        return result.passed, result.detail

    return check


def never_before(first: str, second: str) -> Assertion:
    """No event labelled `first` happens before the first event labelled `second`. If `second`
    never happens, `first` must never happen."""

    @predicate(f"never_before({first}, {second})")
    def check(record: RunRecord) -> tuple[bool, str]:
        for event in record.events:
            labels = event_labels(event)
            if second in labels:
                return True, ""
            if first in labels:
                return False, f"{first} at seq {event.seq} before any {second}"
        return True, ""

    return check


def outcome(expected: str) -> Assertion:
    @predicate(f"outcome({expected})")
    def check(record: RunRecord) -> tuple[bool, str]:
        ended = [e for e in record.events if e.type == "conversation.ended"]
        got = getattr(ended[-1], "outcome", None) if ended else None
        return got == expected, f"outcome was {got!r}"

    return check


def caller_heard(party: str, phrase: str) -> Assertion:
    @predicate(f"caller_heard({party}, {phrase})")
    def check(record: RunRecord) -> tuple[bool, str]:
        caller = record.callers.get(party)
        if caller is None:
            return False, f"no caller {party!r}"
        return phrase.lower() in caller.heard.lower(), f"{party} did not hear {phrase!r}"

    return check


def caller_status(party: str, *statuses: str) -> Assertion:
    @predicate(f"caller_status({party}, {'|'.join(statuses)})")
    def check(record: RunRecord) -> tuple[bool, str]:
        caller = record.callers.get(party)
        got = caller.status if caller else None
        return got in statuses, f"{party} ended {got!r}"

    return check


def no_sentence_released_with_unsupported_claims() -> Assertion:
    """Re-check every released sentence against the evidence that existed when it was released:
    successful tool results and child results earlier in the log, plus `extra_evidence`."""

    @predicate("no_sentence_released_with_unsupported_claims")
    def check(record: RunRecord) -> tuple[bool, str]:
        sources: list[object] = []
        problems: list[str] = []
        for event in record.events:
            if event.type == "tool.executed":
                result = getattr(event, "result", {})
                if "error" not in result:
                    sources.append(result)
            elif event.type == "child.result":
                sources.append(getattr(event, "result", {}))
            elif event.type == "speech.sentence_gated" and getattr(event, "decision", "") == (
                "release"
            ):
                sentence = getattr(event, "sentence", "")
                claims = extract_claims(sentence)
                if not claims:
                    continue
                evidence = Evidence(tool_results=sources, facts=record.extra_evidence)
                bad = [c.raw for c in claims if not evidence.supports(c)]
                if bad:
                    problems.append(f"{sentence!r} ({', '.join(bad)})")
        return not problems, "; ".join(problems)

    return check


def no_corrupted_sentence_released() -> Assertion:
    """No sentence carrying a number the adversary corrupted was released to TTS."""

    @predicate("no_corrupted_sentence_released")
    def check(record: RunRecord) -> tuple[bool, str]:
        corrupted = {f.sentence for f in record.faults if f.sentence and f.token}
        leaked = [
            getattr(e, "sentence", "")
            for e in record.events
            if e.type == "speech.sentence_gated"
            and getattr(e, "decision", "") == "release"
            and getattr(e, "sentence", "") in corrupted
        ]
        return not leaked, "; ".join(leaked)

    return check
