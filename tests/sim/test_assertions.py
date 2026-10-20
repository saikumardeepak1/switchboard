"""Assertion predicates over a run's events, and how they compose."""

from __future__ import annotations

from typing import Any

from switchboard.core import (
    AnyEvent,
    ChildResult,
    ConversationEnded,
    SpeechSentenceGated,
    ToolExecuted,
    ToolProposed,
)
from switchboard.sim import (
    CallerSummary,
    Fault,
    RunRecord,
    all_of,
    any_of,
    caller_heard,
    event_labels,
    never_before,
    no_corrupted_sentence_released,
    no_sentence_released_with_unsupported_claims,
    not_,
    outcome,
    tool_executed_once,
    tool_executed_times,
    tool_never_executed,
)


class Log:
    def __init__(self) -> None:
        self.events: list[AnyEvent] = []

    def add(self, cls: type[Any], **payload: Any) -> Log:
        seq = len(self.events) + 1
        self.events.append(
            cls(
                conversation_id="conv_1",
                leg_id=payload.pop("leg_id", "leg_1"),
                seq=seq,
                at=float(seq),
                bundle_id="bnd_1",
                event_id=f"ev_{seq}",
                **payload,
            )
        )
        return self

    def executed(self, tool: str, result: dict[str, Any] | None = None, **kw: Any) -> Log:
        call_id = f"c{len(self.events)}"
        self.add(ToolProposed, call_id=call_id, tool=tool, args={})
        return self.add(ToolExecuted, call_id=call_id, tool=tool, result=result or {}, **kw)

    def said(self, sentence: str, decision: str = "release") -> Log:
        return self.add(
            SpeechSentenceGated,
            utterance_id="t/u0",
            sentence=sentence,
            decision=decision,
            tier=1,
            claims=[],
        )

    def record(self, **kw: Any) -> RunRecord:
        return RunRecord(events=self.events, **kw)


def confirmed(log: Log, value: bool = True) -> Log:
    return log.add(ChildResult, child_leg_id="leg_2", result={"confirmed": value})


def test_labels_name_the_type_and_its_distinguishing_fields() -> None:
    log = confirmed(Log().executed("mark_delivered", replayed=True))
    assert event_labels(log.events[1]) >= {
        "tool.executed",
        "tool.executed:mark_delivered",
        "tool.executed:replayed",
    }
    assert "child.result:confirmed" in event_labels(log.events[2])
    assert "child.result:declined" in event_labels(confirmed(Log(), False).events[0])


def test_tool_executed_once_counts_executions_of_that_tool() -> None:
    once = Log().executed("mark_delivered").executed("lookup_order")
    twice = Log().executed("mark_delivered").executed("mark_delivered")
    assert tool_executed_once("mark_delivered")(once.record()).passed
    result = tool_executed_once("mark_delivered")(twice.record())
    assert not result.passed and "2" in result.detail
    assert tool_executed_times("mark_delivered", 2)(twice.record()).passed
    assert tool_never_executed("issue_refund")(once.record()).passed
    assert not tool_never_executed("lookup_order")(once.record()).passed


def test_never_before_holds_only_when_the_first_event_waits_for_the_second() -> None:
    check = never_before("tool.executed:mark_delivered", "child.result:confirmed")
    good = Log().executed("lookup_order")
    confirmed(good).executed("mark_delivered")
    bad = confirmed(Log().executed("mark_delivered"))
    declined = confirmed(Log(), False).executed("mark_delivered")
    nothing = Log().executed("lookup_order")

    assert check(good.record()).passed
    assert not check(bad.record()).passed
    assert not check(declined.record()).passed, "a declined result does not unlock it"
    assert check(nothing.record()).passed


def test_unsupported_claims_are_found_in_released_sentences_only() -> None:
    log = Log().executed("get_courier_pay", {"week_total": 182.40, "deliveries": 23})
    log.said("You've earned $182.40 this week.")
    log.said("You've earned $128.40 this week.", decision="block")
    check = no_sentence_released_with_unsupported_claims()
    assert check(log.record()).passed

    log.said("That is $99.00 more than last week.")
    result = check(log.record())
    assert not result.passed and "$99.00" in result.detail


def test_evidence_must_exist_before_the_sentence_that_uses_it() -> None:
    log = Log().said("Your refund of $10.00 is done.")
    log.executed("issue_refund", {"amount": 10.0})
    assert not no_sentence_released_with_unsupported_claims()(log.record()).passed
    extra = no_sentence_released_with_unsupported_claims()(
        log.record(extra_evidence=("refund amount $10.00",))
    )
    assert extra.passed


def test_corrupted_sentences_from_the_fault_log_must_not_be_released() -> None:
    fault = Fault(
        index=0,
        kind="corrupt_money",
        sentence="You've earned $128.40 this week.",
        original="You've earned $182.40 this week.",
        token="$128.40",
    )
    blocked = Log().said(fault.sentence or "", decision="block").record(faults=(fault,))
    released = Log().said(fault.sentence or "").record(faults=(fault,))
    assert no_corrupted_sentence_released()(blocked).passed
    assert not no_corrupted_sentence_released()(released).passed


def test_outcome_and_caller_heard() -> None:
    record = (
        Log()
        .add(ConversationEnded, outcome="resolved", leg_id=None)
        .record(
            callers={"courier": CallerSummary("courier", "Your pay is $182.40.", "completed", 3)}
        )
    )
    assert outcome("resolved")(record).passed
    assert not outcome("handoff")(record).passed
    assert caller_heard("courier", "182.40")(record).passed
    assert not caller_heard("courier", "128.40")(record).passed
    assert not caller_heard("customer", "hello")(record).passed


def test_predicates_compose() -> None:
    record = Log().executed("mark_delivered").record()
    yes = tool_executed_once("mark_delivered")
    no = tool_never_executed("mark_delivered")
    assert all_of(yes, not_(no))(record).passed
    assert not all_of(yes, no)(record).passed
    assert any_of(no, yes)(record).passed
    assert not any_of(no, not_(yes))(record).passed
    assert all_of(yes, no)(record).name == (
        "all_of(tool_executed_once(mark_delivered), tool_never_executed(mark_delivered))"
    )
