from collections.abc import Sequence
from typing import Any

import pytest

from switchboard.core.events import (
    ConversationEnded,
    ConversationStarted,
    Event,
    HandoffRequested,
    ToolExecuted,
    TurnCommitted,
)
from switchboard.memory.consolidate import consolidate, consolidate_with
from switchboard.memory.store import DEFAULT_TTL_S, MemoryKind, MemoryStore, MemoryWrite

NOW = 1_000.0


def base(at: float) -> dict[str, Any]:
    return {"conversation_id": "conv_1", "leg_id": "leg_1", "at": at, "bundle_id": "b1"}


def conversation(
    outcome: str, *, handoff: str | None = None, tools: Sequence[str] = ()
) -> list[Event]:
    events: list[Event] = [
        ConversationStarted(**base(0.0), tenant_id="t1", customer_id="c1"),
        TurnCommitted(**base(1.0), turn_id="turn_1", text="where is my order", endpointing_ms=200),
    ]
    for i, tool in enumerate(tools):
        events.append(ToolExecuted(**base(2.0 + i), call_id=f"call_{i}", tool=tool, result={}))
    if handoff is not None:
        events.append(HandoffRequested(**base(8.0), reason=handoff))
    events.append(ConversationEnded(**base(9.0), outcome=outcome))
    return events


def test_resolved_conversation_writes_a_summary_and_closes_open_issues() -> None:
    writes = consolidate(conversation("resolved", tools=["lookup_order", "issue_refund"]), NOW)
    assert writes == [
        MemoryWrite.put(
            "t1",
            "c1",
            MemoryKind.INTERACTION_SUMMARY,
            {
                "conversation_id": "conv_1",
                "outcome": "resolved",
                "tools": ["lookup_order", "issue_refund"],
                "handoff": False,
                "ended_at": 9.0,
                "text": (
                    "Outcome: resolved. Tools executed: lookup_order, issue_refund. Handoff: no."
                ),
            },
            at=NOW,
        ),
        MemoryWrite.delete("t1", "c1", MemoryKind.OPEN_ISSUE),
    ]


def test_handoff_opens_an_issue() -> None:
    writes = consolidate(conversation("resolved", handoff="refund over limit"), NOW)
    summary, issue = writes
    assert summary.kind is MemoryKind.INTERACTION_SUMMARY
    assert summary.content["text"] == "Outcome: resolved. Tools executed: none. Handoff: yes."
    assert summary.content["handoff"] is True
    assert issue.op == "put"
    assert issue.kind is MemoryKind.OPEN_ISSUE
    assert issue.content == {
        "conversation_id": "conv_1",
        "outcome": "resolved",
        "reason": "refund over limit",
        "text": "Open issue from conv_1: refund over limit.",
    }
    assert issue.ttl_s == DEFAULT_TTL_S[MemoryKind.OPEN_ISSUE]


def test_unresolved_outcome_opens_an_issue_without_handoff() -> None:
    writes = consolidate(conversation("abandoned", tools=["lookup_order"]), NOW)
    assert [(w.op, w.kind) for w in writes] == [
        ("put", MemoryKind.INTERACTION_SUMMARY),
        ("put", MemoryKind.OPEN_ISSUE),
    ]
    assert writes[1].content["reason"] == "abandoned"


def test_tools_are_listed_once_in_first_execution_order() -> None:
    writes = consolidate(
        conversation("resolved", tools=["lookup_order", "notify", "lookup_order"]), NOW
    )
    assert writes[0].content["tools"] == ["lookup_order", "notify"]


def test_consolidation_is_pure_and_deterministic() -> None:
    events = conversation("resolved", handoff="x", tools=["a"])
    snapshot = list(events)
    assert consolidate(events, NOW) == consolidate(events, NOW)
    assert events == snapshot


def test_unfinished_or_unscoped_conversations_are_rejected() -> None:
    with pytest.raises(ValueError, match="ended"):
        consolidate(conversation("resolved")[:-1], NOW)
    with pytest.raises(ValueError, match="started"):
        consolidate(conversation("resolved")[1:], NOW)


def test_round_trip_through_the_store_closes_prior_issue() -> None:
    store = MemoryStore()
    store.apply(consolidate(conversation("transferred", handoff="needs human"), 10.0))
    assert [m.kind for m in store.read("t1", "c1", now=11.0)] == [
        MemoryKind.INTERACTION_SUMMARY,
        MemoryKind.OPEN_ISSUE,
    ]
    store.apply(consolidate(conversation("resolved", tools=["lookup_order"]), 20.0))
    kinds = [m.kind for m in store.read("t1", "c1", now=21.0)]
    assert kinds == [MemoryKind.INTERACTION_SUMMARY, MemoryKind.INTERACTION_SUMMARY]


class EchoSummarizer:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    async def summarize(self, events: Sequence[Event], draft: str) -> str:
        self.calls.append((len(events), draft))
        return f"LLM: {draft}"


async def test_summarizer_hook_rewrites_only_the_summary_text() -> None:
    events = conversation("resolved", handoff="x")
    summarizer = EchoSummarizer()
    writes = await consolidate_with(events, NOW, summarizer)
    plain = consolidate(events, NOW)
    assert summarizer.calls == [(len(events), plain[0].content["text"])]
    assert writes[0].content["text"] == "LLM: " + plain[0].content["text"]
    assert writes[0].content["outcome"] == "resolved"
    assert writes[1:] == plain[1:]
