"""A realistic two-leg conversation built from real core events.

conv_a: a customer call (leg_c) with two turns, plus a courier child leg (leg_k) whose single
turn hits a tool error. conv_b: a short second call whose timestamps tie with conv_a, used to
test pagination ordering.

Latency numbers the tests assert against are spelled out next to each turn.
"""

from __future__ import annotations

from typing import Any

from switchboard.core.events import (
    AnyEvent,
    BrainRouted,
    ChildResult,
    ConversationEnded,
    ConversationStarted,
    HandoffRequested,
    LegClosed,
    LegOpened,
    PolicyVerdict,
    SpeechPlayed,
    SpeechSentenceGated,
    ToolExecuted,
    ToolProposed,
    TurnCommitted,
)
from switchboard.core.store import InMemoryEventStore

BUNDLE = "bundle_v1"


def b(conv: str, at: float, leg: str | None = "leg_c") -> dict[str, Any]:
    return {"conversation_id": conv, "leg_id": leg, "at": at, "bundle_id": BUNDLE}


def conv_a_events() -> list[AnyEvent]:
    c = "conv_a"
    return [
        ConversationStarted(**b(c, 0.0, None), tenant_id="acme", customer_id="cust_9"),
        LegOpened(**b(c, 0.0), role="inbound", party="customer"),
        # turn_1: endpointing 250, router 40, brain first output 260, tool 200, audio at +900.
        # voice-to-voice = 250 + 900 = 1150
        TurnCommitted(**b(c, 1.0), turn_id="turn_1", text="where is my order", endpointing_ms=250),
        BrainRouted(**b(c, 1.04), turn_id="turn_1", model_tier="thinker", reason="needs tool"),
        ToolProposed(**b(c, 1.3), call_id="call_1", tool="lookup_order", args={"order": "48213"}),
        PolicyVerdict(**b(c, 1.301), call_id="call_1", allowed=True, rule_ids=[]),
        ToolExecuted(**b(c, 1.5), call_id="call_1", tool="lookup_order", result={"eta": "7:45"}),
        SpeechSentenceGated(
            **b(c, 1.7),
            utterance_id="u1",
            sentence="It arrives at 7:45 pm.",
            decision="release",
            tier=1,
            claims=[{"kind": "time", "raw": "7:45 pm"}],
        ),
        SpeechPlayed(**b(c, 1.9), utterance_id="u1", text_played="It arrives at", interrupted=True),
        # Courier child leg, interleaved in time with the customer leg.
        LegOpened(
            **b(c, 2.0, "leg_k"),
            role="outbound_child",
            party="courier",
            parent_leg_id="leg_c",
            goal="confirm address",
        ),
        # turn_k1: endpointing 300, audio at +500, voice-to-voice 800. Tool fails.
        TurnCommitted(
            **b(c, 3.0, "leg_k"), turn_id="turn_k1", text="I am outside", endpointing_ms=300
        ),
        ToolProposed(**b(c, 3.05, "leg_k"), call_id="call_3", tool="mark_delivered", args={}),
        PolicyVerdict(**b(c, 3.06, "leg_k"), call_id="call_3", allowed=True, rule_ids=[]),
        ToolExecuted(
            **b(c, 3.2, "leg_k"),
            call_id="call_3",
            tool="mark_delivered",
            result={"error": "timeout"},
        ),
        SpeechPlayed(
            **b(c, 3.5, "leg_k"), utterance_id="uk", text_played="One moment.", interrupted=False
        ),
        ChildResult(**b(c, 4.0), child_leg_id="leg_k", result={"confirmed": True}),
        # turn_2: endpointing 400, router 20, brain first 80, audio at +600, voice-to-voice 1000.
        TurnCommitted(
            **b(c, 5.0), turn_id="turn_2", text="refund me 40 dollars", endpointing_ms=400
        ),
        BrainRouted(**b(c, 5.02), turn_id="turn_2", model_tier="talker", reason="simple"),
        ToolProposed(**b(c, 5.1), call_id="call_2", tool="issue_refund", args={"amount": 40}),
        PolicyVerdict(
            **b(c, 5.101),
            call_id="call_2",
            allowed=False,
            rule_ids=["refund_limit"],
            repair_hint="Refunds above $25 need a human.",
        ),
        SpeechSentenceGated(
            **b(c, 5.3),
            utterance_id="u2",
            sentence="I refunded $40.",
            decision="block",
            tier=1,
            claims=[{"kind": "money", "raw": "$40"}],
        ),
        SpeechSentenceGated(
            **b(c, 5.4),
            utterance_id="u2",
            sentence="Let me get a colleague.",
            decision="release",
            tier=1,
            claims=[],
        ),
        SpeechPlayed(
            **b(c, 5.6), utterance_id="u2", text_played="Let me get a colleague.", interrupted=False
        ),
        HandoffRequested(**b(c, 5.7), reason="refund over limit"),
        LegClosed(**b(c, 6.0, "leg_k"), reason="completed"),
        ConversationEnded(**b(c, 6.1, None), outcome="handoff"),
    ]


def conv_b_events() -> list[AnyEvent]:
    c = "conv_b"
    return [
        ConversationStarted(**b(c, 0.0, None), tenant_id="acme", customer_id="cust_2"),
        LegOpened(**b(c, 1.0, "leg_x"), role="inbound", party="customer"),
        TurnCommitted(**b(c, 1.0, "leg_x"), turn_id="t", text="hi", endpointing_ms=100),
        ConversationEnded(**b(c, 1.0, None), outcome="resolved"),
    ]


async def seeded_store() -> InMemoryEventStore:
    store = InMemoryEventStore()
    a = conv_a_events()
    bb = conv_b_events()
    await store.append("conv_a", a, expected_seq=0)
    await store.append("conv_b", bb, expected_seq=0)
    return store
