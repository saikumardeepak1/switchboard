"""One instance of every event type, used by the event and store tests."""

from typing import Any

from switchboard.core.events import (
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


def base(conversation_id: str = "conv_1", at: float = 1.0) -> dict[str, Any]:
    return {
        "conversation_id": conversation_id,
        "leg_id": "leg_1",
        "at": at,
        "bundle_id": "bundle_abc",
        "causation_id": None,
    }


def every_event_type(conversation_id: str = "conv_1") -> list[AnyEvent]:
    b = base(conversation_id)
    return [
        ConversationStarted(**b, tenant_id="t1", customer_id="c1"),
        LegOpened(**b, role="outbound_child", party="courier", parent_leg_id="leg_0", goal="eta"),
        LegClosed(**b, reason="hangup"),
        TurnSpeculationStarted(**b, turn_id="turn_1", partial_text="where is my"),
        TurnSpeculationCancelled(**b, turn_id="turn_1"),
        TurnCommitted(**b, turn_id="turn_1", text="where is my order", endpointing_ms=212.5),
        TurnBackchannel(**b, text="mm-hm", utterance_id="utt_0"),
        BrainRouted(**b, turn_id="turn_1", model_tier="thinker", reason="tool needed"),
        ToolProposed(**b, call_id="call_1", tool="lookup_order", args={"order_id": "48213"}),
        PolicyVerdict(
            **b,
            call_id="call_1",
            allowed=False,
            rule_ids=["refund_limit"],
            repair_hint="Offer a handoff.",
        ),
        ToolExecuted(
            **b,
            call_id="call_1",
            tool="lookup_order",
            result={"eta": "7:45 pm", "items": [1, 2]},
            idempotency_key="k1",
            replayed=True,
        ),
        ToolCompensated(
            **b, call_id="call_1", compensation_tool="void_refund", result={"ok": True}
        ),
        ToolUnresolved(**b, call_id="call_1", tool="issue_refund", reason="conversation ended"),
        SpeechSentenceGated(
            **b,
            utterance_id="utt_1",
            sentence="It arrives at 7:45 pm.",
            decision="release",
            tier=1,
            claims=[{"kind": "time", "raw": "7:45 pm", "value": 71100.0}],
        ),
        SpeechPlayed(**b, utterance_id="utt_1", text_played="It arrives", interrupted=True),
        ProcedureAdvanced(**b, step="verify_identity", state={"done": ["greet"]}),
        ChildResult(**b, child_leg_id="leg_2", result={"confirmed": True}),
        KnowledgeRetrieved(
            **b, fact_ids=["fact_1", "fact_2"], as_of=99.5, texts=["Open late.", "Closed Monday."]
        ),
        MemoryLoaded(**b, count=1, texts=["Prefers short answers."]),
        HandoffRequested(**b, reason="refund above limit"),
        ConversationEnded(**b, outcome="resolved"),
    ]
