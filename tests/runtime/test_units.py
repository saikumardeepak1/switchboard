from typing import Any

from switchboard.core import (
    ChildResult,
    ConversationStarted,
    LegOpened,
    PolicyVerdict,
    SpeechPlayed,
    SpeechSentenceGated,
    ToolExecuted,
    ToolProposed,
    TurnCommitted,
    TurnSpeculationCancelled,
    TurnSpeculationStarted,
)
from switchboard.runtime import (
    SPECULATION_RULE,
    AdmissionController,
    SentenceSplitter,
    replay,
)


def test_splitter_emits_sentences_as_they_complete() -> None:
    s = SentenceSplitter()
    assert s.feed("You earned ") == []
    assert s.feed("$182.40 this week. ") == ["You earned $182.40 this week."]
    assert s.feed("Anything else? I'm") == ["Anything else?"]
    assert s.flush() == "I'm"
    assert s.flush() is None


def test_admission_caps_per_tenant_and_releases() -> None:
    admission = AdmissionController({"acme": 1})
    assert admission.admit("acme", "c1", default_cap=5).admitted
    rejected = admission.admit("acme", "c2", default_cap=5)
    assert not rejected.admitted
    assert rejected.reason is not None and "cap of 1" in rejected.reason
    assert admission.admit("other", "c3", default_cap=1).admitted
    admission.release("acme", "c1")
    assert admission.admit("acme", "c2", default_cap=5).admitted
    assert admission.active("acme") == 1


BASE: dict[str, Any] = {"conversation_id": "conv_1", "at": 0.0, "bundle_id": "bnd_x"}


def numbered(*events: Any) -> list[Any]:
    return [e.model_copy(update={"seq": i, "event_id": f"e{i}"}) for i, e in enumerate(events, 1)]


def test_fold_drops_cancelled_speculation_from_history_but_keeps_evidence() -> None:
    events = numbered(
        ConversationStarted(**BASE, tenant_id="t", customer_id="c"),
        LegOpened(**BASE, leg_id="leg_a", role="inbound", party="courier"),
        TurnSpeculationStarted(**BASE, leg_id="leg_a", turn_id="turn_1", partial_text="Late."),
        ToolProposed(**BASE, leg_id="leg_a", causation_id="e3", call_id="k1", tool="t", args={}),
        PolicyVerdict(**BASE, leg_id="leg_a", call_id="k1", allowed=True, rule_ids=[]),
        ToolExecuted(
            **BASE, leg_id="leg_a", call_id="k1", tool="t", result={"evidence": {"seen": True}}
        ),
        TurnSpeculationCancelled(**BASE, leg_id="leg_a", turn_id="turn_1"),
        TurnCommitted(**BASE, leg_id="leg_a", turn_id="turn_2", text="Hi.", endpointing_ms=500),
    )
    state = replay(events)
    assert state.evidence == {"seen": True}
    assert [(m.role, m.content) for m in state.messages("leg_a")] == [("user", "Hi.")]


def test_fold_keeps_only_played_text_and_skips_speculation_refusals() -> None:
    events = numbered(
        ConversationStarted(**BASE, tenant_id="t", customer_id="c"),
        LegOpened(**BASE, leg_id="leg_a", role="inbound", party="courier"),
        TurnCommitted(**BASE, leg_id="leg_a", turn_id="turn_1", text="Hi.", endpointing_ms=5),
        SpeechSentenceGated(
            **BASE,
            leg_id="leg_a",
            utterance_id="turn_1/u0",
            sentence="Hello there friend.",
            decision="release",
            tier=2,
            claims=[],
        ),
        SpeechPlayed(
            **BASE, leg_id="leg_a", utterance_id="turn_1/u0", text_played="Hello", interrupted=True
        ),
        ToolProposed(**BASE, leg_id="leg_a", causation_id="e3", call_id="k1", tool="w", args={}),
        PolicyVerdict(
            **BASE, leg_id="leg_a", call_id="k1", allowed=False, rule_ids=[SPECULATION_RULE]
        ),
        PolicyVerdict(**BASE, leg_id="leg_a", call_id="k1", allowed=True, rule_ids=[]),
    )
    state = replay(events)
    messages = state.messages("leg_a")
    # The allowed call has no result yet, so it is left out rather than sent dangling.
    assert [(m.role, m.content) for m in messages] == [("user", "Hi."), ("assistant", "Hello")]
    assert [c.call_id for c in state.pending_calls()] == ["k1"]


def test_child_result_merges_its_evidence_patch() -> None:
    events = numbered(
        ConversationStarted(**BASE, tenant_id="t", customer_id="c"),
        LegOpened(**BASE, leg_id="leg_a", role="inbound", party="courier"),
        LegOpened(
            **BASE, leg_id="leg_b", role="outbound_child", party="customer", parent_leg_id="leg_a"
        ),
        ChildResult(
            **BASE,
            leg_id="leg_a",
            child_leg_id="leg_b",
            result={"confirmed": True, "evidence": {"customer_confirmed_address": True}},
        ),
    )
    state = replay(events)
    assert state.evidence["customer_confirmed_address"] is True
    assert state.messages("leg_a")[-1].role == "system"
    assert "customer" in state.messages("leg_a")[-1].content
