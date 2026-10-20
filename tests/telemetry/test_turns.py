import json
import re

import pytest

from switchboard.core.events import (
    BrainRouted,
    PolicyVerdict,
    SpeechPlayed,
    SpeechSentenceGated,
    ToolExecuted,
    ToolProposed,
    TurnCommitted,
    TurnSpeculationCancelled,
    TurnSpeculationStarted,
)
from switchboard.telemetry.turns import TurnTimeline, percentile, to_otlp_json
from tests.telemetry.fixtures import BUNDLE, b, conv_a_events


def timeline() -> TurnTimeline:
    return TurnTimeline.from_events(conv_a_events())


def test_one_entry_per_committed_turn_in_commit_order() -> None:
    tl = timeline()
    assert tl.conversation_id == "conv_a"
    assert tl.bundle_id == BUNDLE
    assert [t.turn_id for t in tl.turns] == ["turn_1", "turn_k1", "turn_2"]
    assert [t.leg_id for t in tl.turns] == ["leg_c", "leg_k", "leg_c"]


def test_turn_one_decomposition() -> None:
    t = timeline().turns[0]
    assert t.endpointing_ms == 250
    assert t.speech_end_at == pytest.approx(0.75)
    assert t.model_tier == "thinker"
    assert t.router_ms == pytest.approx(40)
    assert t.brain_first_output_ms == pytest.approx(260)
    assert t.first_output_kind == "tool"
    assert t.gate_decision == "release"
    assert t.sentences_released == 1
    assert t.sentences_blocked == 0
    assert t.first_audio_ms == pytest.approx(900)
    assert t.release_to_audio_ms == pytest.approx(200)
    assert t.voice_to_voice_ms == pytest.approx(1150)
    assert t.interrupted is True
    (tool,) = t.tools
    assert tool.tool == "lookup_order"
    assert tool.allowed is True
    assert tool.policy_ms == pytest.approx(1)
    assert tool.duration_ms == pytest.approx(200)
    assert tool.error is None


def test_events_on_another_leg_do_not_leak_into_a_turn() -> None:
    tl = timeline()
    courier = tl.turns[1]
    assert courier.voice_to_voice_ms == pytest.approx(800)
    assert courier.model_tier is None
    assert courier.router_ms is None
    # No routing event: brain time is measured from the commit.
    assert courier.brain_first_output_ms == pytest.approx(50)
    (tool,) = courier.tools
    assert tool.error == "timeout"
    assert tl.turns[0].tools[0].call_id == "call_1"


def test_denied_tool_and_blocked_sentence() -> None:
    t = timeline().turns[2]
    assert t.router_ms == pytest.approx(20)
    assert t.brain_first_output_ms == pytest.approx(80)
    assert t.gate_decision == "block"
    assert t.sentences_blocked == 1
    assert t.sentences_released == 1
    assert t.voice_to_voice_ms == pytest.approx(1000)
    (tool,) = t.tools
    assert tool.allowed is False
    assert tool.rule_ids == ("refund_limit",)
    assert tool.executed_at is None
    assert tool.duration_ms is None


def test_turn_without_audio_has_no_voice_to_voice() -> None:
    events = [TurnCommitted(**b("c", 1.0), turn_id="t", text="x", endpointing_ms=10)]
    (t,) = TurnTimeline.from_events(events).turns
    assert t.first_audio_ms is None
    assert t.voice_to_voice_ms is None


def test_audio_before_any_turn_is_ignored() -> None:
    events = [
        SpeechPlayed(**b("c", 0.5), utterance_id="g", text_played="hello", interrupted=False),
        TurnCommitted(**b("c", 1.0), turn_id="t", text="x", endpointing_ms=10),
    ]
    (t,) = TurnTimeline.from_events(events).turns
    assert t.first_audio_ms is None


def test_empty_events() -> None:
    tl = TurnTimeline.from_events([])
    assert tl.turns == []
    assert tl.spans() == []
    assert tl.voice_to_voice_percentiles() == (None, None)


def test_percentiles() -> None:
    assert percentile([800, 1000, 1150], 50) == pytest.approx(1000)
    assert percentile([800, 1000, 1150], 95) == pytest.approx(1135)
    assert percentile([7], 95) == 7
    assert percentile([], 50) is None
    p50, p95 = timeline().voice_to_voice_percentiles()
    assert p50 == pytest.approx(1000)
    assert p95 == pytest.approx(1135)


def test_spans_use_genai_names_and_nest_under_the_turn() -> None:
    spans = timeline().spans()
    roots = [s for s in spans if s["parent"] is None]
    assert len(roots) == 3
    root = roots[0]
    attrs = root["attributes"]
    assert root["name"] == "invoke_agent customer"
    assert attrs["gen_ai.operation.name"] == "invoke_agent"
    assert attrs["gen_ai.conversation.id"] == "conv_a"
    assert attrs["gen_ai.agent.version"] == BUNDLE
    assert attrs["gen_ai.agent.id"] == "switchboard"
    assert attrs["switchboard.turn.id"] == "turn_1"
    assert attrs["switchboard.latency.voice_to_voice_ms"] == pytest.approx(1150)
    assert root["start"] == pytest.approx(0.75)
    assert root["end"] == pytest.approx(1.9)

    children = [s for s in spans if s["parent"] == root["span_id"]]
    names = {s["name"] for s in children}
    assert {
        "switchboard.endpointing",
        "switchboard.route",
        "chat thinker",
        "execute_tool lookup_order",
        "switchboard.gate",
        "switchboard.first_audio",
    } <= names
    tool_span = next(s for s in children if s["name"] == "execute_tool lookup_order")
    assert tool_span["attributes"]["gen_ai.operation.name"] == "execute_tool"
    assert tool_span["attributes"]["gen_ai.tool.name"] == "lookup_order"
    assert tool_span["attributes"]["gen_ai.tool.call.id"] == "call_1"
    assert tool_span["end"] - tool_span["start"] == pytest.approx(0.2)
    policy = next(s for s in spans if s["parent"] == tool_span["span_id"])
    assert policy["name"] == "switchboard.policy.check"
    assert all(s["trace_id"] == root["trace_id"] for s in spans)
    assert len({s["span_id"] for s in spans}) == len(spans)


def test_content_is_not_captured_by_default() -> None:
    dumped = json.dumps(timeline().spans())
    assert "where is my order" not in dumped
    assert "7:45 pm" not in dumped
    captured = json.dumps(TurnTimeline.from_events(conv_a_events(), capture_content=True).spans())
    assert "where is my order" in captured


def test_span_ids_are_deterministic() -> None:
    assert timeline().spans() == timeline().spans()


def test_otlp_json_shape() -> None:
    payload = to_otlp_json(timeline().spans(), service_name="svc", epoch_offset_s=1000.0)
    json.dumps(payload)
    (rs,) = payload["resourceSpans"]
    resource_attrs = {a["key"]: a["value"] for a in rs["resource"]["attributes"]}
    assert resource_attrs["service.name"] == {"stringValue": "svc"}
    (ss,) = rs["scopeSpans"]
    assert ss["scope"]["name"] == "switchboard.telemetry"
    spans = ss["spans"]
    for s in spans:
        assert re.fullmatch(r"[0-9a-f]{32}", s["traceId"])
        assert re.fullmatch(r"[0-9a-f]{16}", s["spanId"])
        assert int(s["endTimeUnixNano"]) >= int(s["startTimeUnixNano"])
        assert s["kind"] == 1
    root = next(s for s in spans if "parentSpanId" not in s)
    assert root["startTimeUnixNano"] == str(1000_750_000_000)
    attrs = {a["key"]: a["value"] for a in root["attributes"]}
    assert attrs["gen_ai.operation.name"] == {"stringValue": "invoke_agent"}
    assert attrs["switchboard.latency.endpointing_ms"] == {"doubleValue": 250.0}
    tool = next(s for s in spans if s["name"] == "execute_tool mark_delivered")
    assert tool["status"]["code"] == 2
    denied = next(s for s in spans if s["name"] == "execute_tool issue_refund")
    dattrs = {a["key"]: a["value"] for a in denied["attributes"]}
    assert dattrs["switchboard.policy.allowed"] == {"boolValue": False}
    assert dattrs["switchboard.policy.rule_ids"] == {
        "arrayValue": {"values": [{"stringValue": "refund_limit"}]}
    }
    gate = next(s for s in spans if s["name"] == "switchboard.gate")
    gattrs = {a["key"]: a["value"] for a in gate["attributes"]}
    assert gattrs["switchboard.gate.tier"] == {"intValue": "1"}


def test_voice_to_voice_ends_at_first_audio_not_at_the_end_of_playback() -> None:
    events = [
        TurnCommitted(**b("c", 1.0), turn_id="t", text="x", endpointing_ms=200),
        SpeechPlayed(
            **b("c", 4.0),
            utterance_id="t/u0",
            text_played="A long answer.",
            interrupted=False,
            first_audio_at=1.5,
        ),
    ]
    (t,) = TurnTimeline.from_events(events).turns
    assert t.first_audio_at == 1.5
    assert t.voice_to_voice_ms == pytest.approx(700)


def test_played_event_without_first_audio_gives_no_voice_to_voice() -> None:
    # Logs written before speech.played carried first_audio_at: the end of playback is not
    # first audio, so no number is better than a wrong one.
    events = [
        TurnCommitted(**b("c", 1.0), turn_id="t", text="x", endpointing_ms=200),
        SpeechPlayed(**b("c", 4.0), utterance_id="t/u0", text_played="Hi.", interrupted=False),
    ]
    (t,) = TurnTimeline.from_events(events).turns
    assert t.voice_to_voice_ms is None


def test_speculative_turn_owns_what_it_did_before_the_commit() -> None:
    c = "c"
    events = [
        TurnCommitted(**b(c, 0.5), turn_id="t0", text="hello", endpointing_ms=100),
        TurnSpeculationStarted(**b(c, 2.0), turn_id="t1", partial_text="pay"),
        BrainRouted(
            **b(c, 2.01), causation_id="anchor", turn_id="t1", model_tier="thinker", reason="tool"
        ),
        SpeechSentenceGated(
            **b(c, 2.2),
            causation_id="anchor",
            utterance_id="t1/u0",
            sentence="Let me check.",
            decision="release",
            tier=2,
            claims=[],
        ),
        ToolProposed(**b(c, 2.2), causation_id="anchor", call_id="k1", tool="get_pay", args={}),
        PolicyVerdict(**b(c, 2.21), call_id="k1", allowed=True, rule_ids=[]),
        ToolExecuted(**b(c, 2.3), call_id="k1", tool="get_pay", result={"total": 1}),
        TurnCommitted(
            **b(c, 2.5), causation_id="anchor", turn_id="t1", text="pay", endpointing_ms=400
        ),
        SpeechPlayed(
            **b(c, 3.4),
            utterance_id="t1/u0",
            text_played="Let me check.",
            interrupted=False,
            first_audio_at=2.65,
        ),
    ]
    events[1] = events[1].model_copy(update={"event_id": "anchor"})
    numbered = [e.model_copy(update={"seq": i}) for i, e in enumerate(events, 1)]
    t0, t1 = TurnTimeline.from_events(numbered).turns
    assert t0.tools == () and t0.gated == ()
    assert t1.model_tier == "thinker"
    assert t1.speculation_started_at == 2.0
    assert t1.router_ms == pytest.approx(10)
    assert t1.brain_first_output_ms == pytest.approx(190)
    assert [tool.call_id for tool in t1.tools] == ["k1"]
    assert t1.sentences_released == 1
    assert t1.voice_to_voice_ms == pytest.approx(550)


def test_cancelled_speculation_is_not_a_turn_and_takes_nothing_with_it() -> None:
    c = "c"
    events = [
        TurnCommitted(**b(c, 0.5), turn_id="t0", text="hello", endpointing_ms=100),
        TurnSpeculationStarted(**b(c, 2.0), turn_id="t1", partial_text="pay"),
        BrainRouted(
            **b(c, 2.01), causation_id="anchor", turn_id="t1", model_tier="talker", reason="x"
        ),
        TurnSpeculationCancelled(**b(c, 2.1), turn_id="t1"),
    ]
    events[1] = events[1].model_copy(update={"event_id": "anchor"})
    numbered = [e.model_copy(update={"seq": i}) for i, e in enumerate(events, 1)]
    (t0,) = TurnTimeline.from_events(numbered).turns
    assert t0.model_tier is None
