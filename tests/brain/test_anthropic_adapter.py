import sys
from types import SimpleNamespace

import pytest

from switchboard.brain.anthropic_adapter import (
    AnthropicLLM,
    StreamTranslator,
    to_anthropic_params,
)
from switchboard.brain.types import (
    BrainRequest,
    Done,
    Message,
    TextDelta,
    ToolCallChunk,
    ToolSchema,
    Usage,
)

TOOL = ToolSchema(
    name="lookup_order",
    description="Look up an order.",
    parameters={"type": "object", "properties": {"order_id": {"type": "string"}}},
)


def test_constructing_without_package_raises_clear_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(ImportError, match=r"switchboard\[anthropic\]"):
        AnthropicLLM()


def test_request_mapping_system_cache_tools_and_messages() -> None:
    request = BrainRequest(
        messages=(
            Message(role="system", content="STABLE PREFIX"),
            Message(role="system", content="volatile context"),
            Message(role="user", content="where is order 48213"),
            Message(
                role="assistant",
                content='{"order_id":"48213"}',
                tool_call_id="toolu_1",
                name="lookup_order",
            ),
            Message(role="tool", content='{"eta":"19:45"}', tool_call_id="toolu_1"),
        ),
        tools=(TOOL,),
        tier="talker",
        max_tokens=512,
    )
    params = to_anthropic_params(request, model="claude-haiku-5-5", effort="low")
    assert params["model"] == "claude-haiku-5-5"
    assert params["max_tokens"] == 512
    assert params["output_config"] == {"effort": "low"}
    assert params["system"] == [
        {"type": "text", "text": "STABLE PREFIX", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "volatile context"},
    ]
    assert params["tools"] == [
        {
            "name": "lookup_order",
            "description": "Look up an order.",
            "input_schema": TOOL.parameters,
            "eager_input_streaming": True,
        }
    ]
    assert params["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "where is order 48213"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "lookup_order",
                    "input": {"order_id": "48213"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": '{"eta":"19:45"}'}
            ],
        },
    ]


def test_consecutive_same_role_messages_are_merged() -> None:
    request = BrainRequest(
        messages=(
            Message(role="tool", content="a", tool_call_id="t1"),
            Message(role="tool", content="b", tool_call_id="t2"),
            Message(role="user", content="and?"),
        )
    )
    params = to_anthropic_params(request, model="m")
    assert len(params["messages"]) == 1
    assert "system" not in params
    assert "tools" not in params


def ev(**kw: object) -> SimpleNamespace:
    return SimpleNamespace(**kw)


def test_stream_translator_maps_events_to_chunks() -> None:
    t = StreamTranslator()
    events = [
        ev(
            type="message_start",
            message=ev(usage=ev(input_tokens=100, cache_read_input_tokens=80, output_tokens=1)),
        ),
        ev(type="content_block_start", index=0, content_block=ev(type="text", text="")),
        ev(type="content_block_delta", index=0, delta=ev(type="text_delta", text="Let me ")),
        ev(type="content_block_delta", index=0, delta=ev(type="text_delta", text="check.")),
        ev(type="content_block_stop", index=0),
        ev(
            type="content_block_start",
            index=1,
            content_block=ev(type="tool_use", id="toolu_9", name="lookup_order", input={}),
        ),
        ev(
            type="content_block_delta",
            index=1,
            delta=ev(type="input_json_delta", partial_json='{"order_'),
        ),
        ev(
            type="content_block_delta",
            index=1,
            delta=ev(type="input_json_delta", partial_json='id":"48213"}'),
        ),
        ev(type="content_block_stop", index=1),
        ev(type="message_delta", delta=ev(stop_reason="tool_use"), usage=ev(output_tokens=30)),
        ev(type="message_stop"),
    ]
    chunks = [c for e in events for c in t.feed(e)]
    assert chunks == [
        TextDelta("Let me "),
        TextDelta("check."),
        ToolCallChunk(call_id="toolu_9", name="lookup_order", args={"order_id": "48213"}),
        Done(
            usage=Usage(input_tokens=100, cached_input_tokens=80, output_tokens=30),
            finish_reason="tool_use",
        ),
    ]


def test_stream_translator_rejects_invalid_tool_json() -> None:
    t = StreamTranslator()
    t.feed(
        ev(type="content_block_start", index=0, content_block=ev(type="tool_use", id="x", name="n"))
    )
    t.feed(
        ev(
            type="content_block_delta",
            index=0,
            delta=ev(type="input_json_delta", partial_json='{"a":'),
        )
    )
    with pytest.raises(ValueError, match="tool input"):
        t.feed(ev(type="content_block_stop", index=0))
