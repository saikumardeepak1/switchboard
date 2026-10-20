import dataclasses

import pytest

from switchboard.brain.types import (
    BrainRequest,
    Done,
    Message,
    TextDelta,
    ToolCallChunk,
    ToolSchema,
    Usage,
)


def test_message_defaults_and_frozen() -> None:
    m = Message(role="user", content="hi")
    assert m.tool_call_id is None
    assert m.name is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.content = "x"  # type: ignore[misc]


def test_tool_message_requires_call_id() -> None:
    with pytest.raises(ValueError, match="tool_call_id"):
        Message(role="tool", content="{}")


def test_unknown_role_rejected() -> None:
    with pytest.raises(ValueError, match="role"):
        Message(role="robot", content="x")  # type: ignore[arg-type]


def test_request_defaults() -> None:
    req = BrainRequest(messages=(Message(role="user", content="hi"),))
    assert req.tools == ()
    assert req.tier == "thinker"
    assert req.max_tokens > 0
    assert req.metadata == {}


def test_request_rejects_bad_tier_and_tokens() -> None:
    with pytest.raises(ValueError, match="tier"):
        BrainRequest(messages=(), tier="genius")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_tokens"):
        BrainRequest(messages=(), max_tokens=0)


def test_chunks_are_distinct_types() -> None:
    chunks = [
        TextDelta("a"),
        ToolCallChunk(call_id="c1", name="lookup_order", args={"order_id": "48213"}),
        Done(usage=Usage(input_tokens=10, cached_input_tokens=0, output_tokens=2)),
    ]
    assert [type(c).__name__ for c in chunks] == ["TextDelta", "ToolCallChunk", "Done"]
    assert chunks[2] == Done(usage=Usage(10, 0, 2), finish_reason="end_turn")


def test_tool_schema_holds_parameters() -> None:
    t = ToolSchema(name="lookup_order", description="d", parameters={"type": "object"})
    assert t.parameters["type"] == "object"
