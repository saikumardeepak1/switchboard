import asyncio

import pytest

from switchboard.brain.llm import LLM, Reply, ScriptedLLM, ScriptExhaustedError, ScriptStep
from switchboard.brain.types import (
    BrainChunk,
    BrainRequest,
    Done,
    Message,
    TextDelta,
    ToolCallChunk,
)
from switchboard.core.clock import VirtualClock


def req(text: str, role: str = "user") -> BrainRequest:
    if role == "tool":
        msg = Message(role="tool", content=text, tool_call_id="c1", name="lookup_order")
    else:
        msg = Message(role="user", content=text)
    return BrainRequest(messages=(Message(role="system", content="sys"), msg))


async def collect(
    llm: LLM, request: BrainRequest, clock: VirtualClock
) -> list[tuple[float, BrainChunk]]:
    out: list[tuple[float, BrainChunk]] = []

    async def run() -> None:
        async for chunk in llm.stream(request):
            out.append((clock.now(), chunk))

    task = asyncio.create_task(run())
    await clock.run_until_idle()
    await task
    return out


async def test_ttft_then_word_deltas_at_token_rate() -> None:
    clock = VirtualClock()
    llm = ScriptedLLM(
        clock, [ScriptStep("hello", text="Hi there, friend.")], ttft_s=0.4, tokens_per_s=10
    )
    out = await collect(llm, req("hello"), clock)
    texts = [(t, c.text) for t, c in out if isinstance(c, TextDelta)]
    assert texts == [
        (pytest.approx(0.4), "Hi "),
        (pytest.approx(0.5), "there, "),
        (pytest.approx(0.6), "friend."),
    ]
    assert "".join(text for _, text in texts) == "Hi there, friend."
    done = out[-1][1]
    assert isinstance(done, Done)
    assert done.finish_reason == "end_turn"
    assert done.usage.output_tokens == 3
    assert done.usage.input_tokens > 0


async def test_tool_call_step_after_text() -> None:
    clock = VirtualClock()
    call = ToolCallChunk(call_id="c1", name="lookup_order", args={"order_id": "48213"})
    llm = ScriptedLLM(
        clock,
        [ScriptStep("where", text="Let me check.", tool_calls=(call,))],
        ttft_s=0.2,
        tokens_per_s=20,
    )
    out = await collect(llm, req("where is my order"), clock)
    kinds = [type(c).__name__ for _, c in out]
    assert kinds == ["TextDelta", "TextDelta", "TextDelta", "ToolCallChunk", "Done"]
    assert out[3][1] == call
    done = out[-1][1]
    assert isinstance(done, Done) and done.finish_reason == "tool_use"


async def test_tool_only_reply_arrives_at_ttft() -> None:
    clock = VirtualClock()
    call = ToolCallChunk(call_id="c1", name="issue_refund", args={"amount": 400})
    llm = ScriptedLLM(clock, [ScriptStep("refund", tool_calls=(call,))], ttft_s=0.3)
    out = await collect(llm, req("refund me"), clock)
    assert out[0] == (pytest.approx(0.3), call)


async def test_matches_last_user_or_tool_message_with_substring_or_predicate() -> None:
    clock = VirtualClock()
    llm = ScriptedLLM(
        clock,
        [
            ScriptStep(lambda m: m.role == "tool", text="Your order arrives at 7:45 pm."),
            ScriptStep("REFUND", text="I can help with a refund."),
            ScriptStep(None, text="Sorry, could you repeat that?"),
        ],
        ttft_s=0.1,
    )

    async def text_of(request: BrainRequest) -> str:
        out = await collect(llm, request, clock)
        return "".join(c.text for _, c in out if isinstance(c, TextDelta))

    assert await text_of(req('{"eta": "19:45"}', role="tool")) == "Your order arrives at 7:45 pm."
    # substring match is case-insensitive
    assert await text_of(req("i want a refund")) == "I can help with a refund."
    assert await text_of(req("blah")) == "Sorry, could you repeat that?"


async def test_once_steps_are_consumed_in_order() -> None:
    clock = VirtualClock()
    llm = ScriptedLLM(
        clock,
        [ScriptStep("hi", text="First.", once=True), ScriptStep("hi", text="Second.", once=True)],
        ttft_s=0.1,
    )
    first = await collect(llm, req("hi"), clock)
    second = await collect(llm, req("hi"), clock)
    assert isinstance(first[0][1], TextDelta) and first[0][1].text == "First."
    assert isinstance(second[0][1], TextDelta) and second[0][1].text == "Second."
    with pytest.raises(ScriptExhaustedError):
        await collect(llm, req("hi"), clock)


async def test_callable_script_and_request_recording() -> None:
    clock = VirtualClock()
    seen: list[str] = []

    def script(request: BrainRequest) -> Reply:
        seen.append(request.tier)
        return Reply(text=f"tier {request.tier}")

    llm = ScriptedLLM(clock, script, ttft_s=0.1)
    r1 = BrainRequest(messages=(Message(role="user", content="a"),), tier="talker")
    await collect(llm, r1, clock)
    assert seen == ["talker"]
    assert llm.requests == [r1]


async def test_cancellation_mid_stream_propagates_and_leaves_no_timers() -> None:
    clock = VirtualClock()
    llm = ScriptedLLM(
        clock, [ScriptStep(None, text="one two three four five")], ttft_s=0.2, tokens_per_s=10
    )
    got: list[str] = []

    async def run() -> None:
        async for chunk in llm.stream(req("x")):
            if isinstance(chunk, TextDelta):
                got.append(chunk.text)

    task = asyncio.create_task(run())
    await clock.advance(0.35)
    assert got == ["one ", "two "]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert clock.pending_timers() == 0
    assert llm.cancelled == 1
    await clock.run_until_idle()
    assert got == ["one ", "two "]


async def test_aclose_mid_stream_is_clean() -> None:
    clock = VirtualClock()
    llm = ScriptedLLM(clock, [ScriptStep(None, text="a b c")], ttft_s=0.0, tokens_per_s=10)
    gen = llm.stream(req("x"))

    async def first() -> BrainChunk:
        return await anext(gen)

    t = asyncio.create_task(first())
    await clock.run_until_idle()
    assert await t == TextDelta("a ")
    await gen.aclose()
    assert clock.pending_timers() == 0
