"""Cost model: per-minute charges from events plus metered model usage."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

import pytest

from switchboard.brain import BrainRequest, Message, Reply, ScriptedLLM, ToolSchema
from switchboard.core import (
    AnyEvent,
    ConversationEnded,
    ConversationStarted,
    LegClosed,
    LegOpened,
    SpeechSentenceGated,
    VirtualClock,
)
from switchboard.sim import CallUsage, CostConfig, MeteredLLM, ModelPrice, PrefixCache, compute_cost


def ev(cls: type[Any], seq: int, at: float, leg: str | None = "leg_1", **kw: Any) -> AnyEvent:
    return cls(  # type: ignore[no-any-return]
        conversation_id="c", leg_id=leg, seq=seq, at=at, bundle_id="b", event_id=f"e{seq}", **kw
    )


def call_log() -> list[AnyEvent]:
    """Two minutes of conversation; the child leg is open for 30 s of it."""
    return [
        ev(ConversationStarted, 1, 0.0, None, tenant_id="t", customer_id="u"),
        ev(LegOpened, 2, 0.0, role="inbound", party="courier"),
        ev(LegOpened, 3, 30.0, "leg_2", role="outbound_child", party="customer"),
        ev(
            SpeechSentenceGated,
            4,
            31.0,
            "leg_2",
            utterance_id="t1/u0",
            sentence="x" * 600,
            decision="release",
            tier=2,
            claims=[],
        ),
        ev(
            SpeechSentenceGated,
            5,
            40.0,
            utterance_id="t2/u0",
            sentence="y" * 400,
            decision="escalate",
            tier=2,
            claims=[],
        ),
        ev(
            SpeechSentenceGated,
            6,
            41.0,
            utterance_id="t2/u0",
            sentence="y" * 400,
            decision="release",
            tier=3,
            claims=[],
        ),
        ev(
            SpeechSentenceGated,
            7,
            42.0,
            utterance_id="t2/u0",
            sentence="z" * 999,
            decision="block",
            tier=1,
            claims=[],
        ),
        ev(LegClosed, 8, 60.0, "leg_2", reason="completed"),
        ev(LegClosed, 9, 120.0, reason="conversation_ended"),
        ev(ConversationEnded, 10, 120.0, None, outcome="resolved"),
    ]


CONFIG = CostConfig(
    stt_per_min=0.01,
    tts_per_1k_chars=0.05,
    telephony_per_min=0.02,
    models={"talker": ModelPrice(1.0, 5.0), "thinker": ModelPrice(3.0, 15.0)},
    cached_input_multiplier=0.1,
    cache_write_multiplier=1.25,
    tokens_per_word=1.0,
)


def test_media_costs_are_per_leg_minute_and_tts_per_released_char() -> None:
    report = compute_cost(call_log(), [], CONFIG)
    assert report.minutes == pytest.approx(2.0)
    assert report.leg_minutes == pytest.approx(2.5)
    assert report.stt == pytest.approx(2.5 * 0.01)
    assert report.telephony == pytest.approx(2.5 * 0.02)
    # 600 released plus 400 escalated once. The tier 3 re-decision and the block add nothing.
    assert report.tts_chars == 1000
    assert report.tts == pytest.approx(0.05)
    assert report.llm == 0.0
    assert report.per_minute == pytest.approx(report.total / 2.0)


def test_llm_cost_prices_cached_and_cache_write_tokens() -> None:
    usages = [
        CallUsage(
            "thinker",
            input_tokens=1_000_000,
            cached_input_tokens=0,
            cache_write_tokens=400_000,
            output_tokens=100_000,
        ),
        CallUsage(
            "thinker",
            input_tokens=1_000_000,
            cached_input_tokens=400_000,
            cache_write_tokens=0,
            output_tokens=0,
        ),
        CallUsage(
            "talker",
            input_tokens=2_000_000,
            cached_input_tokens=0,
            cache_write_tokens=0,
            output_tokens=1_000_000,
            cancelled=True,
        ),
    ]
    report = compute_cost(call_log(), usages, CONFIG)
    first = 0.6 * 3.0 + 0.4 * 3.0 * 1.25 + 0.1 * 15.0
    second = 0.6 * 3.0 + 0.4 * 3.0 * 0.1
    talker = 2.0 * 1.0 + 1.0 * 5.0
    assert report.llm_by_model == pytest.approx({"thinker": first + second, "talker": talker})
    assert report.llm == pytest.approx(first + second + talker)
    assert report.wasted_llm == pytest.approx(talker)
    assert report.llm_calls == {"thinker": 2, "talker": 1}


def test_tokens_per_word_scales_scripted_token_counts() -> None:
    usages = [CallUsage("talker", 1_000_000, 0, 0, 0)]
    scaled = compute_cost(call_log(), usages, dataclasses.replace(CONFIG, tokens_per_word=1.3))
    assert scaled.llm == pytest.approx(1.3)


def test_prefix_cache_modes() -> None:
    cache = PrefixCache("auto", ttl_s=300.0)
    assert cache.lookup("thinker", "h1", now=0.0) == "write"
    assert cache.lookup("thinker", "h1", now=100.0) == "hit"
    assert cache.lookup("talker", "h1", now=100.0) == "write", "caches are per model"
    assert cache.lookup("thinker", "h1", now=350.0) == "hit", "a hit refreshes the ttl"
    assert cache.lookup("thinker", "h1", now=900.0) == "write"
    assert PrefixCache("always_hit").lookup("m", "h", 0.0) == "hit"
    assert PrefixCache("never").lookup("m", "h", 0.0) == "none"


TOOLS = (ToolSchema("lookup_order", "Read an order.", {}),)


def req(text: str) -> BrainRequest:
    return BrainRequest(
        messages=(Message("system", "stable prefix of five words"), Message("user", text)),
        tools=TOOLS,
        metadata={"prefix_hash": "h1"},
    )


async def test_metered_llm_records_usage_with_cache_and_cancellation() -> None:
    clock = VirtualClock()
    inner = ScriptedLLM(clock, lambda r: Reply("one two three four"), ttft_s=0.1, tokens_per_s=10)
    llm = MeteredLLM(inner, "thinker", PrefixCache("auto"), clock)

    async def full() -> None:
        async for _ in llm.stream(req("hi there")):
            pass

    async def cut() -> None:
        stream = llm.stream(req("hi again"))
        async for _ in stream:
            break
        await stream.aclose()  # type: ignore[attr-defined]

    for job in (full, cut):
        task = asyncio.ensure_future(job())
        await clock.run_until_idle()
        task.result()

    first, second = llm.usages
    # prefix: 5 words of system text plus the tool description (3 words) and one per tool
    assert (first.input_tokens, first.cache_write_tokens, first.cached_input_tokens) == (11, 9, 0)
    assert (first.output_tokens, first.cancelled) == (4, False)
    assert (second.cached_input_tokens, second.cache_write_tokens) == (9, 0)
    assert (second.output_tokens, second.cancelled) == (1, True)
