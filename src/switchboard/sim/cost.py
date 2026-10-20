"""Per-minute cost of a simulated call, from its events plus metered model usage.

Every price below is an ASSUMPTION: a public list price at the time of writing, rounded, for
order-of-magnitude comparison between architectures. None of them is a quote, none includes
volume discounts, and they drift. Change them in `CostConfig`, not in code.

What is billed and how:

- STT and telephony per leg-minute. A leg streams audio from `leg.opened` to `leg.closed` (or the
  end of the conversation), so a two-party call pays for both legs.
- TTS per character synthesized: every sentence whose first gate decision released it or sent it
  to tier 3 (escalated sentences are synthesized while the verifier runs). Blocked sentences are
  never synthesized.
- LLM per token, by model, from `MeteredLLM` records. Cancelled calls (abandoned speculation,
  barge-in) still pay for their input and for the output produced before the cancel.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from switchboard.brain import LLM, BrainChunk, BrainRequest, TextDelta, ToolCallChunk
from switchboard.brain.llm import approx_tokens
from switchboard.core import AnyEvent, Clock


@dataclass(frozen=True)
class ModelPrice:
    input_per_m: float
    output_per_m: float


DEFAULT_MODELS: Mapping[str, ModelPrice] = {
    # Talker: Claude Haiku 5.5 list price, $0.10 in / $0.50 out per 1M tokens
    # (Anthropic API pricing table, cached 2026-10-06). ASSUMPTION.
    "talker": ModelPrice(0.10, 0.50),
    # Thinker: Claude Sonnet 5.5 list price, $2 in / $10 out per 1M tokens (same source).
    # ASSUMPTION.
    "thinker": ModelPrice(2.00, 10.00),
}


@dataclass(frozen=True)
class CostConfig:
    # Deepgram Nova-3 streaming, pay as you go: about $0.0077 per audio minute. ASSUMPTION.
    stt_per_min: float = 0.0077
    # Between Cartesia Sonic (about $0.03 per 1k characters on paid plans) and ElevenLabs Flash
    # API (about $0.05 per 1k). ASSUMPTION.
    tts_per_1k_chars: float = 0.04
    # Twilio US programmable voice: about $0.014 per minute outbound, $0.0085 inbound local.
    # The higher figure is used for every leg. ASSUMPTION.
    telephony_per_min: float = 0.014
    models: Mapping[str, ModelPrice] = field(default_factory=lambda: dict(DEFAULT_MODELS))
    # Anthropic prompt caching: cache reads at 0.1x the input price, 5 minute cache writes at
    # 1.25x. ASSUMPTION (published multipliers at the time of writing).
    cached_input_multiplier: float = 0.1
    cache_write_multiplier: float = 1.25
    # ScriptedLLM counts one token per word. Real tokenizers produce more: roughly 1.3 per
    # English word, more for JSON tool schemas. ASSUMPTION.
    tokens_per_word: float = 1.3


@dataclass(frozen=True)
class CallUsage:
    model: str
    input_tokens: int
    cached_input_tokens: int
    cache_write_tokens: int
    output_tokens: int
    cancelled: bool = False


CacheMode = Literal["auto", "always_hit", "never"]
CacheStatus = Literal["hit", "write", "none"]


class PrefixCache:
    """Provider prefix cache keyed by (model, prefix hash) with a sliding TTL.

    `auto` behaves like the provider: the first call writes, later calls inside the TTL hit and
    refresh it. `always_hit` models a warm cache shared with other conversations on the same
    release bundle. `never` models caching switched off.
    """

    def __init__(self, mode: CacheMode = "auto", ttl_s: float = 300.0) -> None:
        self.mode = mode
        self.ttl_s = ttl_s
        self._expiry: dict[tuple[str, str], float] = {}

    def lookup(self, model: str, prefix_hash: str, now: float) -> CacheStatus:
        if self.mode == "never":
            return "none"
        if self.mode == "always_hit":
            return "hit"
        key = (model, prefix_hash)
        expiry = self._expiry.get(key)
        self._expiry[key] = now + self.ttl_s
        return "hit" if expiry is not None and now < expiry else "write"


def prompt_tokens(request: BrainRequest) -> tuple[int, int]:
    """(input tokens, stable prefix tokens), counted the way ScriptedLLM counts them."""
    tools = sum(approx_tokens(t.description) + 1 for t in request.tools)
    total = sum(approx_tokens(m.content) for m in request.messages) + tools
    prefix = (approx_tokens(request.messages[0].content) if request.messages else 0) + tools
    return total, prefix


class MeteredLLM:
    """Wraps an LLM and records one `CallUsage` per call, including cancelled ones."""

    def __init__(self, inner: LLM, model: str, cache: PrefixCache, clock: Clock) -> None:
        self.inner = inner
        self.model = model
        self.cache = cache
        self.clock = clock
        self.usages: list[CallUsage] = []

    async def stream(self, request: BrainRequest) -> AsyncIterator[BrainChunk]:
        total, prefix = prompt_tokens(request)
        status = self.cache.lookup(
            self.model, str(request.metadata.get("prefix_hash", "")), self.clock.now()
        )
        output = 0
        cancelled = True
        try:
            async with contextlib.aclosing(self.inner.stream(request)) as chunks:  # type: ignore[type-var]
                async for chunk in chunks:
                    if isinstance(chunk, TextDelta | ToolCallChunk):
                        output += 1
                    yield chunk
            cancelled = False
        finally:
            self.usages.append(
                CallUsage(
                    self.model,
                    input_tokens=total,
                    cached_input_tokens=prefix if status == "hit" else 0,
                    cache_write_tokens=prefix if status == "write" else 0,
                    output_tokens=output,
                    cancelled=cancelled,
                )
            )


@dataclass(frozen=True)
class CostReport:
    minutes: float
    leg_minutes: float
    stt: float
    tts: float
    telephony: float
    llm: float
    llm_by_model: Mapping[str, float]
    llm_calls: Mapping[str, int]
    wasted_llm: float
    tts_chars: int
    tokens: Mapping[str, int]

    @property
    def total(self) -> float:
        return self.stt + self.tts + self.telephony + self.llm

    @property
    def per_minute(self) -> float:
        return self.total / self.minutes if self.minutes > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "minutes": self.minutes,
            "leg_minutes": self.leg_minutes,
            "stt": self.stt,
            "tts": self.tts,
            "telephony": self.telephony,
            "llm": self.llm,
            "llm_by_model": dict(self.llm_by_model),
            "llm_calls": dict(self.llm_calls),
            "wasted_llm": self.wasted_llm,
            "tts_chars": self.tts_chars,
            "tokens": dict(self.tokens),
            "total": self.total,
            "per_minute": self.per_minute,
        }


def _usage_cost(u: CallUsage, config: CostConfig) -> float:
    price = config.models.get(u.model)
    if price is None:
        raise KeyError(f"no price for model {u.model!r}; known: {sorted(config.models)}")
    scale = config.tokens_per_word / 1_000_000
    plain = u.input_tokens - u.cached_input_tokens - u.cache_write_tokens
    input_cost = (
        price.input_per_m
        * scale
        * (
            plain
            + u.cached_input_tokens * config.cached_input_multiplier
            + u.cache_write_tokens * config.cache_write_multiplier
        )
    )
    return input_cost + price.output_per_m * scale * u.output_tokens


def compute_cost(
    events: Sequence[AnyEvent], usages: Iterable[CallUsage], config: CostConfig | None = None
) -> CostReport:
    config = config or CostConfig()
    start = next((e.at for e in events if e.type == "conversation.started"), None)
    end = next((e.at for e in reversed(events) if e.type == "conversation.ended"), None)
    if start is None:
        start = events[0].at if events else 0.0
    if end is None:
        end = events[-1].at if events else start
    opened: dict[str, float] = {}
    closed: dict[str, float] = {}
    seen: set[tuple[str, str]] = set()
    chars = 0
    for e in events:
        if e.type == "leg.opened" and e.leg_id:
            opened[e.leg_id] = e.at
        elif e.type == "leg.closed" and e.leg_id:
            closed.setdefault(e.leg_id, e.at)
        elif e.type == "speech.sentence_gated":
            sentence = getattr(e, "sentence", "")
            key = (getattr(e, "utterance_id", ""), sentence)
            if key in seen:
                continue
            seen.add(key)
            if getattr(e, "decision", "") in ("release", "escalate"):
                chars += len(sentence)
    leg_s = sum(closed.get(leg, end) - at for leg, at in opened.items())
    minutes = max(0.0, end - start) / 60.0
    leg_minutes = max(0.0, leg_s) / 60.0

    by_model: dict[str, float] = {}
    calls: dict[str, int] = {}
    tokens = {"input": 0, "cached_input": 0, "cache_write": 0, "output": 0}
    wasted = 0.0
    for u in usages:
        cost = _usage_cost(u, config)
        by_model[u.model] = by_model.get(u.model, 0.0) + cost
        calls[u.model] = calls.get(u.model, 0) + 1
        if u.cancelled:
            wasted += cost
        tokens["input"] += u.input_tokens
        tokens["cached_input"] += u.cached_input_tokens
        tokens["cache_write"] += u.cache_write_tokens
        tokens["output"] += u.output_tokens
    return CostReport(
        minutes=minutes,
        leg_minutes=leg_minutes,
        stt=leg_minutes * config.stt_per_min,
        tts=chars / 1000.0 * config.tts_per_1k_chars,
        telephony=leg_minutes * config.telephony_per_min,
        llm=sum(by_model.values()),
        llm_by_model=by_model,
        llm_calls=calls,
        wasted_llm=wasted,
        tts_chars=chars,
        tokens=tokens,
    )
