"""The LLM protocol and a deterministic scripted fake.

`ScriptedLLM` stands in for a real model in tests and simulations. It runs on the injected
clock, so a test can say "the first token lands at 0.4 s" and check it exactly. Scripts can
contain wrong numbers and policy-violating tool calls on purpose, which is how the policy
checker and grounding gate get exercised end to end.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from switchboard.brain.types import (
    BrainChunk,
    BrainRequest,
    Done,
    Message,
    TextDelta,
    ToolCallChunk,
    Usage,
)
from switchboard.core.clock import Clock


class LLM(Protocol):
    """A streaming model.

    `stream` is a plain method returning an async iterator, so an `async def` generator
    satisfies it. Cancelling the consumer, or calling `aclose()` on the returned generator,
    must stop the provider without leaving tasks or timers behind.
    """

    def stream(self, request: BrainRequest) -> AsyncIterator[BrainChunk]: ...


@dataclass(frozen=True, slots=True)
class Reply:
    """What the scripted model says: text first, then any tool calls."""

    text: str = ""
    tool_calls: tuple[ToolCallChunk, ...] = ()


Matcher = str | Callable[[Message], bool] | None


@dataclass(frozen=True, slots=True)
class ScriptStep:
    """A reply guarded by a matcher on the last user or tool message.

    A string matches as a case-insensitive substring of that message's content, a callable is
    a predicate on the message, and None matches anything (put it last as a fallback). A step
    with `once=True` is used at most one time, so a list can script a conversation in order.
    """

    match: Matcher
    text: str = ""
    tool_calls: tuple[ToolCallChunk, ...] = ()
    once: bool = False

    @property
    def reply(self) -> Reply:
        return Reply(text=self.text, tool_calls=self.tool_calls)


Script = Sequence[ScriptStep] | Callable[[BrainRequest], Reply]


class ScriptExhaustedError(LookupError):
    """No script step matched the request."""


_WORD = re.compile(r"\s*\S+\s*")


def split_words(text: str) -> list[str]:
    """Split into word-sized deltas that concatenate back to `text` exactly."""
    return _WORD.findall(text)


def approx_tokens(text: str) -> int:
    # Deterministic stand-in for a tokenizer: one token per word.
    return len(text.split())


class ScriptedLLM:
    def __init__(
        self,
        clock: Clock,
        script: Script,
        ttft_s: float = 0.3,
        tokens_per_s: float = 40.0,
    ) -> None:
        if ttft_s < 0:
            raise ValueError("ttft_s must be non-negative")
        if tokens_per_s <= 0:
            raise ValueError("tokens_per_s must be positive")
        self._clock = clock
        self._script = script
        self._used: set[int] = set()
        self.ttft_s = ttft_s
        self.tokens_per_s = tokens_per_s
        self.requests: list[BrainRequest] = []
        self.completed = 0
        self.cancelled = 0

    def _reply_for(self, request: BrainRequest) -> Reply:
        if callable(self._script):
            return self._script(request)
        last = next((m for m in reversed(request.messages) if m.role in ("user", "tool")), None)
        for index, step in enumerate(self._script):
            if step.once and index in self._used:
                continue
            if _matches(step.match, last):
                if step.once:
                    self._used.add(index)
                return step.reply
        content = last.content if last else None
        raise ScriptExhaustedError(f"no script step matches the last message: {content!r}")

    async def stream(self, request: BrainRequest) -> AsyncGenerator[BrainChunk, None]:
        self.requests.append(request)
        reply = self._reply_for(request)
        interval = 1.0 / self.tokens_per_s
        words = split_words(reply.text)
        output_tokens = 0
        try:
            await self._clock.sleep(self.ttft_s)
            for i, word in enumerate(words):
                if i:
                    await self._clock.sleep(interval)
                output_tokens += 1
                yield TextDelta(word)
            for call in reply.tool_calls:
                output_tokens += 1
                yield call
            input_tokens = sum(approx_tokens(m.content) for m in request.messages)
            input_tokens += sum(approx_tokens(t.description) + 1 for t in request.tools)
            yield Done(
                usage=Usage(
                    input_tokens=input_tokens,
                    cached_input_tokens=0,
                    output_tokens=output_tokens,
                ),
                finish_reason="tool_use" if reply.tool_calls else "end_turn",
            )
            self.completed += 1
        except asyncio.CancelledError:
            self.cancelled += 1
            raise


def _matches(matcher: Matcher, message: Message | None) -> bool:
    if matcher is None:
        return True
    if message is None:
        return False
    if isinstance(matcher, str):
        return matcher.lower() in message.content.lower()
    return matcher(message)
