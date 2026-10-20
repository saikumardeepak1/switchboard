"""Anthropic Messages API adapter.

The SDK is an optional extra, imported inside the constructor, so this module imports cleanly
without it. Everything that can be checked without a network call lives in two pure pieces:
`to_anthropic_params` (request mapping) and `StreamTranslator` (raw stream events to chunks).

Mapping conventions:
- System messages become top-level system blocks in order. The first block carries
  `cache_control`, because `assemble_prompt` puts the stable prefix first; tools render before
  system, so the cached prefix covers tool schemas too.
- An assistant message with `tool_call_id` is a tool call: `name` is the tool, `content` the
  JSON arguments. A tool message becomes a `tool_result` block in a user turn.
- Consecutive messages with the same API role are merged, which keeps the alternation the API
  requires and returns parallel tool results in one user message.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field
from typing import Any

from switchboard.brain.types import (
    BrainChunk,
    BrainRequest,
    Done,
    Message,
    TextDelta,
    Tier,
    ToolCallChunk,
    Usage,
)

DEFAULT_MODELS: Mapping[Tier, str] = {"talker": "claude-haiku-5-5", "thinker": "claude-opus-5-5"}
DEFAULT_EFFORT: Mapping[Tier, str] = {"talker": "low", "thinker": "medium"}

_INSTALL_HINT = (
    "AnthropicLLM needs the optional 'anthropic' package. "
    "Install it with: pip install 'switchboard[anthropic]' (or uv sync --extra anthropic)."
)


def to_anthropic_params(
    request: BrainRequest,
    model: str,
    effort: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"model": model, "max_tokens": request.max_tokens}
    system = [m.content for m in request.messages if m.role == "system"]
    if system:
        blocks: list[dict[str, Any]] = [{"type": "text", "text": text} for text in system]
        blocks[0]["cache_control"] = {"type": "ephemeral"}
        params["system"] = blocks
    if request.tools:
        params["tools"] = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": dict(t.parameters),
                # Arguments stream as generated; StreamTranslator validates the JSON.
                "eager_input_streaming": True,
            }
            for t in request.tools
        ]
    if effort is not None:
        params["output_config"] = {"effort": effort}
    params["messages"] = _map_messages(request.messages)
    return params


def _map_messages(messages: tuple[Message, ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            continue
        role, block = _block_for(m)
        if out and out[-1]["role"] == role:
            out[-1]["content"].append(block)
        else:
            out.append({"role": role, "content": [block]})
    return out


def _block_for(m: Message) -> tuple[str, dict[str, Any]]:
    if m.role == "tool":
        return "user", {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
    if m.role == "assistant" and m.tool_call_id is not None:
        return "assistant", {
            "type": "tool_use",
            "id": m.tool_call_id,
            "name": m.name,
            "input": json.loads(m.content) if m.content else {},
        }
    return m.role, {"type": "text", "text": m.content}


@dataclass
class _ToolBuffer:
    call_id: str
    name: str
    parts: list[str] = field(default_factory=list)


class StreamTranslator:
    """Turns raw Messages API stream events into BrainChunks. Pure and stateful per stream."""

    def __init__(self) -> None:
        self._tools: dict[int, _ToolBuffer] = {}
        self._input_tokens = 0
        self._cached_tokens = 0
        self._output_tokens = 0
        self._stop_reason = "end_turn"

    def feed(self, event: Any) -> list[BrainChunk]:
        kind = getattr(event, "type", None)
        if kind == "message_start":
            usage = event.message.usage
            self._input_tokens = getattr(usage, "input_tokens", 0) or 0
            self._cached_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
            self._output_tokens = getattr(usage, "output_tokens", 0) or 0
        elif kind == "content_block_start":
            block = event.content_block
            if block.type == "tool_use":
                self._tools[event.index] = _ToolBuffer(block.id, block.name)
        elif kind == "content_block_delta":
            delta = event.delta
            if delta.type == "text_delta" and delta.text:
                return [TextDelta(delta.text)]
            if delta.type == "input_json_delta" and event.index in self._tools:
                self._tools[event.index].parts.append(delta.partial_json)
        elif kind == "content_block_stop":
            buffer = self._tools.pop(event.index, None)
            if buffer is not None:
                return [self._tool_call(buffer)]
        elif kind == "message_delta":
            stop = getattr(event.delta, "stop_reason", None)
            if stop:
                self._stop_reason = stop
            usage = getattr(event, "usage", None)
            if usage is not None and getattr(usage, "output_tokens", None) is not None:
                self._output_tokens = usage.output_tokens
        elif kind == "message_stop":
            usage = Usage(self._input_tokens, self._cached_tokens, self._output_tokens)
            return [Done(usage=usage, finish_reason=self._stop_reason)]
        return []

    @staticmethod
    def _tool_call(buffer: _ToolBuffer) -> ToolCallChunk:
        raw = "".join(buffer.parts) or "{}"
        try:
            args = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"tool input for {buffer.name!r} is not valid JSON: {raw!r}") from exc
        if not isinstance(args, dict):
            raise ValueError(f"tool input for {buffer.name!r} is not an object: {raw!r}")
        return ToolCallChunk(call_id=buffer.call_id, name=buffer.name, args=args)


class AnthropicLLM:
    """LLM backed by the Anthropic SDK's async client. Not exercised in the offline suite."""

    def __init__(
        self,
        models: Mapping[Tier, str] = DEFAULT_MODELS,
        effort: Mapping[Tier, str] = DEFAULT_EFFORT,
        client: Any = None,
        api_key: str | None = None,
    ) -> None:
        try:
            sdk = importlib.import_module("anthropic")
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        self._client = client if client is not None else sdk.AsyncAnthropic(api_key=api_key)
        self._models = dict(models)
        self._effort = dict(effort)

    async def stream(self, request: BrainRequest) -> AsyncGenerator[BrainChunk, None]:
        params = to_anthropic_params(
            request, model=self._models[request.tier], effort=self._effort.get(request.tier)
        )
        translator = StreamTranslator()
        response = await self._client.messages.create(**params, stream=True)
        try:
            async for event in response:
                for chunk in translator.feed(event):
                    yield chunk
        finally:
            await response.close()
