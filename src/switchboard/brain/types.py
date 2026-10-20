"""Value types shared by every brain implementation (ADR-001, ADR-008).

Messages are provider-neutral. Adapters translate them into a provider's wire format, so the
rest of the system never sees a vendor type.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

Role = Literal["system", "user", "assistant", "tool"]
Tier = Literal["talker", "thinker"]

ROLES: frozenset[str] = frozenset(get_args(Role))
TIERS: frozenset[str] = frozenset(get_args(Tier))


@dataclass(frozen=True, slots=True)
class Message:
    """One conversation message.

    A `tool` message carries a tool result and must name the call it answers. An `assistant`
    message with `tool_call_id` set records a tool call the model made: `name` is the tool and
    `content` is the canonical JSON of its arguments.
    """

    role: Role
    content: str
    tool_call_id: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}; expected one of {sorted(ROLES)}")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("a tool message needs the tool_call_id it answers")


@dataclass(frozen=True, slots=True)
class ToolSchema:
    name: str
    description: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class BrainRequest:
    messages: tuple[Message, ...]
    tools: tuple[ToolSchema, ...] = ()
    tier: Tier = "thinker"
    max_tokens: int = 1024
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"unknown tier {self.tier!r}; expected one of {sorted(TIERS)}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ToolCallChunk:
    """A complete tool call. Providers that stream arguments buffer them until valid."""

    call_id: str
    name: str
    args: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Done:
    usage: Usage
    finish_reason: str = "end_turn"


BrainChunk = TextDelta | ToolCallChunk | Done
