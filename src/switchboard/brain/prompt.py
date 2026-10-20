"""Cache-friendly prompt assembly.

Provider prefix caches hit only on byte-identical prefixes, so the stable part of a prompt
(bundle system text, compiled policies, tool schemas) is rendered first and deterministically:
tools sorted by name, schemas as canonical JSON. Everything that changes per turn goes after
it. `prefix_hash` lets tests and telemetry prove that the prefix did not move.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from switchboard.brain.types import Message, ToolSchema


def canonical_json(value: Any) -> str:
    """Sorted keys, no insignificant whitespace, UTF-8 kept as is."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


@dataclass(frozen=True, slots=True)
class VolatileContext:
    evidence: Mapping[str, Any] = field(default_factory=dict)
    memory: tuple[str, ...] = ()
    knowledge: tuple[str, ...] = ()
    open_steps: tuple[str, ...] = ()
    messages: tuple[Message, ...] = ()


@dataclass(frozen=True, slots=True)
class AssembledPrompt:
    """`messages[:prefix_len]` is the stable prefix; the rest is per turn."""

    messages: tuple[Message, ...]
    tools: tuple[ToolSchema, ...]
    prefix_len: int
    prefix_hash: str


def assemble_prompt(
    bundle_system: str,
    policies_text: str,
    tool_schemas: Iterable[ToolSchema],
    volatile: VolatileContext,
) -> AssembledPrompt:
    tools = tuple(sorted(tool_schemas, key=lambda t: t.name))
    tool_doc = canonical_json(
        [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in tools]
    )
    prefix_text = f"{bundle_system}\n\n# Policies\n{policies_text}\n\n# Tools\n{tool_doc}"
    prefix = (Message(role="system", content=prefix_text),)
    prefix_hash = hashlib.sha256("".join(m.content for m in prefix).encode()).hexdigest()

    context = _render_context(volatile)
    suffix = (Message(role="system", content=context),) if context else ()
    return AssembledPrompt(
        messages=prefix + suffix + tuple(volatile.messages),
        tools=tools,
        prefix_len=len(prefix),
        prefix_hash=prefix_hash,
    )


def _render_context(volatile: VolatileContext) -> str:
    sections: list[str] = []
    if volatile.evidence:
        sections.append(f"# Evidence\n{canonical_json(dict(volatile.evidence))}")
    if volatile.open_steps:
        sections.append("# Open procedure steps\n" + "\n".join(volatile.open_steps))
    if volatile.memory:
        sections.append("# Customer memory\n" + "\n".join(volatile.memory))
    if volatile.knowledge:
        sections.append("# Current knowledge\n" + "\n".join(volatile.knowledge))
    return "\n\n".join(sections)
