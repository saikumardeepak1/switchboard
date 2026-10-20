"""Turn a finished conversation's events into memory writes (ADR-011).

Runs after the call ends, off the live path. `consolidate` is a pure function with a fixed
template, so replaying the same log always yields the same writes. `consolidate_with` is the hook
for an LLM summarizer: it rewrites only the summary text and leaves the structured fields, which
the grounding gate may later use as evidence, untouched.

Rules:

- always one interaction_summary: outcome, tools executed (first-execution order, deduplicated),
  handoff yes or no.
- an open_issue when the call ended in a handoff or with an outcome not in RESOLVED_OUTCOMES.
- otherwise a delete of the customer's prior open_issue memories, because the call resolved them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

from switchboard.core.events import (
    ConversationEnded,
    ConversationStarted,
    Event,
    HandoffRequested,
    ToolExecuted,
)
from switchboard.memory.store import MemoryKind, MemoryWrite

RESOLVED_OUTCOMES = frozenset({"resolved", "completed", "success"})


class Summarizer(Protocol):
    async def summarize(self, events: Sequence[Event], draft: str) -> str:
        """Return a better summary of the conversation. `draft` is the template text."""
        ...


def consolidate(events: Sequence[Event], now: float) -> list[MemoryWrite]:
    started = next((e for e in events if isinstance(e, ConversationStarted)), None)
    if started is None:
        raise ValueError("no conversation.started event; cannot scope memory to a customer")
    ended = next((e for e in reversed(events) if isinstance(e, ConversationEnded)), None)
    if ended is None:
        raise ValueError("conversation has not ended; consolidate only finished conversations")

    tools = list(dict.fromkeys(e.tool for e in events if isinstance(e, ToolExecuted)))
    handoffs = [e.reason for e in events if isinstance(e, HandoffRequested)]
    handoff = bool(handoffs)
    outcome = ended.outcome
    tenant, customer, conv = started.tenant_id, started.customer_id, started.conversation_id

    text = (
        f"Outcome: {outcome}. Tools executed: {', '.join(tools) or 'none'}. "
        f"Handoff: {'yes' if handoff else 'no'}."
    )
    writes = [
        MemoryWrite.put(
            tenant,
            customer,
            MemoryKind.INTERACTION_SUMMARY,
            {
                "conversation_id": conv,
                "outcome": outcome,
                "tools": tools,
                "handoff": handoff,
                "ended_at": ended.at,
                "text": text,
            },
            at=now,
        )
    ]

    if handoff or outcome not in RESOLVED_OUTCOMES:
        reason = handoffs[-1] if handoffs else outcome
        writes.append(
            MemoryWrite.put(
                tenant,
                customer,
                MemoryKind.OPEN_ISSUE,
                {
                    "conversation_id": conv,
                    "outcome": outcome,
                    "reason": reason,
                    "text": f"Open issue from {conv}: {reason}.",
                },
                at=now,
            )
        )
    else:
        writes.append(MemoryWrite.delete(tenant, customer, MemoryKind.OPEN_ISSUE))
    return writes


async def consolidate_with(
    events: Sequence[Event], now: float, summarizer: Summarizer
) -> list[MemoryWrite]:
    writes = consolidate(events, now)
    summary = writes[0]
    text = await summarizer.summarize(events, summary.content["text"])
    writes[0] = replace(summary, content={**summary.content, "text": text})
    return writes
