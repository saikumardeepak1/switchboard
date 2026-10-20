"""Scoped, typed, expiring customer memory and post-call consolidation (ADR-011)."""

from switchboard.memory.consolidate import (
    RESOLVED_OUTCOMES,
    Summarizer,
    consolidate,
    consolidate_with,
)
from switchboard.memory.store import (
    DEFAULT_TTL_S,
    MemoryItem,
    MemoryKind,
    MemoryStore,
    MemoryWrite,
)

__all__ = [
    "DEFAULT_TTL_S",
    "RESOLVED_OUTCOMES",
    "MemoryItem",
    "MemoryKind",
    "MemoryStore",
    "MemoryWrite",
    "Summarizer",
    "consolidate",
    "consolidate_with",
]
