"""Scoped, typed, expiring customer memory (ADR-011).

Every item is keyed by the pair (tenant_id, customer_id). There is no API that reads across that
key, and the key is a tuple rather than a joined string, so "a:b" + "c" can never alias
"a" + "b:c". A customer id that exists under two tenants is two unrelated customers.

TTLs per kind (seconds), chosen for a delivery-support deployment:

- profile, 365 days: name, language, address hints; refreshed whenever the customer calls.
- preference, 180 days: contact channel, delivery instructions; long-lived but can go stale.
- interaction_summary, 90 days: one per call; enough history for "you called last week".
- open_issue, 14 days: an unresolved problem; long enough to span a follow-up, short enough
  that a forgotten issue does not haunt every future call.

Capacity: each customer holds at most `max_items_per_customer`. On overflow, expired items go
first, then the oldest interaction summaries, then (only if no summaries remain) the oldest item
of any kind. Summaries are the cheapest to lose: each is one call's recap.

`forget` is a hard delete: rows are removed, not tombstoned.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

_DAY = 86_400.0


class MemoryKind(StrEnum):
    PROFILE = "profile"
    OPEN_ISSUE = "open_issue"
    PREFERENCE = "preference"
    INTERACTION_SUMMARY = "interaction_summary"


DEFAULT_TTL_S: dict[MemoryKind, float] = {
    MemoryKind.PROFILE: 365 * _DAY,
    MemoryKind.OPEN_ISSUE: 14 * _DAY,
    MemoryKind.PREFERENCE: 180 * _DAY,
    MemoryKind.INTERACTION_SUMMARY: 90 * _DAY,
}


@dataclass(frozen=True)
class MemoryItem:
    memory_id: str
    tenant_id: str
    customer_id: str
    kind: MemoryKind
    content: dict[str, Any]
    written_at: float
    expires_at: float

    def live_at(self, now: float) -> bool:
        return now < self.expires_at


@dataclass(frozen=True)
class MemoryWrite:
    """A pending change, produced by consolidation and applied with `MemoryStore.apply`.

    `op="put"` writes one item. `op="delete"` removes every item of `kind` for the customer.
    """

    op: Literal["put", "delete"]
    tenant_id: str
    customer_id: str
    kind: MemoryKind
    content: dict[str, Any] = field(default_factory=dict)
    at: float = 0.0
    ttl_s: float | None = None

    @classmethod
    def put(
        cls,
        tenant_id: str,
        customer_id: str,
        kind: MemoryKind,
        content: dict[str, Any],
        *,
        at: float,
        ttl_s: float | None = None,
    ) -> MemoryWrite:
        ttl = DEFAULT_TTL_S[kind] if ttl_s is None else ttl_s
        return cls("put", tenant_id, customer_id, kind, content, at, ttl)

    @classmethod
    def delete(cls, tenant_id: str, customer_id: str, kind: MemoryKind) -> MemoryWrite:
        return cls("delete", tenant_id, customer_id, kind)


_Key = tuple[str, str]


def _key(tenant_id: str, customer_id: str) -> _Key:
    if not tenant_id or not customer_id:
        raise ValueError("tenant_id and customer_id are both required")
    return (tenant_id, customer_id)


class MemoryStore:
    """In-memory implementation. Items per customer are kept in write order."""

    def __init__(self, max_items_per_customer: int = 50) -> None:
        if max_items_per_customer < 1:
            raise ValueError("max_items_per_customer must be at least 1")
        self._max = max_items_per_customer
        self._items: dict[_Key, list[MemoryItem]] = {}
        self._seq = 0
        self._lock = threading.Lock()

    def write(
        self,
        tenant_id: str,
        customer_id: str,
        kind: MemoryKind,
        content: dict[str, Any],
        *,
        now: float,
        ttl_s: float | None = None,
    ) -> MemoryItem:
        key = _key(tenant_id, customer_id)
        ttl = DEFAULT_TTL_S[kind] if ttl_s is None else ttl_s
        if ttl <= 0:
            raise ValueError("ttl_s must be positive")
        with self._lock:
            self._seq += 1
            digest = hashlib.sha256(f"{tenant_id}\x00{customer_id}\x00{self._seq}".encode())
            item = MemoryItem(
                memory_id=f"mem_{digest.hexdigest()[:16]}",
                tenant_id=tenant_id,
                customer_id=customer_id,
                kind=kind,
                content=dict(content),
                written_at=now,
                expires_at=now + ttl,
            )
            items = self._items.setdefault(key, [])
            items.append(item)
            self._evict(items, now)
        return item

    def read(self, tenant_id: str, customer_id: str, now: float) -> list[MemoryItem]:
        key = _key(tenant_id, customer_id)
        with self._lock:
            return [m for m in self._items.get(key, []) if m.live_at(now)]

    def forget(self, tenant_id: str, customer_id: str) -> int:
        """Hard delete everything for the customer. Returns the number of rows removed."""
        key = _key(tenant_id, customer_id)
        with self._lock:
            removed = self._items.pop(key, [])
        assert key not in self._items
        return len(removed)

    def delete_kind(self, tenant_id: str, customer_id: str, kind: MemoryKind) -> int:
        key = _key(tenant_id, customer_id)
        with self._lock:
            items = self._items.get(key, [])
            kept = [m for m in items if m.kind is not kind]
            removed = len(items) - len(kept)
            if kept:
                self._items[key] = kept
            else:
                self._items.pop(key, None)
        return removed

    def purge_expired(self, now: float) -> int:
        removed = 0
        with self._lock:
            for key in list(self._items):
                items = self._items[key]
                kept = [m for m in items if m.live_at(now)]
                removed += len(items) - len(kept)
                if kept:
                    self._items[key] = kept
                else:
                    del self._items[key]
        return removed

    def count(self, tenant_id: str, customer_id: str, *, now: float | None = None) -> int:
        """Row count for audits. Without `now` it counts every row still physically stored,
        expired or not, which is what a deletion check must look at."""
        key = _key(tenant_id, customer_id)
        with self._lock:
            items = self._items.get(key, [])
            return len(items) if now is None else sum(1 for m in items if m.live_at(now))

    def apply(self, writes: Iterable[MemoryWrite]) -> None:
        for w in writes:
            if w.op == "delete":
                self.delete_kind(w.tenant_id, w.customer_id, w.kind)
            else:
                self.write(w.tenant_id, w.customer_id, w.kind, w.content, now=w.at, ttl_s=w.ttl_s)

    def _evict(self, items: list[MemoryItem], now: float) -> None:
        if len(items) <= self._max:
            return
        items[:] = [m for m in items if m.live_at(now)]
        while len(items) > self._max:
            summaries = [m for m in items if m.kind is MemoryKind.INTERACTION_SUMMARY]
            victim = min(summaries or items, key=lambda m: m.written_at)
            items.remove(victim)
