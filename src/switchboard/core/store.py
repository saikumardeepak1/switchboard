"""Append-only event store with optimistic concurrency (ADR-002).

Both implementations assign `seq` (1-based, gapless per conversation) and `event_id` on append.
Event ids derive from `conversation_id` and `seq` alone, so the same log written to either store,
in any process, gets the same ids. That keeps simulation runs byte-identical per seed without
threading an rng through every writer.

`read_all` yields events in global append order. `since` is inclusive (`at >= since`).
"""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import threading
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, Protocol

from switchboard.core.events import AnyEvent, NewEvent, parse_event_json

_PAGE_SIZE = 500


class ConcurrencyError(Exception):
    """`expected_seq` did not match the conversation's last seq. Nothing was written."""

    def __init__(self, conversation_id: str, expected_seq: int, actual_seq: int | None) -> None:
        self.conversation_id = conversation_id
        self.expected_seq = expected_seq
        self.actual_seq = actual_seq
        actual = "unknown" if actual_seq is None else str(actual_seq)
        super().__init__(
            f"conversation {conversation_id!r}: expected seq {expected_seq}, found {actual}"
        )


class EventStore(Protocol):
    async def append(
        self, conversation_id: str, events: Sequence[NewEvent], expected_seq: int
    ) -> list[AnyEvent]: ...

    async def read(self, conversation_id: str, after_seq: int = 0) -> list[AnyEvent]: ...

    def read_all(
        self, *, types: set[str] | None = None, since: float | None = None
    ) -> AsyncIterator[AnyEvent]: ...

    async def close(self) -> None: ...


def event_id_for(conversation_id: str, seq: int) -> str:
    digest = hashlib.sha256(f"{conversation_id}\x00{seq}".encode()).hexdigest()
    return f"evt_{digest[:16]}"


def _assign(conversation_id: str, events: Sequence[NewEvent], expected_seq: int) -> list[AnyEvent]:
    out: list[AnyEvent] = []
    for offset, event in enumerate(events, start=1):
        if event.conversation_id != conversation_id:
            raise ValueError(f"event for {event.conversation_id!r} appended to {conversation_id!r}")
        seq = expected_seq + offset
        out.append(
            event.model_copy(update={"seq": seq, "event_id": event_id_for(conversation_id, seq)})
        )
    return out


def _matches(event: AnyEvent, types: set[str] | None, since: float | None) -> bool:
    if types is not None and event.type not in types:
        return False
    return since is None or event.at >= since


class InMemoryEventStore:
    def __init__(self) -> None:
        self._by_conversation: dict[str, list[AnyEvent]] = {}
        self._global: list[AnyEvent] = []

    async def append(
        self, conversation_id: str, events: Sequence[NewEvent], expected_seq: int
    ) -> list[AnyEvent]:
        # No await between the check and the write, so this is atomic on the event loop.
        log = self._by_conversation.get(conversation_id, [])
        if len(log) != expected_seq:
            raise ConcurrencyError(conversation_id, expected_seq, len(log))
        assigned = _assign(conversation_id, events, expected_seq)
        if assigned:
            self._by_conversation.setdefault(conversation_id, []).extend(assigned)
            self._global.extend(assigned)
        return assigned

    async def read(self, conversation_id: str, after_seq: int = 0) -> list[AnyEvent]:
        return list(self._by_conversation.get(conversation_id, [])[max(after_seq, 0) :])

    async def read_all(
        self, *, types: set[str] | None = None, since: float | None = None
    ) -> AsyncIterator[AnyEvent]:
        for event in list(self._global):
            if _matches(event, types, since):
                yield event

    async def close(self) -> None:
        return None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    event_id TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL,
    at REAL NOT NULL,
    payload TEXT NOT NULL,
    UNIQUE (conversation_id, seq)
);
CREATE INDEX IF NOT EXISTS events_type_at ON events (type, at);
"""


class SqliteEventStore:
    """SQLite-backed store. Blocking calls run in a worker thread via `asyncio.to_thread`.

    One connection is shared behind a lock, so `":memory:"` works too.
    """

    def __init__(self, path: str | Path) -> None:
        self._lock = threading.Lock()
        self._con = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        with self._lock:
            self._con.execute("PRAGMA journal_mode=WAL")
            self._con.execute("PRAGMA synchronous=NORMAL")
            self._con.executescript(_SCHEMA)

    async def append(
        self, conversation_id: str, events: Sequence[NewEvent], expected_seq: int
    ) -> list[AnyEvent]:
        return await asyncio.to_thread(self._append, conversation_id, list(events), expected_seq)

    def _append(
        self, conversation_id: str, events: list[NewEvent], expected_seq: int
    ) -> list[AnyEvent]:
        assigned = _assign(conversation_id, events, expected_seq)
        rows = [
            (e.conversation_id, e.seq, e.event_id, e.type, e.at, e.model_dump_json())
            for e in assigned
        ]
        with self._lock:
            self._con.execute("BEGIN IMMEDIATE")
            try:
                (last,) = self._con.execute(
                    "SELECT COALESCE(MAX(seq), 0) FROM events WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                if last != expected_seq:
                    raise ConcurrencyError(conversation_id, expected_seq, int(last))
                self._con.executemany(
                    "INSERT INTO events (conversation_id, seq, event_id, type, at, payload)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    rows,
                )
            except sqlite3.IntegrityError as exc:
                self._con.execute("ROLLBACK")
                raise ConcurrencyError(conversation_id, expected_seq, None) from exc
            except BaseException:
                self._con.execute("ROLLBACK")
                raise
            self._con.execute("COMMIT")
        return assigned

    async def read(self, conversation_id: str, after_seq: int = 0) -> list[AnyEvent]:
        rows = await asyncio.to_thread(
            self._query,
            "SELECT payload FROM events WHERE conversation_id = ? AND seq > ? ORDER BY seq",
            (conversation_id, after_seq),
        )
        return [parse_event_json(payload) for (payload,) in rows]

    async def read_all(
        self, *, types: set[str] | None = None, since: float | None = None
    ) -> AsyncIterator[AnyEvent]:
        if types is not None and not types:
            return
        where = ["id > ?"]
        filters: list[Any] = []
        if types is not None:
            where.append(f"type IN ({', '.join('?' for _ in types)})")
            filters.extend(sorted(types))
        if since is not None:
            where.append("at >= ?")
            filters.append(since)
        sql = (
            f"SELECT id, payload FROM events WHERE {' AND '.join(where)}"
            f" ORDER BY id LIMIT {_PAGE_SIZE}"
        )
        cursor = 0
        while True:
            rows = await asyncio.to_thread(self._query, sql, (cursor, *filters))
            for row_id, payload in rows:
                cursor = row_id
                yield parse_event_json(payload)
            if len(rows) < _PAGE_SIZE:
                return

    def _query(self, sql: str, params: tuple[Any, ...]) -> list[Any]:
        with self._lock:
            return self._con.execute(sql, params).fetchall()

    async def close(self) -> None:
        await asyncio.to_thread(self._close)

    def _close(self) -> None:
        with self._lock:
            self._con.close()
