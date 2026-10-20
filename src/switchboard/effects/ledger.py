"""Idempotency ledger: at most one live execution per key, and the recorded result after (ADR-009).

Each key moves through `pending` (claimed by one attempt), then `done` (result recorded, final)
or `failed` (may be claimed again). `claim` is atomic:

- no entry, or a `failed` one: the caller gets `acquired` and a fresh token.
- `pending` within its lease: `in_flight`; someone else is running it.
- `pending` past its lease: the claimer is presumed dead and the caller gets `acquired`.
- `done`: the recorded result.

Tokens are the attempt number. `record` and `fail` only apply while the entry is still pending
under the caller's token, so an attempt that outlived its lease cannot overwrite the attempt that
replaced it. The effect runner keeps every write tool's timeout below the lease, so a live attempt
is cancelled before its claim can go stale.

Results are stored as JSON in both implementations, so they come back as equal copies and
anything that would not survive a restart fails here first.

Both ledgers do their work synchronously inside the coroutine. Each operation is one small
indexed statement, and doing it inline means no other coroutine can interleave between the check
and the write, and virtual-time tests stay deterministic. SQLite's `BEGIN IMMEDIATE` covers
other connections and processes sharing the file.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from switchboard.core.clock import Clock

EntryState = Literal["pending", "done", "failed"]
ClaimStatus = Literal["acquired", "in_flight", "done"]


@dataclass(frozen=True)
class ClaimResult:
    status: ClaimStatus
    token: int | None = None
    result: dict[str, Any] | None = None


@dataclass(frozen=True)
class LedgerEntry:
    key: str
    state: EntryState
    attempts: int
    claimed_at: float
    result: dict[str, Any] | None
    error: str | None


class IdempotencyLedger(Protocol):
    @property
    def lease_s(self) -> float: ...

    async def claim(self, key: str) -> ClaimResult: ...

    async def record(self, key: str, token: int, result: dict[str, Any]) -> bool: ...

    async def fail(self, key: str, token: int, error: str) -> bool: ...

    async def get(self, key: str) -> LedgerEntry | None: ...

    async def close(self) -> None: ...


def _check_lease(lease_s: float) -> float:
    if not lease_s > 0:
        raise ValueError(f"lease_s must be positive, got {lease_s}")
    return float(lease_s)


def _dump(result: dict[str, Any]) -> str:
    if not isinstance(result, dict):
        raise TypeError(f"ledger results must be dicts, got {type(result).__name__}")
    return json.dumps(result, sort_keys=True, allow_nan=False)


def _load(raw: str | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    loaded: dict[str, Any] = json.loads(raw)
    return loaded


@dataclass
class _Row:
    state: EntryState
    attempts: int
    claimed_at: float
    result: str | None = None
    error: str | None = None


class InMemoryLedger:
    def __init__(self, clock: Clock, lease_s: float = 60.0) -> None:
        self._clock = clock
        self._lease_s = _check_lease(lease_s)
        self._rows: dict[str, _Row] = {}

    @property
    def lease_s(self) -> float:
        return self._lease_s

    async def claim(self, key: str) -> ClaimResult:
        now = self._clock.now()
        row = self._rows.get(key)
        if row is None:
            self._rows[key] = _Row(state="pending", attempts=1, claimed_at=now)
            return ClaimResult("acquired", token=1)
        if row.state == "done":
            return ClaimResult("done", result=_load(row.result))
        if row.state == "pending" and now < row.claimed_at + self._lease_s:
            return ClaimResult("in_flight")
        row.state, row.attempts, row.claimed_at, row.error = "pending", row.attempts + 1, now, None
        return ClaimResult("acquired", token=row.attempts)

    async def record(self, key: str, token: int, result: dict[str, Any]) -> bool:
        payload = _dump(result)
        row = self._rows.get(key)
        if row is None or row.state != "pending" or row.attempts != token:
            return False
        row.state, row.result = "done", payload
        return True

    async def fail(self, key: str, token: int, error: str) -> bool:
        row = self._rows.get(key)
        if row is None or row.state != "pending" or row.attempts != token:
            return False
        row.state, row.error = "failed", error
        return True

    async def get(self, key: str) -> LedgerEntry | None:
        row = self._rows.get(key)
        if row is None:
            return None
        return LedgerEntry(
            key, row.state, row.attempts, row.claimed_at, _load(row.result), row.error
        )

    async def close(self) -> None:
        return None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS idempotency_ledger (
    key TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL CHECK (state IN ('pending', 'done', 'failed')),
    attempts INTEGER NOT NULL,
    claimed_at REAL NOT NULL,
    result TEXT,
    error TEXT
);
"""


class SqliteLedger:
    """SQLite-backed ledger. Several instances, or processes, may share one file."""

    def __init__(self, path: str | Path, clock: Clock, lease_s: float = 60.0) -> None:
        self._clock = clock
        self._lease_s = _check_lease(lease_s)
        self._lock = threading.Lock()
        self._con = sqlite3.connect(
            str(path), check_same_thread=False, isolation_level=None, timeout=5.0
        )
        with self._lock:
            self._con.execute("PRAGMA journal_mode=WAL")
            self._con.executescript(_SCHEMA)

    @property
    def lease_s(self) -> float:
        return self._lease_s

    async def claim(self, key: str) -> ClaimResult:
        now = self._clock.now()
        with self._lock:
            self._con.execute("BEGIN IMMEDIATE")
            try:
                row = self._con.execute(
                    "SELECT state, attempts, claimed_at, result FROM idempotency_ledger"
                    " WHERE key = ?",
                    (key,),
                ).fetchone()
                if row is None:
                    self._con.execute(
                        "INSERT INTO idempotency_ledger (key, state, attempts, claimed_at)"
                        " VALUES (?, 'pending', 1, ?)",
                        (key, now),
                    )
                    outcome = ClaimResult("acquired", token=1)
                else:
                    state, attempts, claimed_at, result = row
                    if state == "done":
                        outcome = ClaimResult("done", result=_load(result))
                    elif state == "pending" and now < claimed_at + self._lease_s:
                        outcome = ClaimResult("in_flight")
                    else:
                        self._con.execute(
                            "UPDATE idempotency_ledger SET state = 'pending',"
                            " attempts = attempts + 1, claimed_at = ?, error = NULL"
                            " WHERE key = ?",
                            (now, key),
                        )
                        outcome = ClaimResult("acquired", token=attempts + 1)
            except BaseException:
                self._con.execute("ROLLBACK")
                raise
            self._con.execute("COMMIT")
        return outcome

    async def record(self, key: str, token: int, result: dict[str, Any]) -> bool:
        payload = _dump(result)
        return self._finish(
            "UPDATE idempotency_ledger SET state = 'done', result = ?"
            " WHERE key = ? AND state = 'pending' AND attempts = ?",
            (payload, key, token),
        )

    async def fail(self, key: str, token: int, error: str) -> bool:
        return self._finish(
            "UPDATE idempotency_ledger SET state = 'failed', error = ?"
            " WHERE key = ? AND state = 'pending' AND attempts = ?",
            (error, key, token),
        )

    def _finish(self, sql: str, params: tuple[Any, ...]) -> bool:
        with self._lock:
            return self._con.execute(sql, params).rowcount == 1

    async def get(self, key: str) -> LedgerEntry | None:
        with self._lock:
            row = self._con.execute(
                "SELECT state, attempts, claimed_at, result, error FROM idempotency_ledger"
                " WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        state, attempts, claimed_at, result, error = row
        return LedgerEntry(key, state, attempts, claimed_at, _load(result), error)

    async def close(self) -> None:
        with self._lock:
            self._con.close()
