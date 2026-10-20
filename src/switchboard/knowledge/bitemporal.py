"""Bitemporal fact store (ADR-010).

Every fact version carries two intervals:

- valid time `[valid_from, valid_to)`: when the fact is true in the world. `valid_to=None` means
  open ended.
- transaction time `[recorded_at, superseded_at)`: when this version was our belief.
  `superseded_at=None` means it is still our current belief.

Nothing is ever deleted or edited in place. `close` and a changed `assert_fact` add a new version
and stamp `superseded_at` on the old one; `retract` only stamps `superseded_at`. Stamping that one
field is the single write an old version ever receives, so any past belief can be reconstructed
with `query(as_of=..., known_at=...)`.

The API is synchronous on purpose. Reads happen at call setup and per turn against an in-process
store, where a SQLite lookup costs well under a millisecond; ingestion runs off the hot path. A
caller backed by a networked database can wrap calls in `asyncio.to_thread` without changing this
contract, and `KnowledgeIndex` stays a plain function of its inputs.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class Fact:
    fact_id: str
    subject: str
    predicate: str
    value: Any
    valid_from: float
    valid_to: float | None
    recorded_at: float
    superseded_at: float | None
    source: str

    def valid_at(self, as_of: float) -> bool:
        return self.valid_from <= as_of and (self.valid_to is None or as_of < self.valid_to)

    def known_at(self, known_at: float | None) -> bool:
        if known_at is None:
            return self.superseded_at is None
        return self.recorded_at <= known_at and (
            self.superseded_at is None or known_at < self.superseded_at
        )

    def same_content(self, other: Fact) -> bool:
        return (
            self.subject == other.subject
            and self.predicate == other.predicate
            and _canonical(self.value) == _canonical(other.value)
            and self.valid_from == other.valid_from
            and self.valid_to == other.valid_to
            and self.source == other.source
        )


class FactStore(Protocol):
    def assert_fact(
        self,
        subject: str,
        predicate: str,
        value: Any,
        valid_from: float,
        valid_to: float | None = None,
        *,
        recorded_at: float,
        source: str = "",
        fact_id: str | None = None,
    ) -> Fact: ...

    def close(self, fact_id: str, valid_to: float, recorded_at: float) -> Fact: ...

    def retract(self, fact_id: str, recorded_at: float) -> None: ...

    def query(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        *,
        as_of: float,
        known_at: float | None = None,
    ) -> list[Fact]: ...

    def current(self, fact_id: str) -> Fact | None: ...

    def history(self, fact_id: str) -> list[Fact]: ...

    def close_store(self) -> None: ...


def default_fact_id(
    subject: str, predicate: str, value: Any, valid_from: float, source: str
) -> str:
    """Content-derived id. Excludes valid_to and recorded_at so a closed or re-recorded fact
    keeps its identity."""
    payload = _canonical([subject, predicate, value, valid_from, source])
    return "fact_" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sort_key(fact: Fact) -> tuple[float, str]:
    return (fact.valid_from, fact.fact_id)


class _VersionedStore:
    """Write rules shared by every backend; subclasses supply four storage primitives."""

    def _atomic(self) -> AbstractContextManager[object]:
        raise NotImplementedError

    def current(self, fact_id: str) -> Fact | None:
        raise NotImplementedError

    def _insert(self, fact: Fact) -> None:
        raise NotImplementedError

    def _supersede(self, fact_id: str, at: float) -> None:
        raise NotImplementedError

    def assert_fact(
        self,
        subject: str,
        predicate: str,
        value: Any,
        valid_from: float,
        valid_to: float | None = None,
        *,
        recorded_at: float,
        source: str = "",
        fact_id: str | None = None,
    ) -> Fact:
        if valid_to is not None and valid_to <= valid_from:
            raise ValueError(f"valid_to {valid_to} must be after valid_from {valid_from}")
        _canonical(value)  # reject values that cannot be stored before touching state
        fid = fact_id or default_fact_id(subject, predicate, value, valid_from, source)
        new = Fact(
            fact_id=fid,
            subject=subject,
            predicate=predicate,
            value=value,
            valid_from=valid_from,
            valid_to=valid_to,
            recorded_at=recorded_at,
            superseded_at=None,
            source=source,
        )
        with self._atomic():
            existing = self.current(fid)
            if existing is not None:
                if existing.same_content(new):
                    return existing
                self._check_order(existing, recorded_at)
                self._supersede(fid, recorded_at)
            self._insert(new)
        return new

    def close(self, fact_id: str, valid_to: float, recorded_at: float) -> Fact:
        with self._atomic():
            existing = self._require(fact_id)
            if valid_to <= existing.valid_from:
                raise ValueError(f"valid_to {valid_to} must be after valid_from")
            if existing.valid_to == valid_to:
                return existing
            self._check_order(existing, recorded_at)
            new = replace(existing, valid_to=valid_to, recorded_at=recorded_at)
            self._supersede(fact_id, recorded_at)
            self._insert(new)
        return new

    def retract(self, fact_id: str, recorded_at: float) -> None:
        with self._atomic():
            existing = self._require(fact_id)
            self._check_order(existing, recorded_at)
            self._supersede(fact_id, recorded_at)

    def _require(self, fact_id: str) -> Fact:
        existing = self.current(fact_id)
        if existing is None:
            raise KeyError(f"no current version of fact {fact_id!r}")
        return existing

    @staticmethod
    def _check_order(existing: Fact, recorded_at: float) -> None:
        if recorded_at < existing.recorded_at:
            raise ValueError(
                f"recorded_at {recorded_at} is before the current version's "
                f"recorded_at {existing.recorded_at}; transaction time only moves forward"
            )


class BitemporalStore(_VersionedStore):
    """In-memory store. Versions per fact id are kept in recording order."""

    def __init__(self) -> None:
        self._versions: dict[str, list[Fact]] = {}
        self._lock = threading.RLock()

    def _atomic(self) -> AbstractContextManager[object]:
        return self._lock

    def current(self, fact_id: str) -> Fact | None:
        versions = self._versions.get(fact_id)
        if versions and versions[-1].superseded_at is None:
            return versions[-1]
        return None

    def _insert(self, fact: Fact) -> None:
        self._versions.setdefault(fact.fact_id, []).append(fact)

    def _supersede(self, fact_id: str, at: float) -> None:
        versions = self._versions[fact_id]
        versions[-1] = replace(versions[-1], superseded_at=at)

    def query(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        *,
        as_of: float,
        known_at: float | None = None,
    ) -> list[Fact]:
        with self._lock:
            hits = [
                f
                for versions in self._versions.values()
                for f in versions
                if (subject is None or f.subject == subject)
                and (predicate is None or f.predicate == predicate)
                and f.valid_at(as_of)
                and f.known_at(known_at)
            ]
        return sorted(hits, key=_sort_key)

    def history(self, fact_id: str) -> list[Fact]:
        with self._lock:
            return list(self._versions.get(fact_id, []))

    def close_store(self) -> None:
        pass


_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    value TEXT NOT NULL,
    valid_from REAL NOT NULL,
    valid_to REAL,
    recorded_at REAL NOT NULL,
    superseded_at REAL,
    source TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS facts_by_id ON facts (fact_id, row_id);
CREATE INDEX IF NOT EXISTS facts_by_subject ON facts (subject, predicate);
CREATE UNIQUE INDEX IF NOT EXISTS facts_one_current ON facts (fact_id)
    WHERE superseded_at IS NULL;
"""

_COLUMNS = (
    "fact_id, subject, predicate, value, valid_from, valid_to, recorded_at, superseded_at, source"
)


class SqliteBitemporalStore(_VersionedStore):
    """SQLite backend on stdlib `sqlite3`, synchronous (see module docstring).

    One row per version. A partial unique index guarantees at most one current version per
    fact id, so the "never two beliefs at once" invariant holds even under a buggy writer.
    """

    def __init__(self, path: str | Path) -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._lock = threading.RLock()
        self._depth = 0

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with self._lock:
            outer = self._depth == 0
            if outer:
                self._conn.execute("BEGIN IMMEDIATE")
            self._depth += 1
            try:
                yield
            except BaseException:
                self._depth -= 1
                if outer:
                    self._conn.execute("ROLLBACK")
                raise
            self._depth -= 1
            if outer:
                self._conn.execute("COMMIT")

    def _atomic(self) -> AbstractContextManager[object]:
        return self._transaction()

    @staticmethod
    def _row(row: tuple[Any, ...]) -> Fact:
        return Fact(
            fact_id=row[0],
            subject=row[1],
            predicate=row[2],
            value=json.loads(row[3]),
            valid_from=row[4],
            valid_to=row[5],
            recorded_at=row[6],
            superseded_at=row[7],
            source=row[8],
        )

    def current(self, fact_id: str) -> Fact | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {_COLUMNS} FROM facts WHERE fact_id = ? AND superseded_at IS NULL",
                (fact_id,),
            ).fetchone()
        return None if row is None else self._row(row)

    def _insert(self, fact: Fact) -> None:
        self._conn.execute(
            f"INSERT INTO facts ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                fact.fact_id,
                fact.subject,
                fact.predicate,
                _canonical(fact.value),
                fact.valid_from,
                fact.valid_to,
                fact.recorded_at,
                fact.superseded_at,
                fact.source,
            ),
        )

    def _supersede(self, fact_id: str, at: float) -> None:
        self._conn.execute(
            "UPDATE facts SET superseded_at = ? WHERE fact_id = ? AND superseded_at IS NULL",
            (at, fact_id),
        )

    def query(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        *,
        as_of: float,
        known_at: float | None = None,
    ) -> list[Fact]:
        clauses = ["valid_from <= ?", "(valid_to IS NULL OR valid_to > ?)"]
        params: list[Any] = [as_of, as_of]
        if subject is not None:
            clauses.append("subject = ?")
            params.append(subject)
        if predicate is not None:
            clauses.append("predicate = ?")
            params.append(predicate)
        if known_at is None:
            clauses.append("superseded_at IS NULL")
        else:
            clauses.append("recorded_at <= ? AND (superseded_at IS NULL OR superseded_at > ?)")
            params.extend([known_at, known_at])
        sql = (
            f"SELECT {_COLUMNS} FROM facts WHERE {' AND '.join(clauses)} "
            "ORDER BY valid_from, fact_id"
        )
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row(r) for r in rows]

    def history(self, fact_id: str) -> list[Fact]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_COLUMNS} FROM facts WHERE fact_id = ? ORDER BY row_id", (fact_id,)
            ).fetchall()
        return [self._row(r) for r in rows]

    def close_store(self) -> None:
        with self._lock:
            self._conn.close()


__all__ = [
    "BitemporalStore",
    "Fact",
    "FactStore",
    "SqliteBitemporalStore",
    "default_fact_id",
]
