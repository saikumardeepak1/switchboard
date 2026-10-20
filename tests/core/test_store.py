import asyncio
import sqlite3
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from switchboard.core.events import AnyEvent, ConversationEnded, HandoffRequested
from switchboard.core.store import (
    ConcurrencyError,
    EventStore,
    InMemoryEventStore,
    SqliteEventStore,
)
from tests.core.samples import base, every_event_type

StoreFactory = Callable[[Path], EventStore]


def _memory(_: Path) -> EventStore:
    return InMemoryEventStore()


def _sqlite(tmp: Path) -> EventStore:
    return SqliteEventStore(tmp / "events.sqlite3")


@pytest.fixture(params=[_memory, _sqlite], ids=["memory", "sqlite"])
async def store(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[EventStore]:
    factory: StoreFactory = request.param
    s = factory(tmp_path)
    yield s
    await s.close()


def handoff(conversation_id: str = "conv_1", at: float = 1.0, reason: str = "r") -> AnyEvent:
    return HandoffRequested(**base(conversation_id, at), reason=reason)


async def collect(it: AsyncIterator[AnyEvent]) -> list[AnyEvent]:
    return [e async for e in it]


async def test_append_assigns_gapless_seq_and_ids(store: EventStore) -> None:
    first = await store.append("conv_1", [handoff(), handoff()], expected_seq=0)
    second = await store.append("conv_1", [handoff()], expected_seq=2)
    assert [e.seq for e in first + second] == [1, 2, 3]
    ids = [e.event_id for e in first + second]
    assert all(ids) and len(set(ids)) == 3
    assert await store.read("conv_1") == first + second


async def test_event_ids_are_deterministic(store: EventStore, tmp_path: Path) -> None:
    written = await store.append("conv_1", [handoff()], expected_seq=0)
    other = InMemoryEventStore()
    again = await other.append("conv_1", [handoff()], expected_seq=0)
    assert written[0].event_id == again[0].event_id


async def test_read_after_seq(store: EventStore) -> None:
    await store.append("conv_1", [handoff(reason=str(i)) for i in range(5)], expected_seq=0)
    tail = await store.read("conv_1", after_seq=3)
    assert [e.seq for e in tail] == [4, 5]
    assert await store.read("missing") == []


async def test_conversations_are_independent(store: EventStore) -> None:
    await store.append("conv_1", [handoff("conv_1")], expected_seq=0)
    b = await store.append("conv_2", [handoff("conv_2")], expected_seq=0)
    assert b[0].seq == 1
    assert b[0].event_id != (await store.read("conv_1"))[0].event_id


async def test_stale_expected_seq_raises_and_writes_nothing(store: EventStore) -> None:
    await store.append("conv_1", [handoff()], expected_seq=0)
    with pytest.raises(ConcurrencyError):
        await store.append("conv_1", [handoff(), handoff()], expected_seq=0)
    with pytest.raises(ConcurrencyError):
        await store.append("conv_1", [handoff()], expected_seq=5)
    assert [e.seq for e in await store.read("conv_1")] == [1]


async def test_mismatched_conversation_id_writes_nothing(store: EventStore) -> None:
    with pytest.raises(ValueError):
        await store.append("conv_1", [handoff("conv_1"), handoff("conv_9")], expected_seq=0)
    assert await store.read("conv_1") == []


async def test_empty_append_is_a_noop(store: EventStore) -> None:
    assert await store.append("conv_1", [], expected_seq=0) == []
    with pytest.raises(ConcurrencyError):
        await store.append("conv_1", [], expected_seq=1)


async def test_read_all_filters_by_type_and_since(store: EventStore) -> None:
    await store.append(
        "conv_1",
        [handoff(at=1.0), ConversationEnded(**base("conv_1", 5.0), outcome="ok")],
        expected_seq=0,
    )
    await store.append("conv_2", [handoff("conv_2", at=3.0)], expected_seq=0)

    everything = await collect(store.read_all())
    assert [(e.conversation_id, e.seq) for e in everything] == [
        ("conv_1", 1),
        ("conv_1", 2),
        ("conv_2", 1),
    ]
    handoffs = await collect(store.read_all(types={"handoff.requested"}))
    assert [e.at for e in handoffs] == [1.0, 3.0]
    recent = await collect(store.read_all(since=3.0))
    assert [e.at for e in recent] == [5.0, 3.0]
    both = await collect(store.read_all(types={"handoff.requested"}, since=2.0))
    assert [(e.conversation_id, e.at) for e in both] == [("conv_2", 3.0)]
    assert await collect(store.read_all(types=set())) == []


async def test_every_event_type_round_trips(store: EventStore) -> None:
    events = every_event_type()
    written = await store.append("conv_1", events, expected_seq=0)
    read = await store.read("conv_1")
    assert read == written
    for original, stored in zip(events, read, strict=True):
        assert type(stored) is type(original)
        assert stored.model_copy(update={"seq": 0, "event_id": ""}) == original


async def test_sqlite_persists_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "events.sqlite3"
    s = SqliteEventStore(path)
    written = await s.append("conv_1", every_event_type(), expected_seq=0)
    await s.close()
    reopened = SqliteEventStore(path)
    assert await reopened.read("conv_1") == written
    with pytest.raises(ConcurrencyError):
        await reopened.append("conv_1", [handoff()], expected_seq=0)
    await reopened.close()


async def test_sqlite_schema(tmp_path: Path) -> None:
    path = tmp_path / "events.sqlite3"
    s = SqliteEventStore(path)
    await s.close()
    con = sqlite3.connect(path)
    try:
        assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        index_cols = {
            tuple(r[2] for r in con.execute(f"PRAGMA index_info('{name}')"))
            for (_, name, unique, *_rest) in con.execute("PRAGMA index_list('events')")
        }
        assert ("conversation_id", "seq") in index_cols
        assert ("type", "at") in index_cols
    finally:
        con.close()


async def _writer(store: EventStore, name: str, batches: list[int], yields: list[int]) -> int:
    written = 0
    for i, size in enumerate(batches):
        for _attempt in range(1000):
            current = len(await store.read("conv_1"))
            for _ in range(yields[i % len(yields)]):
                await asyncio.sleep(0)
            try:
                await store.append(
                    "conv_1",
                    [handoff(reason=f"{name}-{i}-{k}") for k in range(size)],
                    expected_seq=current,
                )
            except ConcurrencyError:
                continue
            written += size
            break
        else:
            raise AssertionError("writer starved")
    return written


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    batches_a=st.lists(st.integers(1, 3), min_size=1, max_size=5),
    batches_b=st.lists(st.integers(1, 3), min_size=1, max_size=5),
    yields_a=st.lists(st.integers(0, 3), min_size=1, max_size=5),
    yields_b=st.lists(st.integers(0, 3), min_size=1, max_size=5),
)
def test_interleaved_writers_never_duplicate_or_gap(
    kind: str,
    batches_a: list[int],
    batches_b: list[int],
    yields_a: list[int],
    yields_b: list[int],
) -> None:
    async def run() -> None:
        s: EventStore = InMemoryEventStore() if kind == "memory" else SqliteEventStore(":memory:")
        try:
            a, b = await asyncio.gather(
                _writer(s, "a", batches_a, yields_a), _writer(s, "b", batches_b, yields_b)
            )
            events = await s.read("conv_1")
            assert a + b == sum(batches_a) + sum(batches_b)
            assert [e.seq for e in events] == list(range(1, a + b + 1))
            assert len({e.event_id for e in events}) == len(events)
            # Batches stay contiguous: every batch is written as one unit.
            reasons = [e.reason for e in events if isinstance(e, HandoffRequested)]
            for prefix in {r.rsplit("-", 1)[0] for r in reasons}:
                idx = [n for n, r in enumerate(reasons) if r.rsplit("-", 1)[0] == prefix]
                assert idx == list(range(idx[0], idx[0] + len(idx)))
        finally:
            await s.close()

    asyncio.run(run())
