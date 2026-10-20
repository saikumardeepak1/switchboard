"""Contract tests for both bitemporal fact stores (ADR-010).

Time units are arbitrary seconds. DAY keeps the "yesterday versus today" cases readable.
"""

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from switchboard.knowledge.bitemporal import (
    BitemporalStore,
    Fact,
    FactStore,
    SqliteBitemporalStore,
)

DAY = 86_400.0

StoreFactory = Callable[[Path], FactStore]


def _memory(_: Path) -> FactStore:
    return BitemporalStore()


def _sqlite(tmp: Path) -> FactStore:
    return SqliteBitemporalStore(tmp / "facts.sqlite3")


@pytest.fixture(params=[_memory, _sqlite], ids=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[FactStore]:
    factory: StoreFactory = request.param
    s = factory(tmp_path)
    yield s
    s.close_store()


def ids(facts: list[Fact]) -> list[str]:
    return [f.fact_id for f in facts]


def test_fact_is_valid_inside_its_interval_only(store: FactStore) -> None:
    f = store.assert_fact(
        "Order Delivery", "incident", {"name": "x"}, 10.0, 20.0, recorded_at=5.0, source="t"
    )
    assert ids(store.query(as_of=10.0)) == [f.fact_id]
    assert ids(store.query(as_of=19.9)) == [f.fact_id]
    assert store.query(as_of=9.9) == []
    # valid_to is exclusive: at the instant the interval closes the fact no longer holds.
    assert store.query(as_of=20.0) == []


def test_open_ended_fact_stays_valid(store: FactStore) -> None:
    store.assert_fact("platform", "status", "ok", 0.0, recorded_at=0.0)
    assert len(store.query(as_of=1e12)) == 1


def test_fact_is_unknown_before_it_was_recorded(store: FactStore) -> None:
    # A late-arriving fact: true since day 1, but we only learned it on day 3.
    store.assert_fact("Payments", "incident", "outage", 1 * DAY, recorded_at=3 * DAY)
    assert store.query(as_of=2 * DAY, known_at=2.5 * DAY) == []
    assert len(store.query(as_of=2 * DAY, known_at=3 * DAY)) == 1
    assert len(store.query(as_of=2 * DAY)) == 1


def test_what_we_believed_yesterday_about_today(store: FactStore) -> None:
    # Day 1: incident opens, open-ended. Day 2 we learn it actually ended at day 1.5.
    f = store.assert_fact("Order Delivery", "incident", "delays", 1 * DAY, recorded_at=1 * DAY)
    store.close(f.fact_id, valid_to=1.5 * DAY, recorded_at=2 * DAY)

    # Yesterday's belief about today: the incident was still ongoing.
    believed = store.query(as_of=2 * DAY, known_at=1.9 * DAY)
    assert ids(believed) == [f.fact_id]
    assert believed[0].valid_to is None

    # Current belief about today: it is over.
    assert store.query(as_of=2 * DAY) == []
    assert store.query(as_of=2 * DAY, known_at=2 * DAY) == []

    # Both beliefs agree it held during day 1.2.
    assert ids(store.query(as_of=1.2 * DAY, known_at=1.9 * DAY)) == [f.fact_id]
    current = store.query(as_of=1.2 * DAY)
    assert ids(current) == [f.fact_id]
    assert current[0].valid_to == 1.5 * DAY


def test_close_records_a_new_version_and_keeps_the_old_one(store: FactStore) -> None:
    f = store.assert_fact("s", "p", "v", 0.0, recorded_at=1.0, source="src")
    closed = store.close(f.fact_id, valid_to=50.0, recorded_at=60.0)
    assert closed.valid_to == 50.0
    assert closed.recorded_at == 60.0
    assert closed.superseded_at is None

    history = store.history(f.fact_id)
    assert len(history) == 2
    old, new = history
    assert old.valid_to is None
    assert old.recorded_at == 1.0
    assert old.superseded_at == 60.0
    assert new == closed
    assert new.source == "src"
    assert new.value == "v"


def test_retract_is_a_correction_that_keeps_history(store: FactStore) -> None:
    f = store.assert_fact("Checkout", "incident", "outage", 0.0, recorded_at=10.0)
    store.retract(f.fact_id, recorded_at=20.0)

    assert store.query(as_of=15.0) == []
    # Before the correction we did believe it.
    assert ids(store.query(as_of=15.0, known_at=19.0)) == [f.fact_id]
    history = store.history(f.fact_id)
    assert len(history) == 1
    assert history[0].superseded_at == 20.0
    assert store.current(f.fact_id) is None


def test_close_and_retract_reject_unknown_or_retracted_facts(store: FactStore) -> None:
    with pytest.raises(KeyError):
        store.close("nope", valid_to=1.0, recorded_at=1.0)
    with pytest.raises(KeyError):
        store.retract("nope", recorded_at=1.0)
    f = store.assert_fact("s", "p", "v", 0.0, recorded_at=1.0)
    store.retract(f.fact_id, recorded_at=2.0)
    with pytest.raises(KeyError):
        store.close(f.fact_id, valid_to=5.0, recorded_at=3.0)


def test_close_validates_times(store: FactStore) -> None:
    f = store.assert_fact("s", "p", "v", 10.0, recorded_at=10.0)
    with pytest.raises(ValueError):
        store.close(f.fact_id, valid_to=10.0, recorded_at=11.0)
    with pytest.raises(ValueError):
        store.close(f.fact_id, valid_to=20.0, recorded_at=9.0)
    with pytest.raises(ValueError):
        store.assert_fact("s", "p", "v", 10.0, 5.0, recorded_at=10.0)


def test_reasserting_identical_content_is_idempotent(store: FactStore) -> None:
    a = store.assert_fact("s", "p", {"k": 1}, 0.0, recorded_at=1.0, fact_id="f1")
    b = store.assert_fact("s", "p", {"k": 1}, 0.0, recorded_at=5.0, fact_id="f1")
    assert a == b
    assert len(store.history("f1")) == 1


def test_reasserting_changed_content_supersedes(store: FactStore) -> None:
    store.assert_fact("s", "p", {"k": 1}, 0.0, recorded_at=1.0, fact_id="f1")
    store.assert_fact("s", "p", {"k": 2}, 0.0, recorded_at=5.0, fact_id="f1")
    assert [f.value for f in store.history("f1")] == [{"k": 1}, {"k": 2}]
    assert store.query(as_of=1.0, known_at=4.0)[0].value == {"k": 1}
    assert store.query(as_of=1.0)[0].value == {"k": 2}


def test_default_fact_id_is_deterministic(tmp_path: Path) -> None:
    a = BitemporalStore().assert_fact("s", "p", "v", 0.0, recorded_at=1.0, source="x")
    b = SqliteBitemporalStore(tmp_path / "f.sqlite3").assert_fact(
        "s", "p", "v", 0.0, recorded_at=1.0, source="x"
    )
    assert a.fact_id == b.fact_id
    assert a.fact_id.startswith("fact_")


def test_query_filters_by_subject_and_predicate(store: FactStore) -> None:
    store.assert_fact("A", "incident", 1, 0.0, recorded_at=0.0)
    store.assert_fact("A", "status", 2, 0.0, recorded_at=0.0)
    store.assert_fact("B", "incident", 3, 0.0, recorded_at=0.0)
    assert sorted(f.value for f in store.query(subject="A", as_of=1.0)) == [1, 2]
    assert sorted(f.value for f in store.query(predicate="incident", as_of=1.0)) == [1, 3]
    assert [f.value for f in store.query(subject="B", predicate="incident", as_of=1.0)] == [3]


def test_query_results_are_ordered_deterministically(store: FactStore) -> None:
    store.assert_fact("s", "p", "late", 5.0, recorded_at=5.0, fact_id="b")
    store.assert_fact("s", "p", "early", 1.0, recorded_at=1.0, fact_id="a")
    store.assert_fact("s", "p", "tie", 1.0, recorded_at=1.0, fact_id="c")
    assert ids(store.query(as_of=10.0)) == ["a", "c", "b"]


def test_nested_values_round_trip(store: FactStore) -> None:
    value = {"name": "Delays", "impact": "major", "n": [1, 2.5, None, True]}
    f = store.assert_fact("s", "p", value, 0.0, recorded_at=0.0)
    assert store.query(as_of=0.0)[0].value == value
    assert store.current(f.fact_id) == f


def test_sqlite_store_persists_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "facts.sqlite3"
    first = SqliteBitemporalStore(path)
    f = first.assert_fact("s", "p", "v", 0.0, recorded_at=1.0)
    first.close(f.fact_id, valid_to=10.0, recorded_at=2.0)
    first.close_store()

    second = SqliteBitemporalStore(path)
    assert len(second.history(f.fact_id)) == 2
    assert second.query(as_of=5.0)[0].valid_to == 10.0
    second.close_store()
