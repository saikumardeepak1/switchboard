import copy
import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from switchboard.knowledge.bitemporal import BitemporalStore
from switchboard.knowledge.statuspage import StatusPageIngestor, parse_timestamp

FIXTURE = Path(__file__).parent / "fixtures" / "incidents_open.json"
SOURCE = "statuspage:demo"


def ts(text: str) -> float:
    return datetime.fromisoformat(text).timestamp()


OPENED = ts("2026-10-19T08:12:44.120-07:00")
LAST_UPDATE = ts("2026-10-19T09:41:07.418-07:00")
RESOLVED_TEXT = "2026-10-19T11:30:00-07:00"
RESOLVED = ts(RESOLVED_TEXT)
DURING = OPENED + 3600
AFTER = RESOLVED + 60


def load() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(FIXTURE.read_text())
    return data


def resolved_payload() -> dict[str, Any]:
    payload = copy.deepcopy(load())
    incident = payload["incidents"][0]
    incident["status"] = "resolved"
    incident["resolved_at"] = RESOLVED_TEXT
    incident["updated_at"] = RESOLVED_TEXT
    incident["incident_updates"].insert(
        0,
        {
            "id": "x9",
            "status": "resolved",
            "body": "Deliveries are back on schedule.",
            "created_at": RESOLVED_TEXT,
        },
    )
    return payload


@pytest.fixture
def store() -> BitemporalStore:
    return BitemporalStore()


@pytest.fixture
def ingestor(store: BitemporalStore) -> StatusPageIngestor:
    return StatusPageIngestor(store, SOURCE)


def test_parse_timestamp_handles_offsets_z_and_fractions() -> None:
    assert parse_timestamp("1970-01-01T00:00:00Z") == 0.0
    assert parse_timestamp("1970-01-01T00:00:01.500Z") == 1.5
    assert parse_timestamp("1970-01-01T01:00:00+01:00") == 0.0
    assert parse_timestamp("1969-12-31T17:00:00-07:00") == 0.0
    # A naive timestamp is taken as UTC rather than local time, so results do not vary by host.
    assert parse_timestamp("1970-01-01T00:00:10") == 10.0
    tz = timezone(timedelta(hours=-7))
    expected = datetime(2026, 10, 19, 8, 12, 44, 120000, tzinfo=tz).astimezone(UTC).timestamp()
    assert parse_timestamp("2026-10-19T08:12:44.120-07:00") == expected


def test_open_incident_becomes_a_fact_per_component(
    store: BitemporalStore, ingestor: StatusPageIngestor
) -> None:
    ingestor.ingest(load())
    facts = store.query(subject="Order Delivery", predicate="incident", as_of=DURING)
    assert len(facts) == 1
    fact = facts[0]
    assert fact.valid_from == OPENED
    assert fact.valid_to is None
    assert fact.source == SOURCE
    assert fact.recorded_at == LAST_UPDATE
    assert fact.value == {
        "incident_id": "cp306tmzcl0y",
        "name": "Order delivery delays",
        "impact": "major",
        "status": "identified",
        "update": (
            "A routing fault at the Portland hub is delaying courier assignment. "
            "Deliveries in the Pacific Northwest may arrive up to 90 minutes late."
        ),
    }


def test_incident_without_components_is_about_the_platform(
    store: BitemporalStore, ingestor: StatusPageIngestor
) -> None:
    ingestor.ingest(load())
    facts = store.query(subject="platform", as_of=ts("2026-10-19T10:00:00Z"))
    assert [f.value["name"] for f in facts] == ["Slow app performance"]


def test_already_resolved_incident_is_recorded_as_a_closed_interval(
    store: BitemporalStore, ingestor: StatusPageIngestor
) -> None:
    ingestor.ingest(load())
    during = ts("2026-10-17T14:00:00Z")
    subjects = sorted(f.subject for f in store.query(predicate="incident", as_of=during))
    assert subjects == ["Checkout", "Payments"]
    payments = store.query(subject="Payments", as_of=during)[0]
    assert payments.valid_to == ts("2026-10-17T15:02:30Z")
    assert store.query(subject="Payments", as_of=ts("2026-10-18T00:00:00Z")) == []


def test_repeated_ingest_of_the_same_payload_is_idempotent(
    store: BitemporalStore, ingestor: StatusPageIngestor
) -> None:
    first = ingestor.ingest(load())
    assert len(first) == 4
    second = ingestor.ingest(load())
    assert second == []
    for fact in store.query(predicate="incident", as_of=DURING):
        assert len(store.history(fact.fact_id)) == 1


def test_later_resolution_closes_the_interval_without_losing_history(
    store: BitemporalStore, ingestor: StatusPageIngestor
) -> None:
    ingestor.ingest(load())
    fact_id = store.query(subject="Order Delivery", as_of=DURING)[0].fact_id

    changed = ingestor.ingest(resolved_payload())
    assert [f.fact_id for f in changed] == [fact_id]
    assert changed[0].valid_to == RESOLVED

    assert store.query(subject="Order Delivery", as_of=AFTER) == []
    assert store.query(subject="Order Delivery", as_of=DURING)[0].valid_to == RESOLVED
    # Before we ingested the resolution we believed the incident was still open.
    believed = store.query(subject="Order Delivery", as_of=AFTER, known_at=RESOLVED - 1)
    assert [f.fact_id for f in believed] == [fact_id]
    assert len(store.history(fact_id)) == 2

    assert ingestor.ingest(resolved_payload()) == []
    assert len(store.history(fact_id)) == 2


def test_resolution_with_no_new_update_uses_close(
    store: BitemporalStore, ingestor: StatusPageIngestor
) -> None:
    ingestor.ingest(load())
    payload = copy.deepcopy(load())
    payload["incidents"][0]["resolved_at"] = RESOLVED_TEXT
    payload["incidents"][0]["updated_at"] = RESOLVED_TEXT
    payload["incidents"][0]["status"] = "identified"
    ingestor.ingest(payload)
    old, new = store.history(store.query(subject="Order Delivery", as_of=DURING)[0].fact_id)
    assert old.value == new.value
    assert new.valid_to == RESOLVED
    assert new.recorded_at == RESOLVED


def test_new_update_on_an_open_incident_revises_the_value(
    store: BitemporalStore, ingestor: StatusPageIngestor
) -> None:
    ingestor.ingest(load())
    payload = copy.deepcopy(load())
    incident = payload["incidents"][0]
    incident["updated_at"] = "2026-10-19T10:15:00-07:00"
    incident["incident_updates"].insert(
        0,
        {
            "body": "Fix deployed, monitoring.",
            "status": "monitoring",
            "created_at": incident["updated_at"],
        },
    )
    incident["status"] = "monitoring"
    ingestor.ingest(payload)
    current = store.query(subject="Order Delivery", as_of=DURING)[0]
    assert current.value["update"] == "Fix deployed, monitoring."
    assert current.value["status"] == "monitoring"
    assert current.valid_to is None


def test_explicit_recorded_at_overrides_the_payload_time(
    store: BitemporalStore, ingestor: StatusPageIngestor
) -> None:
    ingestor.ingest(load(), recorded_at=LAST_UPDATE + 30)
    assert store.query(subject="Order Delivery", as_of=DURING, known_at=LAST_UPDATE) == []
    assert len(store.query(subject="Order Delivery", as_of=DURING)) == 1


def test_malformed_payload_is_rejected(ingestor: StatusPageIngestor) -> None:
    with pytest.raises(ValueError):
        ingestor.ingest({"nope": []})
    with pytest.raises(ValueError):
        ingestor.ingest({"incidents": [{"id": "x"}]})
