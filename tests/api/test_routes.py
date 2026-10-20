import asyncio
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from switchboard.api import create_app
from switchboard.telemetry.metrics import Registry
from tests.telemetry.fixtures import conv_a_events, conv_b_events, seeded_store

ALL = len(conv_a_events()) + len(conv_b_events())


@pytest.fixture
def registry() -> Registry:
    return Registry()


@pytest.fixture
def client(registry: Registry) -> TestClient:
    store = asyncio.run(seeded_store())
    return TestClient(create_app(store, registry=registry, epoch_offset_s=1_700_000_000.0))


def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_activity_lists_items_newest_first(client: TestClient) -> None:
    r = client.get("/v1/activity", params={"limit": 500})
    assert r.status_code == 200
    body = r.json()
    assert len(body["items"]) == ALL
    assert body["next_cursor"] is None
    first = body["items"][0]
    assert set(first) >= {
        "at",
        "wall_time",
        "conversation_id",
        "leg_id",
        "seq",
        "event_id",
        "type",
        "severity",
        "summary",
        "payload",
    }
    assert first["wall_time"].startswith("2023-11-14T22:13:26")
    assert "Link" not in r.headers


def test_activity_pagination_headers_and_cursor(client: TestClient) -> None:
    # Five tool.* events in the fixture: a page of three, then the last two.
    r = client.get("/v1/activity", params={"limit": 3, "type": "tool."})
    body = r.json()
    assert len(body["items"]) == 3
    cursor = body["next_cursor"]
    assert r.headers["X-Next-Cursor"] == cursor
    link = r.headers["Link"]
    assert link.endswith('>; rel="next"')
    url = urlparse(link[1 : link.index(">")])
    qs = parse_qs(url.query)
    assert url.path == "/v1/activity"
    assert qs["cursor"] == [cursor]
    assert qs["type"] == ["tool."]
    r2 = client.get("/v1/activity", params={"limit": 3, "type": "tool.", "cursor": cursor})
    rest = r2.json()["items"]
    assert len(rest) == 2
    assert r2.json()["next_cursor"] is None
    ids = [i["event_id"] for i in body["items"] + rest]
    assert len(set(ids)) == 5


def test_activity_filters(client: TestClient) -> None:
    r = client.get(
        "/v1/activity",
        params=[("type", "policy."), ("type", "handoff."), ("conversation_id", "conv_a")],
    )
    assert {i["type"] for i in r.json()["items"]} == {"policy.verdict", "handoff.requested"}
    r = client.get("/v1/activity", params={"q": "colleague", "order": "asc"})
    types = [i["type"] for i in r.json()["items"]]
    assert types == ["speech.sentence_gated", "speech.played"]
    r = client.get("/v1/activity", params={"leg_id": "leg_x", "since": 0, "until": 2})
    assert len(r.json()["items"]) == 2


@pytest.mark.parametrize(
    ("params", "field", "fragment"),
    [
        ({"limit": 0}, "limit", "greater than or equal"),
        ({"limit": "abc"}, "limit", "integer"),
        ({"type": "nonsense."}, "type", "unknown event type"),
        ({"cursor": "garbage"}, "cursor", "cursor"),
        ({"since": 5, "until": 1}, "since", "since"),
        ({"order": "sideways"}, "order", "asc"),
    ],
)
def test_activity_validation_errors(
    client: TestClient, params: dict[str, object], field: str, fragment: str
) -> None:
    r = client.get("/v1/activity", params=params)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert any(d["loc"][-1] == field and fragment in d["msg"] for d in detail), detail


def test_histogram(client: TestClient) -> None:
    r = client.get("/v1/activity/histogram", params={"bucket_s": 2, "conversation_id": "conv_a"})
    assert r.status_code == 200
    body = r.json()
    assert body["bucket_s"] == 2.0
    assert [b["total"] for b in body["buckets"]] == [9, 6, 9, 2]
    assert body["buckets"][0]["counts"]["turn.committed"] == 1
    assert body["buckets"][0]["end"] == 2.0


def test_histogram_validation(client: TestClient) -> None:
    r = client.get("/v1/activity/histogram", params={"bucket_s": 0})
    assert r.status_code == 422
    r = client.get("/v1/activity/histogram", params={"bucket_s": 0.0001, "since": 0, "until": 1e9})
    assert r.status_code == 422
    assert "buckets" in r.json()["detail"][0]["msg"]


def test_conversation_summary_and_timeline(client: TestClient) -> None:
    r = client.get("/v1/conversations/conv_a")
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["outcome"] == "handoff"
    assert body["summary"]["violations_blocked"] == 1
    assert body["summary"]["voice_to_voice_p50_ms"] == pytest.approx(1000)
    turns = body["timeline"]["turns"]
    assert [t["turn_id"] for t in turns] == ["turn_1", "turn_k1", "turn_2"]
    assert turns[0]["voice_to_voice_ms"] == pytest.approx(1150)
    assert turns[0]["tools"][0]["tool"] == "lookup_order"
    assert "text" not in turns[0]


def test_conversation_404(client: TestClient) -> None:
    r = client.get("/v1/conversations/nope")
    assert r.status_code == 404
    assert r.json() == {"detail": "conversation 'nope' not found"}
    assert client.get("/v1/conversations/nope/trace").status_code == 404


def test_trace_is_otlp_json(client: TestClient) -> None:
    r = client.get("/v1/conversations/conv_a/trace")
    assert r.status_code == 200
    spans = r.json()["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert any(s["name"] == "execute_tool issue_refund" for s in spans)
    root = next(s for s in spans if "parentSpanId" not in s)
    assert int(root["startTimeUnixNano"]) > 1_700_000_000 * 10**9


def test_metrics_endpoint(client: TestClient, registry: Registry) -> None:
    registry.counter("probe_total", "probe").inc(3)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain; version=0.0.4")
    assert "probe_total 3.0" in r.text
    assert "switchboard_twilio_streams_active" in r.text


def test_activity_page_html(client: TestClient) -> None:
    r = client.get("/activity")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    html = r.text
    assert html.lstrip().lower().startswith("<!doctype html>")
    assert "/v1/activity/histogram" in html
    assert "/v1/activity?" in html or "'/v1/activity'" in html
    assert 'createElementNS(NS, "svg")' in html
    assert "<script src" not in html
    assert chr(0x2014) not in html


def test_unknown_route_is_404(client: TestClient) -> None:
    assert client.get("/v1/nothing").status_code == 404
