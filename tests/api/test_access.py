"""Activity and conversation reads need a principal, see only its tenant, and hide content."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from switchboard.api import Principal, StaticTokenAuthenticator, create_app
from switchboard.core import InMemoryEventStore
from switchboard.core.events import ConversationStarted, ToolExecuted, TurnCommitted
from tests.telemetry.fixtures import seeded_store

AUTH = StaticTokenAuthenticator(
    {"acme-token": Principal("acme", "ops@acme"), "globex-token": Principal("globex", "ops@gx")}
)
ROUTES = [
    "/v1/activity",
    "/v1/activity/histogram",
    "/v1/conversations/conv_a",
    "/v1/conversations/conv_a/trace",
]


def client(token: str | None, **kw: object) -> TestClient:
    store = asyncio.run(seeded_store())
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return TestClient(create_app(store, authenticator=AUTH, **kw), headers=headers)  # type: ignore[arg-type]


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("token", [None, "wrong-token"])
def test_reads_without_a_valid_bearer_token_are_401(route: str, token: str | None) -> None:
    r = client(token).get(route)
    assert r.status_code == 401
    assert r.headers["WWW-Authenticate"] == "Bearer"


def test_no_authenticator_means_every_read_is_refused() -> None:
    c = TestClient(create_app(asyncio.run(seeded_store())))
    assert all(c.get(route).status_code == 401 for route in ROUTES)
    assert c.get("/healthz").status_code == 200


def test_another_tenant_sees_nothing_and_gets_404_for_conversations() -> None:
    c = client("globex-token")
    assert c.get("/v1/activity").json()["items"] == []
    assert c.get("/v1/activity", params={"conversation_id": "conv_a"}).json()["items"] == []
    assert c.get("/v1/activity/histogram").json()["buckets"] == []
    assert c.get("/v1/conversations/conv_a").status_code == 404
    assert c.get("/v1/conversations/conv_a/trace").status_code == 404


def test_the_owning_tenant_sees_its_conversations() -> None:
    c = client("acme-token")
    assert c.get("/v1/activity").json()["items"]
    assert c.get("/v1/conversations/conv_a").status_code == 200


def seeded_secret_store() -> InMemoryEventStore:
    store = InMemoryEventStore()
    base = {"conversation_id": "conv_s", "bundle_id": "b", "causation_id": None}
    asyncio.run(
        store.append(
            "conv_s",
            [
                ConversationStarted(
                    leg_id=None, at=0, tenant_id="acme", customer_id="cust_9", **base
                ),
                TurnCommitted(
                    leg_id="l",
                    at=1,
                    turn_id="t",
                    text="my card is 4111 1111 1111 1111 and I live at 12 Elm St",
                    endpointing_ms=1,
                    **base,
                ),
                ToolExecuted(
                    leg_id="l",
                    at=2,
                    call_id="c",
                    tool="lookup_order",
                    result={"customer_name": "Ana Ruiz", "pin_address": "12 Elm St"},
                    **base,
                ),
            ],
            0,
        )
    )
    return store


def test_capture_off_redacts_text_and_tool_payloads_to_type_and_length() -> None:
    app = create_app(seeded_secret_store(), authenticator=AUTH, capture_content=False)
    c = TestClient(app, headers={"Authorization": "Bearer acme-token"})
    items = c.get("/v1/activity", params={"limit": 500}).json()["items"]
    dumped = json.dumps(items)
    for secret in ("4111", "Elm", "Ana Ruiz"):
        assert secret not in dumped
    turn = next(i for i in items if i["type"] == "turn.committed")
    said = "my card is 4111 1111 1111 1111 and I live at 12 Elm St"
    assert turn["payload"]["text"] == {"redacted": "string", "length": len(said)}
    tool = next(i for i in items if i["type"] == "tool.executed")
    assert tool["payload"]["result"] == {"redacted": "object", "length": 2}
    assert tool["payload"]["tool"] == "lookup_order"
    # Text search cannot probe redacted content either.
    assert c.get("/v1/activity", params={"q": "elm"}).json()["items"] == []


def test_capture_on_shows_content() -> None:
    app = create_app(seeded_secret_store(), authenticator=AUTH, capture_content=True)
    c = TestClient(app, headers={"Authorization": "Bearer acme-token"})
    assert len(c.get("/v1/activity", params={"q": "elm"}).json()["items"]) == 2
