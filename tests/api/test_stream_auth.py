"""The media stream trusts only a server-signed token for tenant and customer."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from switchboard.api import StreamAuth, StreamTokenSigner, create_app
from switchboard.api.stream_auth import InvalidStreamToken, twilio_signature
from switchboard.core import InMemoryEventStore, VirtualClock

from .test_media import STREAM, FakeFactory, send, start_msg

SECRET = b"test-stream-secret"
MEDIA_URL = "ws://testserver/v1/twilio/media"


def start_with(params: dict[str, str]) -> dict[str, Any]:
    msg = start_msg()
    msg["start"]["customParameters"] = params
    return msg


def stop_msg() -> dict[str, Any]:
    return {
        "event": "stop",
        "sequenceNumber": "9",
        "streamSid": STREAM,
        "stop": {"accountSid": "AC1", "callSid": "CA1"},
    }


def test_tokens_round_trip_and_reject_tampering() -> None:
    signer = StreamTokenSigner(SECRET)
    token = signer.issue(tenant="acme", customer="cust_9", party="courier")
    claims = signer.verify(token)
    assert (claims.tenant, claims.customer, claims.party) == ("acme", "cust_9", "courier")

    body, sig = token.split(".")
    forged = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    forged["tenant"] = "victim"
    forged_body = base64.urlsafe_b64encode(json.dumps(forged).encode()).decode().rstrip("=")
    for bad in [f"{forged_body}.{sig}", token + "x", "nonsense", "", "a.b.c"]:
        with pytest.raises(InvalidStreamToken):
            signer.verify(bad)
    with pytest.raises(InvalidStreamToken):
        StreamTokenSigner(b"another-stream-secret").verify(token)


async def test_tokens_expire_on_the_clock() -> None:
    clock = VirtualClock(start=100.0)
    signer = StreamTokenSigner(SECRET, clock=clock)
    token = signer.issue(tenant="acme", customer="c", ttl_s=60.0)
    assert signer.verify(token).tenant == "acme"
    await clock.advance(61.0)
    with pytest.raises(InvalidStreamToken, match="expired"):
        signer.verify(token)


def test_a_token_bound_to_a_call_fails_for_another_call() -> None:
    signer = StreamTokenSigner(SECRET)
    token = signer.issue(tenant="acme", customer="c", call_sid="CA1")
    assert signer.verify(token, call_sid="CA1").customer == "c"
    with pytest.raises(InvalidStreamToken):
        signer.verify(token, call_sid="CA2")


def test_twilio_signature_is_hmac_sha1_over_url_and_sorted_params() -> None:
    url = "https://example.com/voice?x=1"
    params = {"To": "+1555", "CallSid": "CA1", "From": "+1666"}
    data = url + "CallSidCA1From+1666To+1555"
    expected = base64.b64encode(hmac.new(b"tok", data.encode(), hashlib.sha1).digest()).decode()
    assert twilio_signature("tok", url, params) == expected
    assert (
        twilio_signature("tok", url)
        == base64.b64encode(hmac.new(b"tok", url.encode(), hashlib.sha1).digest()).decode()
    )


def app_with(factory: FakeFactory, **auth: Any) -> TestClient:
    stream_auth = StreamAuth(StreamTokenSigner(SECRET), **auth)
    return TestClient(
        create_app(InMemoryEventStore(), session_factory=factory, stream_auth=stream_auth)
    )


def test_a_start_without_a_valid_token_is_closed_before_any_session() -> None:
    for params in [{"tenant": "victim_tenant", "customer": "victim"}, {"token": "forged.sig"}]:
        factory = FakeFactory(b"")
        with app_with(factory) as client, client.websocket_connect("/v1/twilio/media") as ws:
            send(ws, start_with(params))
            with pytest.raises(WebSocketDisconnect) as closed:
                ws.receive_text()
        assert closed.value.code == 1008
        assert factory.starts == []


def test_tenant_and_customer_come_only_from_the_token() -> None:
    factory = FakeFactory(b"")
    token = StreamTokenSigner(SECRET).issue(tenant="acme", customer="cust_9", party="courier")
    params = {"token": token, "tenant": "victim_tenant", "customer": "victim", "party": "admin"}
    with app_with(factory) as client, client.websocket_connect("/v1/twilio/media") as ws:
        send(ws, start_with(params))
        send(ws, stop_msg())
    assert factory.starts == [
        ("CA1", STREAM, {"tenant": "acme", "customer": "cust_9", "party": "courier"})
    ]


def test_twilio_signature_check_on_the_upgrade_when_configured() -> None:
    factory = FakeFactory(b"")
    token = StreamTokenSigner(SECRET).issue(tenant="acme", customer="c")
    with app_with(factory, twilio_auth_token="tw-secret") as client:
        with (
            pytest.raises(WebSocketDisconnect),
            client.websocket_connect(
                "/v1/twilio/media", headers={"X-Twilio-Signature": "bogus"}
            ) as ws,
        ):
            ws.receive_text()
        good = twilio_signature("tw-secret", MEDIA_URL)
        with client.websocket_connect(
            "/v1/twilio/media", headers={"X-Twilio-Signature": good}
        ) as ws:
            send(ws, start_with({"token": token}))
            send(ws, stop_msg())
    assert [s[2]["tenant"] for s in factory.starts] == ["acme"]


def test_a_session_factory_needs_stream_auth() -> None:
    with pytest.raises(ValueError, match="stream_auth"):
        create_app(InMemoryEventStore(), session_factory=FakeFactory(b""))
