"""Drive /v1/twilio/media with a fake Twilio client and a fake runtime session."""

from __future__ import annotations

import asyncio
import base64
import json
import struct
from collections.abc import AsyncIterator, Mapping
from typing import Any

from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from switchboard.api import StreamAuth, StreamTokenSigner, create_app
from switchboard.api.media import Clear, OutboundAudio, Utterance
from switchboard.core.store import InMemoryEventStore
from switchboard.providers.twilio import FRAME_BYTES, decode_ulaw, encode_ulaw
from switchboard.telemetry.metrics import Registry

STREAM = "MZ123"
AUTH = StreamAuth(StreamTokenSigner(b"media-test-stream-secret"))
TOKEN = AUTH.signer.issue(tenant="acme", customer="cust_1")


def pcm(samples: list[int]) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


class FakeSession:
    """Echoes the scripted behaviour the real runtime will have: speak after the first audio,
    and barge in (clear) once the caller's mark echo shows the first utterance played."""

    def __init__(self, reply: bytes) -> None:
        self.audio: list[bytes] = []
        self.marks: list[str] = []
        self.closed = False
        self._reply = reply
        self._queue: asyncio.Queue[OutboundAudio | None] = asyncio.Queue()

    async def on_audio(self, pcm16: bytes) -> None:
        self.audio.append(pcm16)
        if len(self.audio) == 1:
            await self._queue.put(Utterance(pcm16=self._reply, mark="utt_1"))

    async def on_mark(self, name: str) -> None:
        self.marks.append(name)
        await self._queue.put(Clear())

    async def outbound(self) -> AsyncIterator[OutboundAudio]:
        while (item := await self._queue.get()) is not None:
            yield item

    async def close(self) -> None:
        self.closed = True
        await self._queue.put(None)


class FakeFactory:
    def __init__(self, reply: bytes) -> None:
        self.reply = reply
        self.sessions: list[FakeSession] = []
        self.starts: list[tuple[str, str, dict[str, str]]] = []

    async def start(
        self, call_sid: str, stream_sid: str, custom_parameters: Mapping[str, str]
    ) -> FakeSession:
        self.starts.append((call_sid, stream_sid, dict(custom_parameters)))
        session = FakeSession(self.reply)
        self.sessions.append(session)
        return session


def send(ws: WebSocketTestSession, msg: dict[str, Any]) -> None:
    ws.send_text(json.dumps(msg))


def start_msg() -> dict[str, Any]:
    return {
        "event": "start",
        "sequenceNumber": "1",
        "streamSid": STREAM,
        "start": {
            "accountSid": "AC1",
            "callSid": "CA1",
            "streamSid": STREAM,
            "tracks": ["inbound"],
            "customParameters": {"token": TOKEN},
            "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
        },
    }


def media_msg(seq: int, ulaw: bytes, track: str = "inbound") -> dict[str, Any]:
    return {
        "event": "media",
        "sequenceNumber": str(seq),
        "streamSid": STREAM,
        "media": {
            "track": track,
            "chunk": str(seq),
            "timestamp": str(seq * 20),
            "payload": base64.b64encode(ulaw).decode(),
        },
    }


def test_full_media_stream_round_trip() -> None:
    # 400 samples: two full 160-sample frames and one 80-sample frame padded with silence.
    reply = pcm([(i * 37) % 2000 - 1000 for i in range(400)])
    factory = FakeFactory(reply)
    registry = Registry()
    app = create_app(
        InMemoryEventStore(), session_factory=factory, registry=registry, stream_auth=AUTH
    )
    inbound = pcm([0, 1000, -1000, 32000, -32000] * 32)
    inbound_ulaw = encode_ulaw(inbound)

    with TestClient(app) as client, client.websocket_connect("/v1/twilio/media") as ws:
        send(ws, {"event": "connected", "protocol": "Call", "version": "1.0.0"})
        send(ws, start_msg())
        send(ws, media_msg(2, inbound_ulaw))

        frames = [json.loads(ws.receive_text()) for _ in range(3)]
        for f in frames:
            assert f["event"] == "media"
            assert f["streamSid"] == STREAM
            assert len(base64.b64decode(f["media"]["payload"])) == FRAME_BYTES
        sent = b"".join(base64.b64decode(f["media"]["payload"]) for f in frames)
        assert sent[: len(reply) // 2] == encode_ulaw(reply)
        assert set(sent[len(reply) // 2 :]) == {0xFF}

        mark = json.loads(ws.receive_text())
        assert mark == {"event": "mark", "streamSid": STREAM, "mark": {"name": "utt_1"}}

        send(
            ws,
            {
                "event": "mark",
                "sequenceNumber": "3",
                "streamSid": STREAM,
                "mark": {"name": "utt_1"},
            },
        )
        clear = json.loads(ws.receive_text())
        assert clear == {"event": "clear", "streamSid": STREAM}

        # Outbound-track echoes and DTMF are not caller audio.
        send(ws, media_msg(4, inbound_ulaw, track="outbound"))
        send(
            ws,
            {"event": "dtmf", "sequenceNumber": "5", "streamSid": STREAM, "dtmf": {"digit": "1"}},
        )
        send(
            ws,
            {
                "event": "stop",
                "sequenceNumber": "6",
                "streamSid": STREAM,
                "stop": {"accountSid": "AC1", "callSid": "CA1"},
            },
        )

    assert factory.starts == [
        ("CA1", STREAM, {"tenant": "acme", "customer": "cust_1", "party": "caller"})
    ]
    (session,) = factory.sessions
    assert session.audio == [decode_ulaw(inbound_ulaw)]
    assert session.marks == ["utt_1"]
    assert session.closed
    text = registry.render()
    assert "switchboard_twilio_frames_out_total 3.0" in text
    assert "switchboard_twilio_frames_in_total 1.0" in text
    assert "switchboard_twilio_streams_active 0.0" in text


def test_client_disconnect_closes_session() -> None:
    factory = FakeFactory(b"")
    app = create_app(InMemoryEventStore(), session_factory=factory, stream_auth=AUTH)
    with TestClient(app) as client, client.websocket_connect("/v1/twilio/media") as ws:
        send(ws, start_msg())
        send(ws, media_msg(2, b"\xff" * 160))
    (session,) = factory.sessions
    assert session.closed


def test_media_before_start_and_malformed_messages_are_ignored() -> None:
    factory = FakeFactory(b"")
    registry = Registry()
    app = create_app(
        InMemoryEventStore(), session_factory=factory, registry=registry, stream_auth=AUTH
    )
    with TestClient(app) as client, client.websocket_connect("/v1/twilio/media") as ws:
        send(ws, media_msg(1, b"\xff" * 160))
        ws.send_text("{not json")
        send(ws, start_msg())
        send(ws, media_msg(2, b"\xff" * 160))
        send(
            ws,
            {
                "event": "stop",
                "sequenceNumber": "3",
                "streamSid": STREAM,
                "stop": {"accountSid": "AC1", "callSid": "CA1"},
            },
        )
    (session,) = factory.sessions
    assert len(session.audio) == 1
    assert "switchboard_twilio_protocol_errors_total 1.0" in registry.render()


def test_no_session_factory_means_no_media_route() -> None:
    client = TestClient(create_app(InMemoryEventStore()))
    assert client.get("/v1/twilio/media").status_code == 404
