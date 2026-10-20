"""The Twilio media path end to end: WebSocket frames in, a live conversation, frames out.

There is no real ASR, so the caller's words travel inside the audio: the test sends the text's
bytes as mu-law payloads (ASCII survives the mu-law round trip exactly) and `AudioTextSTT`
decodes them back into text for the simulated recognizer. Everything runs on the real clock with
fast simulated providers, because the WebSocket test client runs the app in its own thread.
"""

from __future__ import annotations

import asyncio
import base64
import json
import random
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

from switchboard.api import StreamAuth, StreamTokenSigner, create_app
from switchboard.brain import ScriptedLLM
from switchboard.bundle import load_bundle
from switchboard.core import InMemoryEventStore, RealClock
from switchboard.effects import InMemoryLedger
from switchboard.providers import LatencyModel, SimulatedSTT, SimulatedTTS, SttEvent
from switchboard.providers.twilio import encode_ulaw
from switchboard.runtime import Runtime
from switchboard.runtime.twilio_session import TwilioSessionFactory

from .harness import EXAMPLE, LONG_GREETING, DeliveryBrain, RecordingTTS

STREAM = "MZ900"
AUTH = StreamAuth(StreamTokenSigner(b"twilio-session-test-secret"))


class AudioTextSTT:
    """Simulated STT that reads the caller's words from the audio itself.

    Each newline-terminated run of mu-law bytes is one utterance, spoken through `SimulatedSTT`.
    """

    def __init__(self, clock: RealClock) -> None:
        self.sim = SimulatedSTT(
            clock, random.Random(1), LatencyModel(p50=0.01, p95=0.02), words_per_s=20.0
        )
        self.frames = 0
        self._bytes = b""
        self._speaking: set[asyncio.Task[None]] = set()

    async def push(self, frame: bytes) -> None:
        self.frames += 1
        self._bytes += encode_ulaw(frame)
        while b"\n" in self._bytes:
            line, _, self._bytes = self._bytes.partition(b"\n")
            task = asyncio.create_task(self.sim.speak(line.decode("ascii")))
            self._speaking.add(task)
            task.add_done_callback(self._speaking.discard)

    def events(self) -> AsyncIterator[SttEvent]:
        return self.sim.events()

    async def close(self) -> None:
        await self.sim.close()


def send(ws: WebSocketTestSession, msg: dict[str, Any]) -> None:
    ws.send_text(json.dumps(msg))


def say(ws: WebSocketTestSession, seq: int, text: str) -> None:
    payload = base64.b64encode((text + "\n").encode("ascii")).decode()
    media = {"track": "inbound", "chunk": str(seq), "timestamp": str(seq * 20), "payload": payload}
    send(ws, {"event": "media", "sequenceNumber": str(seq), "streamSid": STREAM, "media": media})


def start(ws: WebSocketTestSession) -> None:
    send(ws, {"event": "connected", "protocol": "Call", "version": "1.0.0"})
    send(
        ws,
        {
            "event": "start",
            "sequenceNumber": "1",
            "streamSid": STREAM,
            "start": {
                "accountSid": "AC1",
                "callSid": "CA77",
                "streamSid": STREAM,
                "tracks": ["inbound"],
                "customParameters": {
                    "token": AUTH.signer.issue(tenant="dashfleet", customer="C-77", party="courier")
                },
                "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
            },
        },
    )


def test_a_call_over_the_media_stream_speaks_barges_in_and_ends() -> None:
    clock = RealClock()
    store = InMemoryEventStore()
    llm = ScriptedLLM(clock, DeliveryBrain(long_greeting=True), ttft_s=0.01, tokens_per_s=2000.0)
    runtime = Runtime(
        clock=clock, store=store, ledger=InMemoryLedger(clock), llm=llm, rng=random.Random(5)
    )
    stts: list[AudioTextSTT] = []
    tts = RecordingTTS(SimulatedTTS(clock, ttfb_s=0.01, chars_per_s=60.0))

    def new_stt(call_sid: str) -> AudioTextSTT:
        stts.append(AudioTextSTT(clock))
        return stts[-1]

    factory = TwilioSessionFactory(
        runtime, load_bundle(EXAMPLE), stt=new_stt, tts=lambda call_sid: tts
    )
    app = create_app(store, session_factory=factory, stream_auth=AUTH)
    seen: list[dict[str, Any]] = []

    def read_until(done: Any) -> None:
        while not done():
            seen.append(json.loads(ws.receive_text()))

    with TestClient(app) as client, client.websocket_connect("/v1/twilio/media") as ws:
        start(ws)
        say(ws, 2, "Hi, this is courier C-77.")
        read_until(lambda: LONG_GREETING in tts.texts and seen and seen[-1]["event"] == "media")
        # Talk over the greeting: the agent stops and Twilio is told to drop queued audio.
        say(ws, 3, "Sorry, what is my pay this week?")
        read_until(lambda: seen[-1]["event"] == "clear")
        read_until(lambda: any("182.40" in t for t in tts.texts))
        read_until(lambda: seen[-1]["event"] == "mark")
        echo = {"name": seen[-1]["mark"]["name"]}
        send(ws, {"event": "mark", "sequenceNumber": "8", "streamSid": STREAM, "mark": echo})
        send(
            ws,
            {
                "event": "stop",
                "sequenceNumber": "9",
                "streamSid": STREAM,
                "stop": {"accountSid": "AC1", "callSid": "CA77"},
            },
        )
        # The server ends the conversation, then closes the socket.
        with pytest.raises(WebSocketDisconnect):
            while True:
                seen.append(json.loads(ws.receive_text()))

    (session,) = factory.sessions.values()
    assert session.heard_marks == [echo["name"]]
    assert stts[0].frames == 2
    assert {m["event"] for m in seen} == {"media", "mark", "clear"}
    assert all(m["streamSid"] == STREAM for m in seen)

    conv = session.conversation
    assert conv is not None
    events = asyncio.run(store.read(conv.id))
    types = [e.type for e in events]
    assert types[0] == "conversation.started" and types[-1] == "conversation.ended"
    assert events[-1].outcome == "hangup"  # type: ignore[union-attr]
    interrupted = [e for e in events if e.type == "speech.played" and e.interrupted]  # type: ignore[union-attr]
    assert len(interrupted) == 1
    assert LONG_GREETING.startswith(interrupted[0].text_played)  # type: ignore[union-attr]
    committed = [e.text for e in events if e.type == "turn.committed"]  # type: ignore[union-attr]
    assert committed == ["Hi, this is courier C-77.", "Sorry, what is my pay this week?"]
