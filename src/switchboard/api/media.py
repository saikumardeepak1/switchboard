"""Twilio Media Streams WebSocket endpoint.

The endpoint is transport only: it decodes caller audio and hands it to a `MediaSession`, and
frames whatever the session wants to say. The session (built by the runtime) owns turn-taking.

Inbound: `media` messages on the inbound track are decoded from 8 kHz mu-law to PCM16 and passed
to `on_audio`. Twilio echoes each `mark` once the audio queued before it has played; the echo is
forwarded to `on_mark`, which is how the runtime's playback cursor learns what the caller heard.

Outbound: the session yields `Utterance(pcm16, mark)` items (8 kHz mono PCM16) and `Clear()`
items. An utterance is encoded to mu-law, cut into 20 ms frames of 160 bytes (the last padded
with silence), sent as `media` messages, and followed by a `mark` named by the session. `Clear`
sends `clear`, which drops audio Twilio has queued but not played: that is barge-in. Frames are
not paced; Twilio buffers and plays them in real time, and `clear` empties that buffer.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Protocol

from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

from switchboard.api.stream_auth import POLICY_VIOLATION, InvalidStreamToken, StreamAuth
from switchboard.providers.twilio import (
    TwilioMark,
    TwilioMedia,
    TwilioProtocolError,
    TwilioStart,
    TwilioStop,
    build_clear,
    build_mark,
    build_media,
    decode_ulaw,
    encode_ulaw,
    parse_message,
    split_frames,
)
from switchboard.telemetry.metrics import Registry

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Utterance:
    pcm16: bytes
    mark: str


@dataclass(frozen=True, slots=True)
class Clear:
    """Barge-in: drop everything queued for playback."""


OutboundAudio = Utterance | Clear


class MediaSession(Protocol):
    async def on_audio(self, pcm16: bytes) -> None: ...

    def outbound(self) -> AsyncIterator[OutboundAudio]: ...

    async def on_mark(self, name: str) -> None: ...

    async def close(self) -> None: ...


class SessionFactory(Protocol):
    async def start(
        self, call_sid: str, stream_sid: str, custom_parameters: Mapping[str, str]
    ) -> MediaSession: ...


class MediaStreamHandler:
    def __init__(self, factory: SessionFactory, registry: Registry, auth: StreamAuth) -> None:
        self._factory = factory
        self._auth = auth
        self._rejected = registry.counter(
            "switchboard_twilio_streams_rejected_total", "Streams refused by authentication."
        )
        self._active = registry.gauge(
            "switchboard_twilio_streams_active", "Twilio media streams currently open."
        )
        self._frames_in = registry.counter(
            "switchboard_twilio_frames_in_total", "Inbound caller media messages decoded."
        )
        self._frames_out = registry.counter(
            "switchboard_twilio_frames_out_total", "Outbound 20 ms mu-law frames sent."
        )
        self._errors = registry.counter(
            "switchboard_twilio_protocol_errors_total", "Inbound messages that failed to parse."
        )

    async def __call__(self, ws: WebSocket) -> None:
        if not self._auth.check_upgrade(str(ws.url), ws.headers.get("x-twilio-signature")):
            self._rejected.inc()
            await ws.close(POLICY_VIOLATION)
            return
        await ws.accept()
        session: MediaSession | None = None
        pump: asyncio.Task[None] | None = None
        try:
            while True:
                try:
                    raw = await ws.receive_text()
                except WebSocketDisconnect:
                    break
                try:
                    msg = parse_message(raw)
                except TwilioProtocolError as exc:
                    self._errors.inc()
                    log.warning("dropping malformed Twilio message: %s", exc)
                    continue
                if isinstance(msg, TwilioStart) and session is None:
                    try:
                        claims = self._auth.claims(msg.call_sid, msg.custom_parameters)
                    except InvalidStreamToken as exc:
                        self._rejected.inc()
                        log.warning("refusing Twilio stream %s: %s", msg.stream_sid, exc)
                        await _close(ws, code=POLICY_VIOLATION)
                        break
                    # Tenant, customer and party come from the verified token only.
                    session = await self._factory.start(
                        msg.call_sid, msg.stream_sid, claims.parameters()
                    )
                    self._active.inc()
                    pump = asyncio.create_task(self._pump(ws, session, msg.stream_sid))
                elif isinstance(msg, TwilioMedia) and session is not None:
                    if msg.track == "inbound":
                        self._frames_in.inc()
                        await session.on_audio(decode_ulaw(msg.payload))
                elif isinstance(msg, TwilioMark) and session is not None:
                    await session.on_mark(msg.name)
                elif isinstance(msg, TwilioStop):
                    break
        finally:
            if pump is not None:
                pump.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await pump
            if session is not None:
                await session.close()
                self._active.dec()
            await _close(ws)

    async def _pump(self, ws: WebSocket, session: MediaSession, stream_sid: str) -> None:
        try:
            async for item in session.outbound():
                if isinstance(item, Clear):
                    await ws.send_text(build_clear(stream_sid))
                    continue
                pcm = item.pcm16[: len(item.pcm16) - len(item.pcm16) % 2]
                for frame in split_frames(encode_ulaw(pcm)):
                    await ws.send_text(build_media(stream_sid, frame))
                    self._frames_out.inc()
                await ws.send_text(build_mark(stream_sid, item.mark))
        except asyncio.CancelledError:
            raise
        except Exception:
            # Without outbound audio the call is dead air; end it so Twilio can fail over.
            log.exception("outbound audio pump failed for stream %s", stream_sid)
            await _close(ws, code=1011)


async def _close(ws: WebSocket, code: int = 1000) -> None:
    if (
        ws.application_state == WebSocketState.CONNECTED
        and ws.client_state == WebSocketState.CONNECTED
    ):
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await ws.close(code)
