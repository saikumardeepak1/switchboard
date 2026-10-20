"""The Twilio media path: an `api.media.SessionFactory` that runs a conversation per stream.

`api.media.MediaStreamHandler` is transport only. This module is the other half: each Twilio
stream becomes one inbound leg of a new conversation. Inbound caller audio (8 kHz PCM16, already
decoded from mu-law) is pushed into a per-call STT stream. Outbound, the session is the leg's
`AudioSink`: every synthesized chunk becomes an `Utterance` followed by its own mark, and a
barge-in (the speaker clearing its sink) becomes `Clear`, which makes Twilio drop the audio it
has queued but not played.

Mark echoes are recorded on the session. The speaker's playback cursor still runs on the clock;
marks say which chunks Twilio played, which is the hook for a cursor driven by the far end.

`call_sid` names the STT and TTS streams; custom parameters on the stream (`<Parameter>` in
TwiML) carry the tenant, the customer and the party name.
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import AsyncIterator, Callable, Mapping

from switchboard.api.media import Clear, OutboundAudio, Utterance
from switchboard.bundle import Bundle
from switchboard.providers import STT, TTS, AudioChunk
from switchboard.runtime.conversation import Conversation, Runtime
from switchboard.runtime.io import LegIO

SttFactory = Callable[[str], STT]
TtsFactory = Callable[[str], TTS]


class TwilioSession:
    """One Twilio media stream: a `MediaSession` for the API and an `AudioSink` for the leg."""

    def __init__(self, call_sid: str, stt: STT) -> None:
        self.call_sid = call_sid
        self.conversation: Conversation | None = None
        self.heard_marks: list[str] = []
        self._stt = stt
        self._queue: asyncio.Queue[OutboundAudio | None] = asyncio.Queue()
        self._marks = itertools.count(1)
        self._closed = False

    # AudioSink, written by the leg's speaker

    async def send(self, chunk: AudioChunk) -> None:
        if not self._closed and chunk.pcm:
            self._queue.put_nowait(Utterance(chunk.pcm, f"{self.call_sid}:{next(self._marks)}"))

    async def clear(self) -> None:
        if not self._closed:
            self._queue.put_nowait(Clear())

    # MediaSession, driven by the WebSocket handler

    async def on_audio(self, pcm16: bytes) -> None:
        if not self._closed:
            await self._stt.push(pcm16)

    async def outbound(self) -> AsyncIterator[OutboundAudio]:
        while (item := await self._queue.get()) is not None:
            yield item

    async def on_mark(self, name: str) -> None:
        self.heard_marks.append(name)

    async def close(self) -> None:
        """The caller hung up or the stream stopped: end the conversation."""
        if self._closed:
            return
        self._closed = True
        conv = self.conversation
        if conv is not None and not conv.state.ended:
            await conv.end("hangup")
        await self._stt.close()
        self._queue.put_nowait(None)


class TwilioSessionFactory:
    def __init__(
        self,
        runtime: Runtime,
        bundle: Bundle,
        *,
        stt: SttFactory,
        tts: TtsFactory,
        default_tenant: str = "default",
        default_party: str = "caller",
    ) -> None:
        self.runtime = runtime
        self.bundle = bundle
        self._stt = stt
        self._tts = tts
        self._default_tenant = default_tenant
        self._default_party = default_party
        self.sessions: dict[str, TwilioSession] = {}

    async def start(
        self, call_sid: str, stream_sid: str, custom_parameters: Mapping[str, str]
    ) -> TwilioSession:
        """Admit the call and open its inbound leg. Raises `AdmissionRejected` at the cap."""
        stt = self._stt(call_sid)
        session = TwilioSession(call_sid, stt)
        session.conversation = await self.runtime.start(
            self.bundle,
            tenant_id=custom_parameters.get("tenant", self._default_tenant),
            customer_id=custom_parameters.get("customer", call_sid),
            io=LegIO(stt, self._tts(call_sid), session),
            party=custom_parameters.get("party", self._default_party),
        )
        self.sessions[call_sid] = session
        return session
