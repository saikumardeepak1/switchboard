"""What a leg needs from the media plane, and how the runtime reaches other parties."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from switchboard.providers import STT, TTS, AudioChunk


class AudioSink(Protocol):
    """Where a leg's synthesized audio goes. `providers.Transport` satisfies it."""

    async def send(self, chunk: AudioChunk) -> None: ...

    async def clear(self) -> None: ...


@dataclass(frozen=True)
class LegIO:
    stt: STT
    tts: TTS
    sink: AudioSink


class Dialer(Protocol):
    """Places outbound calls for `call_party`. Returns once the far end has answered."""

    async def dial(self, party: str, leg_id: str) -> LegIO: ...
