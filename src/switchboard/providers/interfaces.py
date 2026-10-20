"""Provider interfaces for the media side of a call leg.

Audio inside the control plane is 16-bit signed little-endian mono PCM. Transports convert
from their wire codec (mu-law for Twilio) at the edge. The STT event types are defined here
rather than imported from `turn`, so providers stay independent of the turn package.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class SttPartial:
    at: float
    text: str


@dataclass(frozen=True, slots=True)
class SttFinal:
    at: float
    text: str


@dataclass(frozen=True, slots=True)
class SttVad:
    at: float
    speech: bool


SttEvent = SttPartial | SttFinal | SttVad


@dataclass(frozen=True, slots=True)
class AudioFrame:
    at: float
    pcm: bytes


@dataclass(frozen=True, slots=True)
class AudioChunk:
    at: float
    pcm: bytes
    duration_s: float
    text: str
    # Which provider synthesized it, when a fallback chain is in front (ADR-013).
    provider: str | None = None


@runtime_checkable
class STT(Protocol):
    async def push(self, frame: bytes) -> None:
        """Feed one frame of caller audio."""
        ...

    def events(self) -> AsyncIterator[SttEvent]:
        """Transcript and VAD events, ending after `close`."""
        ...

    async def close(self) -> None: ...


@runtime_checkable
class TTS(Protocol):
    def synthesize(self, text: str) -> AsyncIterator[AudioChunk]: ...


@runtime_checkable
class Transport(Protocol):
    """One call leg's audio connection."""

    def inbound(self) -> AsyncIterator[AudioFrame]:
        """Caller audio, ending when the leg hangs up."""
        ...

    async def send(self, chunk: AudioChunk) -> None:
        """Queue agent audio for playback."""
        ...

    async def clear(self) -> None:
        """Drop queued, unplayed agent audio (barge-in)."""
        ...

    async def flush(self) -> None:
        """Wait until queued agent audio has been handed to the far end."""
        ...

    async def close(self) -> None: ...
