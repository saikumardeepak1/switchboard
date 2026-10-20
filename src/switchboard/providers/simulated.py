"""Simulated STT and TTS on the injected clock (ADR-014).

Both are deterministic given the clock and the `random.Random` passed in, so a simulated call
replays bit for bit from a seed.
"""

from __future__ import annotations

import asyncio
import math
import random
import re
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field

from switchboard.core.clock import Clock
from switchboard.providers.interfaces import AudioChunk, SttEvent, SttFinal, SttPartial, SttVad

# Standard normal quantile at 0.95.
_Z95 = 1.6448536269514722


@dataclass(frozen=True, slots=True)
class LatencyModel:
    """Lognormal latency fitted to a median and a 95th percentile.

    For a lognormal X = exp(mu + sigma * Z), the median is exp(mu) and the q-quantile is
    exp(mu + sigma * z_q). So mu = ln(p50) and sigma = ln(p95 / p50) / z_0.95. Two published
    numbers fix the distribution, and the right skew matches how provider latency behaves:
    most calls near the median, a long tail of slow ones.
    """

    p50: float
    p95: float
    mu: float = field(init=False)
    sigma: float = field(init=False)

    def __post_init__(self) -> None:
        if self.p50 <= 0 or self.p95 < self.p50:
            raise ValueError(f"need 0 < p50 <= p95, got p50={self.p50}, p95={self.p95}")
        object.__setattr__(self, "mu", math.log(self.p50))
        object.__setattr__(self, "sigma", math.log(self.p95 / self.p50) / _Z95)

    def sample(self, rng: random.Random) -> float:
        return rng.lognormvariate(self.mu, self.sigma)


def _pairs(*groups: tuple[str, str]) -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    for a, b in groups:
        out.setdefault(a, ())
        out.setdefault(b, ())
        out[a] += (b,)
        out[b] += (a,)
    return out


# Small, plausible ASR confusions. Teen/ty pairs matter most: they change amounts and times.
CONFUSIONS: Mapping[str, tuple[str, ...]] = _pairs(
    ("thirteen", "thirty"),
    ("fourteen", "forty"),
    ("fifteen", "fifty"),
    ("sixteen", "sixty"),
    ("seventeen", "seventy"),
    ("eighteen", "eighty"),
    ("nineteen", "ninety"),
    ("13", "30"),
    ("14", "40"),
    ("15", "50"),
    ("16", "60"),
    ("17", "70"),
    ("18", "80"),
    ("19", "90"),
    ("four", "for"),
    ("two", "to"),
    ("eight", "ate"),
    ("nine", "five"),
    ("cancel", "council"),
    ("refund", "re fund"),
    ("address", "a dress"),
)

_WORD = re.compile(r"^(\W*)(.*?)(\W*)$")


def inject_word_errors(text: str, wer: float, rng: random.Random) -> str:
    """Corrupt each word with probability `wer`.

    A corrupted word is substituted from CONFUSIONS when it has an entry, else deleted.
    Leading and trailing punctuation and an initial capital survive substitution. Exactly one
    rng draw per word, plus one per substitution, keeps runs reproducible.
    """
    if not 0.0 <= wer <= 1.0:
        raise ValueError(f"wer must be within [0, 1], got {wer}")
    out: list[str] = []
    for token in text.split():
        if rng.random() >= wer:
            out.append(token)
            continue
        lead, core, trail = _split(token)
        options = CONFUSIONS.get(core.lower())
        if not options:
            continue
        replacement = rng.choice(options)
        if core[:1].isupper():
            replacement = replacement[:1].upper() + replacement[1:]
        out.append(f"{lead}{replacement}{trail}")
    return " ".join(out)


def _split(token: str) -> tuple[str, str, str]:
    m = _WORD.match(token)
    if m is None:  # pragma: no cover - the pattern matches every string
        return "", token, ""
    return m.group(1), m.group(2), m.group(3)


_CLOSED = object()


class SimulatedSTT:
    """Turns a simulated caller's spoken text into STT events.

    `speak(text)` plays one utterance: VAD on, one partial per word at the speaking rate,
    VAD off after the last word, then the final after a latency drawn from `latency`. Word
    errors are drawn once per utterance, so partials are always prefixes of the final.
    """

    def __init__(
        self,
        clock: Clock,
        rng: random.Random,
        latency: LatencyModel,
        wer: float = 0.0,
        words_per_s: float = 2.5,
    ) -> None:
        if words_per_s <= 0:
            raise ValueError("words_per_s must be positive")
        self._clock = clock
        self._rng = rng
        self.latency = latency
        self.wer = wer
        self.words_per_s = words_per_s
        self.audio_bytes = 0
        self._queue: asyncio.Queue[SttEvent | object] = asyncio.Queue()
        self._closed = False

    async def push(self, frame: bytes) -> None:
        # Audio is accepted for interface parity; the simulation is driven by `speak`.
        self.audio_bytes += len(frame)

    async def speak(self, text: str) -> None:
        if self._closed:
            raise RuntimeError("STT stream is closed")
        spoken = text.split()
        heard = [inject_word_errors(word, self.wer, self._rng) for word in spoken]
        delay = 1.0 / self.words_per_s
        self._emit(SttVad(at=self._clock.now(), speech=True))
        transcript: list[str] = []
        last_partial = ""
        for word in heard:
            await self._clock.sleep(delay)
            if word:
                transcript.append(word)
            partial = " ".join(transcript)
            if partial and partial != last_partial:
                self._emit(SttPartial(at=self._clock.now(), text=partial))
                last_partial = partial
        self._emit(SttVad(at=self._clock.now(), speech=False))
        await self._clock.sleep(self.latency.sample(self._rng))
        self._emit(SttFinal(at=self._clock.now(), text=" ".join(transcript)))

    async def events(self) -> AsyncGenerator[SttEvent, None]:
        while True:
            item = await self._queue.get()
            if item is _CLOSED:
                return
            assert isinstance(item, SttPartial | SttFinal | SttVad)
            yield item

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._queue.put_nowait(_CLOSED)

    def _emit(self, event: SttEvent) -> None:
        self._queue.put_nowait(event)


class SimulatedTTS:
    """Streams silent PCM chunks with realistic timing.

    The first chunk lands after `ttfb_s`; later chunks are paced at real time, each arriving
    as the previous one's audio would finish. Chunk duration is its text length over
    `chars_per_s`, so playback cursors can map elapsed audio back to characters.
    """

    def __init__(
        self,
        clock: Clock,
        ttfb_s: float,
        chars_per_s: float,
        chunk_chars: int = 40,
        sample_rate: int = 8000,
    ) -> None:
        if ttfb_s < 0 or chars_per_s <= 0 or chunk_chars <= 0 or sample_rate <= 0:
            raise ValueError("ttfb_s >= 0 and positive chars_per_s, chunk_chars, sample_rate")
        self._clock = clock
        self.ttfb_s = ttfb_s
        self.chars_per_s = chars_per_s
        self.chunk_chars = chunk_chars
        self.sample_rate = sample_rate

    async def synthesize(self, text: str) -> AsyncGenerator[AudioChunk, None]:
        pieces = _chunk_text(text, self.chunk_chars)
        if not pieces:
            return
        await self._clock.sleep(self.ttfb_s)
        previous = 0.0
        for i, piece in enumerate(pieces):
            if i:
                await self._clock.sleep(previous)
            duration = len(piece) / self.chars_per_s
            samples = round(duration * self.sample_rate)
            yield AudioChunk(
                at=self._clock.now(), pcm=bytes(2 * samples), duration_s=duration, text=piece
            )
            previous = duration


def _chunk_text(text: str, limit: int) -> list[str]:
    """Group whole words (with their trailing spaces) into pieces of about `limit` chars."""
    pieces: list[str] = []
    current = ""
    for word in re.findall(r"\s*\S+\s*", text):
        if current and len(current) + len(word) > limit:
            pieces.append(current)
            current = ""
        current += word
    if current:
        pieces.append(current)
    return pieces
