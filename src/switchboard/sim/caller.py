"""Simulated callers: scripted beats spoken through a simulated recognizer (ADR-014).

A `SimulatedCaller` is one human on one leg. It speaks through `CallerSTT` (VAD edges, a partial
per word at the persona's speaking rate, a final per segment after a sampled recognizer latency)
and hears the agent as the `AudioSink` the leg writes to. Its script is a list of beats:

    Say          speak a sentence, with optional mid-sentence pauses after given word indexes
    Pause        stop for a while between beats
    Silence      say nothing for a while (logged separately, e.g. to test re-prompts)
    WaitForAgent wait until the agent has said something and then gone quiet
    WaitFor      wait until a phrase is heard; on timeout repeat the last Say, then give up
    BargeIn      speak once the agent has been talking for N ms
    Backchannel  a short acknowledgement during agent speech ("uh huh")
    HangUp       end the script

Persona knobs change how the same script sounds: speaking rate, random mid-sentence pauses
(which is what stresses endpointing), recognizer WER and latency, background noise (VAD blips
with no words), reply gap, patience, and an optional habit of interrupting long agent turns.

Everything runs on the injected clock and on `random.Random` streams derived from the seed and
the party name, so a caller is bit-for-bit reproducible.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from switchboard.core import Clock
from switchboard.providers import (
    AudioChunk,
    LatencyModel,
    SimulatedTTS,
    SttEvent,
    SttFinal,
    SttPartial,
    SttVad,
    inject_word_errors,
)
from switchboard.runtime import LegIO

# How long a noise blip holds VAD on. Short enough that it carries no words.
NOISE_BLIP_S = 0.15


@dataclass(frozen=True)
class Persona:
    name: str = "default"
    words_per_s: float = 2.5
    # Chance of a mid-sentence pause at each word boundary, and the pause length distribution.
    pause_prob: float = 0.0
    pause_p50_s: float = 0.45
    pause_p95_s: float = 0.9
    wer: float = 0.0
    # Spurious VAD blips per second while the caller is not speaking.
    noise: float = 0.0
    reply_gap_s: float = 0.4
    stt_p50_s: float = 0.08
    stt_p95_s: float = 0.12
    patience_s: float = 12.0
    retries: int = 1
    # Interrupt any agent utterance that runs longer than this, once per utterance.
    interrupt_after_s: float | None = None
    interjection: str = "Okay, okay."


PERSONAS: Mapping[str, Persona] = {
    "clean": Persona("clean"),
    "wer_5": Persona("wer_5", wer=0.05),
    "wer_12": Persona("wer_12", wer=0.12),
    "slow_speaker": Persona(
        "slow_speaker",
        words_per_s=1.6,
        pause_prob=0.12,
        pause_p50_s=0.6,
        pause_p95_s=1.2,
        reply_gap_s=0.9,
        stt_p50_s=0.1,
        stt_p95_s=0.2,
    ),
    "barge_in_heavy": Persona("barge_in_heavy", interrupt_after_s=1.5, reply_gap_s=0.2),
}


@dataclass(frozen=True)
class Say:
    text: str
    # (word index, seconds): pause after that word, counting from 0.
    pauses: tuple[tuple[int, float], ...] = ()


@dataclass(frozen=True)
class Pause:
    seconds: float


@dataclass(frozen=True)
class Silence:
    seconds: float


@dataclass(frozen=True)
class WaitForAgent:
    gap_s: float | None = None  # defaults to the persona's reply gap
    timeout_s: float | None = None  # defaults to the persona's patience


@dataclass(frozen=True)
class WaitFor:
    phrase: str
    timeout_s: float | None = None
    repeat: bool = True
    # Match anything heard on the call so far, not only since the caller last spoke.
    since_start: bool = False


@dataclass(frozen=True)
class BargeIn:
    after_ms: float
    text: str


@dataclass(frozen=True)
class Backchannel:
    text: str = "uh huh"
    after_ms: float = 600.0


@dataclass(frozen=True)
class HangUp:
    pass


Beat = Say | Pause | Silence | WaitForAgent | WaitFor | BargeIn | Backchannel | HangUp

UtteranceKind = Literal["say", "repeat", "barge_in", "backchannel", "interjection"]
CallerStatus = Literal["waiting", "active", "completed", "gave_up", "hung_up"]


@dataclass(frozen=True)
class Utterance:
    start: float
    end: float
    text: str
    kind: UtteranceKind


@dataclass(frozen=True)
class CallerSummary:
    """What assertions and reports need from a caller, without the live object."""

    party: str
    heard: str
    status: str
    utterances: int


class CallerSTT:
    """A streaming recognizer for one simulated caller. Satisfies `providers.STT`.

    Like `providers.SimulatedSTT`, but a single utterance can contain pauses: VAD drops at each
    pause and the segment so far is finalized after the recognizer latency (or when speech
    resumes, if that comes first), the way streaming recognizers emit per-segment finals.
    """

    def __init__(
        self,
        clock: Clock,
        rng: random.Random,
        latency: LatencyModel,
        wer: float,
        words_per_s: float,
    ) -> None:
        if words_per_s <= 0:
            raise ValueError("words_per_s must be positive")
        self._clock = clock
        self._rng = rng
        self.latency = latency
        self.wer = wer
        self.words_per_s = words_per_s
        self._queue: asyncio.Queue[SttEvent | None] = asyncio.Queue()
        self._pending: set[asyncio.Task[None]] = set()
        self.closed = False

    async def push(self, frame: bytes) -> None:
        """Audio is accepted for interface parity; the simulation is driven by `speak`."""

    async def events(self) -> AsyncGenerator[SttEvent, None]:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item

    async def close(self) -> None:
        """End the stream once every in-flight final has been delivered."""
        if self._pending:
            await asyncio.gather(*self._pending)
        if not self.closed:
            self.closed = True
            self._queue.put_nowait(None)

    def _emit(self, event: SttEvent) -> None:
        if not self.closed:
            self._queue.put_nowait(event)

    async def speak(self, text: str, pauses: Mapping[int, float]) -> None:
        """Play one utterance and return when the speaker stops talking."""
        words = text.split()
        delay = 1.0 / self.words_per_s
        self._emit(SttVad(self._clock.now(), True))
        segment: list[str] = []
        for i, word in enumerate(words):
            await self._clock.sleep(delay)
            heard = inject_word_errors(word, self.wer, self._rng)
            if heard:
                segment.append(heard)
                self._emit(SttPartial(self._clock.now(), " ".join(segment)))
            pause = pauses.get(i, 0.0)
            if pause > 0 and i < len(words) - 1:
                self._emit(SttVad(self._clock.now(), False))
                lag = min(self.latency.sample(self._rng), pause)
                await self._clock.sleep(lag)
                if segment:
                    self._emit(SttFinal(self._clock.now(), " ".join(segment)))
                segment = []
                await self._clock.sleep(pause - lag)
                self._emit(SttVad(self._clock.now(), True))
        self._emit(SttVad(self._clock.now(), False))
        # The speaker is done here; the final lands after the recognizer's lag, in the background.
        task = asyncio.ensure_future(self._finalize(" ".join(segment)))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _finalize(self, text: str) -> None:
        await self._clock.sleep(self.latency.sample(self._rng))
        self._emit(SttFinal(self._clock.now(), text))

    async def blip(self, seconds: float) -> None:
        self._emit(SttVad(self._clock.now(), True))
        await self._clock.sleep(seconds)
        self._emit(SttVad(self._clock.now(), False))


class _GaveUp(Exception):
    def __init__(self, phrase: str) -> None:
        super().__init__(phrase)
        self.phrase = phrase


class SimulatedCaller:
    """One scripted human on one leg: speaks into `stt`, hears whatever the leg `send`s."""

    def __init__(
        self,
        clock: Clock,
        party: str,
        beats: Sequence[Beat],
        persona: Persona | None = None,
        *,
        seed: int,
        tts_ttfb_s: float = 0.15,
        tts_chars_per_s: float = 16.0,
    ) -> None:
        self.clock = clock
        self.party = party
        self.beats = list(beats)
        self.persona = persona or Persona()
        self.seed = seed
        self._rng = random.Random(f"{seed}:{party}:caller")
        p = self.persona
        self.stt = CallerSTT(
            clock,
            random.Random(f"{seed}:{party}:stt"),
            LatencyModel(p.stt_p50_s, p.stt_p95_s),
            p.wer,
            p.words_per_s,
        )
        self.tts = SimulatedTTS(clock, ttfb_s=tts_ttfb_s, chars_per_s=tts_chars_per_s)
        self._pause_model = LatencyModel(p.pause_p50_s, p.pause_p95_s)
        self.chunks: list[AudioChunk] = []
        self.cleared = 0
        self.audio_until = float("-inf")
        self.segment_starts: list[float] = []
        self.utterances: list[Utterance] = []
        self.log: list[tuple[float, str, str]] = []
        self.status: CallerStatus = "waiting"
        self.gave_up_on: str | None = None
        self.started = False
        self._changed = asyncio.Event()
        self._speaking = False
        self._last_say: Say | None = None
        self._say_mark = 0
        self._interrupted_segments: set[int] = set()

    # the sink the leg writes agent audio to

    @property
    def io(self) -> LegIO:
        return LegIO(self.stt, self.tts, self)

    async def send(self, chunk: AudioChunk) -> None:
        if chunk.at > self.audio_until:
            self.segment_starts.append(chunk.at)
        self.chunks.append(chunk)
        self.audio_until = max(self.audio_until, chunk.at) + chunk.duration_s
        self._changed.set()

    async def clear(self) -> None:
        self.cleared += 1
        self.audio_until = min(self.audio_until, self.clock.now())
        self._changed.set()

    @property
    def heard(self) -> str:
        return "".join(c.text for c in self.chunks)

    def heard_since(self, mark: int) -> str:
        return "".join(c.text for c in self.chunks[mark:])

    @property
    def agent_speaking(self) -> bool:
        return self.clock.now() < self.audio_until

    def summary(self) -> CallerSummary:
        return CallerSummary(self.party, self.heard, self.status, len(self.utterances))

    # the script

    async def run(self) -> None:
        if self.started:
            raise RuntimeError(f"caller {self.party!r} already ran")
        self.started = True
        self.status = "active"
        noise = asyncio.ensure_future(self._noise()) if self.persona.noise > 0 else None
        try:
            for beat in self.beats:
                if isinstance(beat, HangUp):
                    self.status = "hung_up"
                    self._note("hang_up", "")
                    return
                await self._beat(beat)
            self.status = "completed"
        except _GaveUp as exc:
            self.status = "gave_up"
            self.gave_up_on = exc.phrase
            self._note("gave_up", exc.phrase)
        finally:
            if noise is not None:
                noise.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await noise
            await self.stt.close()

    async def _beat(self, beat: Beat) -> None:
        if isinstance(beat, Say):
            self._last_say = beat
            await self._say(beat, "say")
        elif isinstance(beat, Pause | Silence):
            self._note("pause" if isinstance(beat, Pause) else "silence", f"{beat.seconds:g}")
            await self.clock.sleep(beat.seconds)
        elif isinstance(beat, WaitForAgent):
            await self._wait_for_agent(beat)
        elif isinstance(beat, WaitFor):
            await self._wait_for(beat)
        elif isinstance(beat, BargeIn):
            await self._speak_into_agent(beat.after_ms, beat.text, "barge_in")
        elif isinstance(beat, Backchannel):
            await self._speak_into_agent(beat.after_ms, beat.text, "backchannel")

    async def _say(self, beat: Say, kind: UtteranceKind) -> None:
        self._say_mark = len(self.chunks)
        pauses = dict(beat.pauses)
        if self.persona.pause_prob > 0:
            for i in range(len(beat.text.split()) - 1):
                if i not in pauses and self._rng.random() < self.persona.pause_prob:
                    pauses[i] = self._pause_model.sample(self._rng)
        await self._utter(beat.text, kind, pauses)

    async def _utter(
        self, text: str, kind: UtteranceKind, pauses: Mapping[int, float] | None = None
    ) -> None:
        start = self.clock.now()
        self._speaking = True
        try:
            await self.stt.speak(text, pauses or {})
        finally:
            self._speaking = False
        self.utterances.append(Utterance(start, self.clock.now(), text, kind))
        self._note(kind, text)

    async def _wait_for(self, beat: WaitFor) -> None:
        timeout = beat.timeout_s if beat.timeout_s is not None else self.persona.patience_s
        attempts = self.persona.retries if beat.repeat and self._last_say is not None else 0
        phrase = beat.phrase.lower()
        while True:
            mark = 0 if beat.since_start else self._say_mark

            def heard(mark: int = mark) -> bool:
                return phrase in self.heard_since(mark).lower()

            if await self._wait(heard, timeout):
                return
            if attempts <= 0 or self._last_say is None:
                raise _GaveUp(beat.phrase)
            attempts -= 1
            await self._say(self._last_say, "repeat")

    async def _wait_for_agent(self, beat: WaitForAgent) -> None:
        gap = beat.gap_s if beat.gap_s is not None else self.persona.reply_gap_s
        timeout = beat.timeout_s if beat.timeout_s is not None else self.persona.patience_s
        mark = self._say_mark

        def quiet() -> bool:
            return len(self.chunks) > mark and self.clock.now() >= self.audio_until + gap

        def wake() -> float | None:
            return self.audio_until + gap if len(self.chunks) > mark else None

        await self._wait(quiet, timeout, wake)

    async def _speak_into_agent(self, after_ms: float, text: str, kind: UtteranceKind) -> None:
        begun = self.clock.now()
        if not await self._wait(lambda: self.agent_speaking, self.persona.patience_s):
            self._note("no_agent_speech", text)
        start = max(begun, self.segment_starts[-1] if self.segment_starts else begun)
        await self.clock.sleep(start + after_ms / 1000.0 - self.clock.now())
        await self._utter(text, kind)

    async def _wait(
        self,
        cond: Callable[[], bool],
        within_s: float,
        wake: Callable[[], float | None] | None = None,
    ) -> bool:
        """Wait for `cond` on the clock, re-checking on new audio and at `wake()` times."""
        deadline = self.clock.now() + within_s
        while True:
            await self._maybe_interject()
            if cond():
                return True
            now = self.clock.now()
            if now >= deadline:
                return False
            wakes = [deadline]
            if wake is not None and (w := wake()) is not None and w > now:
                wakes.append(w)
            if (i := self._interject_at()) is not None and i > now:
                wakes.append(i)
            await self._changed_or(min(wakes) - now)

    async def _changed_or(self, seconds: float) -> None:
        self._changed.clear()
        changed = asyncio.ensure_future(self._changed.wait())
        timer = asyncio.ensure_future(self.clock.sleep(seconds))
        try:
            await asyncio.wait({changed, timer}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (changed, timer):
                task.cancel()

    def _interject_at(self) -> float | None:
        after = self.persona.interrupt_after_s
        n = len(self.segment_starts)
        if after is None or not n or (n - 1) in self._interrupted_segments:
            return None
        if not self.agent_speaking:
            return None
        return self.segment_starts[-1] + after

    async def _maybe_interject(self) -> None:
        at = self._interject_at()
        if at is not None and self.clock.now() >= at:
            self._interrupted_segments.add(len(self.segment_starts) - 1)
            await self._utter(self.persona.interjection, "interjection")

    async def _noise(self) -> None:
        while True:
            await self.clock.sleep(self._rng.expovariate(self.persona.noise))
            if not self._speaking:
                self._note("noise", "")
                await self.stt.blip(NOISE_BLIP_S)

    def _note(self, kind: str, detail: str) -> None:
        self.log.append((self.clock.now(), kind, detail))


@dataclass
class SimDialer:
    """Places the runtime's outbound calls to simulated callers. Satisfies `runtime.Dialer`."""

    clock: Clock
    callers: Mapping[str, SimulatedCaller]
    ring_s: float = 2.0
    dialed: list[tuple[str, str]] = field(default_factory=list)
    tasks: list[asyncio.Task[None]] = field(default_factory=list)

    async def dial(self, party: str, leg_id: str) -> LegIO:
        caller = self.callers.get(party)
        if caller is None:
            raise LookupError(f"no simulated caller for party {party!r}")
        if caller.started or any(p == party for p, _ in self.dialed):
            raise LookupError(f"{party!r} is already on a call")
        self.dialed.append((party, leg_id))
        await self.clock.sleep(self.ring_s)
        self.tasks.append(asyncio.ensure_future(caller.run()))
        return caller.io
