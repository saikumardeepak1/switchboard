"""One leg's mouth: gated sentences in, synthesized audio out, played text recorded (ADR-006).

Each utterance gets a `PlaybackCursor` and a `BargeInPolicy`. When an utterance finishes playing
the speaker appends `speech.played` with the full text; when the caller barges in it stops, clears
the sink and appends `speech.played` with `interrupted=True` and only the text that was heard.
History is built from those events, so it never holds words the caller did not hear.

Escalated sentences (ADR-005 tier 3) carry a verdict task. The speaker waits for the verdict only
while earlier audio is still playing, plus a short grace. A verdict that arrives in time decides
release or drop; a sentence that had to be played unverified and is later blocked gets a spoken
correction before the utterance closes (block-or-self-correct).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from switchboard.core import Clock
from switchboard.grounding import GateDecision
from switchboard.turn import BargeInConfig, BargeInDecision, BargeInPolicy, InputSignal
from switchboard.turn.playback import PlaybackCursor

if TYPE_CHECKING:
    from switchboard.runtime.io import LegIO

_EPS = 1e-9
CORRECTION = "Sorry, I need to correct that last part, so please ignore it."


@dataclass(frozen=True, slots=True)
class Spoken:
    """A gated sentence on its way to TTS, as the turn engine buffers it.

    `end` marks the close of an utterance and carries no text. `verdict` is set for sentences
    still waiting on the tier 3 verifier.
    """

    text: str
    utterance_id: str
    end: bool = field(default=False, kw_only=True)
    verdict: asyncio.Task[GateDecision] | None = field(default=None, kw_only=True)


@dataclass
class _Utterance:
    utterance_id: str
    cursor: PlaybackCursor = field(default_factory=PlaybackCursor)
    policy: BargeInPolicy = field(default_factory=BargeInPolicy)
    audio_end: float = 0.0
    paused_at: float | None = None
    resumed: asyncio.Event = field(default_factory=asyncio.Event)
    unverified: list[asyncio.Task[GateDecision]] = field(default_factory=list)
    providers: list[str] = field(default_factory=list)

    @property
    def provider(self) -> str | None:
        return "+".join(self.providers) or None


# Called with (utterance_id, text_played, interrupted, tts_provider); appends speech.played.
PlayedCallback = Callable[[str, str, bool, str | None], Awaitable[None]]
# Called with (utterance_id, sentence); records a correction sentence as released speech.
CorrectionCallback = Callable[[str, str], Awaitable[None]]


class Speaker:
    def __init__(
        self,
        io: LegIO,
        clock: Clock,
        *,
        on_played: PlayedCallback,
        on_correction: CorrectionCallback,
        on_barge_in: Callable[[], Awaitable[None]],
        on_error: Callable[[BaseException], None] | None = None,
        barge_in: BargeInConfig | None = None,
        escalation_grace_s: float = 0.3,
    ) -> None:
        self._io = io
        self._clock = clock
        self._on_played = on_played
        self._on_correction = on_correction
        self._on_barge_in = on_barge_in
        self._on_error = on_error
        self._barge_in = barge_in or BargeInConfig()
        self._grace = escalation_grace_s
        self._queue: asyncio.Queue[Spoken] = asyncio.Queue()
        self._utterance: _Utterance | None = None
        self._task: asyncio.Task[None] | None = None
        self._idle = asyncio.Event()
        self._idle.set()
        self.spoken: list[str] = []

    @property
    def active(self) -> bool:
        return self._utterance is not None

    @property
    def speaking(self) -> bool:
        """Audio is playing or queued to play."""
        return not self._idle.is_set()

    @property
    def utterance_id(self) -> str | None:
        return self._utterance.utterance_id if self._utterance is not None else None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="speaker")

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await asyncio.wait([self._task])
        self._task = None

    def say(self, item: Spoken) -> None:
        self._idle.clear()
        self._queue.put_nowait(item)

    async def drain(self) -> None:
        """Wait until everything queued has played."""
        await self._idle.wait()

    async def observe(self, signal: InputSignal) -> None:
        u = self._utterance
        if u is not None:
            await self._apply(u, u.policy.observe(signal))

    async def tick(self, now: float) -> None:
        u = self._utterance
        if u is not None:
            await self._apply(u, u.policy.tick(now))

    async def _apply(self, u: _Utterance, decision: BargeInDecision | None) -> None:
        if decision is None or u is not self._utterance:
            return
        now = self._clock.now()
        if decision.kind == "interrupt":
            u.cursor.pause(now)
            u.paused_at = now
            u.resumed.clear()
        elif decision.kind == "resume":
            u.cursor.resume(now)
            if u.paused_at is not None:
                u.audio_end += now - u.paused_at
            u.paused_at = None
            u.resumed.set()
        else:
            await self._interrupt(u)

    async def _interrupt(self, u: _Utterance) -> None:
        text = u.cursor.interrupt(self._clock.now())
        self._utterance = None
        await self.stop()
        while not self._queue.empty():
            self._queue.get_nowait()
        await self._io.sink.clear()
        for task in u.unverified:
            task.cancel()
        await self._on_played(u.utterance_id, text, True, u.provider)
        self._idle.set()
        self.start()
        await self._on_barge_in()

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                await self._process(item)
            except Exception as exc:
                # One bad sentence must not silence the leg for the rest of the call.
                self._utterance = None
                if self._on_error is not None:
                    self._on_error(exc)
            if self._queue.empty() and self._utterance is None:
                self._idle.set()

    async def _process(self, item: Spoken) -> None:
        u = self._utterance
        if u is not None and u.utterance_id != item.utterance_id:
            # The previous utterance lost its end marker (its turn failed); close it anyway.
            await self._finish(u)
            u = None
        if item.end:
            if u is not None:
                await self._finish(u)
            return
        if u is None:
            u = _Utterance(item.utterance_id, policy=BargeInPolicy(self._barge_in))
            u.resumed.set()
            self._utterance = u
        if item.verdict is not None and not await self._verified(u, item.verdict):
            return
        await self._synthesize(u, item.text)

    async def _verified(self, u: _Utterance, verdict: asyncio.Task[GateDecision]) -> bool:
        """Race the verifier against the audio already queued. False means drop the sentence."""
        if not verdict.done():
            wait_s = max(0.0, u.audio_end - self._clock.now()) + self._grace
            timer = asyncio.ensure_future(self._clock.sleep(wait_s))
            try:
                await asyncio.wait({verdict, timer}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                timer.cancel()
        if not verdict.done():
            u.unverified.append(verdict)
            return True
        if verdict.cancelled() or verdict.exception() is not None:
            return False  # fail closed
        return verdict.result().decision == "release"

    async def _synthesize(self, u: _Utterance, text: str) -> None:
        self.spoken.append(text)
        if u.cursor.full_text:
            # Sentences arrive without the space between them; keep played text readable.
            u.cursor.enqueue(" ", 0.0, self._clock.now())
        async for chunk in self._io.tts.synthesize(text):
            now = self._clock.now()
            u.audio_end = max(now, u.audio_end) + chunk.duration_s
            u.cursor.enqueue(chunk.text, chunk.duration_s, now)
            if chunk.provider is not None and chunk.provider not in u.providers:
                u.providers.append(chunk.provider)
            await self._io.sink.send(chunk)

    async def _finish(self, u: _Utterance) -> None:
        for verdict in u.unverified:
            decision = await _settled(verdict)
            if decision is not None and decision.decision != "release":
                await self._on_correction(u.utterance_id, CORRECTION)
                await self._synthesize(u, CORRECTION)
        while True:
            if u.paused_at is not None:
                await u.resumed.wait()
                continue
            remaining = u.audio_end - self._clock.now()
            if remaining <= _EPS:
                break
            await self._clock.sleep(remaining)
        self._utterance = None
        played = u.cursor.advance(self._clock.now())
        await self._on_played(u.utterance_id, played, False, u.provider)


async def _settled(verdict: asyncio.Task[GateDecision]) -> GateDecision | None:
    """The verdict, or None if the verdict task itself was cancelled.

    A cancellation aimed at the speaker (stop, barge-in, hang-up) arrives here too, because the
    speaker is awaiting the verdict; it must propagate or `stop()` would never return.
    """
    try:
        return await verdict
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            raise
        return None
    except Exception as exc:
        # Fail closed: a sentence that played unverified gets the correction.
        return GateDecision("block", 3, (), (), f"verifier failed: {type(exc).__name__}")
