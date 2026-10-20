"""Playback cursor and barge-in policy (ADR-006: played audio is the source of truth).

`PlaybackCursor` maps elapsed audio time to characters of the utterance text, so on barge-in
history keeps only what the caller actually heard. `BargeInPolicy` decides when user speech
during playback is a real interruption, a backchannel to ignore, or a false start after which
playback should resume from the cursor.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from switchboard.turn.signals import FinalTranscript, InputSignal, PartialTranscript, VadFrame

_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class _Chunk:
    text: str
    duration: float
    start_pos: float  # offset of this chunk on the utterance's audio timeline, seconds


class PlaybackCursor:
    """Tracks how much of one utterance has played.

    Chunks play back to back. If the audio runs dry before the next chunk arrives, the gap is
    silence, not playback, so the playhead holds until `enqueue` re-anchors it.
    """

    def __init__(self) -> None:
        self._chunks: list[_Chunk] = []
        self._total = 0.0
        self._anchor_pos = 0.0
        self._anchor_time: float | None = None
        self._playing = True
        self._stopped = False
        self._played = ""

    @property
    def full_text(self) -> str:
        return "".join(c.text for c in self._chunks)

    @property
    def text_played(self) -> str:
        return self._played

    @property
    def position(self) -> float:
        """Audio seconds played as of the last `advance`, `pause` or `interrupt`."""
        return self._anchor_pos

    def enqueue(self, text: str, duration: float, now: float) -> None:
        if duration < 0:
            raise ValueError(f"duration must be non-negative: {duration}")
        if self._anchor_time is None:
            self._anchor_time = now
        else:
            self._reanchor(now)
        self._chunks.append(_Chunk(text, duration, self._total))
        self._total += duration

    def advance(self, now: float) -> str:
        """Mark text as played up to `now` and return everything played so far."""
        if self._stopped:
            return self._played
        self._reanchor(now)
        played = self._text_at(self._anchor_pos)
        if len(played) > len(self._played):
            self._played = played
        return self._played

    def finished(self, now: float) -> bool:
        self.advance(now)
        return bool(self._chunks) and self._anchor_pos + _EPS >= self._total

    def pause(self, now: float) -> None:
        self.advance(now)
        self._playing = False

    def resume(self, now: float) -> None:
        if self._stopped:
            return
        self._anchor_time = now
        self._playing = True

    def interrupt(self, now: float) -> str:
        """Stop playback and return the played prefix, cut back to the last whole word."""
        self.advance(now)
        self._playing = False
        self._stopped = True
        full = self.full_text
        played = self._played
        cut_mid_word = len(played) < len(full) and not (
            played[-1:].isspace() or full[len(played)].isspace()
        )
        if cut_mid_word:
            cut = max(played.rfind(" "), played.rfind("\n"), played.rfind("\t"))
            played = played[:cut] if cut >= 0 else ""
        self._played = played.rstrip()
        return self._played

    def _reanchor(self, now: float) -> None:
        if self._anchor_time is None:
            return
        if self._playing and not self._stopped:
            elapsed = max(0.0, now - self._anchor_time)
            self._anchor_pos = min(self._total, self._anchor_pos + elapsed)
        self._anchor_time = max(self._anchor_time, now)

    def _text_at(self, pos: float) -> str:
        parts: list[str] = []
        for chunk in self._chunks:
            end = chunk.start_pos + chunk.duration
            if pos + _EPS >= end:
                parts.append(chunk.text)
                continue
            if chunk.duration > 0 and pos > chunk.start_pos:
                frac = (pos - chunk.start_pos) / chunk.duration
                parts.append(chunk.text[: math.floor(len(chunk.text) * frac + _EPS)])
            break
        return "".join(parts)


# "mm-hm", "uh huh" and friends tokenize into these words; a partial made only of them is a
# backchannel, not a turn.
_BACKCHANNEL_WORDS = frozenset(
    {
        "mm",
        "mmm",
        "hm",
        "hmm",
        "mhm",
        "mm-hm",
        "mm-hmm",
        "uh",
        "huh",
        "uh-huh",
        "yeah",
        "yep",
        "yes",
        "okay",
        "ok",
        "right",
        "sure",
        "alright",
    }
)
_TOKEN = re.compile(r"[a-z']+")


def is_backchannel(text: str) -> bool:
    words = _TOKEN.findall(text.lower().replace("-", " "))
    return bool(words) and all(w in _BACKCHANNEL_WORDS for w in words)


def _is_real_utterance(text: str) -> bool:
    return bool(_TOKEN.findall(text.lower())) and not is_backchannel(text)


@dataclass(frozen=True, slots=True)
class BargeInConfig:
    """Barge-in thresholds.

    300 ms of overlapping speech filters coughs, clicks and line noise without making a real
    interruption feel ignored. `settle_ms` is the silence that ends a user burst. A paused
    agent resumes if the burst ends within `resume_window_ms` of the interrupt with no real
    words; sustained wordless speech past the window is treated as a real barge-in.
    """

    min_overlap_ms: float = 300.0
    settle_ms: float = 300.0
    resume_window_ms: float = 1000.0


BargeInKind = Literal["interrupt", "resume", "confirm"]


@dataclass(frozen=True, slots=True)
class BargeInDecision:
    """`interrupt`: pause playback at the cursor. `resume`: false alarm, continue from the
    cursor with no regeneration. `confirm`: real barge-in, truncate history with
    `PlaybackCursor.interrupt`."""

    kind: BargeInKind
    at: float


class BargeInPolicy:
    """Classifies user speech heard while one utterance is playing.

    Feed it the user's VAD frames and transcripts for the lifetime of that utterance, and call
    `tick(now)` on a schedule so a resume can fire without new frames.
    """

    def __init__(self, config: BargeInConfig | None = None) -> None:
        self.config = config or BargeInConfig()
        self._state: Literal["idle", "interrupted", "confirmed"] = "idle"
        self._burst_start: float | None = None
        self._last_speech: float | None = None
        self._partial = ""
        self._interrupt_at = 0.0

    @property
    def state(self) -> str:
        return self._state

    def observe(self, signal: InputSignal) -> BargeInDecision | None:
        if isinstance(signal, VadFrame):
            if signal.speech:
                if self._burst_start is None or self._silence_ms(signal.at) > self.config.settle_ms:
                    self._burst_start = signal.at
                    if self._state == "idle":
                        self._partial = ""
                self._last_speech = signal.at
        elif isinstance(signal, PartialTranscript | FinalTranscript):
            self._partial = signal.text
        return self.tick(signal.at)

    def tick(self, now: float) -> BargeInDecision | None:
        cfg = self.config
        if self._state == "confirmed" or self._burst_start is None or self._last_speech is None:
            return None
        settled = self._silence_ms(now) + _EPS >= cfg.settle_ms
        if self._state == "idle":
            if settled:
                self._end_burst()
                return None
            overlap_ms = (self._last_speech - self._burst_start) * 1000.0
            if overlap_ms + _EPS >= cfg.min_overlap_ms and not is_backchannel(self._partial):
                self._state = "interrupted"
                self._interrupt_at = now
                return BargeInDecision("interrupt", now)
            return None
        if _is_real_utterance(self._partial):
            self._state = "confirmed"
            return BargeInDecision("confirm", now)
        if settled:
            self._state = "idle"
            self._end_burst()
            return BargeInDecision("resume", now)
        if (now - self._interrupt_at) * 1000.0 + _EPS >= cfg.resume_window_ms:
            self._state = "confirmed"
            return BargeInDecision("confirm", now)
        return None

    def _silence_ms(self, now: float) -> float:
        return 0.0 if self._last_speech is None else (now - self._last_speech) * 1000.0

    def _end_burst(self) -> None:
        self._burst_start = None
        self._partial = ""
