"""Hybrid endpointing controller (ADR-007).

End of turn is a logistic over three features: silence since the last speech frame, a text
completeness score, and an optional audio model score. Silence is measured in units of the
caller's own pause habit, so a slow speaker gets proportionally longer thresholds. The module
is pure: it reads no clock and does no I/O. Callers pass signal times and call `tick(now)`
on their own schedule.
"""

from __future__ import annotations

import math
import re
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from switchboard.turn.signals import (
    FinalTranscript,
    InputSignal,
    PartialTranscript,
    TurnSignal,
    TurnSignalKind,
    VadFrame,
)

# Float slack so a frame that lands exactly on a threshold counts as reaching it.
_EPS = 1e-9


class CompletenessScorer(Protocol):
    def score(self, text: str) -> float:
        """Return how likely `text` is a finished utterance, in [0, 1]."""
        ...


class AudioTurnScorer(Protocol):
    def score(self, recent: Sequence[VadFrame]) -> float:
        """Return an end-of-turn probability in [0, 1] from recent audio frames."""
        ...


@dataclass(frozen=True, slots=True)
class EndpointingConfig:
    """Thresholds and logistic weights.

    The logistic is `z = bias + w_silence * s + w_text * (c - 0.5) + w_audio * (a - 0.5)`,
    with `s` the silence in seconds divided by the caller's pause scale, `c` the completeness
    score and `a` the audio score (the audio term is dropped when no audio scorer is set).

    The defaults are hand-calibrated so that, for an average caller:
      * a punctuated, complete sentence (c about 0.9) reaches eager at the 200 ms floor and
        end at about 500 ms of silence;
      * a neutral fragment (c = 0.5) goes eager at about 750 ms and ends at about 1.1 s;
      * a trailing conjunction or filler (c about 0.1) never crosses end before the 1500 ms
        ceiling, so "my order number is" survives a long pause.
    """

    eager_threshold: float = 0.5
    end_threshold: float = 0.8
    min_silence_ms: float = 200.0
    max_silence_ms: float = 1500.0
    bias: float = -3.0
    w_silence: float = 4.0
    w_text: float = 6.0
    w_audio: float = 4.0
    # Adaptive pause model. The prior describes a typical caller; the scale is
    # (mean + std) of observed pauses over (prior mean + prior std), so it starts at 1.0.
    pause_prior_mean_ms: float = 300.0
    pause_prior_std_ms: float = 100.0
    pause_alpha: float = 0.3
    min_pause_ms: float = 100.0
    min_pause_scale: float = 0.75
    max_pause_scale: float = 2.0
    audio_window: int = 50

    def __post_init__(self) -> None:
        if not 0.0 < self.eager_threshold <= self.end_threshold < 1.0:
            raise ValueError("need 0 < eager_threshold <= end_threshold < 1")
        if not 0.0 <= self.min_silence_ms < self.max_silence_ms:
            raise ValueError("need 0 <= min_silence_ms < max_silence_ms")
        if not 0.0 < self.min_pause_scale <= 1.0 <= self.max_pause_scale:
            raise ValueError("need 0 < min_pause_scale <= 1 <= max_pause_scale")
        if not 0.0 < self.pause_alpha <= 1.0:
            raise ValueError("pause_alpha must be in (0, 1]")


_WORD = re.compile(r"[a-z0-9']+")

_SHORT_ANSWERS = frozenset(
    {
        "yes",
        "no",
        "yeah",
        "yep",
        "nope",
        "correct",
        "right",
        "sure",
        "okay",
        "ok",
        "thanks",
        "thank you",
        "that's right",
        "that is right",
        "that's correct",
        "that's it",
        "exactly",
        "no thanks",
        "yes please",
    }
)
_FILLERS = frozenset({"um", "uh", "er", "erm", "hmm", "like"})
_TRAILING_INCOMPLETE = frozenset(
    {
        "and",
        "or",
        "but",
        "so",
        "because",
        "cause",
        "then",
        "if",
        "that",
        "which",
        "the",
        "a",
        "an",
        "to",
        "of",
        "for",
        "with",
        "at",
        "in",
        "on",
        "my",
        "your",
        "is",
        "are",
        "was",
        "it's",
        "i'm",
    }
)
_QUESTION_STARTS = frozenset(
    {
        "what",
        "where",
        "when",
        "why",
        "who",
        "which",
        "how",
        "can",
        "could",
        "would",
        "will",
        "is",
        "are",
        "do",
        "does",
        "did",
    }
)


class HeuristicCompleteness:
    """Rule-based completeness score. Cheap stand-in for a learned text model.

    Trailing conjunctions, fillers, articles and copulas ("my order is") score low regardless
    of punctuation, since STT often appends a period to a cut-off phrase. Terminal punctuation,
    question forms and complete short answers score high.
    """

    def score(self, text: str) -> float:
        stripped = text.strip()
        words = _WORD.findall(stripped.lower())
        if not words:
            return 0.4
        if " ".join(words) in _SHORT_ANSWERS:
            return 0.95
        last = words[-1]
        if last in _FILLERS:
            return 0.05
        if last in _TRAILING_INCOMPLETE:
            return 0.1
        if stripped.endswith("..."):
            return 0.3
        if stripped.endswith("?"):
            return 0.92
        if stripped.endswith((".", "!")):
            return 0.9
        if stripped.endswith((",", ";", ":", "-")):
            return 0.2
        if words[0] in _QUESTION_STARTS and len(words) >= 3:
            return 0.75
        return 0.5


def _logistic(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class _PauseModel:
    """Exponentially weighted mean and variance of a caller's within-turn pauses (ms)."""

    def __init__(self, config: EndpointingConfig) -> None:
        self._cfg = config
        self.mean = config.pause_prior_mean_ms
        self.var = config.pause_prior_std_ms**2

    def observe(self, pause_ms: float) -> None:
        a = self._cfg.pause_alpha
        diff = pause_ms - self.mean
        self.mean += a * diff
        self.var = (1.0 - a) * (self.var + a * diff * diff)

    @property
    def scale(self) -> float:
        cfg = self._cfg
        reference = cfg.pause_prior_mean_ms + cfg.pause_prior_std_ms
        raw = (self.mean + math.sqrt(self.var)) / reference
        return min(cfg.max_pause_scale, max(cfg.min_pause_scale, raw))


class EndpointingController:
    """Turns VAD frames and transcripts into `eager_end`, `end` and `resumed` signals.

    One instance per caller, so the pause model carries across that caller's turns.
    """

    def __init__(
        self,
        config: EndpointingConfig | None = None,
        scorer: CompletenessScorer | None = None,
        audio_scorer: AudioTurnScorer | None = None,
    ) -> None:
        self.config = config or EndpointingConfig()
        self._scorer: CompletenessScorer = scorer or HeuristicCompleteness()
        self._audio = audio_scorer
        self._pauses = _PauseModel(self.config)
        self._frames: deque[VadFrame] = deque(maxlen=self.config.audio_window)
        self._finals: list[str] = []
        self._partial = ""
        self._last_speech_at: float | None = None
        self._turn_active = False
        self._eager_sent = False

    @property
    def pause_scale(self) -> float:
        return self._pauses.scale

    @property
    def effective_min_silence_ms(self) -> float:
        return self.config.min_silence_ms * self._pauses.scale

    @property
    def effective_max_silence_ms(self) -> float:
        return self.config.max_silence_ms * self._pauses.scale

    @property
    def text(self) -> str:
        parts = [*self._finals, self._partial] if self._partial else self._finals
        return " ".join(p.strip() for p in parts if p.strip())

    def observe(self, signal: InputSignal) -> TurnSignal | None:
        if isinstance(signal, VadFrame):
            self._frames.append(signal)
            if signal.speech:
                return self._on_speech(signal.at)
        elif isinstance(signal, PartialTranscript):
            self._partial = signal.text
        elif isinstance(signal, FinalTranscript):
            self._finals.append(signal.text)
            self._partial = ""
        return self.tick(signal.at)

    def tick(self, now: float) -> TurnSignal | None:
        if not self._turn_active or self._last_speech_at is None:
            return None
        silence_ms = (now - self._last_speech_at) * 1000.0
        if silence_ms + _EPS >= self.effective_max_silence_ms:
            return self._emit_end(now, self.end_probability(now))
        if silence_ms + _EPS < self.effective_min_silence_ms:
            return None
        p = self.end_probability(now)
        if p >= self.config.end_threshold:
            return self._emit_end(now, p)
        if p >= self.config.eager_threshold and not self._eager_sent:
            self._eager_sent = True
            return self._signal("eager_end", now, p)
        return None

    def end_probability(self, now: float) -> float:
        cfg = self.config
        silence_s = 0.0 if self._last_speech_at is None else max(0.0, now - self._last_speech_at)
        z = cfg.bias + cfg.w_silence * silence_s / self._pauses.scale
        z += cfg.w_text * (self._scorer.score(self.text) - 0.5)
        if self._audio is not None:
            z += cfg.w_audio * (self._audio.score(tuple(self._frames)) - 0.5)
        return _logistic(z)

    def _on_speech(self, at: float) -> TurnSignal | None:
        resumed: TurnSignal | None = None
        if self._turn_active and self._last_speech_at is not None:
            gap_ms = (at - self._last_speech_at) * 1000.0
            if self.config.min_pause_ms <= gap_ms < self.effective_max_silence_ms:
                self._pauses.observe(gap_ms)
            if self._eager_sent:
                resumed = self._signal("resumed", at, 1.0 - self.end_probability(at))
                self._eager_sent = False
        self._turn_active = True
        self._last_speech_at = at
        return resumed

    def _emit_end(self, now: float, p: float) -> TurnSignal:
        signal = self._signal("end", now, p)
        self._turn_active = False
        self._eager_sent = False
        self._finals = []
        self._partial = ""
        return signal

    def _signal(self, kind: TurnSignalKind, at: float, p: float) -> TurnSignal:
        return TurnSignal(kind=kind, at=at, text=self.text, confidence=p)
