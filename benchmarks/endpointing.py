"""Endpointing (ADR-007): hybrid controller vs a fixed 800 ms silence timer vs text only.

Turns are generated from support-call templates with marked hesitation points ("my order number
is | four eight two one three"). Each simulated caller has their own speaking rate and pause
habit; at each marked point they pause with probability `P_PAUSE` for a lognormal duration
around their habit. Some pauses follow text that already looks complete ("I can't mark it
delivered. | The customer wants it at home."), which is the hard case for any text model.

All three controllers see the same signal stream, fed the way a `Leg` feeds its controller: a
VAD frame every 20 ms while the caller speaks, a partial per word, a final per segment after a
sampled recognizer lag, and a tick every 20 ms of silence.

- hybrid: `turn.EndpointingController` with default config, one instance per caller so its
  pause model adapts across that caller's turns.
- fixed_800ms: end of turn after 800 ms of silence.
- text_only: end of turn at the first 20 ms tick of silence where `HeuristicCompleteness`
  scores the text at or above 0.8, with a 3 s silence fallback so it always terminates.

Partials. Real streaming recognizers usually send interim text without punctuation and put it
on the final, which lands after the silence starts. The headline (`controllers`) feeds bare
partials and punctuated finals. `punctuated_partials` repeats the run with punctuation already on
every partial, which flatters any text model. `noisy_punctuation` keeps bare partials and
punctuates finals the way recognizers often do, by acoustics: a segment final before a pause
gets a period half the time, and the true end loses its punctuation a quarter of the time.

Metrics: premature cutoff rate (turns where an end fired before the caller finished) and end
latency (from the true end of speech to the final end signal).
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from benchmarks.common import cli, envelope, mean, pct, pmap, ratio, table
from switchboard.providers import LatencyModel
from switchboard.turn import (
    EndpointingController,
    FinalTranscript,
    HeuristicCompleteness,
    InputSignal,
    PartialTranscript,
    VadFrame,
)

NAME = "endpointing"
N_DEFAULT = 3000  # turns
TURNS_PER_CALLER = 10
P_PAUSE = 0.7
TICK = 0.02
STT_LAG = LatencyModel(0.08, 0.12)

TEMPLATES = [
    "my order number is | four eight two one three.",
    "I can't mark it delivered. | The customer wants it at home.",
    "I want to, um, | change the delivery address.",
    "Can you check the status of my order?",
    "The address is | fourteen fifty Harbor Drive, | apartment four.",
    "I was wondering if | you could refund the customer.",
    "Yes.",
    "No, that's not right.",
    "It says the geofence | is broken again.",
    "So the customer called and | she said to leave it at the door.",
    "What is my pay this week?",
    "The courier is C seventy seven, | and the order is late.",
    "Okay, thanks.",
    "I'm outside the building but | the app won't let me finish.",
]


class Controller(Protocol):
    def observe(self, signal: InputSignal) -> bool: ...

    def tick(self, now: float) -> bool: ...


class Hybrid:
    def __init__(self) -> None:
        self.inner = EndpointingController()

    def observe(self, signal: InputSignal) -> bool:
        s = self.inner.observe(signal)
        return s is not None and s.kind == "end"

    def tick(self, now: float) -> bool:
        s = self.inner.tick(now)
        return s is not None and s.kind == "end"


class Silence:
    """End after a fixed silence; ignores text entirely."""

    def __init__(self, timeout_s: float) -> None:
        self.timeout_s = timeout_s
        self.last_speech: float | None = None

    def observe(self, signal: InputSignal) -> bool:
        if isinstance(signal, VadFrame) and signal.speech:
            self.last_speech = signal.at
        return False

    def tick(self, now: float) -> bool:
        if self.last_speech is not None and now - self.last_speech + 1e-9 >= self.timeout_s:
            self.last_speech = None
            return True
        return False


class TextOnly:
    """End as soon as the text looks complete; silence only as a 3 s safety net."""

    def __init__(self, threshold: float = 0.8, floor_s: float = TICK, fallback_s: float = 3.0):
        self.scorer = HeuristicCompleteness()
        self.threshold = threshold
        self.floor_s = floor_s
        self.fallback_s = fallback_s
        self.last_speech: float | None = None
        self.finals: list[str] = []
        self.partial = ""

    def observe(self, signal: InputSignal) -> bool:
        if isinstance(signal, VadFrame) and signal.speech:
            self.last_speech = signal.at
        elif isinstance(signal, PartialTranscript):
            self.partial = signal.text
        elif isinstance(signal, FinalTranscript):
            self.finals.append(signal.text)
            self.partial = ""
        return False

    def tick(self, now: float) -> bool:
        if self.last_speech is None:
            return False
        silence = now - self.last_speech + 1e-9
        text = " ".join([*self.finals, self.partial]).strip()
        if silence >= self.fallback_s or (
            silence >= self.floor_s and self.scorer.score(text) >= self.threshold
        ):
            self.last_speech, self.finals, self.partial = None, [], ""
            return True
        return False


MODES: tuple[Mode, ...] = ("bare", "noisy", "punctuated")

FACTORIES: dict[str, Callable[[], Controller]] = {
    "hybrid": Hybrid,
    "fixed_800ms": lambda: Silence(0.8),
    "text_only": TextOnly,
}


@dataclass(frozen=True)
class Turn:
    segments: tuple[tuple[str, ...], ...]  # words per segment
    pauses: tuple[float, ...]  # pause after each segment but the last
    words_per_s: float


def make_turns(rng: random.Random, n: int) -> list[list[Turn]]:
    """Callers, each with a speaking rate and pause habit, and their turns."""
    callers: list[list[Turn]] = []
    while sum(len(c) for c in callers) < n:
        rate = rng.uniform(2.0, 3.2)
        habit = rng.uniform(0.35, 0.9)
        pause_model = LatencyModel(habit, habit * 2.0)
        turns = []
        for _ in range(min(TURNS_PER_CALLER, n - sum(len(c) for c in callers))):
            parts = [p.split() for p in rng.choice(TEMPLATES).split("|")]
            segments: list[list[str]] = [parts[0]]
            pauses: list[float] = []
            for part in parts[1:]:
                if rng.random() < P_PAUSE:
                    pauses.append(pause_model.sample(rng))
                    segments.append(part)
                else:
                    segments[-1] = segments[-1] + part
            turns.append(Turn(tuple(tuple(s) for s in segments), tuple(pauses), rate))
        callers.append(turns)
    return callers


def _bare(text: str) -> str:
    return re.sub(r"[^\w\s']", "", text)


Mode = str  # "bare", "punctuated" or "noisy"


def play(
    controller: Controller, turn: Turn, start: float, rng: random.Random, mode: Mode
) -> dict[str, Any]:
    """Feed one turn; return when (relative to the true end) the controller ended it."""
    events: list[tuple[float, int, InputSignal]] = []  # (time, order, signal)
    t = start
    order = 0
    speech: list[tuple[float, float]] = []
    for i, words in enumerate(turn.segments):
        seg_start = t
        text: list[str] = []
        for word in words:
            t += 1.0 / turn.words_per_s
            text.append(word)
            partial = " ".join(text) if mode == "punctuated" else _bare(" ".join(text))
            events.append((t, order, PartialTranscript(t, partial, 0)))
            order += 1
        speech.append((seg_start, t))
        lag = STT_LAG.sample(rng)
        final = " ".join(text)
        if mode == "noisy":
            # Recognizers punctuate by acoustics: a pause often gets a period, and the true end
            # sometimes gets none.
            last = i == len(turn.segments) - 1
            if not last and rng.random() < 0.5:
                final = final.rstrip(",;:") + "."
            elif last and rng.random() < 0.25:
                final = _bare(final)
        events.append((t, order, VadFrame(t, False, 0.0)))
        events.append((t + lag, order + 1, FinalTranscript(t + lag, final)))
        order += 2
        if i < len(turn.pauses):
            t += turn.pauses[i]
    true_end = t
    events.sort(key=lambda e: (e[0], e[1]))
    ends: list[float] = []
    now = start
    idx = 0
    limit = true_end + 4.0
    while now <= limit:
        while idx < len(events) and events[idx][0] <= now + 1e-9:
            if controller.observe(events[idx][2]):
                ends.append(events[idx][0])
            idx += 1
        if any(a <= now < b for a, b in speech):
            if controller.observe(VadFrame(now, True, 1.0)):
                ends.append(now)
        elif controller.tick(now):
            ends.append(now)
        if ends and ends[-1] >= true_end - 1e-9 and idx >= len(events):
            break
        now = round(now + TICK, 9)
    premature = [e for e in ends if e < true_end - 1e-9]
    ended = next((e for e in ends if e >= true_end - 1e-9), None)
    return {
        "premature": len(premature),
        "end_latency_ms": None if ended is None else (ended - true_end) * 1000.0,
        "paused": bool(turn.pauses),
        "next_start": (ended if ended is not None else limit) + 1.0,
    }


def simulate(job: tuple[str, Mode, int, int]) -> list[dict[str, Any]]:
    name, mode, seed, n = job
    callers = make_turns(random.Random(f"endpointing:{seed}"), n)
    rng = random.Random(f"endpointing:{seed}:stt")
    out: list[dict[str, Any]] = []
    for turns in callers:
        controller = FACTORIES[name]()
        start = 0.0
        for turn in turns:
            row = play(controller, turn, start, rng, mode)
            start = row.pop("next_start")
            out.append(row)
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [r["end_latency_ms"] for r in rows if r["end_latency_ms"] is not None]
    paused = [r for r in rows if r["paused"]]
    return {
        "turns": len(rows),
        "premature_cutoff_rate": ratio(sum(1 for r in rows if r["premature"]), len(rows)),
        "premature_cutoff_rate_paused_turns": ratio(
            sum(1 for r in paused if r["premature"]), len(paused)
        ),
        "mean_end_latency_ms": mean(latencies),
        "p50_end_latency_ms": pct(latencies, 50),
        "p95_end_latency_ms": pct(latencies, 95),
        "unterminated": sum(1 for r in rows if r["end_latency_ms"] is None),
    }


def run(n: int | None = None, workers: int | None = None) -> dict[str, Any]:
    n = n or N_DEFAULT
    jobs = [(name, mode, 0, n) for mode in MODES for name in FACTORIES]
    rows = pmap(simulate, jobs, workers)
    by_mode: dict[str, dict[str, Any]] = {mode: {} for mode in MODES}
    for (name, mode, _, _), r in zip(jobs, rows, strict=True):
        by_mode[mode][name] = summarize(r)
    paused = sum(1 for r in rows[0] if r["paused"])
    return envelope(
        NAME,
        {"turns": n, "p_pause": P_PAUSE, "tick_s": TICK, "templates": len(TEMPLATES)},
        {
            "turns_with_a_pause": paused,
            "controllers": by_mode["bare"],
            "noisy_punctuation": by_mode["noisy"],
            "punctuated_partials": by_mode["punctuated"],
        },
    )


def render(data: dict[str, Any]) -> str:
    head = [
        "controller",
        "partials",
        "premature",
        "premature_paused",
        "mean_end_ms",
        "p50_end_ms",
        "p95_end_ms",
    ]
    rows = [
        [
            k,
            label,
            v["premature_cutoff_rate"],
            v["premature_cutoff_rate_paused_turns"],
            v["mean_end_latency_ms"],
            v["p50_end_latency_ms"],
            v["p95_end_latency_ms"],
        ]
        for key, label in (
            ("controllers", "bare"),
            ("noisy_punctuation", "noisy"),
            ("punctuated_partials", "punctuated"),
        )
        for k, v in data["results"][key].items()
    ]
    return table(head, rows)


def main() -> None:
    cli(NAME, run, render)


if __name__ == "__main__":
    main()
