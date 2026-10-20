"""In-process metrics with Prometheus text exposition (format 0.0.4), no dependencies.

`Registry` holds counters, gauges and histograms; `render()` produces the body for a
`/metrics` scrape. `EventMetrics` projects conversation events and turn timelines onto a
registry, so the runtime only has to call `observe(event)` after each append.
"""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Callable, Sequence
from typing import Literal

from switchboard.core.events import (
    AnyEvent,
    HandoffRequested,
    PolicyVerdict,
    SpeechSentenceGated,
    ToolExecuted,
)
from switchboard.telemetry.turns import TurnTimeline, tool_error

_NAME = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_LABEL = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

LATENCY_BUCKETS_MS: tuple[float, ...] = (100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000)

Kind = Literal["counter", "gauge", "histogram"]


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _escape_help(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\n", "\\n")


def _fmt(value: float) -> str:
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if math.isnan(value):
        return "NaN"
    return repr(float(value))


def _labels(pairs: Sequence[tuple[str, str]]) -> str:
    if not pairs:
        return ""
    return "{" + ",".join(f'{k}="{_escape_label(v)}"' for k, v in pairs) + "}"


class _Metric:
    kind: Kind

    def __init__(self, name: str, help: str, labelnames: Sequence[str]) -> None:
        if not _NAME.match(name):
            raise ValueError(f"invalid metric name {name!r}")
        for label in labelnames:
            if not _LABEL.match(label) or label.startswith("__") or label == "le":
                raise ValueError(f"invalid label name {label!r}")
        self.name = name
        self.help = help
        self.labelnames = tuple(labelnames)
        self._lock = threading.Lock()

    def _key(self, labels: dict[str, str]) -> tuple[str, ...]:
        if set(labels) != set(self.labelnames):
            raise ValueError(
                f"{self.name} expects labels {list(self.labelnames)}, got {sorted(labels)}"
            )
        return tuple(str(labels[n]) for n in self.labelnames)

    def _pairs(self, key: tuple[str, ...]) -> list[tuple[str, str]]:
        return list(zip(self.labelnames, key, strict=True))

    def _check_unlabelled(self) -> None:
        if self.labelnames:
            raise ValueError(f"{self.name} has labels {list(self.labelnames)}; use .labels()")

    def samples(self) -> list[str]:
        raise NotImplementedError


class _Value:
    def __init__(self, metric: _Metric) -> None:
        self._metric = metric
        self.value = 0.0

    def _add(self, amount: float) -> None:
        with self._metric._lock:
            self.value += amount


class CounterChild(_Value):
    def inc(self, amount: float = 1.0) -> None:
        if amount < 0:
            raise ValueError("counters only go up")
        self._add(amount)


class GaugeChild(_Value):
    def inc(self, amount: float = 1.0) -> None:
        self._add(amount)

    def dec(self, amount: float = 1.0) -> None:
        self._add(-amount)

    def set(self, value: float) -> None:
        with self._metric._lock:
            self.value = float(value)


class _Scalar[C: _Value](_Metric):
    def __init__(self, name: str, help: str, labelnames: Sequence[str]) -> None:
        super().__init__(name, help, labelnames)
        self._children: dict[tuple[str, ...], C] = {}
        if not self.labelnames:
            self._children[()] = self._new()

    def _new(self) -> C:
        raise NotImplementedError

    def labels(self, **labels: str) -> C:
        key = self._key(labels)
        with self._lock:
            child = self._children.get(key)
            if child is None:
                child = self._children[key] = self._new()
        return child

    def _only(self) -> C:
        self._check_unlabelled()
        return self._children[()]

    def samples(self) -> list[str]:
        with self._lock:
            items = sorted(self._children.items())
        return [f"{self.name}{_labels(self._pairs(k))} {_fmt(c.value)}" for k, c in items]


class Counter(_Scalar[CounterChild]):
    kind: Kind = "counter"

    def _new(self) -> CounterChild:
        return CounterChild(self)

    def inc(self, amount: float = 1.0) -> None:
        self._only().inc(amount)


class Gauge(_Scalar[GaugeChild]):
    kind: Kind = "gauge"

    def _new(self) -> GaugeChild:
        return GaugeChild(self)

    def inc(self, amount: float = 1.0) -> None:
        self._only().inc(amount)

    def dec(self, amount: float = 1.0) -> None:
        self._only().dec(amount)

    def set(self, value: float) -> None:
        self._only().set(value)


class HistogramChild:
    def __init__(self, metric: Histogram) -> None:
        self._metric = metric
        self.counts = [0] * len(metric.buckets)
        self.sum = 0.0
        self.count = 0

    def observe(self, value: float) -> None:
        with self._metric._lock:
            for i, bound in enumerate(self._metric.buckets):
                if value <= bound:
                    self.counts[i] += 1
                    break
            self.sum += value
            self.count += 1


class Histogram(_Metric):
    kind: Kind = "histogram"

    def __init__(
        self, name: str, help: str, buckets: Sequence[float], labelnames: Sequence[str]
    ) -> None:
        super().__init__(name, help, labelnames)
        bounds = sorted(float(b) for b in buckets)
        if not bounds or len(set(bounds)) != len(bounds):
            raise ValueError("histogram buckets must be non-empty and distinct")
        if not math.isinf(bounds[-1]):
            bounds.append(math.inf)
        self.buckets = tuple(bounds)
        self._children: dict[tuple[str, ...], HistogramChild] = {}
        if not self.labelnames:
            self._children[()] = HistogramChild(self)

    def labels(self, **labels: str) -> HistogramChild:
        key = self._key(labels)
        with self._lock:
            child = self._children.get(key)
            if child is None:
                child = self._children[key] = HistogramChild(self)
        return child

    def observe(self, value: float) -> None:
        self._check_unlabelled()
        self._children[()].observe(value)

    def samples(self) -> list[str]:
        lines: list[str] = []
        with self._lock:
            items = sorted(self._children.items())
            for key, child in items:
                pairs = self._pairs(key)
                running = 0
                for bound, n in zip(self.buckets, child.counts, strict=True):
                    running += n
                    le = _labels([*pairs, ("le", _fmt(bound))])
                    lines.append(f"{self.name}_bucket{le} {running}")
                lines.append(f"{self.name}_sum{_labels(pairs)} {_fmt(child.sum)}")
                lines.append(f"{self.name}_count{_labels(pairs)} {child.count}")
        return lines


class Registry:
    def __init__(self) -> None:
        self._metrics: dict[str, _Metric] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, help: str, labelnames: Sequence[str] = ()) -> Counter:
        return self._get(name, Counter, lambda: Counter(name, help, labelnames))

    def gauge(self, name: str, help: str, labelnames: Sequence[str] = ()) -> Gauge:
        return self._get(name, Gauge, lambda: Gauge(name, help, labelnames))

    def histogram(
        self,
        name: str,
        help: str,
        buckets: Sequence[float] = LATENCY_BUCKETS_MS,
        labelnames: Sequence[str] = (),
    ) -> Histogram:
        return self._get(name, Histogram, lambda: Histogram(name, help, buckets, labelnames))

    def _get[M: _Metric](self, name: str, cls: type[M], make: Callable[[], M]) -> M:
        with self._lock:
            existing = self._metrics.get(name)
            if existing is None:
                metric = make()
                self._metrics[name] = metric
                return metric
        if not isinstance(existing, cls):
            raise ValueError(f"metric {name!r} already registered as a {existing.kind}")
        return existing

    def render(self) -> str:
        with self._lock:
            metrics = sorted(self._metrics.values(), key=lambda m: m.name)
        lines: list[str] = []
        for m in metrics:
            lines.append(f"# HELP {m.name} {_escape_help(m.help)}")
            lines.append(f"# TYPE {m.name} {m.kind}")
            lines.extend(m.samples())
        return "\n".join(lines) + "\n" if lines else ""


class EventMetrics:
    """Counters and latency histograms derived from conversation events."""

    def __init__(self, registry: Registry) -> None:
        self.events = registry.counter(
            "switchboard_events_total", "Events appended, by type.", labelnames=("type",)
        )
        self.denials = registry.counter(
            "switchboard_policy_denials_total", "Tool calls blocked by policy before running."
        )
        self.blocked = registry.counter(
            "switchboard_sentences_blocked_total", "Sentences blocked by the grounding gate."
        )
        self.handoffs = registry.counter(
            "switchboard_handoffs_total", "Handoffs to a human requested."
        )
        self.tool_errors = registry.counter(
            "switchboard_tool_errors_total", "Tool executions that returned an error.", ("tool",)
        )
        self.voice_to_voice = registry.histogram(
            "switchboard_voice_to_voice_ms", "End of caller speech to first agent audio."
        )
        self.endpointing = registry.histogram(
            "switchboard_endpointing_ms", "End of caller speech to turn commit."
        )

    def observe(self, event: AnyEvent) -> None:
        self.events.labels(type=event.type).inc()
        if isinstance(event, PolicyVerdict) and not event.allowed:
            self.denials.inc()
        elif isinstance(event, SpeechSentenceGated) and event.decision == "block":
            self.blocked.inc()
        elif isinstance(event, HandoffRequested):
            self.handoffs.inc()
        elif isinstance(event, ToolExecuted) and tool_error(event) is not None:
            self.tool_errors.labels(tool=event.tool).inc()

    def observe_timeline(self, timeline: TurnTimeline) -> None:
        for turn in timeline.turns:
            self.endpointing.observe(turn.endpointing_ms)
            if turn.voice_to_voice_ms is not None:
                self.voice_to_voice.observe(turn.voice_to_voice_ms)
