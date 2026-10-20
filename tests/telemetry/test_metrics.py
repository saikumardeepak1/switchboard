import pytest

from switchboard.telemetry.metrics import EventMetrics, Registry
from switchboard.telemetry.turns import TurnTimeline
from tests.telemetry.fixtures import conv_a_events


def test_counter_with_labels_renders_prometheus_text() -> None:
    reg = Registry()
    c = reg.counter("calls_total", "Calls handled.", labelnames=("outcome",))
    c.labels(outcome="resolved").inc()
    c.labels(outcome="resolved").inc(2)
    c.labels(outcome='we"ird\n').inc()
    text = reg.render()
    assert "# HELP calls_total Calls handled.\n" in text
    assert "# TYPE calls_total counter\n" in text
    assert 'calls_total{outcome="resolved"} 3.0\n' in text
    assert 'calls_total{outcome="we\\"ird\\n"} 1.0\n' in text
    assert text.endswith("\n")


def test_counter_rejects_negative_and_wrong_labels() -> None:
    reg = Registry()
    c = reg.counter("x_total", "x", labelnames=("a",))
    with pytest.raises(ValueError):
        c.labels(a="1").inc(-1)
    with pytest.raises(ValueError):
        c.labels(b="1")
    with pytest.raises(ValueError):
        reg.counter("bad name", "x")


def test_registering_the_same_name_twice_returns_the_same_metric() -> None:
    reg = Registry()
    assert reg.counter("a_total", "a") is reg.counter("a_total", "a")
    with pytest.raises(ValueError):
        reg.gauge("a_total", "a")


def test_unlabelled_counter_and_gauge() -> None:
    reg = Registry()
    reg.counter("hits_total", "hits").inc()
    g = reg.gauge("active", "active streams")
    g.inc()
    g.inc()
    g.dec()
    text = reg.render()
    assert "hits_total 1.0\n" in text
    assert "# TYPE active gauge\n" in text
    assert "active 1.0\n" in text


def test_histogram_is_cumulative_with_sum_and_count() -> None:
    reg = Registry()
    h = reg.histogram("lat_ms", "latency", buckets=(100, 500, 1000))
    for v in (50, 300, 300, 2000):
        h.observe(v)
    text = reg.render()
    assert "# TYPE lat_ms histogram\n" in text
    assert 'lat_ms_bucket{le="100.0"} 1\n' in text
    assert 'lat_ms_bucket{le="500.0"} 3\n' in text
    assert 'lat_ms_bucket{le="1000.0"} 3\n' in text
    assert 'lat_ms_bucket{le="+Inf"} 4\n' in text
    assert "lat_ms_sum 2650.0\n" in text
    assert "lat_ms_count 4\n" in text


def test_labelled_histogram_orders_le_last() -> None:
    reg = Registry()
    h = reg.histogram("d_ms", "d", buckets=(1,), labelnames=("tool",))
    h.labels(tool="x").observe(0.5)
    assert 'd_ms_bucket{tool="x",le="1.0"} 1\n' in reg.render()


def test_event_metrics_projection() -> None:
    reg = Registry()
    em = EventMetrics(reg)
    for event in conv_a_events():
        em.observe(event)
    em.observe_timeline(TurnTimeline.from_events(conv_a_events()))
    text = reg.render()
    assert 'switchboard_events_total{type="policy.verdict"} 3.0\n' in text
    assert "switchboard_policy_denials_total 1.0\n" in text
    assert "switchboard_sentences_blocked_total 1.0\n" in text
    assert "switchboard_handoffs_total 1.0\n" in text
    assert 'switchboard_tool_errors_total{tool="mark_delivered"} 1.0\n' in text
    assert "switchboard_voice_to_voice_ms_count 3\n" in text
