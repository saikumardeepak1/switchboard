import asyncio
import math
import random

import pytest

from switchboard.core.clock import VirtualClock
from switchboard.providers.interfaces import (
    STT,
    TTS,
    AudioChunk,
    SttEvent,
    SttFinal,
    SttPartial,
    SttVad,
)
from switchboard.providers.simulated import (
    CONFUSIONS,
    LatencyModel,
    SimulatedSTT,
    SimulatedTTS,
    inject_word_errors,
)


def test_latency_model_fit_matches_quantiles() -> None:
    m = LatencyModel(p50=0.3, p95=0.9)
    assert math.exp(m.mu) == pytest.approx(0.3)
    assert math.exp(m.mu + 1.6448536269514722 * m.sigma) == pytest.approx(0.9)
    rng = random.Random(7)
    samples = sorted(m.sample(rng) for _ in range(20_000))
    assert samples[10_000] == pytest.approx(0.3, rel=0.05)
    assert samples[19_000] == pytest.approx(0.9, rel=0.08)


def test_latency_model_degenerate_and_invalid() -> None:
    m = LatencyModel(p50=0.2, p95=0.2)
    assert m.sigma == 0.0
    assert m.sample(random.Random(1)) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        LatencyModel(p50=0.5, p95=0.4)
    with pytest.raises(ValueError):
        LatencyModel(p50=0.0, p95=0.4)


def test_word_errors_zero_rate_is_identity() -> None:
    text = "my order fifteen is late"
    assert inject_word_errors(text, 0.0, random.Random(1)) == text


def test_word_errors_full_rate_substitutes_confusables_and_deletes_others() -> None:
    out = inject_word_errors("fifteen apples 15 thirteen", 1.0, random.Random(3))
    words = out.split()
    assert "apples" not in words
    assert words[0] in CONFUSIONS["fifteen"]
    assert words[1] in CONFUSIONS["15"]
    assert words[2] in CONFUSIONS["thirteen"]


def test_word_errors_keep_trailing_punctuation_and_case() -> None:
    out = inject_word_errors("Fifteen.", 1.0, random.Random(0))
    assert out in {"Fifty."}


def test_word_errors_deterministic_per_seed_and_rate_is_about_right() -> None:
    # all words are confusable, so every error is a substitution and positions stay aligned
    text = " ".join(["fifteen", "four", "thirteen", "eight"] * 1000)
    a = inject_word_errors(text, 0.1, random.Random(42))
    b = inject_word_errors(text, 0.1, random.Random(42))
    c = inject_word_errors(text, 0.1, random.Random(43))
    assert a == b
    assert a != c
    original, noisy = text.split(), a.split()
    assert len(noisy) == len(original)
    changed = sum(1 for x, y in zip(original, noisy, strict=True) if x != y)
    assert 0.08 < changed / len(original) < 0.12


async def drain(stt: SimulatedSTT, clock: VirtualClock) -> list[SttEvent]:
    events: list[SttEvent] = []

    async def reader() -> None:
        async for e in stt.events():
            events.append(e)

    task = asyncio.create_task(reader())
    await clock.run_until_idle()
    await stt.close()
    await clock.run_until_idle()
    await task
    return events


async def speak_and_drain(seed: int, wer: float) -> list[SttEvent]:
    clock = VirtualClock()
    stt = SimulatedSTT(
        clock,
        random.Random(seed),
        latency=LatencyModel(p50=0.3, p95=0.6),
        wer=wer,
        words_per_s=2.0,
    )
    speak = asyncio.create_task(stt.speak("my order fifteen is late"))
    events = await drain(stt, clock)
    await speak
    return events


async def test_stt_emits_vad_partials_word_by_word_then_final() -> None:
    events = await speak_and_drain(seed=1, wer=0.0)
    assert events[0] == SttVad(at=0.0, speech=True)
    partials = [e for e in events if isinstance(e, SttPartial)]
    assert [p.text for p in partials] == [
        "my",
        "my order",
        "my order fifteen",
        "my order fifteen is",
        "my order fifteen is late",
    ]
    assert [p.at for p in partials] == pytest.approx([0.5, 1.0, 1.5, 2.0, 2.5])
    vad_off = [e for e in events if isinstance(e, SttVad) and not e.speech]
    assert vad_off == [SttVad(at=2.5, speech=False)]
    final = events[-1]
    assert isinstance(final, SttFinal)
    assert final.text == "my order fifteen is late"
    assert final.at > 2.5


async def test_stt_is_deterministic_per_seed_with_errors() -> None:
    a = await speak_and_drain(seed=5, wer=0.5)
    b = await speak_and_drain(seed=5, wer=0.5)
    c = await speak_and_drain(seed=6, wer=0.5)
    assert a == b
    assert a != c
    final = a[-1]
    assert isinstance(final, SttFinal)
    partials = [e for e in a if isinstance(e, SttPartial)]
    # partials are prefixes of the final, so the transcript never rewrites itself
    assert all(final.text.startswith(p.text) for p in partials)


async def test_stt_accepts_audio_frames_and_satisfies_protocol() -> None:
    clock = VirtualClock()
    stt: STT = SimulatedSTT(clock, random.Random(0), LatencyModel(0.1, 0.2))
    await stt.push(b"\x00" * 320)
    await stt.close()
    assert [e async for e in stt.events()] == []


async def test_tts_ttfb_and_chunk_durations() -> None:
    clock = VirtualClock()
    tts: TTS = SimulatedTTS(clock, ttfb_s=0.2, chars_per_s=10.0, chunk_chars=10, sample_rate=8000)
    chunks: list[AudioChunk] = []

    async def run() -> None:
        async for c in tts.synthesize("Your order arrives at 7:45 pm."):
            chunks.append(c)

    task = asyncio.create_task(run())
    await clock.run_until_idle()
    await task
    assert "".join(c.text for c in chunks) == "Your order arrives at 7:45 pm."
    assert chunks[0].at == pytest.approx(0.2)
    for c in chunks:
        assert c.duration_s == pytest.approx(len(c.text) / 10.0)
        assert len(c.pcm) == 2 * round(c.duration_s * 8000)
    # real-time pacing: each chunk follows the previous one's audio
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.at == pytest.approx(prev.at + prev.duration_s)


async def test_tts_empty_text_yields_nothing() -> None:
    clock = VirtualClock()
    tts = SimulatedTTS(clock, ttfb_s=0.2, chars_per_s=10.0)
    assert [c async for c in tts.synthesize("")] == []
