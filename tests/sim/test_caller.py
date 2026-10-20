"""Simulated callers: beat timing on the virtual clock, persona knobs, the dialer."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable

import pytest

from switchboard.core import VirtualClock
from switchboard.providers import AudioChunk, SttEvent, SttFinal, SttPartial, SttVad
from switchboard.sim import (
    Backchannel,
    BargeIn,
    HangUp,
    Pause,
    Persona,
    Say,
    Silence,
    SimDialer,
    SimulatedCaller,
    WaitFor,
    WaitForAgent,
)

START = 100.0


async def collect(caller: SimulatedCaller) -> list[SttEvent]:
    return [e async for e in caller.stt.events()]


async def agent_says(clock: VirtualClock, caller: SimulatedCaller, at: float, text: str) -> None:
    """Deliver one chunk of agent audio at `at`, lasting len(text) / 10 seconds."""
    await clock.sleep(at - clock.now())
    await caller.send(AudioChunk(at=clock.now(), pcm=b"", duration_s=len(text) / 10, text=text))


async def play(
    clock: VirtualClock, caller: SimulatedCaller, *extra: Awaitable[None]
) -> list[SttEvent]:
    heard = asyncio.ensure_future(collect(caller))
    run = asyncio.ensure_future(caller.run())
    others = [asyncio.ensure_future(x) for x in extra]
    await clock.run_until_idle()
    assert run.done() and heard.done()
    run.result()
    for o in others:
        o.result()
    return heard.result()


def caller(clock: VirtualClock, *beats: object, persona: Persona | None = None) -> SimulatedCaller:
    return SimulatedCaller(clock, "courier", list(beats), persona or Persona(), seed=1)  # type: ignore[arg-type]


async def test_say_plays_one_word_per_beat_of_the_speaking_rate() -> None:
    clock = VirtualClock(START)
    c = caller(clock, Say("one two three four"), persona=Persona(words_per_s=2.0))
    events = await play(clock, c)

    u = c.utterances[0]
    assert (u.start, u.end, u.text, u.kind) == (START, START + 2.0, "one two three four", "say")
    vad = [(e.at, e.speech) for e in events if isinstance(e, SttVad)]
    assert vad == [(START, True), (START + 2.0, False)]
    partials = [(e.at, e.text) for e in events if isinstance(e, SttPartial)]
    assert partials == [
        (START + 0.5, "one"),
        (START + 1.0, "one two"),
        (START + 1.5, "one two three"),
        (START + 2.0, "one two three four"),
    ]
    final = [e for e in events if isinstance(e, SttFinal)]
    assert len(final) == 1 and final[0].text == "one two three four"
    assert START + 2.0 < final[0].at < START + 2.5
    assert c.status == "completed"


async def test_mid_sentence_pause_drops_vad_and_finalizes_the_first_segment() -> None:
    clock = VirtualClock(START)
    c = caller(
        clock, Say("my order is 48213", pauses=((2, 0.7),)), persona=Persona(words_per_s=2.0)
    )
    events = await play(clock, c)

    vad = [(round(e.at, 6), e.speech) for e in events if isinstance(e, SttVad)]
    assert vad == [(START, True), (START + 1.5, False), (START + 2.2, True), (START + 2.7, False)]
    finals = [e.text for e in events if isinstance(e, SttFinal)]
    assert finals == ["my order is", "48213"]
    assert c.utterances[0].end == pytest.approx(START + 2.7)


async def test_pause_and_silence_hold_the_floor_for_their_duration() -> None:
    clock = VirtualClock(START)
    c = caller(
        clock,
        Say("hi"),
        Pause(1.5),
        Say("there"),
        Silence(2.0),
        Say("bye"),
        persona=Persona(words_per_s=2.0),
    )
    await play(clock, c)
    starts = [u.start for u in c.utterances]
    assert starts == pytest.approx([START, START + 2.0, START + 4.5])
    assert [kind for _, kind, _ in c.log if kind in ("pause", "silence")] == ["pause", "silence"]


async def test_wait_for_phrase_speaks_as_soon_as_it_is_heard() -> None:
    clock = VirtualClock(START)
    c = caller(clock, WaitFor("verified"), Say("thanks"))
    await play(clock, c, agent_says(clock, c, START + 3.0, "Thanks Sam, you're verified."))
    assert c.utterances[0].start == pytest.approx(START + 3.0)


async def test_wait_for_agent_waits_for_the_reply_gap_after_audio_ends() -> None:
    clock = VirtualClock(START)
    c = caller(clock, WaitForAgent(gap_s=0.5), Say("ok"))
    # 20 chars at 10 chars per second of audio: plays from +1.0 to +3.0.
    await play(clock, c, agent_says(clock, c, START + 1.0, "x" * 20))
    assert c.utterances[0].start == pytest.approx(START + 3.5)


async def test_wait_for_times_out_repeats_the_last_say_then_gives_up() -> None:
    clock = VirtualClock(START)
    persona = Persona(words_per_s=2.0, patience_s=5.0, retries=1)
    c = caller(clock, Say("hello there"), WaitFor("verified"), Say("never said"), persona=persona)
    await play(clock, c)

    assert [(u.text, u.kind) for u in c.utterances] == [
        ("hello there", "say"),
        ("hello there", "repeat"),
    ]
    assert c.utterances[1].start == pytest.approx(START + 1.0 + 5.0)
    assert c.status == "gave_up"
    assert c.gave_up_on == "verified"


async def test_barge_in_starts_after_the_given_ms_of_agent_speech() -> None:
    clock = VirtualClock(START)
    c = caller(clock, BargeIn(after_ms=500, text="sorry, stop"), Backchannel(after_ms=200))
    agent = agent_says(clock, c, START + 2.0, "a long greeting that keeps going" * 3)
    await play(clock, c, agent)
    barge, back = c.utterances
    assert barge.kind == "barge_in" and barge.start == pytest.approx(START + 2.5)
    # The backchannel waits for the agent to be speaking, which it still is.
    assert back.kind == "backchannel" and back.text == "uh huh"
    assert back.start == pytest.approx(barge.end + 0.2)


async def test_hang_up_stops_the_script_and_closes_the_line() -> None:
    clock = VirtualClock(START)
    c = caller(clock, Say("bye"), HangUp(), Say("never"))
    events = await play(clock, c)
    assert [u.text for u in c.utterances] == ["bye"]
    assert c.status == "hung_up"
    assert isinstance(events[-1], SttFinal)


async def test_word_errors_and_noise_are_reproducible_per_seed() -> None:
    persona = Persona(wer=0.5, noise=0.5)

    async def transcript(seed: int) -> tuple[list[str], int]:
        clock = VirtualClock(START)
        c = SimulatedCaller(
            clock,
            "courier",
            [Say("thirteen fourteen fifteen sixteen seventeen eighteen"), Silence(6.0)],
            persona,
            seed=seed,
        )
        events = await play(clock, c)
        finals = [e.text for e in events if isinstance(e, SttFinal)]
        blips = sum(1 for e in events if isinstance(e, SttVad) and e.speech) - 1
        return finals, blips

    first, second, other = await transcript(3), await transcript(3), await transcript(4)
    assert first == second
    assert first[0][0] != "thirteen fourteen fifteen sixteen seventeen eighteen"
    assert first[1] > 0
    assert other != first


async def test_impatient_persona_interjects_once_per_long_agent_utterance() -> None:
    clock = VirtualClock(START)
    persona = Persona(interrupt_after_s=1.0, interjection="Okay, okay.", patience_s=20.0)
    c = caller(clock, WaitFor("goodbye"), persona=persona)
    long_text = "x" * 50  # five seconds of audio
    await play(
        clock,
        c,
        agent_says(clock, c, START + 1.0, long_text),
        agent_says(clock, c, START + 12.0, "short goodbye"),
    )
    interjections = [u for u in c.utterances if u.kind == "interjection"]
    assert len(interjections) == 1
    assert interjections[0].start == pytest.approx(START + 2.0)


async def test_dialer_runs_the_callee_script_once_answered() -> None:
    clock = VirtualClock(START)
    callee = SimulatedCaller(clock, "customer", [Say("hello")], Persona(), seed=2)
    dialer = SimDialer(clock, {"customer": callee}, ring_s=2.0)

    async def dial() -> None:
        io = await dialer.dial("customer", "leg_x")
        assert io.stt is callee.stt
        assert clock.now() == START + 2.0
        with pytest.raises(LookupError):
            await dialer.dial("customer", "leg_y")
        with pytest.raises(LookupError):
            await dialer.dial("nobody", "leg_z")

    heard = asyncio.ensure_future(collect(callee))
    task = asyncio.ensure_future(dial())
    await clock.run_until_idle()
    task.result()
    assert heard.done()
    assert dialer.dialed == [("customer", "leg_x")]
    assert callee.utterances[0].start == START + 2.0
