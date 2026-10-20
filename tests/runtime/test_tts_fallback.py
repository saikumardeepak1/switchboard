"""A slow TTS falls to the backup mid-call, and the log says who spoke (ADR-013)."""

from __future__ import annotations

from switchboard.providers import FallbackTTS, SimulatedTTS
from switchboard.runtime import replay

from .harness import GREETING, every, make_world
from .test_scenarios import finished_call, verify


async def test_speech_played_records_the_tts_provider_that_served() -> None:
    world = make_world()
    clock = world.clock
    world.courier.tts.inner = FallbackTTS(  # type: ignore[assignment]
        [
            ("slow_primary", SimulatedTTS(clock, ttfb_s=3.0, chars_per_s=16.0)),
            ("backup", SimulatedTTS(clock, ttfb_s=0.15, chars_per_s=16.0)),
        ],
        clock,
        first_byte_deadline_s=0.4,
    )
    events = await finished_call(world, verify)
    played = every(events, "speech.played")
    assert [(p.text_played, p.tts_provider) for p in played] == [
        ("One moment.", "backup"),
        (GREETING, "backup"),
    ]
    assert replay(events).last_seq == events[-1].seq


async def test_plain_tts_leaves_the_provider_unset() -> None:
    world = make_world()
    events = await finished_call(world, verify)
    assert {p.tts_provider for p in every(events, "speech.played")} == {None}
