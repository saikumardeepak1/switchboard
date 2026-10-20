"""`speech.played` records when the utterance's first audio reached the caller (ADR-015)."""

from __future__ import annotations

from switchboard.core import SpeechPlayed

from .harness import make_world
from .test_scenarios import finished_call, verify


async def test_speech_played_carries_the_time_the_first_chunk_was_sent() -> None:
    world = make_world()
    events = await finished_call(world, verify)

    played = [e for e in events if isinstance(e, SpeechPlayed)]
    assert played
    starts = {c.at for c in world.courier.chunks}
    for p in played:
        assert p.first_audio_at is not None
        assert p.first_audio_at in starts
        # The event is appended when playback ends, after the first audio went out.
        assert p.first_audio_at < p.at
    # Each utterance's first audio is the earliest chunk sent from then on.
    greeting = played[-1]
    assert greeting.first_audio_at == min(
        c.at for c in world.courier.chunks if c.at >= (greeting.first_audio_at or 0.0)
    )
