import pytest

from switchboard.core.clock import VirtualClock
from switchboard.turn.playback import (
    BargeInConfig,
    BargeInDecision,
    BargeInPolicy,
    PlaybackCursor,
    is_backchannel,
)
from switchboard.turn.signals import PartialTranscript, VadFrame

FRAME = 0.02
SENTENCE = "Your order will arrive at seven."  # 32 characters


async def test_advance_marks_text_proportionally_within_a_chunk() -> None:
    clock = VirtualClock()
    cursor = PlaybackCursor()
    cursor.enqueue(SENTENCE, duration=3.2, now=clock.now())
    await clock.advance(1.0)
    assert cursor.advance(clock.now()) == SENTENCE[:10]
    await clock.advance(5.0)
    assert cursor.advance(clock.now()) == SENTENCE
    assert cursor.finished(clock.now())


async def test_multiple_chunks_play_back_to_back() -> None:
    clock = VirtualClock()
    cursor = PlaybackCursor()
    cursor.enqueue("Hello there. ", duration=1.0, now=clock.now())
    cursor.enqueue("Your order shipped.", duration=2.0, now=clock.now())
    await clock.advance(2.0)
    played = cursor.advance(clock.now())
    assert played == "Hello there. " + "Your order shipped."[: len("Your order shipped.") // 2]


async def test_underrun_does_not_count_silence_as_playback() -> None:
    clock = VirtualClock()
    cursor = PlaybackCursor()
    cursor.enqueue("One. ", duration=0.5, now=clock.now())
    await clock.advance(2.0)
    cursor.enqueue("Two three four.", duration=1.5, now=clock.now())
    await clock.advance(0.1)
    assert cursor.advance(clock.now()) == "One. T"


async def test_interrupt_rounds_back_to_last_word_boundary() -> None:
    clock = VirtualClock()
    cursor = PlaybackCursor()
    cursor.enqueue(SENTENCE, duration=3.2, now=clock.now())
    await clock.advance(0.9)  # 9 characters: "Your orde"
    assert cursor.interrupt(clock.now()) == "Your"
    assert cursor.text_played == "Your"
    await clock.advance(5.0)
    assert cursor.advance(clock.now()) == "Your"


async def test_interrupt_on_a_space_keeps_the_whole_word() -> None:
    clock = VirtualClock()
    cursor = PlaybackCursor()
    cursor.enqueue(SENTENCE, duration=3.2, now=clock.now())
    await clock.advance(1.0)  # "Your order"; next char is a space
    assert cursor.interrupt(clock.now()) == "Your order"


async def test_interrupt_before_first_word_completes_is_empty() -> None:
    clock = VirtualClock()
    cursor = PlaybackCursor()
    cursor.enqueue(SENTENCE, duration=3.2, now=clock.now())
    await clock.advance(0.2)
    assert cursor.interrupt(clock.now()) == ""


async def test_interrupt_after_full_playback_returns_everything() -> None:
    clock = VirtualClock()
    cursor = PlaybackCursor()
    cursor.enqueue(SENTENCE, duration=3.2, now=clock.now())
    await clock.advance(4.0)
    assert cursor.interrupt(clock.now()) == SENTENCE


async def test_pause_and_resume_continue_from_cursor() -> None:
    clock = VirtualClock()
    cursor = PlaybackCursor()
    cursor.enqueue(SENTENCE, duration=3.2, now=clock.now())
    await clock.advance(1.0)
    cursor.pause(clock.now())
    await clock.advance(2.0)
    assert cursor.advance(clock.now()) == SENTENCE[:10]
    assert cursor.position == pytest.approx(1.0)
    cursor.resume(clock.now())
    await clock.advance(0.5)
    assert cursor.advance(clock.now()) == SENTENCE[:15]


def test_enqueue_rejects_negative_duration() -> None:
    with pytest.raises(ValueError):
        PlaybackCursor().enqueue("x", duration=-1.0, now=0.0)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mm-hm", True),
        ("Mm hmm.", True),
        ("yeah", True),
        ("okay", True),
        ("uh huh", True),
        ("Right.", True),
        ("yeah okay", True),
        ("", False),
        ("wait", False),
        ("yeah but my order is wrong", False),
    ],
)
def test_is_backchannel(text: str, expected: bool) -> None:
    assert is_backchannel(text) is expected


class Listener:
    """Feeds user audio to a barge-in policy while the agent is speaking."""

    def __init__(self, policy: BargeInPolicy, clock: VirtualClock) -> None:
        self.policy = policy
        self.clock = clock
        self.decisions: list[BargeInDecision] = []

    def _collect(self, decision: BargeInDecision | None) -> None:
        if decision is not None:
            self.decisions.append(decision)

    async def speak(self, seconds: float, partial_at: float | None = None, text: str = "") -> None:
        start = self.clock.now()
        sent = False
        for _ in range(round(seconds / FRAME)):
            await self.clock.advance(FRAME)
            self._collect(self.policy.observe(VadFrame(self.clock.now(), True, 0.6)))
            if partial_at is not None and not sent and self.clock.now() - start >= partial_at:
                sent = True
                self._collect(self.policy.observe(PartialTranscript(self.clock.now(), text, 0)))

    async def silence(self, seconds: float) -> None:
        for _ in range(round(seconds / FRAME)):
            await self.clock.advance(FRAME)
            self._collect(self.policy.observe(VadFrame(self.clock.now(), False, 0.01)))

    def kinds(self) -> list[str]:
        return [d.kind for d in self.decisions]


async def test_short_overlap_is_ignored() -> None:
    clock = VirtualClock()
    listener = Listener(BargeInPolicy(), clock)
    await listener.silence(0.5)
    await listener.speak(0.2)
    await listener.silence(1.0)
    assert listener.kinds() == []


async def test_backchannel_is_ignored_even_when_long() -> None:
    clock = VirtualClock()
    listener = Listener(BargeInPolicy(), clock)
    await listener.speak(0.6, partial_at=0.2, text="mm-hm")
    await listener.silence(1.0)
    assert listener.kinds() == []


async def test_real_utterance_interrupts_then_confirms() -> None:
    clock = VirtualClock()
    listener = Listener(BargeInPolicy(), clock)
    await listener.speak(0.8, partial_at=0.5, text="wait, that's the wrong address")
    assert listener.kinds() == ["interrupt", "confirm"]
    assert listener.decisions[0].at == pytest.approx(0.3, abs=FRAME + 1e-6)
    assert listener.decisions[1].at == pytest.approx(0.5, abs=FRAME + 1e-6)


async def test_backchannel_partial_before_threshold_then_real_words_interrupt() -> None:
    clock = VirtualClock()
    listener = Listener(BargeInPolicy(), clock)
    await listener.speak(0.4, partial_at=0.1, text="yeah")
    assert listener.kinds() == []
    listener._collect(listener.policy.observe(PartialTranscript(clock.now(), "yeah but no", 0)))
    await listener.speak(0.1)
    assert listener.kinds() == ["interrupt", "confirm"]


async def test_false_interrupt_resumes_playback_without_regeneration() -> None:
    clock = VirtualClock()
    cursor = PlaybackCursor()
    cursor.enqueue(SENTENCE, duration=3.2, now=clock.now())
    policy = BargeInPolicy()
    listener = Listener(policy, clock)

    await listener.silence(1.0)
    await listener.speak(0.4)  # a cough: VAD fires, no words
    assert listener.kinds() == ["interrupt"]
    cursor.pause(listener.decisions[0].at)
    paused_text = cursor.advance(clock.now())

    await listener.silence(1.0)
    assert listener.kinds() == ["interrupt", "resume"]
    resume_at = listener.decisions[1].at
    cfg = BargeInConfig()
    assert resume_at - 1.4 == pytest.approx(cfg.settle_ms / 1000, abs=FRAME + 1e-6)
    cursor.resume(resume_at)

    assert cursor.full_text == SENTENCE
    later = cursor.advance(clock.now())
    assert later.startswith(paused_text)
    assert len(later) > len(paused_text)


async def test_sustained_speech_without_words_confirms_after_window() -> None:
    clock = VirtualClock()
    policy = BargeInPolicy(BargeInConfig(resume_window_ms=600))
    listener = Listener(policy, clock)
    await listener.speak(1.2)
    assert listener.kinds() == ["interrupt", "confirm"]
    assert listener.decisions[1].at - listener.decisions[0].at == pytest.approx(0.6, abs=0.03)


async def test_tick_resumes_without_new_frames() -> None:
    clock = VirtualClock()
    policy = BargeInPolicy()
    listener = Listener(policy, clock)
    await listener.speak(0.4)
    assert listener.kinds() == ["interrupt"]
    await clock.advance(0.2)
    assert policy.tick(clock.now()) is None
    await clock.advance(0.2)
    decision = policy.tick(clock.now())
    assert decision is not None and decision.kind == "resume"


async def test_policy_rearms_after_resume() -> None:
    clock = VirtualClock()
    listener = Listener(BargeInPolicy(), clock)
    await listener.speak(0.4)
    await listener.silence(0.5)
    await listener.speak(0.4)
    assert listener.kinds() == ["interrupt", "resume", "interrupt"]
