from collections.abc import Sequence

import pytest

from switchboard.core.clock import VirtualClock
from switchboard.turn.endpointing import (
    EndpointingConfig,
    EndpointingController,
    HeuristicCompleteness,
)
from switchboard.turn.signals import (
    FinalTranscript,
    PartialTranscript,
    TurnSignal,
    VadFrame,
)

FRAME = 0.02


class Caller:
    """Drives a controller with 20 ms VAD frames on a virtual clock."""

    def __init__(self, controller: EndpointingController, clock: VirtualClock) -> None:
        self.controller = controller
        self.clock = clock
        self.signals: list[TurnSignal] = []
        self.last_speech_at = 0.0

    def _collect(self, signal: TurnSignal | None) -> None:
        if signal is not None:
            self.signals.append(signal)

    async def speak(self, seconds: float) -> None:
        for _ in range(round(seconds / FRAME)):
            await self.clock.advance(FRAME)
            self._collect(self.controller.observe(VadFrame(self.clock.now(), True, 0.7)))
            self.last_speech_at = self.clock.now()

    async def pause(self, seconds: float) -> None:
        for _ in range(round(seconds / FRAME)):
            await self.clock.advance(FRAME)
            self._collect(self.controller.observe(VadFrame(self.clock.now(), False, 0.01)))

    def partial(self, text: str) -> None:
        self._collect(self.controller.observe(PartialTranscript(self.clock.now(), text, 0)))

    def final(self, text: str) -> None:
        self._collect(self.controller.observe(FinalTranscript(self.clock.now(), text)))

    def kinds(self) -> list[str]:
        return [s.kind for s in self.signals]


def first(signals: Sequence[TurnSignal], kind: str) -> TurnSignal:
    return next(s for s in signals if s.kind == kind)


class ConstantScorer:
    def __init__(self, value: float) -> None:
        self.value = value

    def score(self, text: str) -> float:
        return self.value


class ConstantAudio:
    def __init__(self, value: float) -> None:
        self.value = value

    def score(self, recent: Sequence[VadFrame]) -> float:
        return self.value


async def test_complete_sentence_emits_eager_end_then_end() -> None:
    clock = VirtualClock()
    caller = Caller(EndpointingController(), clock)
    await caller.speak(1.0)
    caller.partial("I want to check on my order.")
    await caller.pause(1.0)

    assert caller.kinds() == ["eager_end", "end"]
    eager = first(caller.signals, "eager_end")
    end = first(caller.signals, "end")
    assert eager.at - caller.last_speech_at == pytest.approx(0.2, abs=FRAME + 1e-6)
    assert 0.3 <= end.at - caller.last_speech_at <= 0.6
    assert end.text == "I want to check on my order."
    assert end.confidence >= 0.8


async def test_trailing_incomplete_phrase_does_not_end_on_600ms_pause() -> None:
    clock = VirtualClock()
    caller = Caller(EndpointingController(), clock)
    await caller.speak(0.8)
    caller.partial("my order number is")
    await caller.pause(0.6)
    assert caller.kinds() == []


async def test_speech_after_eager_end_emits_resumed() -> None:
    clock = VirtualClock()
    caller = Caller(EndpointingController(), clock)
    await caller.speak(0.6)
    caller.partial("I want to check my order.")
    await caller.pause(0.3)
    await caller.speak(0.4)
    assert caller.kinds() == ["eager_end", "resumed"]

    caller.partial("I want to check my order. And the refund.")
    await caller.pause(1.0)
    assert caller.kinds() == ["eager_end", "resumed", "eager_end", "end"]
    assert caller.signals[-1].text == "I want to check my order. And the refund."


async def test_max_silence_is_a_hard_ceiling() -> None:
    clock = VirtualClock()
    controller = EndpointingController(scorer=ConstantScorer(0.0))
    caller = Caller(controller, clock)
    await caller.speak(0.5)
    caller.partial("well")
    await caller.pause(2.0)
    assert caller.kinds() == ["end"]
    end = caller.signals[0]
    assert end.at - caller.last_speech_at == pytest.approx(1.5, abs=FRAME + 1e-6)
    assert end.confidence < 0.8


async def test_tick_drives_end_without_frames() -> None:
    clock = VirtualClock()
    controller = EndpointingController(scorer=ConstantScorer(0.0))
    caller = Caller(controller, clock)
    await caller.speak(0.5)
    await clock.advance(1.0)
    assert controller.tick(clock.now()) is None
    await clock.advance(0.6)
    signal = controller.tick(clock.now())
    assert signal is not None and signal.kind == "end"


async def test_audio_scorer_pulls_end_earlier() -> None:
    async def end_delay(audio: ConstantAudio | None) -> float:
        clock = VirtualClock()
        caller = Caller(EndpointingController(audio_scorer=audio), clock)
        await caller.speak(0.5)
        caller.partial("I see what you mean")
        await caller.pause(2.0)
        return first(caller.signals, "end").at - caller.last_speech_at

    assert await end_delay(ConstantAudio(1.0)) < await end_delay(None)


async def test_final_transcripts_accumulate_into_turn_text() -> None:
    clock = VirtualClock()
    caller = Caller(EndpointingController(), clock)
    await caller.speak(0.5)
    caller.final("My order is late.")
    await caller.speak(0.5)
    caller.partial("Can you check it?")
    await caller.pause(1.0)
    assert caller.signals[-1].kind == "end"
    assert caller.signals[-1].text == "My order is late. Can you check it?"


async def test_turn_state_resets_after_end() -> None:
    clock = VirtualClock()
    controller = EndpointingController()
    caller = Caller(controller, clock)
    await caller.speak(0.5)
    caller.partial("Yes.")
    await caller.pause(1.0)
    assert caller.kinds() == ["eager_end", "end"]
    await caller.pause(3.0)
    assert caller.kinds() == ["eager_end", "end"]


async def test_slow_speaker_lengthens_thresholds() -> None:
    async def end_delay(caller: Caller) -> float:
        caller.signals.clear()
        await caller.speak(0.6)
        caller.partial("I want to check on my order.")
        await caller.pause(4.0)
        return first(caller.signals, "end").at - caller.last_speech_at

    fast = EndpointingController()
    slow = EndpointingController()
    clock = VirtualClock()
    slow_caller = Caller(slow, clock)
    slow_caller.partial("so um")
    for _ in range(4):
        await slow_caller.speak(0.5)
        await slow_caller.pause(0.8)
    await slow_caller.speak(0.3)
    assert slow_caller.kinds() == []

    assert slow.pause_scale > 1.5
    assert slow.effective_max_silence_ms > 1500
    assert slow.effective_max_silence_ms <= 1500 * EndpointingConfig().max_pause_scale
    assert slow.effective_min_silence_ms > 200

    # Let the slow caller's hesitant turn close before the comparison turn.
    await slow_caller.pause(4.0)
    assert slow_caller.kinds()[-1] == "end"

    fast_delay = await end_delay(Caller(fast, VirtualClock()))
    slow_delay = await end_delay(slow_caller)
    assert slow_delay > fast_delay + 0.3
    assert fast.pause_scale == pytest.approx(1.0)


async def test_short_frame_gaps_are_not_counted_as_pauses() -> None:
    clock = VirtualClock()
    controller = EndpointingController()
    caller = Caller(controller, clock)
    await caller.speak(2.0)
    assert controller.pause_scale == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("text", "low"),
    [
        ("my order number is", True),
        ("I went there and", True),
        ("so", True),
        ("um", True),
        ("I called because", True),
        ("I was thinking, um", True),
        ("I want to check on my order.", False),
        ("yes", False),
        ("No", False),
        ("that's right", False),
        ("where is my package", False),
        ("Can you help me?", False),
    ],
)
def test_heuristic_completeness(text: str, low: bool) -> None:
    score = HeuristicCompleteness().score(text)
    assert 0.0 <= score <= 1.0
    if low:
        assert score <= 0.2
    else:
        assert score >= 0.7


def test_heuristic_orders_punctuated_above_trailing_comma() -> None:
    scorer = HeuristicCompleteness()
    assert scorer.score("I need a refund.") > scorer.score("I need a refund,")


def test_config_rejects_inverted_thresholds() -> None:
    with pytest.raises(ValueError):
        EndpointingConfig(eager_threshold=0.9, end_threshold=0.8)
    with pytest.raises(ValueError):
        EndpointingConfig(min_silence_ms=2000, max_silence_ms=1500)
