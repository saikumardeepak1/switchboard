"""STT, TTS and transport interfaces, simulated providers, and the Twilio codec."""

from switchboard.providers.interfaces import (
    STT,
    TTS,
    AudioChunk,
    AudioFrame,
    SttEvent,
    SttFinal,
    SttPartial,
    SttVad,
    Transport,
)
from switchboard.providers.simulated import (
    CONFUSIONS,
    LatencyModel,
    SimulatedSTT,
    SimulatedTTS,
    inject_word_errors,
)

__all__ = [
    "CONFUSIONS",
    "STT",
    "TTS",
    "AudioChunk",
    "AudioFrame",
    "LatencyModel",
    "SimulatedSTT",
    "SimulatedTTS",
    "SttEvent",
    "SttFinal",
    "SttPartial",
    "SttVad",
    "Transport",
    "inject_word_errors",
]
