"""Turn taking: endpointing, speculative turns, playback cursor and barge-in (ADR-006, ADR-007)."""

from switchboard.turn.endpointing import (
    AudioTurnScorer,
    CompletenessScorer,
    EndpointingConfig,
    EndpointingController,
    HeuristicCompleteness,
)
from switchboard.turn.playback import (
    BargeInConfig,
    BargeInDecision,
    BargeInPolicy,
    PlaybackCursor,
    is_backchannel,
)
from switchboard.turn.signals import (
    FinalTranscript,
    InputSignal,
    PartialTranscript,
    TurnSignal,
    VadFrame,
)
from switchboard.turn.speculative import (
    CancelToken,
    GenerationContext,
    SpeculationMetrics,
    SpeculationRecord,
    SpeculativeTurnEngine,
    StartGeneration,
    normalize_text,
)

__all__ = [
    "AudioTurnScorer",
    "BargeInConfig",
    "BargeInDecision",
    "BargeInPolicy",
    "CancelToken",
    "CompletenessScorer",
    "EndpointingConfig",
    "EndpointingController",
    "FinalTranscript",
    "GenerationContext",
    "HeuristicCompleteness",
    "InputSignal",
    "PartialTranscript",
    "PlaybackCursor",
    "SpeculationMetrics",
    "SpeculationRecord",
    "SpeculativeTurnEngine",
    "StartGeneration",
    "TurnSignal",
    "VadFrame",
    "is_backchannel",
    "normalize_text",
]
