"""Input and output signals for the turn layer.

Times are clock seconds (`Clock.now()`), the same timeline the rest of the runtime uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class VadFrame:
    """One voice-activity decision for a short audio frame (typically 20 ms)."""

    at: float
    speech: bool
    energy: float


@dataclass(frozen=True, slots=True)
class PartialTranscript:
    """Interim STT hypothesis for the current segment.

    `stable_prefix_len` is how many leading characters the recognizer says will not change.
    """

    at: float
    text: str
    stable_prefix_len: int


@dataclass(frozen=True, slots=True)
class FinalTranscript:
    """Finalized STT segment. Several finals can make up one user turn."""

    at: float
    text: str


InputSignal = VadFrame | PartialTranscript | FinalTranscript

TurnSignalKind = Literal["eager_end", "end", "resumed"]


@dataclass(frozen=True, slots=True)
class TurnSignal:
    """End-of-turn decision emitted by the endpointing controller.

    `confidence` is the combined end-of-turn probability at the moment of emission.
    """

    kind: TurnSignalKind
    at: float
    text: str
    confidence: float
