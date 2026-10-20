"""Claim extraction, normalization and the tiered grounding gate (ADR-005)."""

from switchboard.grounding.claims import Claim, ClaimKind, ClaimValue, extract_claims, format_value
from switchboard.grounding.evidence import Evidence
from switchboard.grounding.gate import (
    Decision,
    GateDecision,
    GroundingGate,
    SentenceScorer,
    Verifier,
    VerifierVerdict,
    rewrite_hint,
)
from switchboard.grounding.scorer import LexicalScorer

__all__ = [
    "Claim",
    "ClaimKind",
    "ClaimValue",
    "Decision",
    "Evidence",
    "GateDecision",
    "GroundingGate",
    "LexicalScorer",
    "SentenceScorer",
    "Verifier",
    "VerifierVerdict",
    "extract_claims",
    "format_value",
    "rewrite_hint",
]
