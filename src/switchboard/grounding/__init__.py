"""Claim extraction, normalization and the tiered grounding gate (ADR-005)."""

from switchboard.grounding.claims import Claim, ClaimKind, ClaimValue, extract_claims, format_value
from switchboard.grounding.evidence import Evidence
from switchboard.grounding.gate import (
    POLICY_WORDS,
    Decision,
    GateDecision,
    GroundingGate,
    SentenceScorer,
    Verifier,
    VerifierVerdict,
    makes_assertion,
    rewrite_hint,
)
from switchboard.grounding.scorer import LexicalScorer

__all__ = [
    "POLICY_WORDS",
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
    "makes_assertion",
    "rewrite_hint",
]
