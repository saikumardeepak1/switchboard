"""Tiered grounding gate for spoken sentences (ADR-005).

Tier 1 checks every extracted claim against the turn's evidence and decides on its own whenever
a sentence carries a claim: one unsupported claim blocks, all supported releases. Hedged claims
get the tolerance described in `evidence`.

Tier 2 handles sentences with no checkable claim. With a `SentenceScorer` it releases at or above
`release_at`, blocks at or below `block_at`, and escalates in between. Without a scorer a
claimless sentence releases: it states no number, time, date, amount or ID that could be wrong,
which is also the rule that lets bridge speech through (ADR-008).

Tier 3 is the async `Verifier`. `review` never awaits it; it returns `escalate` and the caller
runs `resolve` while earlier audio plays. With no verifier configured an escalation fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from switchboard.grounding.claims import Claim, ClaimValue, extract_claims, format_value
from switchboard.grounding.evidence import Evidence

Decision = Literal["release", "block", "escalate"]
Tier = Literal[1, 2, 3]


@runtime_checkable
class SentenceScorer(Protocol):
    def score(self, sentence: str, evidence: Evidence) -> float:
        """Probability in [0, 1] that the sentence is supported by the evidence."""
        ...


@dataclass(frozen=True)
class VerifierVerdict:
    supported: bool
    reason: str = ""


@runtime_checkable
class Verifier(Protocol):
    async def verify(self, sentence: str, evidence: Evidence) -> VerifierVerdict: ...


@dataclass(frozen=True)
class GateDecision:
    decision: Decision
    tier: Tier
    claims: tuple[Claim, ...]
    unsupported: tuple[Claim, ...]
    reason: str
    score: float | None = None
    # Aligned with `unsupported`: the closest evidence value of the same kind, if any.
    nearest: tuple[ClaimValue | None, ...] = ()

    def claim_dicts(self) -> list[dict[str, Any]]:
        """Claims in the plain form `SpeechSentenceGated.claims` stores."""
        return [c.to_dict() for c in self.claims]


class GroundingGate:
    def __init__(
        self,
        scorer: SentenceScorer | None = None,
        verifier: Verifier | None = None,
        *,
        release_at: float = 0.8,
        block_at: float = 0.2,
    ) -> None:
        if not 0.0 <= block_at < release_at <= 1.0:
            raise ValueError("thresholds need 0 <= block_at < release_at <= 1")
        self.scorer = scorer
        self.verifier = verifier
        self.release_at = release_at
        self.block_at = block_at

    def review(self, sentence: str, evidence: Evidence) -> GateDecision:
        claims = tuple(extract_claims(sentence))
        if claims:
            return self._tier_one(claims, evidence)
        return self._tier_two(sentence, evidence)

    def _tier_one(self, claims: tuple[Claim, ...], evidence: Evidence) -> GateDecision:
        unsupported = tuple(c for c in claims if not evidence.supports(c))
        if not unsupported:
            return GateDecision("release", 1, claims, (), f"all {len(claims)} claims supported")
        listed = ", ".join(f"{c.kind} {format_value(c.kind, c.value)}" for c in unsupported)
        return GateDecision(
            "block",
            1,
            claims,
            unsupported,
            f"unsupported: {listed}",
            nearest=tuple(evidence.nearest(c) for c in unsupported),
        )

    def _tier_two(self, sentence: str, evidence: Evidence) -> GateDecision:
        if self.scorer is None:
            return GateDecision("release", 2, (), (), "no checkable claims and no scorer")
        score = self.scorer.score(sentence, evidence)
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"scorer returned {score}, expected a value in [0, 1]")
        decision: Decision
        if score >= self.release_at:
            decision = "release"
        elif score <= self.block_at:
            decision = "block"
        else:
            decision = "escalate"
        return GateDecision(decision, 2, (), (), f"support score {score:.2f}", score=score)

    async def resolve(
        self, decision: GateDecision, sentence: str, evidence: Evidence
    ) -> GateDecision:
        """Settle an escalation with the tier 3 verifier. Other decisions pass through as is."""
        if decision.decision != "escalate":
            return decision
        if self.verifier is None:
            return GateDecision(
                "block",
                3,
                decision.claims,
                decision.unsupported,
                "escalated with no verifier configured",
                score=decision.score,
            )
        verdict = await self.verifier.verify(sentence, evidence)
        return GateDecision(
            "release" if verdict.supported else "block",
            3,
            decision.claims,
            decision.unsupported,
            verdict.reason,
            score=decision.score,
        )


def rewrite_hint(decision: GateDecision) -> str | None:
    """A short instruction for the brain to regenerate a blocked sentence."""
    if decision.decision != "block":
        return None
    if not decision.unsupported:
        return "Rewrite the sentence using only facts present in the evidence, or offer to check."
    nearest = decision.nearest or (None,) * len(decision.unsupported)
    lines = []
    for claim, near in zip(decision.unsupported, nearest, strict=True):
        said = format_value(claim.kind, claim.value)
        if near is None:
            lines.append(f"Do not state {said}; it is not in the evidence.")
        else:
            lines.append(f"Do not state {said}; evidence shows {format_value(claim.kind, near)}.")
    return " ".join(lines)
