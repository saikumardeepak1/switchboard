import pytest

from switchboard.grounding.evidence import Evidence
from switchboard.grounding.gate import (
    GateDecision,
    GroundingGate,
    SentenceScorer,
    Verifier,
    VerifierVerdict,
    rewrite_hint,
)
from switchboard.grounding.scorer import LexicalScorer

EVIDENCE = Evidence(tool_results=[{"refund": 12.5, "eta": "7:52 pm"}], now_minutes=19 * 60 + 30)


class FixedScorer:
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = 0

    def score(self, sentence: str, evidence: Evidence) -> float:
        self.calls += 1
        return self.value


class FixedVerifier:
    def __init__(self, supported: bool) -> None:
        self.supported = supported
        self.calls = 0

    async def verify(self, sentence: str, evidence: Evidence) -> VerifierVerdict:
        self.calls += 1
        return VerifierVerdict(supported=self.supported, reason="checked by fake")


def test_fakes_satisfy_protocols() -> None:
    assert isinstance(FixedScorer(0.5), SentenceScorer)
    assert isinstance(FixedVerifier(True), Verifier)
    assert isinstance(LexicalScorer(), SentenceScorer)


def test_realistic_support_sentence_blocks_on_money_accepts_duration() -> None:
    sentence = "Your refund of $18 will arrive in about 20 minutes"
    decision = GroundingGate().review(sentence, EVIDENCE)
    assert decision.decision == "block"
    assert decision.tier == 1
    assert [c.kind for c in decision.claims] == ["money", "duration"]
    assert [(c.kind, c.value) for c in decision.unsupported] == [("money", 1800)]
    duration = decision.claims[1]
    assert duration.approximate
    assert duration not in decision.unsupported
    assert rewrite_hint(decision) == "Do not state $18.00; evidence shows $12.50."


def test_all_claims_supported_releases_at_tier_one() -> None:
    scorer = FixedScorer(0.0)
    decision = GroundingGate(scorer=scorer).review(
        "Your refund of $12.50 arrives in about 20 minutes.", EVIDENCE
    )
    assert decision.decision == "release"
    assert decision.tier == 1
    assert decision.unsupported == ()
    assert scorer.calls == 0
    assert rewrite_hint(decision) is None


def test_exact_duration_outside_tolerance_blocks() -> None:
    decision = GroundingGate().review("It arrives in 20 minutes.", EVIDENCE)
    assert decision.decision == "block"
    assert rewrite_hint(decision) == "Do not state 20 minutes; evidence shows 22 minutes."


def test_unsupported_claim_without_nearest_value() -> None:
    decision = GroundingGate().review("I opened ticket AB-1029.", EVIDENCE)
    assert decision.decision == "block"
    assert rewrite_hint(decision) == "Do not state AB-1029; it is not in the evidence."


def test_hint_lists_every_unsupported_claim() -> None:
    decision = GroundingGate().review("Your refund of $18 arrives at 9 pm.", EVIDENCE)
    assert rewrite_hint(decision) == (
        "Do not state $18.00; evidence shows $12.50. Do not state 9:00 pm; evidence shows 7:52 pm."
    )


def test_claimless_without_scorer_releases_at_tier_two() -> None:
    decision = GroundingGate().review("Let me check that for you.", EVIDENCE)
    assert decision.decision == "release"
    assert decision.tier == 2
    assert decision.claims == ()


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.95, "release"), (0.8, "release"), (0.5, "escalate"), (0.2, "block"), (0.0, "block")],
)
def test_tier_two_thresholds(score: float, expected: str) -> None:
    gate = GroundingGate(scorer=FixedScorer(score))
    decision = gate.review("Your package is with the courier.", EVIDENCE)
    assert decision.tier == 2
    assert decision.decision == expected
    assert decision.score == score


def test_custom_thresholds() -> None:
    gate = GroundingGate(scorer=FixedScorer(0.6), release_at=0.5, block_at=0.1)
    assert gate.review("It is with the courier.", EVIDENCE).decision == "release"


@pytest.mark.parametrize(("release_at", "block_at"), [(0.3, 0.5), (1.5, 0.2), (0.8, -0.1)])
def test_invalid_thresholds_rejected(release_at: float, block_at: float) -> None:
    with pytest.raises(ValueError):
        GroundingGate(release_at=release_at, block_at=block_at)


def test_scorer_out_of_range_rejected() -> None:
    gate = GroundingGate(scorer=FixedScorer(1.2))
    with pytest.raises(ValueError):
        gate.review("It is with the courier.", EVIDENCE)


def test_tier_two_block_has_generic_hint() -> None:
    decision = GroundingGate(scorer=FixedScorer(0.0)).review("It is with the courier.", EVIDENCE)
    hint = rewrite_hint(decision)
    assert hint is not None
    assert "evidence" in hint


async def test_resolve_passes_through_non_escalations() -> None:
    verifier = FixedVerifier(False)
    gate = GroundingGate(verifier=verifier)
    sentence = "Your refund of $12.50 is on its way."
    decision = gate.review(sentence, EVIDENCE)
    assert await gate.resolve(decision, sentence, EVIDENCE) is decision
    assert verifier.calls == 0


@pytest.mark.parametrize(("supported", "expected"), [(True, "release"), (False, "block")])
async def test_resolve_escalation_with_verifier(supported: bool, expected: str) -> None:
    verifier = FixedVerifier(supported)
    gate = GroundingGate(scorer=FixedScorer(0.5), verifier=verifier)
    sentence = "It is with the courier."
    escalated = gate.review(sentence, EVIDENCE)
    assert escalated.decision == "escalate"
    resolved = await gate.resolve(escalated, sentence, EVIDENCE)
    assert resolved.decision == expected
    assert resolved.tier == 3
    assert resolved.reason == "checked by fake"
    assert verifier.calls == 1


async def test_resolve_without_verifier_fails_closed() -> None:
    gate = GroundingGate(scorer=FixedScorer(0.5))
    sentence = "It is with the courier."
    resolved = await gate.resolve(gate.review(sentence, EVIDENCE), sentence, EVIDENCE)
    assert resolved.decision == "block"
    assert resolved.tier == 3


def test_decision_claims_serialize_for_events() -> None:
    decision = GroundingGate().review("Your refund of $18 is ready.", EVIDENCE)
    assert isinstance(decision, GateDecision)
    assert decision.claim_dicts() == [
        {"kind": "money", "raw": "$18", "value": 1800, "span": [15, 18], "approximate": False}
    ]


def test_lexical_scorer_overlap() -> None:
    scorer = LexicalScorer()
    evidence = Evidence(tool_results=[{"status": "Package is with the courier near Main Street"}])
    assert scorer.score("Your package is with the courier.", evidence) == pytest.approx(1.0)
    assert scorer.score("Your package was destroyed in a fire.", evidence) < 0.5
    assert scorer.score("Okay.", evidence) == 1.0
