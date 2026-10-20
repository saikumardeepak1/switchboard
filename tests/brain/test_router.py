import pytest

from switchboard.brain.router import (
    DEFAULT_ROUTER_CONFIG,
    HeuristicRouter,
    RouterConfig,
    RouteRule,
    RouteState,
)

KNOWN = RouteState(answer_in_evidence=True)
UNKNOWN = RouteState()


@pytest.mark.parametrize(
    ("text", "state", "tier", "reason"),
    [
        ("Hi there", UNKNOWN, "talker", "greeting"),
        ("hello", UNKNOWN, "talker", "greeting"),
        ("Good morning!", UNKNOWN, "talker", "greeting"),
        ("yes", UNKNOWN, "talker", "acknowledgement"),
        ("Yeah, that's right.", UNKNOWN, "talker", "acknowledgement"),
        ("okay thanks", UNKNOWN, "talker", "acknowledgement"),
        ("got it", UNKNOWN, "talker", "acknowledgement"),
        ("where is my order", KNOWN, "talker", "status_read_in_evidence"),
        ("when will it arrive?", KNOWN, "talker", "status_read_in_evidence"),
        ("where is my order", UNKNOWN, "thinker", "default"),
        ("yes", RouteState(tool_needed=True), "thinker", "tool_needed"),
        ("I want a refund", UNKNOWN, "thinker", "policy_sensitive"),
        ("yes please refund it", UNKNOWN, "thinker", "policy_sensitive"),
        ("Can you cancel the order", KNOWN, "thinker", "policy_sensitive"),
        ("mark it delivered", UNKNOWN, "thinker", "policy_sensitive"),
        ("you charged me twice", UNKNOWN, "thinker", "policy_sensitive"),
        ("I need to change my address", UNKNOWN, "thinker", "policy_sensitive"),
        ("hi, I need a refund", UNKNOWN, "thinker", "policy_sensitive"),
        ("Where is it? And who is the driver?", KNOWN, "thinker", "multiple_questions"),
        ("where is it and when will it get here", KNOWN, "thinker", "multiple_questions"),
        (
            "so the thing is I ordered this last week and my neighbor said a box showed up "
            "at their place but it was the wrong one",
            UNKNOWN,
            "thinker",
            "long_turn",
        ),
        ("what's the weather like", UNKNOWN, "thinker", "default"),
    ],
)
def test_router_table(text: str, state: RouteState, tier: str, reason: str) -> None:
    decision = HeuristicRouter().route(text, state)
    assert (decision.tier, decision.reason) == (tier, reason)


def test_explain_lists_every_rule_in_order_with_first_match_marked() -> None:
    router = HeuristicRouter()
    trace = router.explain("hi, I need a refund", UNKNOWN)
    names = [check.rule for check in trace.checks]
    assert names == [r.name for r in DEFAULT_ROUTER_CONFIG.rules]
    matched = [c.rule for c in trace.checks if c.matched]
    assert "policy_sensitive" in matched
    assert trace.decision.reason == "policy_sensitive"
    assert trace.decision == router.route("hi, I need a refund", UNKNOWN)


def test_rules_are_data_driven() -> None:
    config = RouterConfig(
        rules=(
            RouteRule(name="pizza", tier="thinker", patterns=(r"\bpizza\b",)),
            RouteRule(name="anything_short", tier="talker", max_words=3),
        ),
        default_tier="thinker",
    )
    router = HeuristicRouter(config)
    assert router.route("pizza now", UNKNOWN).reason == "pizza"
    assert router.route("ok fine", UNKNOWN).tier == "talker"
    assert router.route("this is a long sentence", UNKNOWN).reason == "default"


def test_invalid_pattern_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="bad"):
        HeuristicRouter(
            RouterConfig(rules=(RouteRule(name="bad", tier="talker", patterns=("(",)),))
        )
