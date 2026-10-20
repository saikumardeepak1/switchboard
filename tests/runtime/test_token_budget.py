"""A tenant over its tokens-per-minute budget is served by the talker (ADR-013)."""

from __future__ import annotations

import pytest

from switchboard.runtime import AdmissionController

from .harness import TENANT, SimParty, every, make_world, run_script
from .test_scenarios import verify


def test_the_budget_is_a_sliding_window() -> None:
    admission = AdmissionController(token_budgets={"t": 100})
    admission.record_tokens("t", 60, now=0.0)
    assert not admission.over_budget("t", now=1.0)
    admission.record_tokens("t", 50, now=30.0)
    assert admission.tokens_used("t", now=30.0) == 110
    assert admission.over_budget("t", now=30.0)
    # The first record leaves the 60 s window; the second is still inside it.
    assert admission.tokens_used("t", now=60.5) == 50
    assert not admission.over_budget("t", now=60.5)
    assert not admission.over_budget("other", now=0.0)


def test_a_bundle_default_applies_when_the_tenant_has_no_override() -> None:
    admission = AdmissionController()
    admission.record_tokens("t", 10, now=0.0)
    assert not admission.over_budget("t", now=0.0)
    assert admission.over_budget("t", now=0.0, default_budget=10)
    with pytest.raises(ValueError):
        AdmissionController(token_budgets={"t": 0})


async def test_an_exhausted_budget_routes_to_the_talker_with_reason_budget() -> None:
    world = make_world()
    world.admission = AdmissionController(token_budgets={TENANT: 1})
    world.runtime = world.new_runtime()
    conv = await world.start()

    async def courier(c: SimParty) -> None:
        await verify(c)
        await c.say("What is my pay this week?")
        await c.hear("182.40")
        await c.quiet()

    await run_script(world, courier(world.courier))
    assert conv.failures == []
    await conv.end("resolved")
    routed = every(await world.events(conv), "brain.routed")
    assert routed[0].reason != "budget"
    later = routed[1:]
    assert later and all((r.model_tier, r.reason) == ("talker", "budget") for r in later)
    assert world.llm.requests[-1].tier == "talker"
