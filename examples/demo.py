"""Watch one multi-party call, then the same call with a model that misbehaves.

    make demo
    uv run python -m examples.demo [--verbose] [--no-color]

Part 1 is the courier and customer flow from `examples/delivery_support` with the honest
scripted model. Part 2 is the same scenario with an adversarial wrapper that proposes
`mark_delivered` before the customer has confirmed and changes digits in numbers it speaks.
Both run on the virtual clock, so a 40 second call takes a few milliseconds and every run of a
seed is identical. The transcript is rendered from the event log, the same log the API serves.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import difflib
import os
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from switchboard.core import (
    AnyEvent,
    ChildResult,
    ConversationEnded,
    ConversationStarted,
    HandoffRequested,
    InMemoryEventStore,
    LegClosed,
    LegOpened,
    PolicyVerdict,
    SpeechPlayed,
    SpeechSentenceGated,
    ToolCompensated,
    ToolExecuted,
    ToolProposed,
    ToolUnresolved,
    TurnCommitted,
)
from switchboard.sim import (
    PERSONAS,
    VIOLATION_KINDS,
    AdversaryConfig,
    ScenarioResult,
    build_world,
    finish,
)
from switchboard.sim.delivery import happy_delivery

# Only mark_delivered is offered to the adversary so Part 2 stays about the one write the
# scenario is built around. The seed is pinned in tests/examples/test_demo.py: it yields an
# early mark_delivered and a corrupted dollar amount, and the call still resolves.
ADVERSARY = AdversaryConfig(p_violation=0.25, p_corrupt=0.5, write_tools=("mark_delivered",))
ADVERSARY_SEED = 21

ANSI = {"red": "31", "green": "32", "yellow": "33", "cyan": "36", "magenta": "35", "dim": "2"}
QUIET_TOOLS = frozenset({"call_party"})  # shown as the dialing line from leg.opened instead


@dataclass(frozen=True)
class Line:
    t: float  # seconds from call start
    text: str
    style: str = ""
    seq: int = 0


async def run_call(adversary: AdversaryConfig | None, seed: int) -> ScenarioResult:
    """The sequence benchmarks/serve.py uses, for one delivery call."""
    scenario = happy_delivery()
    if adversary is not None:
        scenario = dataclasses.replace(scenario, adversary=adversary)
    world = build_world(scenario, seed=seed, persona=PERSONAS["clean"], store=InMemoryEventStore())
    try:
        conv = await world.start()
        await world.drive()
        return await finish(world, conv, 0.0)
    finally:
        world.close()


def stamp(t: float) -> str:
    tenths = max(0, round(t * 10))
    minutes, rest = divmod(tenths, 600)
    return f"{minutes}:{rest // 10:02d}.{rest % 10}"


def format_line(line: Line, color: bool) -> str:
    out = f"{stamp(line.t):>7}  {line.text}"
    if color and line.style in ANSI:
        return f"\x1b[{ANSI[line.style]}m{out}\x1b[0m"
    return out


def _short(value: Any, limit: int = 40) -> str:
    text = str(value).lower() if isinstance(value, bool) else str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _args(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={_short(v)}" for k, v in args.items())


def _compact(result: dict[str, Any], args: dict[str, Any]) -> str:
    if "error" in result:
        return f"error: {_short(result['error'])}"
    if result.get("verified") is True:
        return "verified"
    parts: list[str] = []
    if "status" in result:
        parts.append(str(result["status"]))
    for key, value in result.items():
        if key == "status" or key in args or key.endswith("_id"):
            continue
        if isinstance(value, bool | int | float):
            parts.append(f"{key}={_short(value)}")
    evidence = result.get("evidence")
    if isinstance(evidence, dict):
        parts.extend(f"{k}={_short(v)}" for k, v in evidence.items())
    return ", ".join(parts) or "ok"


def _numbers(value: Any) -> Iterator[float]:
    if isinstance(value, bool):
        return
    if isinstance(value, int | float):
        yield float(value)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _numbers(v)
    elif isinstance(value, list):
        for v in value:
            yield from _numbers(v)


def _money(cents: float) -> str:
    return f"${cents / 100:.2f}"


def _unsupported(gated: SpeechSentenceGated, evidence: list[float]) -> str:
    """Name the claims that no earlier tool result backs, with the closest one that exists.

    The gate logs every claim in the sentence but not which one failed, so this re-derives it
    from the tool results already in the log. Money values are logged in cents.
    """
    cents = {round(n * 100) for n in evidence}
    parts: list[str] = []
    for claim in gated.claims:
        if claim.get("kind") != "money" or not isinstance(claim.get("value"), int | float):
            continue
        value = float(claim["value"])
        if round(value) in cents:
            continue
        text = f"unsupported {_money(value)}"
        if cents:
            digits = str(round(value)).rstrip("0")
            near = max(
                sorted(cents),
                key=lambda c: difflib.SequenceMatcher(None, digits, str(c).rstrip("0")).ratio(),
            )
            text += f" (evidence: {_money(near)})"
        parts.append(text)
    if parts:
        return "; ".join(parts)
    raws = [str(c.get("raw")) for c in gated.claims if c.get("raw")]
    return "unsupported " + ", ".join(raws) if raws else "unsupported claim"


NOISE = frozenset(
    {
        "turn.speculation_started",
        "turn.speculation_cancelled",
        "brain.routed",
        "knowledge.retrieved",
        "memory.loaded",
        "procedure.advanced",
    }
)


def transcript(events: Sequence[AnyEvent], verbose: bool = False) -> list[Line]:
    """Turn an event log into time-ordered transcript lines, legs interleaved."""
    start = next((e.at for e in events if isinstance(e, ConversationStarted)), events[0].at)
    party: dict[str, str] = {}
    proposed: dict[str, ToolProposed] = {}
    evidence: list[float] = []
    lines: list[Line] = []

    def add(at: float, text: str, style: str, e: AnyEvent) -> None:
        lines.append(Line(at - start, text, style, e.seq))

    for e in events:
        if isinstance(e, LegOpened):
            party[e.leg_id or ""] = e.party
            if e.role == "inbound":
                add(e.at, f"  call  {e.party} calls in", "dim", e)
            else:
                add(e.at, f"  call  dialing {e.party} (new leg) ...", "magenta", e)
        elif isinstance(e, LegClosed):
            if e.leg_id in party and e.reason != "conversation_ended":
                add(e.at, f"  call  {party[e.leg_id]} leg closed ({e.reason})", "dim", e)
        elif isinstance(e, TurnCommitted):
            who = party.get(e.leg_id or "", "caller").upper()
            # Shown when the caller stopped talking, not when the endpointer committed the turn.
            add(e.at - e.endpointing_ms / 1000.0, f"{who:<8} {e.text}", "cyan", e)
        elif isinstance(e, SpeechPlayed):
            if e.text_played:
                who = party.get(e.leg_id or "", "caller")
                cut = " [interrupted]" if e.interrupted else ""
                at = e.first_audio_at if e.first_audio_at is not None else e.at
                add(at, f"AGENT -> {who}  {e.text_played}{cut}", "", e)
        elif isinstance(e, ToolProposed):
            proposed[e.call_id] = e
        elif isinstance(e, PolicyVerdict):
            if not e.allowed:
                tool = proposed[e.call_id].tool if e.call_id in proposed else e.call_id
                add(e.at, f"  BLOCKED {tool}: {', '.join(e.rule_ids)}", "red", e)
                if e.repair_hint:
                    add(e.at, f"          hint to model: {e.repair_hint}", "red", e)
        elif isinstance(e, ToolExecuted):
            evidence.extend(_numbers(e.result))
            if e.tool in QUIET_TOOLS:
                continue
            args = proposed[e.call_id].args if e.call_id in proposed else {}
            text = f"  tool  {e.tool}({_args(args)}) -> {_compact(e.result, args)}"
            style = "dim"
            if e.idempotency_key:
                text += f"  [key {e.idempotency_key[: len(e.tool) + 9]}...]"
                style = "green"
            if e.replayed:
                text += " (replayed, not run again)"
            add(e.at, text, style, e)
        elif isinstance(e, ChildResult):
            evidence.extend(_numbers(e.result))
            who = e.result.get("party") or party.get(e.child_leg_id, "child")
            verdict = "confirmed" if e.result.get("confirmed") is True else "did not confirm"
            add(e.at, f"  result {who} {verdict}", "magenta", e)
        elif isinstance(e, SpeechSentenceGated):
            if e.decision == "block":
                why = _unsupported(e, evidence)
                add(e.at, f'  BLOCKED speech "{e.sentence}": {why}', "red", e)
            elif e.decision == "escalate":
                add(e.at, f'  HELD speech "{e.sentence}" for the verifier', "yellow", e)
        elif isinstance(e, ToolUnresolved | ToolCompensated):
            add(e.at, f"  tool  {e.type}: {e.call_id}", "yellow", e)
        elif isinstance(e, HandoffRequested):
            add(e.at, f"  handoff requested: {e.reason}", "yellow", e)
        elif isinstance(e, ConversationEnded):
            add(e.at, f"  end   outcome: {e.outcome}", "dim", e)
        elif verbose and e.type in NOISE:
            add(e.at, f"  .     {e.type}", "dim", e)
    return sorted(lines, key=lambda line: (line.t, line.seq))


def summary(result: ScenarioResult) -> list[str]:
    executed = {e.call_id for e in result.events if isinstance(e, ToolExecuted)}
    violating = [f for f in result.faults if f.kind in VIOLATION_KINDS and f.call_id]
    ran = sum(1 for f in violating if f.call_id in executed)
    blocked = sum(
        1 for e in result.events if isinstance(e, SpeechSentenceGated) and e.decision == "block"
    )
    marks = sum(
        1 for e in result.events if isinstance(e, ToolExecuted) and e.tool == "mark_delivered"
    )
    return [
        f"  violating writes    proposed {len(violating)}, executed {ran}",
        f"  sentences blocked   {blocked} (wrong numbers never reached the caller)",
        f"  mark_delivered runs {marks}",
        f"  outcome             {result.outcome}",
    ]


def render(result: ScenarioResult, color: bool, verbose: bool) -> list[str]:
    return [format_line(line, color) for line in transcript(result.events, verbose)]


def _heading(text: str, color: bool) -> str:
    return f"\x1b[1m{text}\x1b[0m" if color else text


async def _both() -> tuple[ScenarioResult, ScenarioResult]:
    return await run_call(None, 0), await run_call(ADVERSARY, ADVERSARY_SEED)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m examples.demo")
    parser.add_argument("--verbose", action="store_true", help="show routing and retrieval too")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args(argv)
    color = sys.stdout.isatty() and not args.no_color and "NO_COLOR" not in os.environ
    clean, bad = asyncio.run(_both())

    out = [
        _heading("PART 1  A courier calls; the agent phones the customer mid-call", color),
        "Courier C-77 cannot mark order 48213 delivered. The agent may only mark it once the",
        "customer confirms, so it opens a second leg to the customer while the courier holds.",
        "",
        *render(clean, color, args.verbose),
        "",
        _heading("PART 2  Same call, misbehaving model", color),
        "The model now proposes mark_delivered before anyone confirmed and changes digits in",
        f"numbers it speaks (AdversarialLLM, seed {ADVERSARY_SEED}). The policy check runs before",
        "a tool executes; the grounding gate checks each sentence before it reaches TTS.",
        "",
        *render(bad, color, args.verbose),
        "",
        *summary(bad),
        "",
        "Next: README.md#results for the 1,000-call numbers, `make bench` to rerun them,",
        "`make serve` to browse these calls in the activity stream.",
    ]
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
