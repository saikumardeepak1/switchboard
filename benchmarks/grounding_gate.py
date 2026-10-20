"""Grounding gate: tiered checking (ADR-005) against sending every sentence to a verifier.

A corpus of agent sentences is generated from templates that mirror what a delivery support
agent says (pay totals, refunds, order ids, clock ETAs, minute ETAs, claimless bridge speech),
each paired with the tool result it should be grounded in plus distractor values from the same
call. Some templates say the number the way people do ("twelve fifty", "a quarter to eight",
"in about 20 minutes"), so the false-block rate includes paraphrase, not only exact strings.
Each sample also has the time of day the sentence is spoken at: the relative ETA template sets
it (the ETA minus the minutes said), every other sentence gets one drawn from its own stream.

With probability `P_CORRUPT` per sentence that carries a money amount, ETA or order id, the
adversary's `corrupt_numbers` changes one of them. Evidence is built the way the runtime builds
it (tool results plus the time of day, `Runtime.day_start`), so the gate sees what it would see
in a call.

Configurations:

- `tier1_only`: the runtime default gate. Claims are checked against evidence; a sentence with no
  claims is released (it states nothing checkable). There is no verifier, so an escalation (a
  hedged or ambiguous claim that misses) fails closed and counts as a block, as in a call.
- `with_scorer`: tier 2 is the reference `LexicalScorer`; sentences it cannot decide escalate to
  tier 3, a simulated verifier that catches a corrupted sentence with recall `VERIFIER_RECALL`.
- baseline: every sentence goes to the verifier (the reference design's per-sentence reasoning
  model), same recall.

Tier 1 latency is real `perf_counter` time around `GroundingGate.review`.
"""

from __future__ import annotations

import random
import time
from collections import Counter
from collections.abc import Callable
from typing import Any

from benchmarks.common import cli, envelope, pct, pmap, ratio, table
from switchboard.grounding import Evidence, GroundingGate, LexicalScorer
from switchboard.sim.adversarial import corrupt_numbers, has_corruptible_number

NAME = "grounding_gate"
N_DEFAULT = 20_000
P_CORRUPT = 0.05
VERIFIER_RECALL = 0.95
CHUNKS = 8

# category, sentence, tool result, and the minute of day it is spoken at when the template fixes it
Sample = tuple[str, str, dict[str, Any], int | None]

_ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
]
_TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty"}

BRIDGE = [
    "One moment.",
    "Let me check that for you.",
    "I'm calling the customer now, please stay on the line.",
    "Thanks for your patience.",
    "The customer confirmed the new address.",
    "The chat doesn't show an address change, so I'll call the customer.",
    "I can help with that.",
    "Sorry, could you say that again?",
    "I need to verify you first.",
    "Thanks for calling, goodbye.",
    "The courier is on the way.",
    "Your refund is on its way to the original card.",
    "I've passed that to the delivery team.",
    "The geofence service is degraded in your area right now.",
    "I'll stay on the line with you.",
]


def _order(rng: random.Random) -> str:
    return str(rng.randrange(10_000, 99_999))


def _distractors(rng: random.Random) -> dict[str, Any]:
    return {
        "previous_order_id": _order(rng),
        "previous_refund": float(rng.randrange(3, 40)),
        "courier_id": f"C-{rng.randrange(10, 99)}",
    }


def pay(rng: random.Random) -> Sample:
    total = rng.randrange(5_000, 40_000) / 100
    n = rng.randrange(5, 60)
    return (
        "pay",
        f"You've earned ${total:.2f} this week across {n} deliveries.",
        {"week_total": total, "deliveries": n},
        None,
    )


def refund(rng: random.Random) -> Sample:
    amount = float(rng.choice([5, 8, 10, 12.5, 15, 20, 25]))
    return "refund", f"Done, I refunded ${amount:.2f} to the customer.", {"amount": amount}, None


def order(rng: random.Random) -> Sample:
    oid = _order(rng)
    return "order_id", f"All set, order {oid} is marked delivered.", {"order_id": oid}, None


def eta_clock(rng: random.Random) -> Sample:
    h, m = rng.randrange(1, 12), rng.choice([0, 5, 10, 15, 20, 30, 40, 45, 50])
    eta = f"{h}:{m:02d} PM"
    return "eta_clock", f"Your order should arrive at {h}:{m:02d} pm.", {"eta": eta}, None


def eta_minutes(rng: random.Random) -> Sample:
    n = rng.choice([5, 10, 12, 15, 20, 25, 30, 40])
    return "eta_minutes", f"The courier is about {n} minutes away.", {"eta_minutes": n}, None


def spoken_money(rng: random.Random) -> Sample:
    """'twelve fifty' for $12.50: how people say amounts."""
    dollars, cents = rng.randrange(5, 13), rng.choice([20, 30, 40, 50])
    words = f"{_ONES[dollars]} {_TENS[cents]}"
    amount = dollars + cents / 100
    return "paraphrase_money", f"I refunded {words} to the customer.", {"amount": amount}, None


def spoken_quarter(rng: random.Random) -> Sample:
    """'a quarter to eight' for 7:45 PM."""
    h = rng.randrange(2, 12)
    words = _ONES[h] if h <= 12 else str(h)
    eta = f"{h - 1}:45 PM"
    return "paraphrase_clock", f"It should be there at a quarter to {words}.", {"eta": eta}, None


def relative_eta(rng: random.Random) -> Sample:
    """'in about 20 minutes' when the tool result is a clock time (needs the current time)."""
    h, m = rng.randrange(1, 11), rng.choice([10, 20, 30, 40])
    ahead = rng.choice([10, 15, 20, 25, 30])
    total = h * 60 + m + ahead
    eta = f"{total // 60}:{total % 60:02d} PM"
    now = (12 + h) * 60 + m
    sentence = f"Your order should arrive in about {ahead} minutes."
    return "relative_eta", sentence, {"eta": eta}, now


def bridge(rng: random.Random) -> Sample:
    return "claimless", rng.choice(BRIDGE), {"status": "out_for_delivery"}, None


GENERATORS: list[tuple[Callable[[random.Random], Sample], float]] = [
    (pay, 0.12),
    (refund, 0.10),
    (order, 0.12),
    (eta_clock, 0.08),
    (eta_minutes, 0.08),
    (spoken_money, 0.03),
    (spoken_quarter, 0.02),
    (relative_eta, 0.03),
    (bridge, 0.42),
]


class Oracle:
    """Tier 3 stand-in: a verifier that catches a corrupted sentence with fixed recall."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.calls = 0

    def supported(self, corrupted: bool) -> bool:
        self.calls += 1
        return not (corrupted and self.rng.random() < VERIFIER_RECALL)


def chunk(job: tuple[int, int]) -> dict[str, Any]:
    seed, n = job
    rng = random.Random(f"grounding:{seed}")
    gens = [g for g, _ in GENERATORS]
    weights = [w for _, w in GENERATORS]
    plain = GroundingGate()
    scored = GroundingGate(scorer=LexicalScorer())
    oracle = Oracle(random.Random(f"grounding:{seed}:verifier"))
    baseline = Oracle(random.Random(f"grounding:{seed}:baseline"))
    clock = random.Random(f"grounding:{seed}:now")
    c: Counter[str] = Counter()
    latency: list[float] = []
    for _ in range(n):
        category, sentence, result, fixed_now = rng.choices(gens, weights)[0](rng)
        now = fixed_now if fixed_now is not None else clock.randrange(1440)
        corrupted = False
        if has_corruptible_number(sentence) and rng.random() < P_CORRUPT:
            out = corrupt_numbers(sentence, rng)
            if out is not None:
                sentence, corrupted = out[0], True
        evidence = Evidence(tool_results=[result, _distractors(rng)], now_minutes=now)
        start = time.perf_counter()
        d = plain.review(sentence, evidence)
        latency.append((time.perf_counter() - start) * 1e6)
        s = scored.review(sentence, evidence)
        c["sentences"] += 1
        c[f"cat:{category}"] += 1
        has_claims = bool(d.claims)
        c["with_claims" if has_claims else "claimless"] += 1
        # The runtime default has no verifier, so `resolve` turns every escalation into a block.
        blocked = d.decision in ("block", "escalate")
        if d.decision == "escalate":
            c["tier1_escalated"] += 1
        if corrupted:
            c["corrupted"] += 1
            c[f"corrupted:{category}"] += 1
            if blocked:
                c["tier1_caught"] += 1
            if d.decision == "release":
                c["tier1_only_released_corrupt"] += 1
        else:
            c["clean"] += 1
            if has_claims:
                c["clean_with_claims"] += 1
            if blocked:
                c["false_block"] += 1
                c[f"false_block:{category}"] += 1
        # with a scorer, plus a verifier for escalations
        if s.decision == "escalate":
            c["escalated"] += 1
            final_release = oracle.supported(corrupted)
        else:
            final_release = s.decision == "release"
            if s.tier == 2 and s.decision == "block" and not corrupted:
                c["scorer_false_block"] += 1
        if corrupted and final_release:
            c["scored_released_corrupt"] += 1
        if not corrupted and not final_release:
            c["scored_false_block"] += 1
        # baseline: the verifier sees every sentence
        if baseline.supported(corrupted) and corrupted:
            c["baseline_released_corrupt"] += 1
    return {"counts": dict(c), "latency_us": latency, "verifier_calls": oracle.calls}


def run(n: int | None = None, workers: int | None = None) -> dict[str, Any]:
    n = n or N_DEFAULT
    sizes = [n // CHUNKS + (1 if i < n % CHUNKS else 0) for i in range(CHUNKS)]
    parts = pmap(chunk, [(i, size) for i, size in enumerate(sizes) if size], workers)
    c: Counter[str] = Counter()
    latency: list[float] = []
    calls = 0
    for p in parts:
        c.update(p["counts"])
        latency += p["latency_us"]
        calls += p["verifier_calls"]
    categories = sorted(k.split(":", 1)[1] for k in c if k.startswith("cat:"))
    by_category = {
        cat: {
            "sentences": c[f"cat:{cat}"],
            "corrupted": c[f"corrupted:{cat}"],
            "false_blocks": c[f"false_block:{cat}"],
            "false_block_rate": ratio(
                c[f"false_block:{cat}"], c[f"cat:{cat}"] - c[f"corrupted:{cat}"]
            ),
        }
        for cat in categories
    }
    sentences = c["sentences"]
    results = {
        "sentences": sentences,
        "corrupted": c["corrupted"],
        "corruption_rate_of_all_sentences": ratio(c["corrupted"], sentences),
        "tier1_caught": c["tier1_caught"],
        "tier1_catch_rate": ratio(c["tier1_caught"], c["corrupted"]),
        "tier1_only_released_corrupt": c["tier1_only_released_corrupt"],
        "false_blocks_clean": c["false_block"],
        "false_block_rate_clean": ratio(c["false_block"], c["clean"]),
        "false_block_rate_clean_with_claims": ratio(c["false_block"], c["clean_with_claims"]),
        "by_category": by_category,
        "tier1_latency_us": {"p50": pct(latency, 50), "p99": pct(latency, 99)},
        "tier1_only": {
            "verifier_calls": 0,
            "tier3_share": 0.0,
            "escalated": c["tier1_escalated"],
            "escalated_share": ratio(c["tier1_escalated"], sentences),
        },
        "with_scorer": {
            "verifier_calls": calls,
            "tier3_share": ratio(c["escalated"], sentences),
            "scorer_false_blocks": c["scorer_false_block"],
            "false_blocks_clean": c["scored_false_block"],
            "released_corrupt": c["scored_released_corrupt"],
        },
        "baseline_verifier_calls": sentences,
        "baseline_released_corrupt": c["baseline_released_corrupt"],
        "baseline_catch_rate": ratio(
            c["corrupted"] - c["baseline_released_corrupt"], c["corrupted"]
        ),
        "verifier_calls_avoided_with_scorer": ratio(sentences - calls, sentences),
    }
    return envelope(
        NAME,
        {
            "n": n,
            "p_corrupt": P_CORRUPT,
            "verifier_recall": VERIFIER_RECALL,
            "mix": {g.__name__: w for g, w in GENERATORS},
        },
        results,
    )


def render(data: dict[str, Any]) -> str:
    r = data["results"]
    head = ["config", "verifier_calls", "catch_rate", "released_corrupt", "false_block_rate"]
    rows = [
        [
            "tier1_only",
            0,
            r["tier1_catch_rate"],
            r["tier1_only_released_corrupt"],
            r["false_block_rate_clean"],
        ],
        [
            "with_scorer",
            r["with_scorer"]["verifier_calls"],
            None,
            r["with_scorer"]["released_corrupt"],
            ratio(r["with_scorer"]["false_blocks_clean"], r["sentences"] - r["corrupted"]),
        ],
        [
            "baseline_all_tier3",
            r["baseline_verifier_calls"],
            r["baseline_catch_rate"],
            r["baseline_released_corrupt"],
            0.0,
        ],
    ]
    cats = table(
        ["category", "sentences", "corrupted", "false_blocks", "false_block_rate"],
        [
            [k, v["sentences"], v["corrupted"], v["false_blocks"], v["false_block_rate"]]
            for k, v in r["by_category"].items()
        ],
    )
    return f"{table(head, rows)}\n\n{cats}"


def main() -> None:
    cli(NAME, run, render)


if __name__ == "__main__":
    main()
