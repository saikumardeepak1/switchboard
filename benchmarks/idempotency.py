"""Idempotent writes (ADR-009): crash during a write, recover from the log, count applies.

Each seed runs the refund flow (two writes: mark_delivered, then a $10 refund), picks one of the
two writes and one crash point, and kills the process there:

- before_downstream: the tool is about to send its request; nothing applied.
- after_downstream: the downstream service applied the effect, the response is lost.
- before_log: the result is in the ledger, but `tool.executed` never reaches the event log.
- after_log: `tool.executed` is in the log; the process dies right after.

"Killed" means every later append to the event store fails (nothing more is written) and the
conversation is abandoned. Then a new runtime recovers the conversation from the log through
`Runtime.recover`, which re-runs allowed calls whose result is missing, and the caller says
hello on a fresh line. Afterwards the downstream service's applied count is compared with the
writes the agent intended: allowed write proposals in the final log, one apply each.

Configurations:

- switchboard: the ledger survives (it is durable, e.g. SQLite) and the downstream honours the
  idempotency key, as the example's `DownstreamAPI` does.
- switchboard_ledger_lost: the ledger is lost with the process; the deterministic key still
  reaches the downstream.
- switchboard_downstream_ignores_key: the ledger survives, but the downstream applies every
  request (it does not support idempotency keys).
- naive_retry: the baseline. No ledger survives and the downstream ignores keys, which is what
  checkpoint-and-retry without dedupe amounts to (ADR-009's "random UUIDs per retry").

Simulation detail: a killed tool is modeled by raising inside it, so the effect runner marks the
ledger claim failed rather than leaving it pending until its lease expires. Recovery outcome is
the same (the key can be claimed again); a real recovery would wait out the lease.
"""

from __future__ import annotations

import asyncio
import dataclasses
import random
import time
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from benchmarks.common import cli, envelope, pmap, ratio, table
from switchboard.core import AnyEvent, InMemoryEventStore, NewEvent
from switchboard.effects import InMemoryLedger
from switchboard.runtime import SPECULATION_RULE, replay
from switchboard.sim import HangUp, Say, SimulatedCaller, WaitForAgent, build_world, finish
from switchboard.sim.delivery import refund_delivery

NAME = "idempotency"
N_DEFAULT = 500
POINTS = ("before_downstream", "after_downstream", "before_log", "after_log")
TARGETS = ("mark_delivered", "issue_refund")
CONFIGS: dict[str, tuple[bool, bool]] = {
    # name: (ledger survives, downstream honours the key)
    "switchboard": (True, True),
    "switchboard_ledger_lost": (False, True),
    "switchboard_downstream_ignores_key": (True, False),
    "naive_retry": (False, False),
}
AFTER = (Say("Hello, are you still there?"), WaitForAgent(timeout_s=8.0), HangUp())


class ProcessKilled(ConnectionResetError):
    pass


@dataclasses.dataclass
class Switch:
    point: str
    tool: str
    fired: bool = False
    down: bool = False

    def kill(self) -> None:
        self.fired = True
        self.down = True


class CrashingStore(InMemoryEventStore):
    def __init__(self, switch: Switch) -> None:
        super().__init__()
        self.switch = switch

    async def append(
        self, conversation_id: str, events: Sequence[NewEvent], expected_seq: int
    ) -> list[AnyEvent]:
        sw = self.switch
        if sw.down:
            raise ProcessKilled("the process is down")
        batch = list(events)
        hit = not sw.fired and any(
            e.type == "tool.executed" and getattr(e, "tool", "") == sw.tool for e in batch
        )
        if hit and sw.point == "before_log":
            sw.kill()
            raise ProcessKilled("killed before the result reached the log")
        stored = await super().append(conversation_id, batch, expected_seq)
        if hit and sw.point == "after_log":
            sw.kill()
        return stored


class CrashingDownstream:
    """Wraps the example's DownstreamAPI: kills the process around one request, and can drop
    the idempotency key so every request applies."""

    def __init__(self, inner: Any, switch: Switch, honours_key: bool) -> None:
        self.inner = inner
        self.switch = switch
        self.honours_key = honours_key
        self.sent = 0

    @property
    def applied(self) -> Counter[str]:
        result: Counter[str] = self.inner.applied
        return result

    def call(self, op: str, key: str, apply: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        sw = self.switch
        armed = not sw.fired and op == sw.tool
        if armed and sw.point == "before_downstream":
            sw.kill()
            raise ProcessKilled("killed before the request was sent")
        self.sent += 1
        response: dict[str, Any] = self.inner.call(
            op, key if self.honours_key else f"{key}#{self.sent}", apply
        )
        if armed and sw.point == "after_downstream":
            sw.kill()
            raise ProcessKilled("killed after the downstream applied it")
        return response


def intended(events: list[AnyEvent]) -> Counter[str]:
    """Allowed write proposals in the log: each should apply exactly once downstream."""
    state = replay(events)
    out: Counter[str] = Counter()
    for call in state.calls.values():
        if call.tool not in TARGETS or not call.verdicts:
            continue
        allowed, rules = call.verdicts[-1]
        if allowed and SPECULATION_RULE not in rules:
            out[call.tool] += 1
    return out


def _quiet_kills(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
    # Tasks of the killed process die with ProcessKilled; that is the point, not a bug report.
    if not isinstance(context.get("exception"), ProcessKilled):
        loop.default_exception_handler(context)


async def crash_and_recover(config: str, seed: int) -> dict[str, Any]:
    asyncio.get_running_loop().set_exception_handler(_quiet_kills)
    survives, honours = CONFIGS[config]
    rng = random.Random(f"idempotency:{seed}")
    switch = Switch(point=rng.choice(POINTS), tool=rng.choice(TARGETS))
    started = time.perf_counter()
    store = CrashingStore(switch)
    world = build_world(refund_delivery(), seed, store=store)
    backend = world.bundle.tools_module.backend
    backend.downstream = CrashingDownstream(backend.downstream, switch, honours)
    conv = await world.start()
    await world.drive()
    recovered = conv
    if switch.fired:
        await conv.abandon()
        switch.down = False  # the process restarts
        ledger = world.ledger if survives else InMemoryLedger(world.clock)
        runtime = world.new_runtime(ledger=ledger)
        line = SimulatedCaller(
            world.clock,
            "courier",
            AFTER,
            world.inbound.persona,
            seed=seed + 1_000_000,
            tts_ttfb_s=world.bundle.config.voice.ttfb_s,
            tts_chars_per_s=world.bundle.config.voice.chars_per_s,
        )
        recovered = await runtime.recover(world.bundle, conv.id, io=line.io)
        await world.drive(line)
    result = await finish(world, recovered, started)
    world.close()
    want = intended(result.events)
    applied = Counter({op: n for op, n in backend.downstream.applied.items() if op in TARGETS})
    doubles = sum(max(0, applied[op] - want[op]) for op in TARGETS)
    missed = sum(max(0, want[op] - applied[op]) for op in TARGETS)
    return {
        "point": switch.point,
        "tool": switch.tool,
        "crashed": switch.fired,
        "intended": sum(want.values()),
        "applied": sum(applied.values()),
        "double_applied": doubles,
        "missed": missed,
    }


def one(job: tuple[str, int]) -> dict[str, Any]:
    return asyncio.run(crash_and_recover(*job))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    crashed = [r for r in rows if r["crashed"]]
    by_point = {
        p: {
            "runs": sum(1 for r in crashed if r["point"] == p),
            "double_applied": sum(r["double_applied"] for r in crashed if r["point"] == p),
            "missed": sum(r["missed"] for r in crashed if r["point"] == p),
        }
        for p in POINTS
    }
    return {
        "runs": len(rows),
        "crashed_runs": len(crashed),
        "intended_writes": sum(r["intended"] for r in rows),
        "applied_writes": sum(r["applied"] for r in rows),
        "double_applied": sum(r["double_applied"] for r in rows),
        "runs_with_double_apply": sum(1 for r in rows if r["double_applied"]),
        "double_apply_rate_per_crash": ratio(
            sum(1 for r in crashed if r["double_applied"]), len(crashed)
        ),
        "missed": sum(r["missed"] for r in rows),
        "by_crash_point": by_point,
    }


def run(n: int | None = None, workers: int | None = None) -> dict[str, Any]:
    n = n or N_DEFAULT
    jobs = [(c, s) for c in CONFIGS for s in range(n)]
    rows = pmap(one, jobs, workers)
    results = {
        c: summarize([r for (jc, _), r in zip(jobs, rows, strict=True) if jc == c]) for c in CONFIGS
    }
    return envelope(
        NAME,
        {
            "n": n,
            "crash_points": list(POINTS),
            "targets": list(TARGETS),
            "configs": {
                k: {"ledger_survives": v[0], "downstream_honours_key": v[1]}
                for k, v in CONFIGS.items()
            },
        },
        results,
    )


def render(data: dict[str, Any]) -> str:
    head = ["config", "crashed", "intended", "applied", "double", "missed", *POINTS]
    rows = [
        [
            k,
            v["crashed_runs"],
            v["intended_writes"],
            v["applied_writes"],
            v["double_applied"],
            v["missed"],
            *(v["by_crash_point"][p]["double_applied"] for p in POINTS),
        ]
        for k, v in data["results"].items()
    ]
    return table(head, rows) + "\n(last four columns: double applies by crash point)"


def main() -> None:
    cli(NAME, run, render)


if __name__ == "__main__":
    main()
