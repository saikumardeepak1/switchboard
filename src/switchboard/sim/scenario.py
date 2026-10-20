"""Scenarios: a bundle, simulated callers and a scripted brain, run on a virtual clock.

`run_scenario(scenario, seed)` builds a `World` (clock, stores, providers, callers, runtime),
starts the conversation through the public runtime API, plays the inbound caller's script until
the clock is idle, ends the conversation with an outcome, and scores the log with the scenario's
assertions. Everything that varies is drawn from `random.Random` streams derived from the seed,
so a seed reproduces a run byte for byte.

`pass_hat_k` is the tau-bench estimator: for a task with n trials and c successes, the chance
that k trials drawn without replacement all pass is C(c, k) / C(n, k); pass^k is its mean over
tasks. A task here is one scenario under one persona.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import random
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from math import comb
from pathlib import Path

from switchboard.brain import ScriptedLLM
from switchboard.brain.llm import Script
from switchboard.bundle import Bundle, load_bundle
from switchboard.core import AnyEvent, EventStore, InMemoryEventStore, VirtualClock
from switchboard.effects import IdempotencyLedger, InMemoryLedger
from switchboard.grounding import GroundingGate
from switchboard.knowledge import BitemporalStore, StatusPageIngestor, parse_timestamp
from switchboard.memory import MemoryStore
from switchboard.providers import LatencyModel
from switchboard.runtime import Conversation, ConversationState, Runtime
from switchboard.sim.adversarial import AdversarialLLM, AdversaryConfig, Fault
from switchboard.sim.assertions import Assertion, AssertionResult, RunRecord
from switchboard.sim.caller import Beat, CallerSummary, Persona, SimDialer, SimulatedCaller
from switchboard.sim.cost import (
    CacheMode,
    CallUsage,
    CostReport,
    MeteredLLM,
    PrefixCache,
    compute_cost,
)
from switchboard.sim.llm import SampledLatencyLLM
from switchboard.telemetry import TurnTimeline
from switchboard.turn import EndpointingConfig

# An eager threshold equal to the end threshold never fires before `end`: speculation off.
SPECULATION_OFF = EndpointingConfig().end_threshold
DEFAULT_START = parse_timestamp("2026-10-20T11:30:00-07:00")
# The simulated tenant is in UTC-7, so its local midnight anchors the time of day (ADR-005).
DEFAULT_DAY_START = parse_timestamp("2026-10-20T00:00:00-07:00")


@dataclass(frozen=True)
class SimConfig:
    """Provider behaviour for a run. Latency distributions are configuration, not measurements."""

    thinker_ttft: LatencyModel = LatencyModel(0.35, 0.8)
    talker_ttft: LatencyModel = LatencyModel(0.15, 0.3)
    tokens_per_s: float = 60.0
    use_talker: bool = True
    ring_s: float = 2.0
    eager_threshold: float | None = None
    cache_mode: CacheMode = "auto"
    gate_factory: Callable[[], GroundingGate] | None = None
    start_at: float = DEFAULT_START
    day_start: float | None = DEFAULT_DAY_START
    max_timer_steps: int = 400_000


@dataclass(frozen=True)
class CallerSpec:
    beats: Sequence[Beat]
    persona: Persona | None = None


OutcomeFn = Callable[[ConversationState, Bundle, Mapping[str, CallerSummary]], str]


def default_outcome(
    state: ConversationState, bundle: Bundle, callers: Mapping[str, CallerSummary]
) -> str:
    """handoff, resolved (the last procedure step completed), abandoned, or unresolved."""
    if state.handoffs:
        return "handoff"
    steps = bundle.policy.procedure.steps
    if steps and steps[-1] in state.completed_steps:
        return "resolved"
    inbound = state.inbound_leg()
    caller = callers.get(inbound.party) if inbound else None
    if caller is not None and caller.status == "gave_up":
        return "abandoned"
    return "unresolved"


@dataclass(frozen=True)
class Scenario:
    name: str
    bundle_path: Path
    # The first entry is the inbound caller; the rest answer outbound calls by party name.
    callers: Mapping[str, CallerSpec]
    # A factory, because scripts may keep per-call state such as call id counters.
    llm_script: Callable[[], Script]
    assertions: Sequence[Assertion] = ()
    tenant_id: str = "tenant"
    customer_id: str = "customer"
    statuspage: Path | None = None
    adversary: AdversaryConfig | None = None
    config: SimConfig = field(default_factory=SimConfig)
    outcome: OutcomeFn = default_outcome

    @property
    def inbound_party(self) -> str:
        return next(iter(self.callers))


@dataclass
class World:
    """Everything one simulated run is made of. Benchmarks reach in to inject crashes."""

    scenario: Scenario
    seed: int
    persona: str
    clock: VirtualClock
    bundle: Bundle
    store: EventStore
    ledger: IdempotencyLedger
    callers: dict[str, SimulatedCaller]
    dialer: SimDialer
    thinker: MeteredLLM
    talker: MeteredLLM | None
    scripted: list[ScriptedLLM]
    adversary: AdversarialLLM | None
    facts: BitemporalStore
    memory: MemoryStore
    gate: GroundingGate
    runtime: Runtime = field(init=False)
    failures: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.runtime = self.new_runtime()

    @property
    def inbound(self) -> SimulatedCaller:
        return self.callers[self.scenario.inbound_party]

    @property
    def usages(self) -> list[CallUsage]:
        out = list(self.thinker.usages)
        if self.talker is not None:
            out += self.talker.usages
        return out

    def close(self) -> None:
        """Unload the bundle's tool modules."""
        self.bundle.close()

    def new_runtime(
        self, *, ledger: IdempotencyLedger | None = None, store: EventStore | None = None
    ) -> Runtime:
        return Runtime(
            clock=self.clock,
            store=store if store is not None else self.store,
            ledger=ledger if ledger is not None else self.ledger,
            llm=self.thinker,
            talker=self.talker,
            rng=random.Random(f"{self.seed}:runtime"),
            gate=self.gate,
            memory=self.memory,
            facts=self.facts,
            dialer=self.dialer,
            day_start=self.scenario.config.day_start,
        )

    async def start(self) -> Conversation:
        return await self.runtime.start(
            self.bundle,
            tenant_id=self.scenario.tenant_id,
            customer_id=self.scenario.customer_id,
            io=self.inbound.io,
            party=self.scenario.inbound_party,
        )

    async def drive(self, caller: SimulatedCaller | None = None) -> None:
        """Play a caller's script (the inbound one by default) until the clock is idle."""
        caller = caller or self.inbound
        task = asyncio.ensure_future(caller.run())
        try:
            await self.clock.run_until_idle(self.scenario.config.max_timer_steps)
        except RuntimeError as exc:
            self.failures.append(f"clock: {exc}")
        for t in [task, *self.dialer.tasks]:
            if not t.done():
                t.cancel()
                self.failures.append("a caller script was still waiting when the clock went idle")
            elif not t.cancelled() and (err := t.exception()) is not None:
                self.failures.append(f"caller script failed: {err!r}")


def build_world(
    scenario: Scenario,
    seed: int,
    persona: Persona | None = None,
    *,
    store: EventStore | None = None,
    ledger: IdempotencyLedger | None = None,
) -> World:
    cfg = scenario.config
    clock = VirtualClock(cfg.start_at)
    bundle = load_bundle(scenario.bundle_path)
    if cfg.eager_threshold is not None:
        config = dataclasses.replace(bundle.config, eager_threshold=cfg.eager_threshold)
        bundle = dataclasses.replace(bundle, config=config)
    facts = BitemporalStore()
    if scenario.statuspage is not None:
        StatusPageIngestor(facts, "statuspage").ingest(json.loads(scenario.statuspage.read_text()))

    script = scenario.llm_script()
    adversary: AdversarialLLM | None = None
    if scenario.adversary is not None:
        if not callable(script):
            raise TypeError("an adversary wraps a callable script, not a list of steps")
        adversary = AdversarialLLM(script, scenario.adversary, seed=seed)
        script = adversary.reply
    cache = PrefixCache(cfg.cache_mode)
    scripted: list[ScriptedLLM] = []

    def tier(model: str, ttft: LatencyModel) -> MeteredLLM:
        inner = ScriptedLLM(clock, script, ttft_s=0.0, tokens_per_s=cfg.tokens_per_s)
        scripted.append(inner)
        sampled = SampledLatencyLLM(inner, clock, ttft, random.Random(f"{seed}:{model}"))
        return MeteredLLM(sampled, model, cache, clock)

    thinker = tier("thinker", cfg.thinker_ttft)
    talker = tier("talker", cfg.talker_ttft) if cfg.use_talker else None

    callers: dict[str, SimulatedCaller] = {}
    for party, spec in scenario.callers.items():
        callers[party] = SimulatedCaller(
            clock,
            party,
            spec.beats,
            persona or spec.persona or Persona(),
            seed=seed,
            tts_ttfb_s=bundle.config.voice.ttfb_s,
            tts_chars_per_s=bundle.config.voice.chars_per_s,
        )
    outbound = {p: c for p, c in callers.items() if p != scenario.inbound_party}
    return World(
        scenario=scenario,
        seed=seed,
        persona=persona.name if persona else _persona_name(scenario),
        clock=clock,
        bundle=bundle,
        store=store if store is not None else InMemoryEventStore(),
        ledger=ledger if ledger is not None else InMemoryLedger(clock),
        callers=callers,
        dialer=SimDialer(clock, outbound, ring_s=cfg.ring_s),
        thinker=thinker,
        talker=talker,
        scripted=scripted,
        adversary=adversary,
        facts=facts,
        memory=MemoryStore(),
        gate=cfg.gate_factory() if cfg.gate_factory else GroundingGate(),
    )


def _persona_name(scenario: Scenario) -> str:
    spec = scenario.callers[scenario.inbound_party]
    return spec.persona.name if spec.persona else "default"


@dataclass
class ScenarioResult:
    scenario: str
    seed: int
    persona: str
    passed: bool
    assertion_results: list[AssertionResult]
    events: list[AnyEvent]
    timeline: TurnTimeline
    cost: CostReport
    faults: list[Fault]
    outcome: str
    callers: dict[str, CallerSummary]
    voice_to_voice_ms: list[float]
    usages: list[CallUsage]
    failures: list[str]
    wall_s: float

    @property
    def task(self) -> tuple[str, str]:
        return (self.scenario, self.persona)


def reply_latencies_ms(caller: SimulatedCaller) -> list[float]:
    """Voice to voice as the caller hears it: end of each thing they said to the first agent
    audio that arrives before they speak again."""
    out: list[float] = []
    spoken = [u for u in caller.utterances if u.kind in ("say", "repeat")]
    starts = sorted(u.start for u in caller.utterances)
    for u in spoken:
        nxt = next((s for s in starts if s > u.start), float("inf"))
        first = next((c.at for c in caller.chunks if u.end <= c.at < nxt), None)
        if first is not None:
            out.append((first - u.end) * 1000.0)
    return out


async def finish(world: World, conv: Conversation, started: float) -> ScenarioResult:
    """End the conversation with its outcome, read the log back and score it."""
    scenario = world.scenario
    summaries = {p: c.summary() for p, c in world.callers.items()}
    outcome = scenario.outcome(conv.state, world.bundle, summaries)
    await conv.end(outcome)
    events = await world.store.read(conv.id)
    failures = [*world.failures, *(repr(f) for f in conv.failures)]
    faults = list(world.adversary.faults) if world.adversary else []
    record = RunRecord(
        events=events,
        callers=summaries,
        faults=faults,
        extra_evidence=(*conv.knowledge_texts(), *conv.memory_texts),
    )
    results = [a(record) for a in scenario.assertions]
    results.append(AssertionResult("runtime_failures", not failures, "; ".join(failures)))
    usages = world.usages
    return ScenarioResult(
        scenario=scenario.name,
        seed=world.seed,
        persona=world.persona,
        passed=all(r.passed for r in results),
        assertion_results=results,
        events=list(events),
        timeline=TurnTimeline.from_events(events),
        cost=compute_cost(events, usages),
        faults=faults,
        outcome=outcome,
        callers=summaries,
        voice_to_voice_ms=[v for c in world.callers.values() for v in reply_latencies_ms(c)],
        usages=usages,
        failures=failures,
        wall_s=time.perf_counter() - started,
    )


async def run_scenario_async(
    scenario: Scenario, seed: int, persona: Persona | None = None
) -> ScenarioResult:
    started = time.perf_counter()
    world = build_world(scenario, seed, persona)
    try:
        conv = await world.start()
        await world.drive()
        return await finish(world, conv, started)
    finally:
        world.close()


def run_scenario(scenario: Scenario, seed: int, persona: Persona | None = None) -> ScenarioResult:
    return asyncio.run(run_scenario_async(scenario, seed, persona))


@dataclass
class MatrixResult:
    results: list[ScenarioResult]

    def tasks(self) -> dict[tuple[str, str], list[bool]]:
        out: dict[tuple[str, str], list[bool]] = {}
        for r in self.results:
            out.setdefault(r.task, []).append(r.passed)
        return out

    def pass_hat_k(self, k: int) -> float:
        return pass_hat_k(self.results, k)


def run_matrix(
    scenarios: Iterable[Scenario],
    seeds: Iterable[int],
    personas: Sequence[Persona | None] = (None,),
) -> MatrixResult:
    seed_list = list(seeds)
    results = [run_scenario(s, seed, p) for s in scenarios for p in personas for seed in seed_list]
    return MatrixResult(results)


def pass_hat_k(
    results: Iterable[ScenarioResult]
    | Mapping[str, Sequence[bool]]
    | Mapping[tuple[str, str], Sequence[bool]],
    k: int,
) -> float:
    """Mean over tasks of C(c, k) / C(n, k), the unbiased pass^k estimator from tau-bench."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if isinstance(results, Mapping):
        tasks: dict[object, list[bool]] = {key: list(v) for key, v in results.items()}
    else:
        tasks = {}
        for r in results:
            tasks.setdefault(r.task, []).append(r.passed)
    if not tasks:
        raise ValueError("no tasks to score")
    total = 0.0
    for key, trials in tasks.items():
        n, c = len(trials), sum(trials)
        if n < k:
            raise ValueError(f"task {key!r} has {n} trials; pass^{k} needs at least {k} trials")
        total += comb(c, k) / comb(n, k)
    return total / len(tasks)
