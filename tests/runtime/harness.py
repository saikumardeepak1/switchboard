"""Virtual-clock harness for runtime scenarios: simulated parties, a dialer, a scripted brain."""

from __future__ import annotations

import asyncio
import itertools
import json
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from switchboard.brain import BrainRequest, Message, Reply, ScriptedLLM, ToolCallChunk
from switchboard.bundle import Bundle, load_bundle
from switchboard.core import AnyEvent, InMemoryEventStore, VirtualClock
from switchboard.effects import InMemoryLedger
from switchboard.grounding import GroundingGate
from switchboard.knowledge import BitemporalStore, StatusPageIngestor, parse_timestamp
from switchboard.memory import MemoryStore
from switchboard.providers import AudioChunk, LatencyModel, SimulatedSTT, SimulatedTTS
from switchboard.runtime import AdmissionController, Conversation, LegIO, Runtime

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "delivery_support"
CALL_START = parse_timestamp("2026-10-20T11:30:00-07:00")
TENANT = "dashfleet"

GREETING = "Thanks Sam, you're verified."
LONG_GREETING = (
    "Thanks Sam, you're verified, and I can help with orders, refunds, pay questions "
    "and anything else that comes up on your route today."
)
CUSTOMER_QUESTION = (
    "Hi Ana, this is courier support about your order 48213. Did you ask the courier "
    "to bring it to your home address instead of the office?"
)


class RecordingTTS:
    """Records every text handed to TTS, so a test can prove a sentence never got there."""

    def __init__(self, inner: SimulatedTTS) -> None:
        self.inner = inner
        self.texts: list[str] = []

    def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        self.texts.append(text)
        return self.inner.synthesize(text)


class SimParty:
    """A simulated person on one leg: speaks through simulated STT, hears the agent's audio."""

    def __init__(self, clock: VirtualClock, name: str, seed: int) -> None:
        self.clock = clock
        self.name = name
        self.stt = SimulatedSTT(clock, random.Random(seed), LatencyModel(p50=0.08, p95=0.1))
        self.tts = RecordingTTS(SimulatedTTS(clock, ttfb_s=0.15, chars_per_s=16.0))
        self.chunks: list[AudioChunk] = []
        self.cleared = 0
        self.audio_until = 0.0
        self._changed = asyncio.Event()

    @property
    def io(self) -> LegIO:
        return LegIO(self.stt, self.tts, self)

    def fresh_line(self, seed: int) -> None:
        """New STT stream, as when a recovered conversation reattaches the caller."""
        self.stt = SimulatedSTT(self.clock, random.Random(seed), LatencyModel(p50=0.08, p95=0.1))

    async def send(self, chunk: AudioChunk) -> None:
        self.chunks.append(chunk)
        self.audio_until = max(self.audio_until, chunk.at) + chunk.duration_s
        self._changed.set()

    async def clear(self) -> None:
        self.cleared += 1
        self.audio_until = self.clock.now()

    @property
    def heard(self) -> str:
        return "".join(c.text for c in self.chunks)

    async def hear(self, cue: str) -> None:
        while cue.lower() not in self.heard.lower():
            self._changed.clear()
            await self._changed.wait()

    async def quiet(self, gap: float = 0.4) -> None:
        while (wait := self.audio_until + gap - self.clock.now()) > 0:
            await self.clock.sleep(wait)

    async def say(self, text: str) -> None:
        await self.stt.speak(text)

    async def reply(self, cue: str, text: str, gap: float = 0.4) -> None:
        await self.hear(cue)
        await self.quiet(gap)
        await self.say(text)


PartyScript = Callable[[SimParty], Awaitable[None]]


class SimDialer:
    def __init__(self, clock: VirtualClock, ring_s: float = 2.0) -> None:
        self.clock = clock
        self.ring_s = ring_s
        self.parties: dict[str, SimParty] = {}
        self.scripts: dict[str, PartyScript] = {}
        self.dialed: list[tuple[str, str]] = []
        self.tasks: list[asyncio.Task[None]] = []

    def add(self, party: SimParty, script: PartyScript) -> None:
        self.parties[party.name] = party
        self.scripts[party.name] = script

    async def dial(self, party: str, leg_id: str) -> LegIO:
        self.dialed.append((party, leg_id))
        await self.clock.sleep(self.ring_s)
        sim = self.parties[party]
        self.tasks.append(asyncio.create_task(self.scripts[party](sim)))
        return sim.io


def _tool_names(messages: tuple[Message, ...]) -> dict[str, str]:
    return {
        m.tool_call_id: m.name or "" for m in messages if m.role == "assistant" and m.tool_call_id
    }


@dataclass
class DeliveryBrain:
    """A deterministic stand-in for the model, driven by what the request contains.

    Knobs switch on the misbehaviour each scenario needs: marking delivered too early, saying
    the wrong pay amount, insisting on a denied call, or talking for a long time.
    """

    mark_early: bool = False
    wrong_pay: bool = False
    stubborn: bool = False
    long_greeting: bool = False
    _ids: itertools.count[int] = field(default_factory=lambda: itertools.count(1))

    def call(self, name: str, **args: Any) -> ToolCallChunk:
        return ToolCallChunk(f"call_{next(self._ids)}", name, args)

    def mark(self) -> ToolCallChunk:
        return self.call("mark_delivered", order_id="48213", location="customer's home address")

    def call_customer(self) -> ToolCallChunk:
        return self.call(
            "call_party",
            party="customer",
            goal="Confirm the customer asked for delivery to their home instead of the pin.",
            questions=["Did you ask the courier to bring it to your home address?"],
        )

    def __call__(self, request: BrainRequest) -> Reply:
        if request.messages[0].content.startswith("You are on an outbound phone call"):
            return self.child(request)
        return self.parent(request)

    def parent(self, request: BrainRequest) -> Reply:
        last = request.messages[-1]
        names = _tool_names(request.messages)
        if last.role == "system" and last.content.startswith("Your last sentence failed"):
            return Reply("You've earned $182.40 this week across 23 deliveries.")
        if last.role == "system" and "Result of the call to the customer" in last.content:
            if '"confirmed":true' in last.content:
                return Reply("The customer confirmed the new address.", (self.mark(),))
            if self.stubborn:
                return Reply("Let me try marking it anyway.", (self.mark(),))
            return Reply("The customer didn't confirm the change.")
        if last.role == "tool":
            tool = names.get(last.tool_call_id or "", "")
            if last.content.startswith("Denied by policy"):
                if self.stubborn:
                    return Reply("", (self.mark(),))
                return Reply("I need the customer to confirm first.", (self.call_customer(),))
            if tool == "verify_courier":
                return Reply(LONG_GREETING if self.long_greeting else GREETING)
            if tool == "lookup_order":
                if self.mark_early:
                    return Reply("", (self.mark(),))
                return Reply(
                    "The chat doesn't show an address change, so I'll call the customer.",
                    (self.call_customer(),),
                )
            if tool == "call_party":
                return Reply("I'm calling the customer now, please stay on the line.")
            if tool == "get_courier_pay":
                amount = "$128.40" if self.wrong_pay else "$182.40"
                return Reply(f"You've earned {amount} this week across 23 deliveries.")
            if tool == "mark_delivered":
                return Reply("All set, order 48213 is marked delivered.")
            if tool == "issue_refund":
                return Reply("Done, I refunded $10 to the customer.")
            return Reply("Okay.")
        text = last.content.lower()
        if "c-77" in text and "verify_courier" not in names.values():
            return Reply("One moment.", (self.call("verify_courier", courier_id="C-77"),))
        if "refund" in text:
            return Reply("", (self.call("issue_refund", order_id="48213", amount=10),))
        if "pay" in text:
            return Reply("Let me check.", (self.call("get_courier_pay", courier_id="C-77"),))
        if "48213" in text:
            return Reply(
                "Let me pull up that order.", (self.call("lookup_order", order_id="48213"),)
            )
        return Reply("Got it.")

    def child(self, request: BrainRequest) -> Reply:
        users = [m for m in request.messages if m.role == "user"]
        if not users:
            return Reply(CUSTOMER_QUESTION)
        if "yes" in users[-1].content.lower():
            return Reply(
                "Thanks Ana, I'll let the courier know. Goodbye.",
                (self.call("report_result", confirmed=True, notes="Asked for home delivery."),),
            )
        return Reply(
            "Understood, thanks for your time. Goodbye.",
            (self.call("report_result", confirmed=False, notes="Did not ask for a change."),),
        )


async def customer_confirms(party: SimParty) -> None:
    await party.reply("home address", "Yes, I asked the courier to bring it to my home.", gap=1.0)


async def customer_declines(party: SimParty) -> None:
    await party.reply("home address", "No, it should go to the office.", gap=1.0)


@dataclass
class World:
    clock: VirtualClock
    store: InMemoryEventStore
    ledger: InMemoryLedger
    bundle: Bundle
    brain: DeliveryBrain
    llm: ScriptedLLM
    courier: SimParty
    customer: SimParty
    dialer: SimDialer
    memory: MemoryStore
    facts: BitemporalStore
    admission: AdmissionController
    runtime: Runtime
    gate: GroundingGate = field(default_factory=GroundingGate)

    @property
    def backend(self) -> Any:
        return self.bundle.tools_module.backend

    def new_runtime(
        self,
        *,
        ledger: InMemoryLedger | None = None,
        store: Any = None,
        day_start: float | None = None,
    ) -> Runtime:
        return Runtime(
            clock=self.clock,
            store=store if store is not None else self.store,
            ledger=ledger if ledger is not None else self.ledger,
            llm=self.llm,
            rng=random.Random(99),
            gate=self.gate,
            memory=self.memory,
            facts=self.facts,
            dialer=self.dialer,
            admission=self.admission,
            day_start=day_start,
        )

    async def start(self, runtime: Runtime | None = None) -> Conversation:
        rt = runtime or self.runtime
        return await rt.start(
            self.bundle, tenant_id=TENANT, customer_id="C-77", io=self.courier.io, party="courier"
        )

    async def events(self, conv: Conversation) -> list[AnyEvent]:
        return await self.store.read(conv.id)


def make_world(
    brain: DeliveryBrain | None = None,
    *,
    ttft_s: float = 0.2,
    customer: PartyScript = customer_confirms,
    tenant_caps: dict[str, int] | None = None,
    store: Any = None,
    gate: GroundingGate | None = None,
) -> World:
    clock = VirtualClock(start=CALL_START)
    facts = BitemporalStore()
    status = json.loads((EXAMPLE / "fixtures" / "statuspage.json").read_text())
    StatusPageIngestor(facts, "statuspage").ingest(status)
    brain = brain or DeliveryBrain()
    llm = ScriptedLLM(clock, brain, ttft_s=ttft_s, tokens_per_s=50.0)
    dialer = SimDialer(clock)
    customer_party = SimParty(clock, "customer", seed=2)
    dialer.add(customer_party, customer)
    world = World(
        clock=clock,
        store=store if store is not None else InMemoryEventStore(),
        ledger=InMemoryLedger(clock),
        bundle=load_bundle(EXAMPLE),
        brain=brain,
        llm=llm,
        courier=SimParty(clock, "courier", seed=1),
        customer=customer_party,
        dialer=dialer,
        memory=MemoryStore(),
        facts=facts,
        admission=AdmissionController(tenant_caps),
        runtime=None,  # type: ignore[arg-type]
    )
    world.gate = gate or GroundingGate()
    world.runtime = world.new_runtime()
    return world


async def run_script(world: World, script: Awaitable[None]) -> None:
    """Run a caller script to completion on the virtual clock and surface its errors."""
    task = asyncio.ensure_future(script)
    await world.clock.run_until_idle()
    assert task.done(), "the caller script is stuck waiting for something the agent never said"
    task.result()
    for dial_task in world.dialer.tasks:
        assert dial_task.done(), "a dialed party's script never finished"
        dial_task.result()


def first(events: list[AnyEvent], type_: str, **fields: Any) -> AnyEvent:
    return next(e for e in events if e.type == type_ and _has(e, fields))


def every(events: list[AnyEvent], type_: str, **fields: Any) -> list[AnyEvent]:
    return [e for e in events if e.type == type_ and _has(e, fields)]


def _has(event: AnyEvent, fields: dict[str, Any]) -> bool:
    return all(getattr(event, k, None) == v for k, v in fields.items())


def is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    it = iter(haystack)
    return all(any(item == h for h in it) for item in needle)
