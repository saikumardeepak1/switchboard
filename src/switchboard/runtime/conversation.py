"""The conversation actor and the runtime that starts, recovers and ends conversations.

`Conversation` is the single writer for one conversation (ADR-002). Every leg, parent or child,
appends through `emit`, which goes through the conversation's mailbox in the core `ActorRuntime`;
the actor appends to the store with the expected sequence number and folds the stored events into
`state`. Nothing else mutates `state`, which is why `replay(log)` reproduces it.

Things the log does not hold live beside it: memory read at setup (ADR-011), the text of retrieved
knowledge (the log records fact and document ids with `as_of`, ADR-010), and running tasks.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from switchboard.brain import (
    DEFAULT_ROUTER_CONFIG,
    LLM,
    HeuristicRouter,
    Router,
    RouterConfig,
    RouteRule,
    Tier,
    ToolCallChunk,
    canonical_json,
)
from switchboard.bundle import Bundle
from switchboard.core import (
    ActorRuntime,
    AnyEvent,
    ChildResult,
    Clock,
    ConversationEnded,
    ConversationStarted,
    Event,
    EventStore,
    KnowledgeRetrieved,
    LegClosed,
    LegOpened,
    ProcedureAdvanced,
    ToolExecuted,
    new_id,
)
from switchboard.effects import (
    EffectContext,
    EffectRunner,
    IdempotencyLedger,
    ToolCall,
    ToolSpec,
)
from switchboard.grounding import Evidence, GroundingGate
from switchboard.knowledge import BitemporalStore, Document, FactStore, KnowledgeIndex
from switchboard.memory import MemoryStore, consolidate
from switchboard.policy import CheckContext, ProcedureState, ProposedCall, Verdict
from switchboard.policy.procedure import tool_succeeded
from switchboard.runtime.admission import AdmissionController, AdmissionRejected
from switchboard.runtime.io import Dialer, LegIO
from switchboard.runtime.leg import Leg
from switchboard.runtime.state import ConversationState, apply_event, replay

E = TypeVar("E", bound=Event)

# Offered only to child legs: how a sub-agent call hands its answer back to the parent.
REPORT_RESULT = ToolSpec(
    name="report_result",
    description="Report the outcome of this call. Ends the call after your goodbye.",
    effect="read",
    params={
        "type": "object",
        "properties": {
            "confirmed": {"type": "boolean", "description": "Did the party confirm?"},
            "notes": {"type": "string", "description": "One short sentence of detail."},
        },
        "required": ["confirmed"],
    },
)

CHILD_PROMPT = (
    "You are on an outbound phone call to the {party} on behalf of {agent}. {description}\n"
    "Goal: {goal}\n{questions}"
    "Keep it short and polite. Once you have the answer, call report_result with confirmed set "
    "to true or false and a short note, and say goodbye in the same reply."
)


class RecoveryError(RuntimeError):
    pass


@dataclass(frozen=True)
class _Append:
    events: tuple[Event, ...]


class Conversation:
    def __init__(
        self,
        runtime: Runtime,
        bundle: Bundle,
        conversation_id: str,
        *,
        state: ConversationState | None = None,
        events: Sequence[AnyEvent] = (),
    ) -> None:
        self.runtime = runtime
        self.bundle = bundle
        self.id = conversation_id
        self.clock = runtime.clock
        self.gate = runtime.gate
        self.max_llm_calls = runtime.max_llm_calls
        self.state = state or ConversationState()
        self.events: list[AnyEvent] = list(events)
        self.legs: dict[str, Leg] = {}
        self.memory_texts: list[str] = []
        self.failures: list[BaseException] = []
        self.router: Router = runtime.router or _bundle_router(bundle)
        self.runner = EffectRunner(bundle.registry, runtime.ledger, runtime.clock)
        self._knowledge: dict[str, str] = {}
        self._index = _knowledge_index(bundle, runtime.facts)
        self._tasks: set[asyncio.Task[Any]] = set()

    # the actor

    async def handle(self, message: object) -> object:
        if not isinstance(message, _Append):
            raise TypeError(f"conversation actor got {type(message).__name__}")
        events = cast(list[AnyEvent], list(message.events))
        stored = await self.runtime.store.append(self.id, events, self.state.last_seq)
        for event in stored:
            apply_event(self.state, event)
            self.events.append(event)
        return stored

    def new_id(self, prefix: str) -> str:
        return new_id(prefix, self.runtime.rng)

    async def emit(
        self,
        cls: type[E],
        *,
        leg_id: str | None,
        causation_id: str | None = None,
        **payload: Any,
    ) -> E:
        """Append one event through the actor and return it as stored, with seq and id."""
        event = cls(
            conversation_id=self.id,
            leg_id=leg_id,
            at=self.clock.now(),
            bundle_id=self.bundle.bundle_id,
            causation_id=causation_id,
            **payload,
        )
        stored = await self.runtime.actors.ask(self.id, _Append((event,)))
        assert isinstance(stored, list)
        out = stored[0]
        assert isinstance(out, cls)
        return out

    def record_failure(self, exc: BaseException) -> None:
        self.failures.append(exc)

    def _spawn(self, coro: Any) -> asyncio.Task[Any]:
        task: asyncio.Task[Any] = asyncio.create_task(coro)
        self._tasks.add(task)

        def done(t: asyncio.Task[Any]) -> None:
            self._tasks.discard(t)
            if not t.cancelled() and (exc := t.exception()) is not None:
                self.record_failure(exc)

        task.add_done_callback(done)
        return task

    # services the legs call

    def llm(self, tier: Tier) -> LLM:
        if tier == "talker" and self.runtime.talker is not None:
            return self.runtime.talker
        return self.runtime.llm

    def knowledge_texts(self) -> list[str]:
        return list(self._knowledge.values())

    def open_steps(self) -> tuple[str, ...]:
        graph = self.bundle.policy.procedure
        return graph.open_steps(ProcedureState(frozenset(self.state.completed_steps)))

    def grounding_evidence(self) -> Evidence:
        return Evidence(
            tool_results=[*self.state.tool_results, *self.state.child_results.values()],
            facts=self.knowledge_texts(),
            memory=self.memory_texts,
        )

    def check(self, call: ToolCallChunk, tool_effects: Mapping[str, str]) -> Verdict:
        ctx = CheckContext(
            evidence=dict(self.state.evidence),
            state={"completed_steps": list(self.state.completed_steps)},
            events=tuple(self.events),
            tool_effects=tool_effects,
        )
        return self.bundle.policy.check(ProposedCall(call.name, call.args, call.call_id), ctx)

    async def retrieve(self, text: str, leg_id: str, causation_id: str) -> None:
        now = self.clock.now()
        results = self._index.retrieve(text, as_of=now, k=self.runtime.knowledge_k)
        if not results:
            return
        for result in results:
            self._knowledge[result.source_id] = result.text
        await self.emit(
            KnowledgeRetrieved,
            leg_id=leg_id,
            causation_id=causation_id,
            fact_ids=[r.source_id for r in results],
            as_of=now,
        )

    async def execute(
        self,
        leg_id: str | None,
        call_id: str,
        tool: str,
        args: dict[str, Any],
        *,
        step_seq: int,
        speculative: bool,
        causation_id: str | None,
    ) -> dict[str, Any]:
        """Run a registry tool and record it. Shielded: once started, a write is recorded even
        if the turn that asked for it is cancelled (barge-in, hang-up)."""

        async def run() -> dict[str, Any]:
            result = await self.runner.run(
                ToolCall(call_id, tool, args), EffectContext(self.id, step_seq, speculative)
            )
            payload = result.result if result.ok and result.result is not None else None
            payload = payload if payload is not None else {"error": result.error or "failed"}
            executed = await self.emit(
                ToolExecuted,
                leg_id=leg_id,
                causation_id=causation_id,
                call_id=call_id,
                tool=tool,
                result=payload,
                idempotency_key=result.idempotency_key,
                replayed=result.replayed,
            )
            step = self.bundle.policy.procedure.completes.get(tool)
            if step and tool_succeeded(payload) and step not in self.state.completed_steps:
                await self.emit(
                    ProcedureAdvanced,
                    leg_id=leg_id,
                    causation_id=executed.event_id,
                    step=step,
                    state={"completed": [*self.state.completed_steps, step]},
                )
            return payload

        return await asyncio.shield(self._spawn(run()))

    # multi-party (ADR-003)

    async def call_party(self, parent: Leg, call: ToolCallChunk, causation_id: str) -> None:
        """Open a child leg and return at once; its answer arrives later as `child.result`."""
        party = str(call.args.get("party", ""))
        goal = str(call.args.get("goal", ""))
        questions = [str(q) for q in call.args.get("questions", []) or []]
        config = self.bundle.config.parties.get(party)
        result: dict[str, Any]
        if config is None:
            result = {
                "error": f"no party named {party!r}; known: {sorted(self.bundle.config.parties)}"
            }
        elif self.runtime.dialer is None:
            result = {"error": "outbound calling is not configured"}
        else:
            child_id = self.new_id("leg")
            opened = await self.emit(
                LegOpened,
                leg_id=child_id,
                causation_id=causation_id,
                role="outbound_child",
                party=party,
                parent_leg_id=parent.leg_id,
                goal=goal,
            )
            prompt = CHILD_PROMPT.format(
                party=party,
                agent=self.bundle.config.name,
                description=config.description,
                goal=goal,
                questions="".join(f"Ask: {q}\n" for q in questions),
            )
            self._spawn(self._run_child(child_id, party, prompt, parent, opened.event_id))
            result = {"status": "calling", "party": party, "child_leg_id": child_id}
        await self.emit(
            ToolExecuted,
            leg_id=parent.leg_id,
            causation_id=causation_id,
            call_id=call.call_id,
            tool=call.name,
            result=result,
        )

    async def _run_child(
        self, child_id: str, party: str, prompt: str, parent: Leg, opened_id: str
    ) -> None:
        dialer = self.runtime.dialer
        assert dialer is not None
        try:
            io = await dialer.dial(party, child_id)
        except Exception as exc:
            failed = await self.emit(
                ChildResult,
                leg_id=parent.leg_id,
                causation_id=opened_id,
                child_leg_id=child_id,
                result={"party": party, "confirmed": False, "notes": f"call failed: {exc}"},
            )
            await self.emit(LegClosed, leg_id=child_id, reason="dial_failed")
            parent.start_agent_turn(failed.event_id, "child_result")
            return
        child = Leg(
            self,
            child_id,
            "outbound_child",
            party,
            io,
            system_prompt=prompt,
            policies_text="",
            tool_specs=[REPORT_RESULT],
            parent_leg_id=parent.leg_id,
            eager_threshold=self.bundle.config.eager_threshold,
        )
        self.legs[child_id] = child
        child.start()
        child.start_agent_turn(opened_id, "outbound_call")

    async def finish_child(self, child: Leg, call: ToolCallChunk, causation_id: str) -> None:
        assert child.parent_leg_id is not None
        confirmed = call.args.get("confirmed") is True
        notes = str(call.args.get("notes", ""))
        executed = await self.emit(
            ToolExecuted,
            leg_id=child.leg_id,
            causation_id=causation_id,
            call_id=call.call_id,
            tool=call.name,
            result={"recorded": True, "confirmed": confirmed},
        )
        config = self.bundle.config.parties.get(child.party)
        evidence = {config.confirms: confirmed} if config and config.confirms else {}
        result = {
            "party": child.party,
            "confirmed": confirmed,
            "notes": notes,
            "evidence": evidence,
        }
        delivered = await self.emit(
            ChildResult,
            leg_id=child.parent_leg_id,
            causation_id=executed.event_id,
            child_leg_id=child.leg_id,
            result=result,
        )
        parent = self.legs.get(child.parent_leg_id)
        if parent is not None and not parent.closed:
            parent.start_agent_turn(delivered.event_id, "child_result")
        self._spawn(self._hang_up(child))

    async def _hang_up(self, child: Leg) -> None:
        # The goodbye was yielded before report_result ran, so it is already queued to speak.
        await child.idle()
        await child.close("completed")

    # lifecycle

    def attach_leg(self, leg: Leg) -> None:
        self.legs[leg.leg_id] = leg
        leg.start()

    async def resume_pending(self) -> None:
        """Re-run calls the policy allowed but whose result never reached the log.

        The idempotency key derives from the proposal's seq, which is in the log, so a write that
        did land before the crash comes back from the ledger instead of running again.
        """
        for call in self.state.pending_calls():
            if self.bundle.registry.get(call.tool) is None:
                continue
            await self.execute(
                call.leg_id,
                call.call_id,
                call.tool,
                dict(call.args),
                step_seq=call.proposed_seq,
                speculative=False,
                causation_id=None,
            )

    async def end(self, outcome: str) -> None:
        if self.state.ended:
            return
        for leg in sorted(self.legs.values(), key=lambda leg: leg.role != "outbound_child"):
            if not leg.closed:
                await leg.close("conversation_ended")
        for task in list(self._tasks):
            task.cancel()
        await self.emit(ConversationEnded, leg_id=None, outcome=outcome)
        if self.runtime.memory is not None:
            self.runtime.memory.apply(consolidate(self.events, self.clock.now()))
        self.runtime.admission.release(self.state.tenant_id, self.id)
        await self.runtime.actors.stop(self.id)
        self.runtime.conversations.pop(self.id, None)

    async def abandon(self) -> None:
        """Simulate the process dying: stop every task and write nothing more."""
        for leg in self.legs.values():
            leg.abandon()
        for task in list(self._tasks):
            task.cancel()
        self.runtime.admission.release(self.state.tenant_id, self.id)
        self.runtime.conversations.pop(self.id, None)
        await asyncio.sleep(0)

    async def wait_idle(self) -> None:
        for leg in list(self.legs.values()):
            if not leg.closed:
                await leg.idle()


class Runtime:
    def __init__(
        self,
        *,
        clock: Clock,
        store: EventStore,
        ledger: IdempotencyLedger,
        llm: LLM,
        rng: random.Random,
        gate: GroundingGate | None = None,
        talker: LLM | None = None,
        router: Router | None = None,
        memory: MemoryStore | None = None,
        facts: FactStore | None = None,
        dialer: Dialer | None = None,
        admission: AdmissionController | None = None,
        knowledge_k: int = 3,
        max_llm_calls: int = 6,
    ) -> None:
        self.clock = clock
        self.store = store
        self.ledger = ledger
        self.llm = llm
        self.talker = talker
        self.rng = rng
        self.gate = gate or GroundingGate()
        self.router = router
        self.memory = memory
        self.facts = facts
        self.dialer = dialer
        self.admission = admission or AdmissionController()
        self.knowledge_k = knowledge_k
        self.max_llm_calls = max_llm_calls
        self.conversations: dict[str, Conversation] = {}
        self.actors = ActorRuntime(self._actor, clock)

    def _actor(self, key: str) -> Conversation:
        return self.conversations[key]

    async def start(
        self,
        bundle: Bundle,
        *,
        tenant_id: str,
        customer_id: str,
        io: LegIO,
        party: str = "caller",
    ) -> Conversation:
        """Admit, read memory, open the inbound leg and start listening. Raises
        `AdmissionRejected` when the tenant is at its cap."""
        conversation_id = new_id("conv", self.rng)
        cap = bundle.config.tenant_limits.max_concurrent_conversations
        decision = self.admission.admit(tenant_id, conversation_id, cap)
        if not decision.admitted:
            raise AdmissionRejected(decision)
        conv = Conversation(self, bundle, conversation_id)
        self.conversations[conversation_id] = conv
        if self.memory is not None:
            items = self.memory.read(tenant_id, customer_id, self.clock.now())
            conv.memory_texts = [
                str(i.content.get("text") or canonical_json(i.content)) for i in items
            ]
        await conv.emit(
            ConversationStarted, leg_id=None, tenant_id=tenant_id, customer_id=customer_id
        )
        leg_id = new_id("leg", self.rng)
        await conv.emit(LegOpened, leg_id=leg_id, role="inbound", party=party)
        conv.attach_leg(_inbound_leg(conv, leg_id, party, io))
        return conv

    async def recover(self, bundle: Bundle, conversation_id: str, *, io: LegIO) -> Conversation:
        """Rebuild a conversation from its log after a crash and reattach the caller.

        Pending allowed calls are re-run through the ledger. Child legs that were open are
        closed: their media is gone with the process.
        """
        events = await self.store.read(conversation_id)
        if not events:
            raise RecoveryError(f"no events for conversation {conversation_id!r}")
        state = replay(events)
        if state.bundle_id != bundle.bundle_id:
            raise RecoveryError(
                f"conversation {conversation_id!r} is pinned to {state.bundle_id}, "
                f"not {bundle.bundle_id}"
            )
        if state.ended:
            raise RecoveryError(f"conversation {conversation_id!r} has already ended")
        inbound = state.inbound_leg()
        if inbound is None or not inbound.open:
            raise RecoveryError(f"conversation {conversation_id!r} has no open inbound leg")
        cap = bundle.config.tenant_limits.max_concurrent_conversations
        decision = self.admission.admit(state.tenant_id, conversation_id, cap)
        if not decision.admitted:
            raise AdmissionRejected(decision)
        conv = Conversation(self, bundle, conversation_id, state=state, events=events)
        self.conversations[conversation_id] = conv
        if self.memory is not None:
            items = self.memory.read(state.tenant_id, state.customer_id, self.clock.now())
            conv.memory_texts = [
                str(i.content.get("text") or canonical_json(i.content)) for i in items
            ]
        for leg in list(state.legs.values()):
            if leg.open and leg.role == "outbound_child":
                await conv.emit(LegClosed, leg_id=leg.leg_id, reason="lost_in_recovery")
        await conv.resume_pending()
        conv.attach_leg(_inbound_leg(conv, inbound.leg_id, inbound.party, io))
        return conv


def _inbound_leg(conv: Conversation, leg_id: str, party: str, io: LegIO) -> Leg:
    bundle = conv.bundle
    return Leg(
        conv,
        leg_id,
        "inbound",
        party,
        io,
        system_prompt=bundle.config.system_prompt,
        policies_text=bundle.policies_text,
        tool_specs=bundle.tool_specs(),
        eager_threshold=bundle.config.eager_threshold,
    )


def _bundle_router(bundle: Bundle) -> HeuristicRouter:
    rules = tuple(
        RouteRule(
            name=r.name,
            tier=r.tier,
            patterns=r.patterns,
            state_flag=r.state_flag,
            min_words=r.min_words,
            max_words=r.max_words,
            min_questions=r.min_questions,
        )
        for r in bundle.config.router_rules
    )
    return HeuristicRouter(
        RouterConfig(
            rules=rules + DEFAULT_ROUTER_CONFIG.rules,
            default_tier=bundle.config.router_default_tier,
            question_joiners=DEFAULT_ROUTER_CONFIG.question_joiners,
        )
    )


def _knowledge_index(bundle: Bundle, facts: FactStore | None) -> KnowledgeIndex:
    index = KnowledgeIndex(facts if facts is not None else BitemporalStore())
    for doc in bundle.documents:
        index.add_document(Document(doc.doc_id, doc.title, doc.body))
    return index
