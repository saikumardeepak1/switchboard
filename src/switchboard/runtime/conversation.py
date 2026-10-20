"""The conversation actor and the runtime that starts, recovers and ends conversations.

`Conversation` is the single writer for one conversation (ADR-002). Every leg, parent or child,
appends through `emit`, which goes through the conversation's mailbox in the core `ActorRuntime`;
the actor appends to the store with the expected sequence number and folds the stored events into
`state`. Nothing else mutates `state`, which is why `replay(log)` reproduces it.

The log is the system of record, so it also holds the inputs the grounding gate reads that do
not come from a tool: memory read at setup (`memory.loaded`, ADR-011) and the text of retrieved
knowledge (`knowledge.retrieved`, ADR-010). `replay(log)` therefore rebuilds the exact evidence
behind every gate decision. `Runtime(capture_content=False)` keeps that text out of the log; the
live call still uses it, but replay can no longer reproduce those decisions. Telemetry export
has its own content flag, off by default (ADR-015).
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
    Usage,
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
    MemoryLoaded,
    ProcedureAdvanced,
    SpeechPlayed,
    ToolCompensated,
    ToolExecuted,
    ToolUnresolved,
    TurnCommitted,
    new_id,
)
from switchboard.effects import (
    EffectContext,
    EffectRunner,
    IdempotencyLedger,
    Saga,
    ToolCall,
    ToolResult,
    ToolSpec,
    idempotency_key,
)
from switchboard.grounding import Evidence, GroundingGate
from switchboard.knowledge import BitemporalStore, Document, FactStore, KnowledgeIndex
from switchboard.memory import MemoryStore, consolidate
from switchboard.policy import (
    CheckContext,
    ProcedureState,
    ProposedCall,
    RequestDetector,
    Verdict,
)
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

# Rule id of a child's confirmation refused because the party never answered.
NO_PARTY_ANSWER = "runtime:no_party_answer"

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
        # Knowledge and memory text held only here when content capture is off.
        self._unlogged = Evidence()
        self._index = _knowledge_index(bundle, runtime.facts)
        self._tasks: set[asyncio.Task[Any]] = set()
        # Tool executions in flight, kept apart so `end` waits for them instead of cancelling.
        self._executions: dict[asyncio.Task[Any], tuple[str | None, str, str, str | None]] = {}

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

    def spend_tokens(self, usage: Usage) -> None:
        tokens = usage.input_tokens + usage.output_tokens
        self.runtime.admission.record_tokens(self.state.tenant_id, tokens, now=self.clock.now())

    def over_token_budget(self) -> bool:
        """The tenant spent its tokens-per-minute budget: route to the talker (ADR-013)."""
        return self.runtime.admission.over_budget(
            self.state.tenant_id,
            now=self.clock.now(),
            default_budget=self.bundle.config.tenant_limits.tokens_per_minute,
        )

    def requested_tools(self, text: str) -> list[str]:
        """Write tools a committed user turn newly asks for (ADR-009)."""
        writes = [name for name, effect in self.bundle.tool_effects().items() if effect == "write"]
        return self.runtime.request_detector.requested(text, writes)

    def knowledge_texts(self) -> list[str]:
        return list(self._knowledge.values())

    def open_steps(self) -> tuple[str, ...]:
        graph = self.bundle.policy.procedure
        return graph.open_steps(ProcedureState(self.state.procedure_done(graph)))

    def grounding_evidence(self) -> Evidence:
        if self.runtime.capture_content:
            # Kept current as events are appended, and exactly what replay rebuilds.
            return self.state.grounding_evidence()
        # The log lacks the text, so merge in what this process holds. Costs the index size.
        return self.state.grounding_evidence().union(self._unlogged)

    async def load_memory(self) -> None:
        """Read memory once at setup and record what was read (ADR-011)."""
        memory = self.runtime.memory
        if memory is None:
            return
        items = memory.read(self.state.tenant_id, self.state.customer_id, self.clock.now())
        self.memory_texts = [str(i.content.get("text") or canonical_json(i.content)) for i in items]
        if not self.runtime.capture_content:
            self._unlogged.add(self.memory_texts)
        await self.emit(
            MemoryLoaded,
            leg_id=None,
            count=len(self.memory_texts),
            texts=self.memory_texts if self.runtime.capture_content else [],
        )

    def party_answer(self, child_id: str) -> str | None:
        """Event id of the latest turn the party committed on a child leg after the agent
        first spoke on its current dial, or None if the party has said nothing since."""
        spoke = False
        answer: str | None = None
        for event in self.events:
            if event.leg_id != child_id:
                continue
            if isinstance(event, LegOpened):
                spoke, answer = False, None
            elif isinstance(event, SpeechPlayed) and event.text_played.strip():
                spoke = True
            elif isinstance(event, TurnCommitted) and spoke:
                answer = event.event_id
        return answer

    def check_child_report(self, child: Leg, call: ToolCallChunk) -> Verdict | None:
        """Refuse a confirmation the other party never gave."""
        if call.args.get("confirmed") is not True or self.party_answer(child.leg_id):
            return None
        return Verdict(
            False,
            [NO_PARTY_ANSWER],
            f"The {child.party} has not answered yet. Ask your question and wait for their "
            "reply, or report confirmed false.",
        )

    def check(self, call: ToolCallChunk, tool_effects: Mapping[str, str]) -> Verdict:
        ctx = CheckContext(
            evidence=self.state.evidence,
            state={"completed_steps": list(self.state.completed_steps)},
            tool_effects=tool_effects,
            index=self.state.index,
            completed_steps=self.state.procedure_done(self.bundle.policy.procedure),
        )
        return self.bundle.policy.check(ProposedCall(call.name, call.args, call.call_id), ctx)

    async def retrieve(self, text: str, leg_id: str, causation_id: str) -> None:
        now = self.clock.now()
        results = self._index.retrieve(text, as_of=now, k=self.runtime.knowledge_k)
        if not results:
            return
        for result in results:
            if not self.runtime.capture_content and result.source_id not in self._knowledge:
                self._unlogged.add(result.text)
            self._knowledge[result.source_id] = result.text
        await self.emit(
            KnowledgeRetrieved,
            leg_id=leg_id,
            causation_id=causation_id,
            fact_ids=[r.source_id for r in results],
            as_of=now,
            texts=[r.text for r in results] if self.runtime.capture_content else [],
        )

    async def execute(
        self,
        leg_id: str | None,
        call_id: str,
        tool: str,
        args: dict[str, Any],
        *,
        intent: int,
        speculative: bool,
        causation_id: str | None,
    ) -> dict[str, Any]:
        """Run a registry tool and record it. Shielded: once started, a write is recorded even
        if the turn that asked for it is cancelled (barge-in, hang-up)."""

        async def run() -> dict[str, Any]:
            result = await self.runner.run(
                ToolCall(call_id, tool, args), EffectContext(self.id, intent, speculative)
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

        task = self._spawn(run())
        self._tasks.discard(task)
        spec = self.bundle.registry.get(tool)
        key = None
        if spec is not None and spec.spec.effect == "write":
            key = idempotency_key(self.id, intent, tool, args, spec.spec.idempotency)
        self._executions[task] = (leg_id, call_id, tool, key)
        task.add_done_callback(lambda t: self._executions.pop(t, None))
        return await asyncio.shield(task)

    async def rollback(self, call_ids: Sequence[str] | None = None) -> None:
        """Compensate executed writes, newest first, with the compensations their specs declare.

        `call_ids` limits the rollback to those calls. Every compensation goes through the
        ledger under a key derived from the original, and a call already compensated in the log
        is skipped, so a second rollback runs nothing twice (ADR-009).
        """
        saga = Saga(self.runner)
        done = set(self.state.compensations)
        chosen = None if call_ids is None else set(call_ids)
        calls = [
            c
            for c in self.state.calls.values()
            if c.executed
            and not c.replayed
            and c.call_id not in done
            and (chosen is None or c.call_id in chosen)
            and self.runner.compensation_for(c.tool) is not None
        ]
        for call in sorted(calls, key=lambda c: c.executed_seq or 0):
            saga.record(
                ToolResult(
                    call.call_id,
                    call.tool,
                    dict(call.args),
                    ok=tool_succeeded(call.result or {}),
                    result=call.result,
                    idempotency_key=call.idempotency_key,
                )
            )
        report = await saga.abort(EffectContext(self.id, 0))
        legs = {c.call_id: c.leg_id for c in calls}
        for outcome in report.outcomes:
            undo = outcome.result
            if outcome.compensation_tool is None or undo is None:
                continue
            payload = undo.result if undo.ok and undo.result is not None else None
            await self.emit(
                ToolCompensated,
                leg_id=legs.get(outcome.call_id),
                call_id=outcome.call_id,
                compensation_tool=outcome.compensation_tool,
                result=payload if payload is not None else {"error": undo.error or "failed"},
            )

    async def abort_turn(self, turn_id: str) -> None:
        """The runtime gave up on a turn: undo its writes if the bundle asks for that."""
        if "turn_aborted" not in self.bundle.config.rollback_on:
            return
        turn = self.state.turns.get(turn_id)
        if turn is not None and turn.call_ids:
            await self.rollback(list(turn.call_ids))

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
            await self._dial_child(child_id, party, goal, questions, parent, causation_id)
            result = {"status": "calling", "party": party, "child_leg_id": child_id}
        await self.emit(
            ToolExecuted,
            leg_id=parent.leg_id,
            causation_id=causation_id,
            call_id=call.call_id,
            tool=call.name,
            result=result,
        )

    async def _dial_child(
        self,
        child_id: str,
        party: str,
        goal: str,
        questions: Sequence[str],
        parent: Leg,
        causation_id: str | None,
    ) -> None:
        """Record the child leg as opened and start dialing it in the background.

        A redial after recovery appends a second `leg.opened` for the same leg id; the fold
        counts it, which is what stops a third dial.
        """
        config = self.bundle.config.parties[party]
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

    async def recover_children(self) -> None:
        """Settle child legs that were open when the process died (ADR-003).

        A child whose result never arrived is redialed once, under the same leg id (the
        dialer's idempotency key), with the goal and questions of the `call_party` that opened
        it. A child already redialed once, or whose party is gone from the bundle, is closed as
        lost; one whose result did arrive is closed as completed.
        """
        parents = {leg_id: leg for leg_id, leg in self.legs.items() if not leg.closed}
        for leg in list(self.state.legs.values()):
            if not leg.open or leg.role != "outbound_child":
                continue
            parent = parents.get(leg.parent_leg_id or "")
            if leg.leg_id in self.state.child_results:
                await self.emit(LegClosed, leg_id=leg.leg_id, reason="completed")
            elif (
                leg.dials >= 2
                or parent is None
                or self.runtime.dialer is None
                or leg.party not in self.bundle.config.parties
            ):
                await self.emit(LegClosed, leg_id=leg.leg_id, reason="lost_in_recovery")
            else:
                await self._dial_child(
                    leg.leg_id,
                    leg.party,
                    leg.goal or "",
                    self._child_questions(leg.leg_id),
                    parent,
                    None,
                )

    def _child_questions(self, child_id: str) -> list[str]:
        for call in self.state.calls.values():
            if call.tool == "call_party" and (call.result or {}).get("child_leg_id") == child_id:
                return [str(q) for q in call.args.get("questions", []) or []]
        return []

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
        confirmed = call.args.get("confirmed") is True
        executed = await self.emit(
            ToolExecuted,
            leg_id=child.leg_id,
            causation_id=causation_id,
            call_id=call.call_id,
            tool=call.name,
            result={"recorded": True, "confirmed": confirmed},
        )
        await self.report_child(
            child, confirmed, str(call.args.get("notes", "")), executed.event_id
        )

    async def report_child(
        self, child: Leg, confirmed: bool, notes: str, causation_id: str | None
    ) -> None:
        """Deliver a child call's answer to its parent and hang the child up.

        The confirmation flag is set only with provenance: the child leg and the party's turn
        that answered. Without an answer on the leg the flag is not set at all.
        """
        assert child.parent_leg_id is not None
        config = self.bundle.config.parties.get(child.party)
        answer = self.party_answer(child.leg_id)
        evidence: dict[str, Any] = {}
        provenance: dict[str, Any] = {}
        if config and config.confirms and answer is not None:
            evidence = {config.confirms: confirmed}
            provenance = {config.confirms: {"child_leg_id": child.leg_id, "turn_event_id": answer}}
        result = {
            "party": child.party,
            "confirmed": confirmed and answer is not None,
            "notes": notes,
            "evidence": evidence,
            "provenance": provenance,
        }
        delivered = await self.emit(
            ChildResult,
            leg_id=child.parent_leg_id,
            causation_id=causation_id,
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

        The idempotency key derives from the call's intent, which is in the log, so a write that
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
                intent=call.intent,
                speculative=False,
                causation_id=None,
            )

    async def end(self, outcome: str) -> None:
        if self.state.ended:
            return
        for leg in sorted(self.legs.values(), key=lambda leg: leg.role != "outbound_child"):
            if not leg.closed:
                await leg.close("conversation_ended")
        await self._settle_executions(outcome)
        for task in list(self._tasks):
            task.cancel()
        if outcome in self.bundle.config.rollback_on:
            await self.rollback()
        await self.emit(ConversationEnded, leg_id=None, outcome=outcome)
        if self.runtime.memory is not None:
            self.runtime.memory.apply(consolidate(self.events, self.clock.now()))
        self.runtime.admission.release(self.state.tenant_id, self.id)
        await self.runtime.actors.stop(self.id)
        self.runtime.conversations.pop(self.id, None)

    async def _settle_executions(self, outcome: str) -> None:
        """Let in-flight tool calls finish, up to `end_write_timeout_s` on the clock.

        Cancelling a write mid-call would leave its effect applied downstream with nothing in
        the log. One still running at the bound is cancelled without touching its ledger claim,
        and recorded as `tool.unresolved`.
        """
        if not self._executions:
            return
        timer = asyncio.ensure_future(self.clock.sleep(self.runtime.end_write_timeout_s))
        try:
            while self._executions and not timer.done():
                await asyncio.wait({*self._executions, timer}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            timer.cancel()
        for task, (leg_id, call_id, tool, key) in list(self._executions.items()):
            task.cancel()
            await asyncio.wait([task])
            await self.emit(
                ToolUnresolved,
                leg_id=leg_id,
                call_id=call_id,
                tool=tool,
                idempotency_key=key,
                reason=f"conversation ended ({outcome})",
            )

    async def abandon(self) -> None:
        """Simulate the process dying: stop every task and write nothing more."""
        for leg in self.legs.values():
            leg.abandon()
        for task in [*self._tasks, *self._executions]:
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
        capture_content: bool = True,
        request_detector: RequestDetector | None = None,
        end_write_timeout_s: float = 10.0,
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
        # Record knowledge and memory text in the log, the system of record (see module doc).
        self.capture_content = capture_content
        self.request_detector = request_detector or RequestDetector()
        # How long ending a conversation waits for tool calls already running.
        self.end_write_timeout_s = end_write_timeout_s
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
        await conv.emit(
            ConversationStarted, leg_id=None, tenant_id=tenant_id, customer_id=customer_id
        )
        await conv.load_memory()
        leg_id = new_id("leg", self.rng)
        await conv.emit(LegOpened, leg_id=leg_id, role="inbound", party=party)
        conv.attach_leg(_inbound_leg(conv, leg_id, party, io))
        return conv

    async def recover(self, bundle: Bundle, conversation_id: str, *, io: LegIO) -> Conversation:
        """Rebuild a conversation from its log after a crash and reattach the caller.

        Pending allowed calls are re-run through the ledger. A child call whose result never
        arrived is redialed once (see `Conversation.recover_children`).
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
        if state.memory_captured:
            conv.memory_texts = list(state.memory_texts)
        else:
            await conv.load_memory()
        conv._knowledge.update(state.knowledge)
        await conv.resume_pending()
        conv.attach_leg(_inbound_leg(conv, inbound.leg_id, inbound.party, io))
        await conv.recover_children()
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
