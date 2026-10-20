"""One party's pipeline: STT in, endpointing, speculative turns, the brain loop, speech out.

Input. STT events become turn signals: a VAD start or a new partial counts as a speech frame,
since simulated and streaming recognizers report VAD edges rather than a frame per 20 ms. The
endpointing controller is ticked on the clock while the caller holds the floor.

Turns. `eager_end` opens a speculative turn (ADR-006) whose generation runs through the brain
loop below into the turn engine's private buffer. `resumed` cancels it, `end` commits it. Write
tools proposed while the turn is still speculative get a `speculation:uncommitted_turn` refusal
and wait; once the turn commits they are checked by policy and run once. Agent-initiated turns
(a child call result, an outbound call connecting) run the same loop without the engine.

Backchannels. A burst of caller speech that starts while the agent is speaking and turns out to
be only "mm-hm", "okay" and the like is recorded as `turn.backchannel` and nothing else: no
speculation, no committed turn, and the barge-in policy already let playback continue.

Brain loop, per turn: route, then up to `max_llm_calls` model calls. Text is split into
sentences and each passes the grounding gate before it can reach TTS; one blocked sentence buys
one regeneration with a rewrite hint. Tool calls are proposed, checked, and executed; a denial
feeds the repair hint back as a tool message, and more than `max_policy_retries` denials in a
turn hand off to a human.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from switchboard.brain import (
    BrainRequest,
    Done,
    Message,
    RouteDecision,
    RouteState,
    TextDelta,
    Tier,
    ToolCallChunk,
    ToolSchema,
    VolatileContext,
    assemble_prompt,
)
from switchboard.core import (
    BrainRouted,
    HandoffRequested,
    LegClosed,
    PolicyVerdict,
    SpeechPlayed,
    SpeechSentenceGated,
    ToolExecuted,
    ToolProposed,
    TurnBackchannel,
    TurnCommitted,
    TurnSpeculationCancelled,
    TurnSpeculationStarted,
)
from switchboard.effects import ToolSpec
from switchboard.grounding import GateDecision, rewrite_hint
from switchboard.providers import SttEvent, SttFinal, SttPartial, SttVad
from switchboard.runtime.sentences import SentenceSplitter
from switchboard.runtime.speaker import Speaker, Spoken
from switchboard.runtime.state import SPECULATION_RULE
from switchboard.turn import (
    CancelToken,
    EndpointingConfig,
    EndpointingController,
    FinalTranscript,
    GenerationContext,
    InputSignal,
    PartialTranscript,
    SpeculativeTurnEngine,
    TurnSignal,
    VadFrame,
    is_backchannel,
    normalize_text,
)

if TYPE_CHECKING:
    from switchboard.runtime.conversation import Conversation
    from switchboard.runtime.io import LegIO

TICK_S = 0.02
FALLBACK_SENTENCE = "Let me double check that before I say it."
REGENERATE_NOTE = (
    "Your last sentence failed the fact check and was not spoken. {hint} "
    "Continue the reply using only values from the evidence."
)

LegRole = Literal["inbound", "outbound_child"]
ToolOutcome = Literal["ok", "denied", "stop"]


@dataclass
class TurnContext:
    turn_id: str
    anchor_id: str
    text: str
    user_turn: bool = True
    reason: str = ""
    committed: asyncio.Event = field(default_factory=asyncio.Event)


class Leg:
    def __init__(
        self,
        conversation: Conversation,
        leg_id: str,
        role: LegRole,
        party: str,
        io: LegIO,
        *,
        system_prompt: str,
        policies_text: str,
        tool_specs: Sequence[ToolSpec],
        parent_leg_id: str | None = None,
        eager_threshold: float = 0.5,
    ) -> None:
        self.conversation = conversation
        self.leg_id = leg_id
        self.role = role
        self.party = party
        self.io = io
        self.parent_leg_id = parent_leg_id
        self.system_prompt = system_prompt
        self.policies_text = policies_text
        self.tool_effects: Mapping[str, str] = {s.name: s.effect for s in tool_specs}
        self._schemas = tuple(ToolSchema(s.name, s.description, s.params) for s in tool_specs)
        self._clock = conversation.clock
        self._endpointing = EndpointingController(
            EndpointingConfig(eager_threshold=eager_threshold)
        )
        self._engine: SpeculativeTurnEngine[Spoken, TurnContext] = SpeculativeTurnEngine(
            self._clock, self._generate
        )
        self.speaker = Speaker(
            io,
            self._clock,
            on_played=self._played,
            on_correction=self._correction,
            on_barge_in=self._barge_in,
            on_error=conversation.record_failure,
        )
        self._turn_lock = asyncio.Lock()
        self._signal_lock = asyncio.Lock()
        self._spec: TurnContext | None = None
        self._in_turn = False
        self._in_speech = False
        self._last_speech_at: float | None = None
        # The utterance playing when the current caller burst began, if any.
        self._over_utterance: str | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._agent_turns: set[asyncio.Task[None]] = set()
        # Tier 3 verdicts in flight; cancelled with the leg so none outlives it.
        self._verdicts: set[asyncio.Task[GateDecision]] = set()
        self._ticker: asyncio.Task[None] | None = None
        self._bridges = 0
        self.closed = False

    # lifecycle

    def start(self) -> None:
        self.speaker.start()
        self._spawn(self._input_loop(), "input")
        self._spawn(self._consume(), "consume")

    async def close(self, reason: str) -> None:
        if self.closed:
            return
        self.closed = True
        tasks: list[asyncio.Task[Any]] = [*self._tasks, *self._agent_turns, *self._verdicts]
        if self._ticker is not None:
            tasks.append(self._ticker)
        current = asyncio.current_task()
        for task in tasks:
            if task is not current:
                task.cancel()
        others = [t for t in tasks if t is not current]
        if others:
            await asyncio.wait(others)
        await self._engine.aclose()
        await self.speaker.stop()
        await self.io.stt.close()
        await self.conversation.emit(LegClosed, leg_id=self.leg_id, reason=reason)

    def abandon(self) -> None:
        """Stop every task without writing anything, as a crashed process would."""
        self.closed = True
        tasks: list[asyncio.Task[Any] | None] = [
            *self._tasks,
            *self._agent_turns,
            *self._verdicts,
            self._ticker,
        ]
        for task in tasks:
            if task is not None:
                task.cancel()

    def start_agent_turn(self, anchor_id: str, reason: str) -> None:
        """Run a turn nobody spoke: a child result arrived, or an outbound call connected."""
        ctx = TurnContext(self.conversation.new_id("turn"), anchor_id, "", False, reason)
        ctx.committed.set()
        task = asyncio.create_task(self._agent_turn(ctx), name=f"agent-turn:{ctx.turn_id}")
        self._agent_turns.add(task)
        task.add_done_callback(self._agent_turns.discard)

    async def idle(self) -> None:
        """Wait until no turn is running and all queued speech has played."""
        async with self._turn_lock:
            pass
        await self.speaker.drain()

    def _spawn(self, coro: object, name: str) -> None:
        assert asyncio.iscoroutine(coro)
        task: asyncio.Task[None] = asyncio.create_task(coro, name=f"{name}:{self.leg_id}")
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            exc = task.exception()
            assert exc is not None
            self.conversation.record_failure(exc)

    # input

    async def _input_loop(self) -> None:
        async for event in self.io.stt.events():
            signals, self._in_speech = _to_signals(event, self._in_speech)
            for signal in signals:
                await self._observe(signal)

    async def _observe(self, signal: InputSignal) -> None:
        await self.speaker.observe(signal)
        if isinstance(signal, VadFrame) and signal.speech:
            self._last_speech_at = signal.at
            if not self._in_turn:
                self._in_turn = True
                self._over_utterance = self.speaker.utterance_id if self.speaker.speaking else None
                if self._ticker is None or self._ticker.done():
                    self._ticker = asyncio.create_task(self._tick_loop())
        if isinstance(signal, FinalTranscript) and not self._in_turn:
            return  # a late final for a turn that already ended
        turn_signal = self._endpointing.observe(signal)
        if turn_signal is not None:
            await self._on_turn_signal(turn_signal)

    async def _tick_loop(self) -> None:
        while self._in_turn and not self.closed:
            await self._clock.sleep(TICK_S)
            now = self._clock.now()
            if self._in_speech:
                # VAD reports edges only; between them the caller is speaking on every frame.
                await self._observe(VadFrame(now, True, 1.0))
                continue
            await self.speaker.tick(now)
            turn_signal = self._endpointing.tick(now)
            if turn_signal is not None:
                await self._on_turn_signal(turn_signal)

    async def _on_turn_signal(self, signal: TurnSignal) -> None:
        async with self._signal_lock:
            if signal.kind == "end":
                self._in_turn = False
            if signal.kind == "resumed" or not signal.text.strip():
                await self._cancel_speculation()
                return
            if self._over_utterance is not None and is_backchannel(signal.text):
                if signal.kind == "end":
                    await self.conversation.emit(
                        TurnBackchannel,
                        leg_id=self.leg_id,
                        text=signal.text,
                        utterance_id=self._over_utterance,
                    )
                    self._over_utterance = None
                return
            if signal.kind == "eager_end":
                await self._on_eager(signal)
            else:
                await self._on_end(signal)

    async def _on_eager(self, signal: TurnSignal) -> None:
        spec = self._spec
        if spec is not None and normalize_text(spec.text) == normalize_text(signal.text):
            return
        await self._cancel_speculation()
        turn_id = self.conversation.new_id("turn")
        started = await self.conversation.emit(
            TurnSpeculationStarted, leg_id=self.leg_id, turn_id=turn_id, partial_text=signal.text
        )
        self._spec = ctx = TurnContext(turn_id, started.event_id, signal.text)
        await self._engine.handle(signal, ctx, turn_id=turn_id)

    async def _on_end(self, signal: TurnSignal) -> None:
        spec, self._spec = self._spec, None
        waited_ms = 0.0
        if self._last_speech_at is not None:
            waited_ms = max(0.0, signal.at - self._last_speech_at) * 1000.0
        if spec is not None and normalize_text(spec.text) == normalize_text(signal.text):
            await self.conversation.emit(
                TurnCommitted,
                leg_id=self.leg_id,
                causation_id=spec.anchor_id,
                turn_id=spec.turn_id,
                text=signal.text,
                endpointing_ms=waited_ms,
                requests=self._requests(signal.text),
            )
            spec.text = signal.text
            await self._engine.handle(signal)  # promotes: the token stops being speculative
            spec.committed.set()
            return
        if spec is not None:
            # The final text differs from the speculation: record its cancellation first.
            self._spec = spec
            await self._cancel_speculation()
        turn_id = self.conversation.new_id("turn")
        committed = await self.conversation.emit(
            TurnCommitted,
            leg_id=self.leg_id,
            turn_id=turn_id,
            text=signal.text,
            endpointing_ms=waited_ms,
            requests=self._requests(signal.text),
        )
        ctx = TurnContext(turn_id, committed.event_id, signal.text)
        ctx.committed.set()
        await self._engine.handle(signal, ctx, turn_id=turn_id)

    def _requests(self, text: str) -> list[str]:
        # A child call's party answers the agent's questions; only the inbound caller asks.
        return self.conversation.requested_tools(text) if self.role == "inbound" else []

    async def _cancel_speculation(self) -> None:
        spec, self._spec = self._spec, None
        if spec is None:
            return
        await self.conversation.emit(
            TurnSpeculationCancelled, leg_id=self.leg_id, turn_id=spec.turn_id
        )
        await self._engine.handle(TurnSignal("resumed", self._clock.now(), spec.text, 1.0))

    # output

    async def _consume(self) -> None:
        while True:
            try:
                async for item in self._engine.committed_stream():
                    self.speaker.say(item)
            except Exception as exc:
                self.conversation.record_failure(exc)

    async def _agent_turn(self, ctx: TurnContext) -> None:
        try:
            async for item in self._run_turn(ctx, CancelToken(speculative=False)):
                self.speaker.say(item)
        except Exception as exc:
            self.conversation.record_failure(exc)

    async def _played(
        self,
        utterance_id: str,
        text: str,
        interrupted: bool,
        tts_provider: str | None,
        first_audio_at: float | None,
    ) -> None:
        await self.conversation.emit(
            SpeechPlayed,
            leg_id=self.leg_id,
            utterance_id=utterance_id,
            text_played=text,
            interrupted=interrupted,
            tts_provider=tts_provider,
            first_audio_at=first_audio_at,
        )

    async def _correction(self, utterance_id: str, sentence: str) -> None:
        await self.conversation.emit(
            SpeechSentenceGated,
            leg_id=self.leg_id,
            utterance_id=utterance_id,
            sentence=sentence,
            decision="release",
            tier=1,
            claims=[],
        )

    async def _barge_in(self) -> None:
        """The caller took the floor: drop the rest of whatever the agent was saying.

        A speculation is never the agent's output; any live one started from the caller's
        speech, usually the barge-in itself, so it is left to endpointing to promote or cancel.
        """
        for task in list(self._agent_turns):
            task.cancel()
        await self._engine.cancel_committed()

    # the brain loop

    async def _generate(
        self, text: str, gen: GenerationContext[TurnContext]
    ) -> AsyncIterator[Spoken]:
        async for item in self._run_turn(gen.payload, gen.token):
            yield item

    async def _run_turn(self, ctx: TurnContext, token: CancelToken) -> AsyncIterator[Spoken]:
        conv = self.conversation
        async with self._turn_lock:
            if ctx.user_turn:
                await conv.retrieve(ctx.text, self.leg_id, ctx.anchor_id)
                decision = conv.router.route(ctx.text, RouteState())
            else:
                decision = RouteDecision("thinker", ctx.reason)
            if conv.over_token_budget():
                decision = RouteDecision("talker", "budget")
            await conv.emit(
                BrainRouted,
                leg_id=self.leg_id,
                causation_id=ctx.anchor_id,
                turn_id=ctx.turn_id,
                model_tier=decision.tier,
                reason=decision.reason,
            )
            denials = 0
            for n in range(conv.max_llm_calls):
                utterance_id = f"{ctx.turn_id}/u{n}"
                calls: list[ToolCallChunk] = []
                spoke = False
                async for spoken in self._speak(ctx, decision.tier, utterance_id, calls):
                    spoke = True
                    yield spoken
                if spoke:
                    yield Spoken("", utterance_id, end=True)
                if not calls:
                    return
                outcomes = [await self._tool(ctx, call, token) for call in calls]
                if "stop" in outcomes:
                    return
                if "denied" in outcomes:
                    denials += 1
                    if denials > conv.bundle.config.max_policy_retries:
                        await conv.abort_turn(ctx.turn_id)
                        async for spoken in self._handoff(ctx, f"{denials} policy denials"):
                            yield spoken
                        await self._give_up_child(ctx)
                        return
            await conv.abort_turn(ctx.turn_id)
            async for spoken in self._handoff(ctx, "model call budget exhausted"):
                yield spoken
            await self._give_up_child(ctx)

    async def _give_up_child(self, ctx: TurnContext) -> None:
        # A child call that cannot finish still owes its parent an answer: not confirmed.
        if self.role == "outbound_child":
            await self.conversation.report_child(
                self, False, "The call could not be completed.", ctx.anchor_id
            )

    async def _handoff(self, ctx: TurnContext, reason: str) -> AsyncIterator[Spoken]:
        await self.conversation.emit(
            HandoffRequested, leg_id=self.leg_id, causation_id=ctx.anchor_id, reason=reason
        )
        utterance_id = f"{ctx.turn_id}/handoff"
        spoken, _ = await self._gate(
            ctx, utterance_id, self.conversation.bundle.config.handoff_message
        )
        if spoken is not None:
            yield spoken
            yield Spoken("", utterance_id, end=True)

    def _request(self, tier: Tier, extra: Sequence[Message]) -> BrainRequest:
        conv = self.conversation
        state = conv.state
        volatile = VolatileContext(
            evidence=dict(state.evidence),
            memory=tuple(conv.memory_texts),
            knowledge=tuple(conv.knowledge_texts()),
            open_steps=conv.open_steps() if self.role == "inbound" else (),
            messages=(*state.messages(self.leg_id), *extra),
        )
        prompt = assemble_prompt(self.system_prompt, self.policies_text, self._schemas, volatile)
        return BrainRequest(
            messages=prompt.messages,
            tools=prompt.tools,
            tier=tier,
            metadata={
                "conversation_id": conv.id,
                "leg_id": self.leg_id,
                "party": self.party,
                "prefix_hash": prompt.prefix_hash,
            },
        )

    async def _speak(
        self, ctx: TurnContext, tier: Tier, utterance_id: str, calls: list[ToolCallChunk]
    ) -> AsyncIterator[Spoken]:
        extra: list[Message] = []
        for _attempt in range(2):
            blocked: GateDecision | None = None
            splitter = SentenceSplitter()
            stream = self.conversation.llm(tier).stream(self._request(tier, extra))
            try:
                async for chunk in stream:
                    if isinstance(chunk, ToolCallChunk):
                        calls.append(chunk)
                    elif isinstance(chunk, Done):
                        self.conversation.spend_tokens(chunk.usage)
                    elif isinstance(chunk, TextDelta):
                        for sentence in splitter.feed(chunk.text):
                            spoken, blocked = await self._gate(ctx, utterance_id, sentence)
                            if spoken is not None:
                                yield spoken
                            if blocked is not None:
                                break
                        if blocked is not None:
                            break
                if blocked is None and (rest := splitter.flush()) is not None:
                    spoken, blocked = await self._gate(ctx, utterance_id, rest)
                    if spoken is not None:
                        yield spoken
            finally:
                aclose = getattr(stream, "aclose", None)
                if aclose is not None:
                    await aclose()
            if blocked is None:
                return
            calls.clear()
            hint = rewrite_hint(blocked) or ""
            extra = [Message("system", REGENERATE_NOTE.format(hint=hint).replace("  ", " "))]
        spoken, _ = await self._gate(ctx, utterance_id, FALLBACK_SENTENCE)
        if spoken is not None:
            yield spoken

    async def _gate(
        self, ctx: TurnContext, utterance_id: str, sentence: str
    ) -> tuple[Spoken | None, GateDecision | None]:
        conv = self.conversation
        evidence = conv.grounding_evidence()
        decision = conv.gate.review(sentence, evidence)
        await conv.emit(
            SpeechSentenceGated,
            leg_id=self.leg_id,
            causation_id=ctx.anchor_id,
            utterance_id=utterance_id,
            sentence=sentence,
            decision=decision.decision,
            tier=decision.tier,
            claims=decision.claim_dicts(),
        )
        if decision.decision == "release":
            return Spoken(sentence, utterance_id), None
        if decision.decision == "block":
            return None, decision

        async def settle() -> GateDecision:
            try:
                final = await conv.gate.resolve(decision, sentence, evidence)
            except Exception as exc:
                # Fail closed: an unanswered verification is a block, and the call goes on.
                conv.record_failure(exc)
                final = GateDecision(
                    "block",
                    3,
                    decision.claims,
                    decision.unsupported,
                    f"verifier failed: {type(exc).__name__}",
                    score=decision.score,
                )
            await conv.emit(
                SpeechSentenceGated,
                leg_id=self.leg_id,
                causation_id=ctx.anchor_id,
                utterance_id=utterance_id,
                sentence=sentence,
                decision=final.decision,
                tier=final.tier,
                claims=final.claim_dicts(),
            )
            return final

        verdict = asyncio.create_task(settle(), name=f"verdict:{utterance_id}")
        self._verdicts.add(verdict)
        verdict.add_done_callback(self._verdicts.discard)
        return Spoken(sentence, utterance_id, verdict=verdict), None

    async def _tool(self, ctx: TurnContext, call: ToolCallChunk, token: CancelToken) -> ToolOutcome:
        conv = self.conversation
        proposed = await conv.emit(
            ToolProposed,
            leg_id=self.leg_id,
            causation_id=ctx.anchor_id,
            call_id=call.call_id,
            tool=call.name,
            args=dict(call.args),
        )
        if call.name not in self.tool_effects and call.name not in conv.bundle.compensation_tools:
            await conv.emit(
                ToolExecuted,
                leg_id=self.leg_id,
                causation_id=proposed.event_id,
                call_id=call.call_id,
                tool=call.name,
                result={"error": f"tool {call.name!r} is not available on this call"},
            )
            return "ok"
        if self.tool_effects.get(call.name, "write") == "write" and token.speculative:
            await conv.emit(
                PolicyVerdict,
                leg_id=self.leg_id,
                causation_id=proposed.event_id,
                call_id=call.call_id,
                allowed=False,
                rule_ids=[SPECULATION_RULE],
                repair_hint="Write tools run only once the caller has finished speaking.",
            )
            await ctx.committed.wait()  # cancelled along with the speculation
        verdict = conv.check(call, self.tool_effects)
        if verdict.allowed and self.role == "outbound_child" and call.name == "report_result":
            verdict = conv.check_child_report(self, call) or verdict
        await conv.emit(
            PolicyVerdict,
            leg_id=self.leg_id,
            causation_id=proposed.event_id,
            call_id=call.call_id,
            allowed=verdict.allowed,
            rule_ids=verdict.rule_ids,
            repair_hint=verdict.repair_hint,
        )
        if not verdict.allowed:
            return "denied"
        if call.name == "call_party":
            await conv.call_party(self, call, proposed.event_id)
            return "ok"
        if self.role == "outbound_child" and call.name == "report_result":
            await conv.finish_child(self, call, proposed.event_id)
            return "stop"
        bridge = asyncio.create_task(self._bridge(ctx), name=f"bridge:{call.call_id}")
        try:
            await conv.execute(
                self.leg_id,
                call.call_id,
                call.name,
                dict(call.args),
                intent=conv.state.calls[call.call_id].intent,
                speculative=token.speculative,
                causation_id=proposed.event_id,
            )
        finally:
            bridge.cancel()
        return "ok"

    async def _bridge(self, ctx: TurnContext) -> None:
        """Fill the silence of a slow tool with a fixed phrase from the bundle (ADR-008).

        It speaks once the caller has heard `bridge_after_ms` of silence on a committed turn,
        and only a phrase the gate finds no claim in: bridge speech must never state a fact.
        """
        config = self.conversation.bundle.config
        if not config.bridge_phrases:
            return
        await ctx.committed.wait()
        while True:
            await self.speaker.drain()
            await self._clock.sleep(config.bridge_after_ms / 1000.0)
            if not self.speaker.speaking:
                break
        if self.closed:
            return
        phrase = config.bridge_phrases[self._bridges % len(config.bridge_phrases)]
        decision = self.conversation.gate.review(phrase, self.conversation.grounding_evidence())
        if decision.claims or decision.decision == "block":
            return
        utterance_id = f"{ctx.turn_id}/bridge{self._bridges}"
        self._bridges += 1
        await self.conversation.emit(
            SpeechSentenceGated,
            leg_id=self.leg_id,
            causation_id=ctx.anchor_id,
            utterance_id=utterance_id,
            sentence=phrase,
            decision="release",
            tier=decision.tier,
            claims=[],
        )
        self.speaker.say(Spoken(phrase, utterance_id))
        self.speaker.say(Spoken("", utterance_id, end=True))


def _to_signals(event: SttEvent, in_speech: bool) -> tuple[list[InputSignal], bool]:
    if isinstance(event, SttVad):
        return [VadFrame(event.at, event.speech, 1.0 if event.speech else 0.0)], event.speech
    if isinstance(event, SttPartial):
        frames: list[InputSignal] = [VadFrame(event.at, True, 1.0)] if in_speech else []
        return [*frames, PartialTranscript(event.at, event.text, 0)], in_speech
    assert isinstance(event, SttFinal)
    return [FinalTranscript(event.at, event.text)], in_speech
