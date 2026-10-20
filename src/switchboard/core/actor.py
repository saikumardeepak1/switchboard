"""Keyed actors with one FIFO mailbox and one worker task per key (ADR-002).

Messages for a key are handled strictly one at a time in arrival order; different keys run
concurrently. A failed `send` is recorded in `ActorRuntime.failures` and the key keeps
processing. A failed `ask` raises to the asker instead and is not recorded.

Actors live until `stop(key)` or `stop_all()`. There is no idle eviction yet; callers that
want it stop keys explicitly. A message sent while a key is stopping waits for the stop to
finish and then activates a fresh actor, so a key never has two live actors.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from switchboard.core.clock import Clock

M_contra = TypeVar("M_contra", contravariant=True)


class Actor(Protocol[M_contra]):
    async def handle(self, message: M_contra) -> object: ...


@dataclass(frozen=True)
class ActorFailure:
    key: str
    message: object
    error: Exception


class _Stop:
    pass


_STOP = _Stop()
_Item = tuple[object, "asyncio.Future[object] | None"] | _Stop


@dataclass
class _Mailbox:
    actor: Actor[Any]
    queue: asyncio.Queue[_Item] = field(default_factory=asyncio.Queue)
    task: asyncio.Task[None] | None = None
    stopping: bool = False


class ActorRuntime:
    def __init__(self, factory: Callable[[str], Actor[Any]], clock: Clock) -> None:
        self._factory = factory
        # Held for timeouts and idle eviction built on this runtime; handlers get their own.
        self.clock = clock
        self._boxes: dict[str, _Mailbox] = {}
        self.failures: list[ActorFailure] = []

    def active_keys(self) -> set[str]:
        return set(self._boxes)

    async def send(self, key: str, message: object) -> None:
        box = await self._box(key)
        box.queue.put_nowait((message, None))

    async def ask(self, key: str, message: object) -> object:
        box = await self._box(key)
        reply: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        box.queue.put_nowait((message, reply))
        return await reply

    async def stop(self, key: str) -> None:
        """Process everything already queued for `key`, then retire its actor."""
        box = self._boxes.get(key)
        if box is None:
            return
        if not box.stopping:
            box.stopping = True
            box.queue.put_nowait(_STOP)
        assert box.task is not None
        await asyncio.shield(box.task)

    async def stop_all(self) -> None:
        await asyncio.gather(*(self.stop(key) for key in list(self._boxes)))

    async def _box(self, key: str) -> _Mailbox:
        while (box := self._boxes.get(key)) is not None and box.stopping:
            assert box.task is not None
            await asyncio.shield(box.task)
        if box is None:
            box = _Mailbox(actor=self._factory(key))
            box.task = asyncio.create_task(self._run(key, box), name=f"actor:{key}")
            self._boxes[key] = box
        return box

    async def _run(self, key: str, box: _Mailbox) -> None:
        try:
            while True:
                item = await box.queue.get()
                if isinstance(item, _Stop):
                    return
                message, reply = item
                if reply is not None and reply.done():
                    continue  # asker gave up before we got to it
                try:
                    result = await box.actor.handle(message)
                except Exception as exc:
                    if reply is None:
                        self.failures.append(ActorFailure(key, message, exc))
                    elif not reply.done():
                        reply.set_exception(exc)
                else:
                    if reply is not None and not reply.done():
                        reply.set_result(result)
        finally:
            if self._boxes.get(key) is box:
                del self._boxes[key]
