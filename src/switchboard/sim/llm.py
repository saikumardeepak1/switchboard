"""Model latency for simulations: time to first token drawn from a distribution per call."""

from __future__ import annotations

import contextlib
import random
from collections.abc import AsyncIterator

from switchboard.brain import LLM, BrainChunk, BrainRequest
from switchboard.core import Clock
from switchboard.providers import LatencyModel


class SampledLatencyLLM:
    """Waits a sampled time to first token, then streams the inner model.

    Pair it with a `ScriptedLLM` whose own `ttft_s` is 0, so the scripted model keeps its
    deterministic token pacing and this wrapper owns the provider-shaped latency tail.
    """

    def __init__(self, inner: LLM, clock: Clock, ttft: LatencyModel, rng: random.Random) -> None:
        self.inner = inner
        self.clock = clock
        self.ttft = ttft
        self.rng = rng
        self.samples: list[float] = []

    async def stream(self, request: BrainRequest) -> AsyncIterator[BrainChunk]:
        wait = self.ttft.sample(self.rng)
        self.samples.append(wait)
        await self.clock.sleep(wait)
        async with contextlib.aclosing(self.inner.stream(request)) as chunks:  # type: ignore[type-var]
            async for chunk in chunks:
                yield chunk
