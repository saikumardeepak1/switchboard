"""Serve the API over a demo store filled by simulated multi-party calls.

    uv run --extra api python -m benchmarks.serve [--port 8000] [--calls 3]

Each call is the courier and customer flow from the delivery example run on the virtual clock,
appended to one in-memory event store that the activity stream, conversation timelines and
traces then read from.
"""

from __future__ import annotations

import argparse
import asyncio

from switchboard.api.app import create_app
from switchboard.core import InMemoryEventStore
from switchboard.sim import PERSONAS, build_world, finish
from switchboard.sim.delivery import declined_delivery, happy_delivery


async def populate(store: InMemoryEventStore, calls: int = 3) -> list[str]:
    """Run `calls` simulated conversations into `store`; return their ids."""
    ids: list[str] = []
    personas = list(PERSONAS.values())
    for i in range(calls):
        scenario = declined_delivery() if i % 3 == 2 else happy_delivery()
        world = build_world(scenario, seed=i, persona=personas[i % len(personas)], store=store)
        try:
            conv = await world.start()
            await world.drive()
            await finish(world, conv, 0.0)
            ids.append(conv.id)
        finally:
            world.close()
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.serve")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--calls", type=int, default=3)
    args = parser.parse_args()
    try:
        import uvicorn  # type: ignore[import-not-found, unused-ignore]
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit("uvicorn is missing: run with `uv run --extra api ...`") from exc
    store = InMemoryEventStore()
    ids = asyncio.run(populate(store, args.calls))
    for conv_id in ids:
        print(f"http://{args.host}:{args.port}/v1/conversations/{conv_id}")
    print(f"activity viewer: http://{args.host}:{args.port}/activity")
    uvicorn.run(create_app(store), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
