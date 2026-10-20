"""Serve the API over a demo store filled by simulated multi-party calls.

    uv run --extra api python -m benchmarks.serve [--port 8000] [--calls 3] [--token T]

Each call is the courier and customer flow from the delivery example run on the virtual clock,
appended to one in-memory event store that the activity stream, conversation timelines and
traces then read from.

Reads need a bearer token (the API refuses them otherwise). The demo maps one token to the
simulated tenant, fresh per run unless `--token` is given, and prints the activity viewer URL
with the token in its fragment (`/activity#token=...`), which is where the page reads it from.
The fragment never reaches the server or its logs. This is a local demo, not an auth setup.
"""

from __future__ import annotations

import argparse
import asyncio
import secrets

from fastapi import FastAPI

from switchboard.api import Principal, StaticTokenAuthenticator
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


async def prepare(
    host: str, port: int, calls: int = 3, token: str | None = None
) -> tuple[FastAPI, list[str]]:
    """Fill a store, build the app with a demo token for its tenant, and the lines to print."""
    store = InMemoryEventStore()
    ids = await populate(store, calls)
    tenants = {getattr((await store.read(i))[0], "tenant_id", "") for i in ids}
    if len(tenants) != 1:
        raise RuntimeError(f"demo calls span tenants {sorted(tenants)}; one token covers one")
    token = token or secrets.token_urlsafe(24)
    auth = StaticTokenAuthenticator({token: Principal(tenants.pop(), subject="demo")})
    base = f"http://{host}:{port}"
    lines = [f"{base}/v1/conversations/{conv_id}" for conv_id in ids]
    lines.append(f"API reads need the header: Authorization: Bearer {token}")
    lines.append("activity viewer:")
    lines.append(f"{base}/activity#token={token}")
    return create_app(store, authenticator=auth), lines


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.serve")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--token", default=None, help="demo bearer token (default: random)")
    args = parser.parse_args()
    try:
        import uvicorn  # type: ignore[import-not-found, unused-ignore]
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit("uvicorn is missing: run with `uv run --extra api ...`") from exc
    app, lines = asyncio.run(prepare(args.host, args.port, args.calls, args.token))
    for line in lines:
        print(line, flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
