"""FastAPI app factory: activity stream, conversations, traces, metrics, Twilio media.

Routes:

    GET  /healthz
    GET  /metrics                         Prometheus text exposition
    GET  /activity                        self-contained HTML viewer
    GET  /v1/activity                     filtered, cursor-paginated activity items
    GET  /v1/activity/histogram           per-type counts in fixed time buckets
    GET  /v1/conversations/{id}           summary plus per-turn latency timeline
    GET  /v1/conversations/{id}/trace     OTLP/JSON traces payload
    WS   /v1/twilio/media                 only when a session factory is given

Validation failures are 422 with FastAPI's usual body, `{"detail": [{"loc", "msg", "type"}]}`,
including the ones the activity stream raises itself (bad cursor, unknown type). Pages carry
the next cursor in the body, in `X-Next-Cursor`, and as a `Link: <...>; rel="next"` header.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from switchboard.api.media import MediaStreamHandler, SessionFactory
from switchboard.api.page import ACTIVITY_HTML
from switchboard.core.store import EventStore
from switchboard.telemetry.activity import (
    MAX_LIMIT,
    ActivityFilter,
    ActivityItem,
    ActivityStream,
    InvalidFilter,
    summarize,
)
from switchboard.telemetry.metrics import Registry
from switchboard.telemetry.turns import TurnTimeline, to_otlp_json

PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

TypeQuery = Annotated[
    list[str] | None,
    Query(
        alias="type",
        description="Exact event type or prefix ending in '.', e.g. 'policy.'. Repeatable.",
    ),
]
OptStr = Annotated[str | None, Query()]
OptFloat = Annotated[float | None, Query()]


def _invalid(exc: InvalidFilter) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": [{"loc": ["query", exc.field], "msg": exc.message, "type": "value_error"}]
        },
    )


def create_app(
    store: EventStore,
    *,
    session_factory: SessionFactory | None = None,
    registry: Registry | None = None,
    epoch_offset_s: float | None = None,
    service_name: str = "switchboard",
    capture_content: bool = False,
) -> FastAPI:
    """Build the app.

    `epoch_offset_s` maps clock seconds (event `at`) onto Unix time. With it, activity items get
    an ISO `wall_time` and traces get real timestamps; without it, traces start at the epoch.
    """
    app = FastAPI(title="Switchboard", version="0.1.0")
    stream = ActivityStream(store)
    metrics = registry if registry is not None else Registry()

    def wall_time(at: float) -> str | None:
        if epoch_offset_s is None:
            return None
        return datetime.fromtimestamp(at + epoch_offset_s, tz=UTC).isoformat()

    def item_json(item: ActivityItem) -> dict[str, Any]:
        d = item.to_dict()
        d["wall_time"] = wall_time(item.at)
        return d

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics_text() -> PlainTextResponse:
        return PlainTextResponse(metrics.render(), media_type=PROMETHEUS_CONTENT_TYPE)

    @app.get("/activity", response_class=HTMLResponse)
    async def activity_page() -> HTMLResponse:
        return HTMLResponse(ACTIVITY_HTML)

    @app.get("/v1/activity", response_model=None)
    async def activity(
        request: Request,
        types: TypeQuery = None,
        conversation_id: OptStr = None,
        leg_id: OptStr = None,
        since: OptFloat = None,
        until: OptFloat = None,
        q: Annotated[str | None, Query(description="Case-insensitive text in the payload")] = None,
        limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = 50,
        cursor: OptStr = None,
        order: Literal["asc", "desc"] = "desc",
    ) -> JSONResponse:
        filters = ActivityFilter(
            conversation_id=conversation_id,
            leg_id=leg_id,
            types=tuple(types or ()),
            since=since,
            until=until,
            text=q or None,
            limit=limit,
            cursor=cursor or None,
            order=order,
        )
        try:
            page = await stream.query(filters)
        except InvalidFilter as exc:
            return _invalid(exc)
        headers: dict[str, str] = {}
        if page.next_cursor is not None:
            next_url = request.url.include_query_params(cursor=page.next_cursor)
            headers["Link"] = f'<{next_url}>; rel="next"'
            headers["X-Next-Cursor"] = page.next_cursor
        body = {"items": [item_json(i) for i in page.items], "next_cursor": page.next_cursor}
        return JSONResponse(body, headers=headers)

    @app.get("/v1/activity/histogram", response_model=None)
    async def activity_histogram(
        bucket_s: Annotated[float, Query(gt=0, description="Bucket width in seconds")] = 60.0,
        types: TypeQuery = None,
        conversation_id: OptStr = None,
        leg_id: OptStr = None,
        since: OptFloat = None,
        until: OptFloat = None,
        q: OptStr = None,
    ) -> JSONResponse | dict[str, Any]:
        filters = ActivityFilter(
            conversation_id=conversation_id,
            leg_id=leg_id,
            types=tuple(types or ()),
            since=since,
            until=until,
            text=q or None,
        )
        try:
            buckets = await stream.histogram(filters, bucket_s)
        except InvalidFilter as exc:
            return _invalid(exc)
        return {"bucket_s": bucket_s, "buckets": [b.to_dict() for b in buckets]}

    @app.get("/v1/conversations/{conversation_id}")
    async def conversation(conversation_id: str) -> dict[str, Any]:
        events = await store.read(conversation_id)
        if not events:
            raise HTTPException(404, f"conversation {conversation_id!r} not found")
        timeline = TurnTimeline.from_events(events, capture_content=capture_content)
        return {"summary": summarize(events).to_dict(), "timeline": timeline.to_dict()}

    @app.get("/v1/conversations/{conversation_id}/trace")
    async def conversation_trace(conversation_id: str) -> dict[str, Any]:
        timeline = await stream.timeline(conversation_id, capture_content=capture_content)
        if timeline is None:
            raise HTTPException(404, f"conversation {conversation_id!r} not found")
        return to_otlp_json(
            timeline.spans(), service_name=service_name, epoch_offset_s=epoch_offset_s or 0.0
        )

    if session_factory is not None:
        app.add_api_websocket_route(
            "/v1/twilio/media", MediaStreamHandler(session_factory, metrics)
        )
    else:
        # Register the gauges anyway so dashboards see zeros instead of missing series.
        metrics.gauge("switchboard_twilio_streams_active", "Twilio media streams currently open.")

    return app
