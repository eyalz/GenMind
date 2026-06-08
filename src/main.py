from __future__ import annotations

import asyncio
import json
import os
from urllib.parse import urlparse
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from src.admin.routes import router as admin_router
from src.db import close_db, init_db
from src.mcp.routes import router as mcp_router


class SSEHub:
    """In-memory fan-out event hub for MCP-compatible SSE streams."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    async def publish(self, event: dict[str, Any]) -> None:
        dead_queues: list[asyncio.Queue[dict[str, Any]]] = []
        for queue in self._subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                dead_queues.append(queue)

        for queue in dead_queues:
            self._subscribers.discard(queue)

    async def subscribe(self) -> AsyncGenerator[str, None]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        self._subscribers.add(queue)

        try:
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield self._format_sse_event(payload.get("event", "message"), payload)
                except asyncio.TimeoutError:
                    heartbeat = {
                        "event": "heartbeat",
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                    yield self._format_sse_event("heartbeat", heartbeat)
        finally:
            self._subscribers.discard(queue)

    @staticmethod
    def _format_sse_event(event_name: str, data: dict[str, Any]) -> str:
        return f"event: {event_name}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    app.state.sse_hub = SSEHub()
    yield
    await close_db()


app = FastAPI(
    title="GenMind MCP Server",
    version="0.1.0",
    description=(
        "Context-as-a-Service MCP server with multi-tenant isolation, "
        "AUDN ingestion, and hybrid memory retrieval."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _allowed_mcp_origins() -> set[str]:
    configured = os.getenv("GENMIND_ALLOWED_MCP_ORIGINS", "").strip()
    if configured:
        return {origin.strip().lower() for origin in configured.split(",") if origin.strip()}
    return {
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    }


def _is_allowed_origin(origin: str, allowed_origins: set[str]) -> bool:
    parsed = urlparse(origin.strip())
    if parsed.scheme not in {"http", "https"}:
        return False
    if not parsed.netloc:
        return False
    normalized = f"{parsed.scheme}://{parsed.netloc}".lower()
    return normalized in allowed_origins


def _accepts_required_stream_types(accept_header: str) -> bool:
    media_types = {
        item.split(";", 1)[0].strip().lower()
        for item in accept_header.split(",")
        if item.strip()
    }
    required = {"application/json", "text/event-stream"}
    return required.issubset(media_types)


@app.middleware("http")
async def enforce_mcp_transport_security(request: Request, call_next):
    if request.method == "POST" and request.url.path == "/mcp/stream":
        origin = request.headers.get("origin", "").strip()
        if not origin or not _is_allowed_origin(origin, _allowed_mcp_origins()):
            return JSONResponse(
                status_code=403,
                content={"detail": "Forbidden: invalid Origin header."},
            )

        accept = request.headers.get("accept", "").strip()
        if not _accepts_required_stream_types(accept):
            return JSONResponse(
                status_code=406,
                content={
                    "detail": "Not Acceptable: Accept must include application/json and text/event-stream.",
                },
            )

    return await call_next(request)

app.include_router(mcp_router)
app.include_router(admin_router)


@app.get("/healthz")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/mcp/events")
async def mcp_events() -> StreamingResponse:
    generator = app.state.sse_hub.subscribe()
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse(
        content={
            "name": "GenMind MCP Server",
            "version": "0.1.0",
            "docs": "/docs",
            "mcp": {
                "stream": "/mcp/stream",
                "initialization": "/mcp/initialization",
                "resources": "/mcp/resources",
                "tools": "/mcp/tools",
                "events": "/mcp/events",
            },
        }
    )
