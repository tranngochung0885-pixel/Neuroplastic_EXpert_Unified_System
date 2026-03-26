"""
nexus/api/server.py
===================
Production FastAPI server for the NEXUS cognitive agent.

Endpoints
---------
GET  /              — root info
GET  /health        — liveness + memory stats (k8s probe compatible)
GET  /status        — full cognitive status dump
GET  /metrics       — Prometheus text/plain exposition
POST /chat          — single-turn request/response
POST /consolidate   — trigger immediate dream consolidation
POST /reflect       — invoke identity self-reflection
GET  /goals         — active goal list
GET  /memory        — memory retrieval (query param: q, k)
WS   /ws            — streaming WebSocket chat

Authentication
--------------
Optional bearer-token gate via NEXUS_API_KEY.  If the env var is empty the
server runs open (suitable for local / dev use).

Streaming
---------
The WebSocket endpoint accepts JSON frames:

    {"text": "…", "stream": true}      → sends partial tokens as they arrive
    {"text": "…", "stream": false}     → single full response frame

REST /chat always returns a complete TurnResult JSON object.

Middleware
----------
- CORS (configurable, open by default)
- Request-ID injection (X-Request-ID header)
- Turn-latency logging
"""

from __future__ import annotations

import asyncio
import dataclasses
import io
import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

# ── Optional deps ─────────────────────────────────────────────────────────────
try:
    from fastapi import (
        BackgroundTasks,
        Depends,
        FastAPI,
        Header,
        HTTPException,
        Query,
        WebSocket,
        WebSocketDisconnect,
        status as http_status,
    )
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
    from pydantic import BaseModel, Field

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

try:
    from prometheus_client import generate_latest

    HAS_PROM = True
except ImportError:
    HAS_PROM = False

from nexus.brain import NexusBrain
from nexus.core.config import NexusSettings, TurnResult, now_ts, utc_iso
from nexus.core.observability import LOG, METRICS

# ══════════════════════════════════════════════════════════════════════════════
# §1  REQUEST / RESPONSE SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

if HAS_FASTAPI:

    class ChatRequest(BaseModel):
        text: str = Field(..., min_length=1, max_length=16_000)
        stream: bool = Field(False, description="Enable token-streaming via SSE body")

    class MemoryQueryParams(BaseModel):
        q: str = Field("recent conversation", description="Retrieval query")
        k: int = Field(8, ge=1, le=40, description="Max results")

    class ChatResponse(BaseModel):
        turn: int
        response: str
        mode: str
        confidence: float
        actual_quality: float
        surprise: float
        free_energy: float
        memory_hits: int
        tool_calls: List[Dict[str, Any]]
        latency_ms: float
        affect: str
        metadata: Dict[str, Any]

    class HealthResponse(BaseModel):
        ok: bool
        version: str
        uptime_s: float
        timestamp: str
        memory: Dict[str, Any]
        gateway: Dict[str, Any]


# ══════════════════════════════════════════════════════════════════════════════
# §2  AUTH HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _make_auth_dependency(api_key: Optional[str]):
    """Return a FastAPI dependency that enforces bearer-token auth (or is a no-op)."""

    async def _no_auth() -> None:
        return None

    if not api_key:
        return _no_auth

    async def _check_auth(authorization: Optional[str] = Header(default=None)) -> None:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=http_status.HTTP_401_UNAUTHORIZED,
                detail="Missing or malformed Authorization header (expected: Bearer <token>)",
            )
        token = authorization.removeprefix("Bearer ").strip()
        if token != api_key:
            raise HTTPException(
                status_code=http_status.HTTP_403_FORBIDDEN,
                detail="Invalid API key.",
            )

    return _check_auth


# ══════════════════════════════════════════════════════════════════════════════
# §3  WEBSOCKET CONNECTION MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class ConnectionManager:
    """Thin manager for active WebSocket connections."""

    def __init__(self) -> None:
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        with __import__("contextlib").suppress(ValueError):
            self.active.remove(ws)

    async def broadcast(self, payload: Dict[str, Any]) -> None:
        dead: List[WebSocket] = []
        for ws in list(self.active):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


# ══════════════════════════════════════════════════════════════════════════════
# §4  APP FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_app(brain: NexusBrain) -> "FastAPI":
    """
    Create and configure the FastAPI application.

    The brain must already be initialised (``await brain.initialize()``).
    """
    if not HAS_FASTAPI:
        raise RuntimeError(
            "FastAPI is not installed. Install with: pip install fastapi uvicorn"
        )

    cfg: NexusSettings = brain.cfg
    _started_at = now_ts()
    ws_manager = ConnectionManager()
    _auth = _make_auth_dependency(cfg.api_key)

    app = FastAPI(
        title="NEXUS Cognitive Agent API",
        version=NexusBrain.VERSION,
        description=(
            "NEXUS — production-grade cognitive agent with layered memory, "
            "free-energy minimization, neurochemical modulation, and "
            "multi-tier reasoning."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    if cfg.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_headers=["*"],
            allow_methods=["*"],
        )

    # ── Request-ID middleware ─────────────────────────────────────────────────
    @app.middleware("http")
    async def _request_id_middleware(request, call_next):
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
        return response

    # ══════════════════════════════════════════════════════════════════════════
    # §4.1  BASIC ROUTES
    # ══════════════════════════════════════════════════════════════════════════

    @app.get("/", tags=["meta"])
    async def root() -> Dict[str, Any]:
        return {
            "name": "NEXUS Cognitive Agent",
            "version": NexusBrain.VERSION,
            "status": "ok",
            "docs": "/docs",
        }

    @app.get("/health", response_model=HealthResponse if HAS_FASTAPI else None, tags=["meta"])
    async def health() -> Dict[str, Any]:
        """
        Liveness probe.  Returns 200 if the brain is responsive.
        Kubernetes-compatible: any non-200 is treated as unhealthy.
        """
        st = brain.status()
        return {
            "ok": True,
            "version": NexusBrain.VERSION,
            "uptime_s": st["uptime_s"],
            "timestamp": utc_iso(),
            "memory": st.get("memory", {}),
            "gateway": st.get("gateway", {}),
        }

    @app.get("/status", tags=["meta"])
    async def status() -> Dict[str, Any]:
        """Full cognitive state dump."""
        return brain.status()

    @app.get("/metrics", tags=["meta"])
    async def metrics() -> PlainTextResponse:
        """Prometheus text/plain exposition format."""
        if not HAS_PROM:
            return PlainTextResponse("# prometheus_client not installed\n")
        return PlainTextResponse(
            generate_latest().decode("utf-8"),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # §4.2  CHAT  (REST)
    # ══════════════════════════════════════════════════════════════════════════

    @app.post("/chat", tags=["cognition"])
    async def chat(
        req: ChatRequest,
        _auth=Depends(_auth),
    ) -> Dict[str, Any]:
        """
        Single-turn cognitive cycle.

        Returns a full TurnResult as JSON, including response text, mode,
        confidence, memory hits, tool calls, latency, and all metadata.
        """
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="text must be non-empty.")

        try:
            result: TurnResult = await brain.think(req.text)
        except Exception as exc:  # noqa: BLE001
            LOG.error("api.chat_error", error=str(exc))
            METRICS.inc_error("api_chat")
            raise HTTPException(status_code=500, detail=f"Cognitive cycle failed: {exc}") from exc

        return dataclasses.asdict(result)

    # ══════════════════════════════════════════════════════════════════════════
    # §4.3  MEMORY RETRIEVAL
    # ══════════════════════════════════════════════════════════════════════════

    @app.get("/memory", tags=["memory"])
    async def memory_retrieve(
        q: str = Query("recent conversation", description="Retrieval query"),
        k: int = Query(8, ge=1, le=40),
        _auth=Depends(_auth),
    ) -> Dict[str, Any]:
        """
        Retrieve episodic / semantic memories relevant to query string.
        Returns scored list of engrams.
        """
        if not brain.lattice:
            raise HTTPException(status_code=503, detail="Memory subsystem not initialised.")

        q_emb = await brain.embedder.embed(q)
        hits = await brain.lattice.retrieve(q, k=k)

        results = []
        for engram, score, tier in hits:
            results.append({
                "uid": engram.uid,
                "tier": tier.value,
                "score": round(score, 4),
                "content": engram.content[:300],
                "importance": round(engram.importance, 4),
                "retrievability": round(engram.retrievability(), 4),
                "age_hours": round(engram.age_hours(), 2),
                "intent": engram.intent,
                "keywords": engram.keywords[:8],
            })

        return {"query": q, "k": k, "hits": len(results), "results": results}

    # ══════════════════════════════════════════════════════════════════════════
    # §4.4  GOALS
    # ══════════════════════════════════════════════════════════════════════════

    @app.get("/goals", tags=["planning"])
    async def goals(_auth=Depends(_auth)) -> Dict[str, Any]:
        """Return active goal list with priority and progress."""
        return {
            "goals": brain.planner.summary(),
            "turn": brain._turn,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # §4.5  CONSOLIDATION
    # ══════════════════════════════════════════════════════════════════════════

    @app.post("/consolidate", tags=["memory"])
    async def consolidate(
        background: BackgroundTasks,
        _auth=Depends(_auth),
    ) -> Dict[str, Any]:
        """
        Trigger an immediate dream-consolidation cycle.
        Runs synchronously in the foreground and returns stats.
        """
        try:
            stats = await brain.consolidate()
        except Exception as exc:  # noqa: BLE001
            LOG.error("api.consolidate_error", error=str(exc))
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {"ok": True, "stats": stats, "timestamp": utc_iso()}

    # ══════════════════════════════════════════════════════════════════════════
    # §4.6  REFLECTION
    # ══════════════════════════════════════════════════════════════════════════

    @app.post("/reflect", tags=["cognition"])
    async def reflect(_auth=Depends(_auth)) -> Dict[str, Any]:
        """
        Invoke the identity self-reflection endpoint.
        Generates a prose self-assessment of recent cognitive performance.
        """
        system = (
            "You are NEXUS reflecting honestly on your own cognitive "
            "performance. Be specific, identify patterns, acknowledge "
            "limitations. First person. No hollow self-praise."
        )
        sm = brain.identity.get_status()
        mc = brain.metacog.get_stats()
        topics = list(brain.cortex.beliefs.topic_history)[-5:]
        prompt = (
            f"Reflect on your recent performance.\n"
            f"Identity status: {json.dumps(sm)}\n"
            f"Metacognition stats: {json.dumps(mc)}\n"
            f"Recent topics: {', '.join(topics)}\n\n"
            "What patterns do you notice? Where are your gaps? Be specific."
        )
        try:
            from nexus.core.config import CognitionMode
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: brain.gateway.complete(
                    system_prompt=system,
                    user_message=prompt,
                    mode=CognitionMode.DEEP,
                    temperature=0.62,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            LOG.error("api.reflect_error", error=str(exc))
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {"reflection": response, "timestamp": utc_iso()}

    # ══════════════════════════════════════════════════════════════════════════
    # §4.7  WEBSOCKET  (streaming + synchronous turns)
    # ══════════════════════════════════════════════════════════════════════════

    @app.websocket("/ws")
    async def ws_chat(ws: WebSocket) -> None:
        """
        Streaming WebSocket chat endpoint.

        Client sends:   {"text": "...", "stream": false}
        Server replies: {"type": "result", ...TurnResult fields...}
                        {"type": "error",  "detail": "..."}

        When stream=true, the server sends incremental token frames:
                        {"type": "token", "token": "..."}  (repeated)
                        {"type": "done",  ...TurnResult fields...}

        Because NexusBrain.think() is a single async call (not a streaming
        generator yet), stream=true currently sends the full response in one
        "token" frame immediately before "done".  Real token streaming requires
        gateway-level SSE support.
        """
        await ws_manager.connect(ws)
        LOG.info("ws.connect", remote=str(ws.client))

        try:
            while True:
                raw = await ws.receive_json()
                text = (raw.get("text") or "").strip()
                do_stream = bool(raw.get("stream", False))

                if not text:
                    await ws.send_json({"type": "error", "detail": "text is required"})
                    continue

                try:
                    result: TurnResult = await brain.think(text)
                except Exception as exc:  # noqa: BLE001
                    LOG.error("ws.think_error", error=str(exc))
                    await ws.send_json({"type": "error", "detail": str(exc)})
                    continue

                payload = dataclasses.asdict(result)

                if do_stream:
                    # Emit incremental token frame, then done
                    await ws.send_json({"type": "token", "token": result.response})
                    await ws.send_json({"type": "done", **payload})
                else:
                    await ws.send_json({"type": "result", **payload})

        except WebSocketDisconnect:
            LOG.info("ws.disconnect", remote=str(ws.client))
        except Exception as exc:  # noqa: BLE001
            LOG.error("ws.unexpected_error", error=str(exc))
        finally:
            ws_manager.disconnect(ws)

    # ══════════════════════════════════════════════════════════════════════════
    # §4.8  LIFECYCLE
    # ══════════════════════════════════════════════════════════════════════════

    @app.on_event("shutdown")
    async def _on_shutdown() -> None:
        LOG.info("api.shutdown")
        await brain.shutdown()

    return app


# ══════════════════════════════════════════════════════════════════════════════
# §5  STANDALONE RUNNER  (used by __main__.py when mode=api)
# ══════════════════════════════════════════════════════════════════════════════

async def run_server(brain: NexusBrain) -> None:
    """
    Initialise the brain and launch the uvicorn server.
    Blocks until SIGTERM / keyboard interrupt.
    """
    try:
        import uvicorn  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn is required to run the API server. "
            "Install with: pip install uvicorn"
        ) from exc

    cfg = brain.cfg
    app = build_app(brain)

    config = uvicorn.Config(
        app,
        host=cfg.api_host,
        port=cfg.api_port,
        log_level=cfg.log_level.lower(),
        access_log=True,
        loop="asyncio",
    )
    server = uvicorn.Server(config)

    LOG.info(
        "api.server_start",
        host=cfg.api_host,
        port=cfg.api_port,
        docs=f"http://{cfg.api_host}:{cfg.api_port}/docs",
    )

    try:
        await server.serve()
    finally:
        await brain.shutdown()
