"""
server.py — FastAPI backend that exposes Atlas over HTTP for the web UI.

It reuses the EXACT same graph as the CLI (graph.build_graph()); this is just a
thin HTTP + streaming layer on top of the validated agent pipeline.

Endpoints:
  GET /            -> basic info
  GET /health      -> liveness check (Render pings this)
  GET /research?goal=...  -> Server-Sent Events (SSE): one 'step' event per agent
                            log line as it happens, then a 'done' event with the
                            final cited brief + stats (or an 'error' event).

Run locally:
  uvicorn server:app --reload --port 8000
Deploy (Render): start command  uvicorn server:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import config
import tracing
from graph import RECURSION_LIMIT, build_graph
from state import new_state

app = FastAPI(title="Atlas API")

# Let the Vercel frontend (and local dev) call us. In production, set
# ALLOWED_ORIGINS to your Vercel URL instead of "*".
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Build the graph once at startup (cheap; no network).
_graph = build_graph()


@app.get("/")
def root():
    return {"service": "atlas", "docs": "/docs", "research": "/research?goal=..."}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": f"{config.LLM_PROVIDER}:{config.LLM_MODEL}",
        "gemini_keys": len(config.gemini_keys()),
        "tracing": tracing.enabled(),
    }


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _run_stream(goal: str):
    """SSE generator.

    The agent graph runs in a dedicated WORKER THREAD so Langfuse's
    OpenTelemetry span context lives entirely on one thread. (Starlette would
    otherwise drive this generator across different threadpool threads, which
    breaks the span's thread-local enter/exit.) The worker pushes log lines onto
    a queue; we read them off and yield them as Server-Sent Events.
    """
    events: queue.Queue = queue.Queue()
    DONE = object()
    result: dict = {}

    def work():
        try:
            cfg = {"recursion_limit": RECURSION_LIMIT}
            handler = tracing.callback_handler()
            if handler is not None:
                cfg["callbacks"] = [handler]
            client = tracing.langfuse_client()

            state = new_state(goal)
            printed = 0
            trace_cm = (
                client.start_as_current_observation(name="atlas-research", as_type="chain", input=goal)
                if client
                else contextlib.nullcontext()
            )
            with trace_cm:
                for snapshot in _graph.stream(state, cfg, stream_mode="values"):
                    state = snapshot
                    log = snapshot.get("log") or []
                    for line in log[printed:]:
                        events.put(("step", {"message": line}))
                    printed = len(log)
                if client:
                    try:
                        result["trace_id"] = client.get_current_trace_id()
                    except Exception:
                        pass
            if client:
                client.flush()
            result["state"] = state
        except Exception as err:  # noqa: BLE001
            result["error"] = f"{type(err).__name__}: {err}"
        finally:
            events.put(DONE)

    threading.Thread(target=work, daemon=True).start()

    # Stream log lines as the worker produces them.
    while True:
        item = events.get()
        if item is DONE:
            break
        name, data = item
        yield _sse(name, data)

    # Worker finished — emit the final result (or the error).
    if "error" in result:
        yield _sse("error", {"message": result["error"]})
        return

    state = result.get("state", {})
    trace_url = None
    client = tracing.langfuse_client()
    tid = result.get("trace_id")
    if client and tid:
        trace_url = client.get_trace_url(trace_id=tid)

    yield _sse(
        "done",
        {
            "draft": state.get("draft", ""),
            "sources": len(state.get("search_results") or []),
            "evidence_count": len(state.get("evidence") or []),
            "revisions": state.get("revisions", 0),
            "steps": state.get("iterations", 0),
            "status": state.get("status"),
            "trace_url": trace_url,
            "evidence": state.get("evidence") or [],
        },
    )


@app.get("/research")
def research(goal: str = ""):
    goal = (goal or "").strip()
    if not goal:
        return {"error": "Provide a ?goal= query parameter."}
    return StreamingResponse(
        _run_stream(goal),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # don't let proxies buffer the stream
        },
    )
