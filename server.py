"""FastAPI backend that exposes the Atlas agent graph over HTTP for the web UI."""

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

# Lock this down to your frontend's origin in production via ALLOWED_ORIGINS.
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)

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
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _run_stream(goal: str):
    # The graph runs in a dedicated worker thread so Langfuse's OpenTelemetry span
    # context stays on a single thread -- Starlette would otherwise drive this
    # generator across threadpool threads and break the span's enter/exit. The
    # worker pushes log lines onto a queue that we drain out as SSE events.
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

    while True:
        item = events.get()
        if item is DONE:
            break
        name, data = item
        yield _sse(name, data)

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
            "X-Accel-Buffering": "no",
        },
    )
