"""
tracing.py — optional Langfuse tracing for Atlas.

When the LANGFUSE_* keys are set in .env, every Atlas run is recorded as ONE
trace tree in Langfuse: the supervisor, each worker agent, every LLM call (with
tokens + latency), and the revise-loop — all nested and clickable. If the keys
are absent, tracing silently no-ops so the app still runs fine without Langfuse.

This targets langfuse 4.x (OpenTelemetry-based):
  - Langfuse(...)                       -> the client (also becomes the default)
  - langfuse.langchain.CallbackHandler  -> bridges LangChain/LangGraph events
  - start_as_current_observation(...)   -> wraps a run so it's ONE named trace
"""

from __future__ import annotations

import config

_client = None
_initialized = False


def langfuse_client():
    """Return a configured Langfuse client, or None if the keys are missing."""
    global _client, _initialized
    if not _initialized:
        _initialized = True
        if config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY:
            from langfuse import Langfuse

            _client = Langfuse(
                public_key=config.LANGFUSE_PUBLIC_KEY,
                secret_key=config.LANGFUSE_SECRET_KEY,
                host=config.LANGFUSE_HOST,
            )
    return _client


def callback_handler():
    """A LangChain/LangGraph CallbackHandler if tracing is on, else None."""
    if langfuse_client() is None:
        return None
    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


def flush() -> None:
    """Send any buffered events to Langfuse (call once at the end of a run)."""
    client = langfuse_client()
    if client is not None:
        client.flush()


def enabled() -> bool:
    return langfuse_client() is not None


if __name__ == "__main__":
    # Verify the keys actually work:  python tracing.py
    client = langfuse_client()
    if client is None:
        print("Langfuse tracing OFF (no LANGFUSE_PUBLIC_KEY/SECRET_KEY in .env).")
    else:
        ok = client.auth_check()
        print(f"Langfuse host : {config.LANGFUSE_HOST}")
        print(f"auth_check()  : {ok}", "(keys valid - tracing will work)" if ok else "(keys REJECTED - check them)")
