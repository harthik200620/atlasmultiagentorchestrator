"""
tracing.py - optional Langfuse tracing for Atlas (targets langfuse 4.x).

When the LANGFUSE_* keys are set in .env, each run is recorded as one nested
trace tree. If the keys are absent, tracing silently no-ops.
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
    client = langfuse_client()
    if client is None:
        print("Langfuse tracing OFF (no LANGFUSE_PUBLIC_KEY/SECRET_KEY in .env).")
    else:
        ok = client.auth_check()
        print(f"Langfuse host : {config.LANGFUSE_HOST}")
        print(f"auth_check()  : {ok}", "(keys valid - tracing will work)" if ok else "(keys REJECTED - check them)")
