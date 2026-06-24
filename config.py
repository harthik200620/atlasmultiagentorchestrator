"""
config.py — one central place for all of Atlas's configuration.

Anything that depends on the environment (which LLM to use, API keys, behavior
knobs) is read HERE, once. The rest of the code imports from this file and never
touches os.environ directly.

    import config
    llm = config.get_llm()          # a ready-to-use chat model (with key rotation)

Command-line helpers:
    python config.py                # show config + which keys are present
    python config.py check-keys     # make ONE real Gemini + Tavily call to test
"""

from __future__ import annotations

import itertools
import os
import re
import sys
import threading

from dotenv import load_dotenv

# Load variables from a local .env file into the environment.
# Real OS environment variables take precedence (handy in deployment).
load_dotenv()


# ----------------------------------------------------------------------
# LLM selection  (swap providers by editing .env, not code)
# ----------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))


# ----------------------------------------------------------------------
# API keys & service endpoints
# ----------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")


# ----------------------------------------------------------------------
# Atlas behavior knobs
# ----------------------------------------------------------------------
# Hard cap on supervisor loops — the safety net against infinite agent loops.
MAX_ITERATIONS = int(os.getenv("ATLAS_MAX_ITERATIONS", "6"))
# How many results the Searcher asks Tavily for per query.
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))


# ----------------------------------------------------------------------
# Gemini API key pool  (rotation across many free-tier keys)
# ----------------------------------------------------------------------
def gemini_keys() -> list[str]:
    """Collect every Gemini/Google API key from the environment.

    Picks up GOOGLE_API_KEY plus any numbered pool keys such as
    GEMINI_API_KEY_4 or GOOGLE_API_KEY_2 (any number works, any order).
    Atlas round-robins across all of them and fails over to the next when one
    hits its free-tier rate limit. Use keys from DIFFERENT Google projects —
    keys in the same project share one quota.
    """
    found: list[tuple[int, str]] = []
    if GOOGLE_API_KEY:
        found.append((0, GOOGLE_API_KEY))  # the un-numbered key sorts first
    for name, value in os.environ.items():
        if not value:
            continue
        match = re.fullmatch(r"(?:GEMINI|GOOGLE)_API_KEY_(\d+)", name)
        if match:
            found.append((int(match.group(1)), value))
    found.sort(key=lambda pair: pair[0])

    # De-duplicate while preserving order (in case the same key appears twice).
    seen: set[str] = set()
    keys: list[str] = []
    for _, key in found:
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def _is_rate_limit(err: Exception) -> bool:
    """True if the error means 'this key is temporarily tapped out' — an HTTP
    429 quota error, or a transient 503 — as opposed to a real bug (bad key,
    bad request) that we should NOT hide behind a rotation."""
    text = f"{type(err).__name__}: {err}".lower()
    signals = (
        "429", "resource_exhausted", "resourceexhausted", "rate limit",
        "rate_limit", "quota", "too many requests", "503", "overloaded",
        "service unavailable", "unavailable",
    )
    return any(s in text for s in signals)


# We subclass LangChain's Runnable so this object behaves like a real chat model
# everywhere in Atlas — agents can .invoke() it, pipe it in a chain with `|`,
# and (Phase 4) pass Langfuse callbacks through `config`, all transparently.
from langchain_core.runnables import Runnable  # noqa: E402  (after stdlib block)


class RotatingGemini(Runnable):
    """A chat model that spreads calls across a POOL of Gemini API keys.

    WHY: free-tier Gemini keys have low per-minute / per-day limits. Holding
    several keys (ideally from different Google projects) and rotating multiplies
    throughput and keeps Atlas running when any single key is throttled — which
    matters a lot when the Phase 5 eval fires off dozens of calls.

    HOW:
      * round-robin  - each .invoke() starts on the NEXT key in the pool;
      * failover     - if a call raises a 429 / quota error, it retries on the
                       following keys until one succeeds or all are exhausted;
      * transparent  - forwards .invoke / .ainvoke and proxies .bind_tools /
                       .with_structured_output, so agents treat it like any LLM.
    """

    def __init__(self, pool: list):
        if not pool:
            raise ValueError(
                "No Gemini API keys found. Add GOOGLE_API_KEY or "
                "GEMINI_API_KEY_<n> to your .env file."
            )
        self._pool = pool
        self._n = len(pool)
        self._cycle = itertools.cycle(range(self._n))
        self._lock = threading.Lock()  # round-robin must be thread-safe

    @classmethod
    def from_keys(cls, keys: list[str], model: str, temperature: float):
        """Build one ChatGoogleGenerativeAI client per key (no network yet)."""
        from langchain_google_genai import ChatGoogleGenerativeAI

        clients = [
            ChatGoogleGenerativeAI(
                model=model, temperature=temperature, google_api_key=key
            )
            for key in keys
        ]
        return cls(clients)

    def _start_index(self) -> int:
        with self._lock:
            return next(self._cycle)

    def invoke(self, input, config=None, **kwargs):  # noqa: A002 (LangChain name)
        start = self._start_index()
        last_err: Exception | None = None
        for offset in range(self._n):
            idx = (start + offset) % self._n
            try:
                return self._pool[idx].invoke(input, config=config, **kwargs)
            except Exception as err:  # noqa: BLE001
                if _is_rate_limit(err):
                    last_err = err
                    continue  # this key is throttled — try the next one
                raise         # a real error — surface it, don't mask it
        raise RuntimeError(
            f"All {self._n} Gemini keys are rate-limited right now. Add more "
            "keys (from different Google projects) or wait a minute."
        ) from last_err

    async def ainvoke(self, input, config=None, **kwargs):  # noqa: A002
        start = self._start_index()
        last_err: Exception | None = None
        for offset in range(self._n):
            idx = (start + offset) % self._n
            try:
                return await self._pool[idx].ainvoke(input, config=config, **kwargs)
            except Exception as err:  # noqa: BLE001
                if _is_rate_limit(err):
                    last_err = err
                    continue
                raise
        raise RuntimeError(
            f"All {self._n} Gemini keys are rate-limited right now."
        ) from last_err

    # Proxy the two builder methods agents may use in later phases, applying the
    # transform to every key in the pool so rotation still works afterwards.
    def bind_tools(self, *args, **kwargs):
        return RotatingGemini([r.bind_tools(*args, **kwargs) for r in self._pool])

    def with_structured_output(self, *args, **kwargs):
        return RotatingGemini(
            [r.with_structured_output(*args, **kwargs) for r in self._pool]
        )

    def __len__(self):
        return self._n


# ----------------------------------------------------------------------
# LLM factory  — every agent gets its model from here
# ----------------------------------------------------------------------
def get_llm(model: str | None = None, temperature: float | None = None):
    """Return the chat model Atlas should use, based on LLM_PROVIDER.

    * google    -> a RotatingGemini pool (auto key-rotation + failover)
    * openai    -> standard ChatOpenAI
    * anthropic -> standard ChatAnthropic

    Provider SDKs are imported lazily so you only need the one you use.
    """
    model = model or LLM_MODEL
    temperature = LLM_TEMPERATURE if temperature is None else temperature

    if LLM_PROVIDER == "google":
        return RotatingGemini.from_keys(gemini_keys(), model, temperature)

    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature, api_key=OPENAI_API_KEY)

    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model, temperature=temperature, api_key=ANTHROPIC_API_KEY
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}. Use: google | openai | anthropic."
    )


# ----------------------------------------------------------------------
# Self-checks
# ----------------------------------------------------------------------
def _status(value: str | None) -> str:
    """Report whether a secret is present WITHOUT printing the secret itself."""
    return f"set ({len(value)} chars)" if value else "MISSING"


def check() -> None:
    pool = gemini_keys()
    print("Atlas configuration")
    print("-" * 46)
    print(f"LLM_PROVIDER      : {LLM_PROVIDER}")
    print(f"LLM_MODEL         : {LLM_MODEL}")
    print(f"LLM_TEMPERATURE   : {LLM_TEMPERATURE}")
    print(f"MAX_ITERATIONS    : {MAX_ITERATIONS}")
    print(f"TAVILY_MAX_RESULTS: {TAVILY_MAX_RESULTS}")
    print("-" * 46)
    print(f"Gemini key pool   : {len(pool)} key(s) for rotation")
    print(f"TAVILY_API_KEY    : {_status(TAVILY_API_KEY)}")
    print(f"OPENAI_API_KEY    : {_status(OPENAI_API_KEY)}")
    print(f"ANTHROPIC_API_KEY : {_status(ANTHROPIC_API_KEY)}")
    print(f"LANGFUSE_PUBLIC   : {_status(LANGFUSE_PUBLIC_KEY)}")
    print(f"LANGFUSE_SECRET   : {_status(LANGFUSE_SECRET_KEY)}")
    print("-" * 46)

    ready = True
    if LLM_PROVIDER == "google" and not pool:
        print("!  No Gemini keys. Add GEMINI_API_KEY_<n> to .env before Phase 1.")
        ready = False
    if not TAVILY_API_KEY:
        print("!  TAVILY_API_KEY missing. The Searcher (Phase 1) needs it.")
        ready = False
    if ready:
        print("OK  Core keys present - you're ready for Phase 1.")
        if LLM_PROVIDER == "google" and len(pool) > 1:
            print(f"    Gemini calls round-robin across {len(pool)} keys, with failover.")
    print("    (Tip: run `python config.py check-keys` to test the keys live.)")


def check_keys() -> None:
    """Make ONE real call to Gemini and Tavily to confirm the keys are live."""
    print("Live key check (uses a tiny bit of quota)")
    print("-" * 46)
    try:
        resp = get_llm().invoke("Reply with exactly one word: pong")
        text = getattr(resp, "content", resp)
        print(f"OK  Gemini replied: {str(text).strip()[:50]!r}")
    except Exception as err:  # noqa: BLE001
        print(f"X   Gemini call FAILED: {type(err).__name__}: {str(err)[:200]}")

    try:
        from tavily import TavilyClient

        results = TavilyClient(api_key=TAVILY_API_KEY).search(
            "what is langgraph", max_results=1
        )
        print(f"OK  Tavily replied: {len(results.get('results', []))} result(s)")
    except Exception as err:  # noqa: BLE001
        print(f"X   Tavily call FAILED: {type(err).__name__}: {str(err)[:200]}")


if __name__ == "__main__":
    check()
    if len(sys.argv) > 1 and sys.argv[1] in ("check-keys", "keys", "llm"):
        print()
        check_keys()
