"""
config.py - central configuration for Atlas.

All environment-dependent settings (LLM choice, API keys, behavior knobs) are
read here once; the rest of the code imports from this module.
"""

from __future__ import annotations

import itertools
import os
import re
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

# Load atlas/.env next to this file so launch directory doesn't matter. Real OS
# env vars still take precedence.
load_dotenv(Path(__file__).with_name(".env"))


# ----------------------------------------------------------------------
# LLM selection
# ----------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# Gemini 2.5 flash defaults to a "thinking" pass that adds large per-call
# latency; 0 disables it for this many-call pipeline. Ignored on non-2.5/pro.
GEMINI_THINKING_BUDGET = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))


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
# Behavior knobs
# ----------------------------------------------------------------------
# Max Critic revise-rounds before Atlas accepts the current draft.
MAX_ITERATIONS = int(os.getenv("ATLAS_MAX_ITERATIONS", "2"))
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))


# ----------------------------------------------------------------------
# Gemini API key pool
# ----------------------------------------------------------------------
def gemini_keys() -> list[str]:
    """Collect every Gemini/Google API key from the environment.

    Picks up GOOGLE_API_KEY plus any numbered pool keys such as
    GEMINI_API_KEY_4 or GOOGLE_API_KEY_2. Keys should come from different Google
    projects, since keys in the same project share one quota.
    """
    found: list[tuple[int, str]] = []
    if GOOGLE_API_KEY:
        found.append((0, GOOGLE_API_KEY))
    for name, value in os.environ.items():
        if not value:
            continue
        match = re.fullmatch(r"(?:GEMINI|GOOGLE)_API_KEY_(\d+)", name)
        if match:
            found.append((int(match.group(1)), value))
    found.sort(key=lambda pair: pair[0])

    seen: set[str] = set()
    keys: list[str] = []
    for _, key in found:
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def _is_rate_limit(err: Exception) -> bool:
    """True for a transient throttle (429 quota, 503) that warrants rotating to
    the next key, as opposed to a real error (bad key, bad request)."""
    text = f"{type(err).__name__}: {err}".lower()
    signals = (
        "429", "resource_exhausted", "resourceexhausted", "rate limit",
        "rate_limit", "quota", "too many requests", "503", "overloaded",
        "service unavailable", "unavailable",
    )
    return any(s in text for s in signals)


# Subclassing Runnable lets this behave like a chat model everywhere agents use
# it: .invoke(), the `|` pipe, and passing callbacks through `config`.
from langchain_core.runnables import Runnable  # noqa: E402


class RotatingGemini(Runnable):
    """A chat model that spreads calls across a pool of Gemini API keys.

    Free-tier keys have low per-minute/day limits, so holding several and
    rotating raises throughput and keeps running when one key is throttled:
      * round-robin - each .invoke() starts on the next key in the pool;
      * failover    - on a 429/quota error it retries on the following keys;
      * patient     - if every key is throttled in one pass, it waits and
                      retries the whole rotation once per-minute windows recover;
      * transparent - forwards .invoke / .ainvoke and proxies .bind_tools /
                      .with_structured_output.
    """

    def __init__(self, pool: list, max_retries: int = 2, retry_wait: int = 15):
        if not pool:
            raise ValueError(
                "No Gemini API keys found. Add GOOGLE_API_KEY or "
                "GEMINI_API_KEY_<n> to your .env file."
            )
        self._pool = pool
        self._n = len(pool)
        self._cycle = itertools.cycle(range(self._n))
        self._lock = threading.Lock()  # guards the round-robin cursor
        self._max_retries = max_retries
        self._retry_wait = retry_wait

    @classmethod
    def from_keys(cls, keys: list[str], model: str, temperature: float):
        """Build one ChatGoogleGenerativeAI client per key."""
        from langchain_google_genai import ChatGoogleGenerativeAI

        extra = {}
        m = model.lower()
        # Only 2.5 flash models support disabling 'thinking'.
        if "gemini-2.5" in m and "pro" not in m:
            extra["thinking_budget"] = GEMINI_THINKING_BUDGET

        clients = [
            # max_retries=0 so a throttled key fails over immediately to the
            # next key, instead of LangChain backing off ~60s on the same one.
            ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=key,
                max_retries=0,
                **extra,
            )
            for key in keys
        ]
        return cls(clients)

    def _start_index(self) -> int:
        with self._lock:
            return next(self._cycle)

    def invoke(self, input, config=None, **kwargs):  # noqa: A002 (LangChain name)
        last_err: Exception | None = None
        for attempt in range(self._max_retries + 1):
            start = self._start_index()
            for offset in range(self._n):
                idx = (start + offset) % self._n
                try:
                    return self._pool[idx].invoke(input, config=config, **kwargs)
                except Exception as err:  # noqa: BLE001
                    if _is_rate_limit(err):
                        last_err = err
                        continue  # throttled key, try the next one
                    raise
            # Every key was throttled this pass; wait for a per-minute window.
            if attempt < self._max_retries:
                wait = self._retry_wait * (attempt + 1)
                print(f"[llm] all {self._n} keys rate-limited; waiting {wait}s then retrying...")
                time.sleep(wait)
        raise RuntimeError(
            f"All {self._n} Gemini keys are rate-limited after "
            f"{self._max_retries + 1} passes. Wait a minute or add more keys."
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
        raise RuntimeError(f"All {self._n} Gemini keys are rate-limited.") from last_err

    def bind_tools(self, *args, **kwargs):
        return RotatingGemini([r.bind_tools(*args, **kwargs) for r in self._pool])

    def with_structured_output(self, *args, **kwargs):
        return RotatingGemini(
            [r.with_structured_output(*args, **kwargs) for r in self._pool]
        )

    def __len__(self):
        return self._n


# ----------------------------------------------------------------------
# LLM factory
# ----------------------------------------------------------------------
_LLM_CACHE: dict = {}


def _build_llm(model: str, temperature: float):
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


def get_llm(model: str | None = None, temperature: float | None = None):
    """Return the chat model Atlas should use, based on LLM_PROVIDER.

    The model is built once and cached so the rotating pool's round-robin index
    persists across calls. Provider SDKs are imported lazily.
    """
    cache_key = (
        model or LLM_MODEL,
        LLM_TEMPERATURE if temperature is None else temperature,
    )
    if cache_key not in _LLM_CACHE:
        _LLM_CACHE[cache_key] = _build_llm(*cache_key)
    return _LLM_CACHE[cache_key]


# ----------------------------------------------------------------------
# Self-checks
# ----------------------------------------------------------------------
def _status(value: str | None) -> str:
    """Report whether a secret is present without printing the secret itself."""
    return f"set ({len(value)} chars)" if value else "MISSING"


def check() -> None:
    pool = gemini_keys()
    print("Atlas configuration")
    print("-" * 46)
    print(f"LLM_PROVIDER      : {LLM_PROVIDER}")
    print(f"LLM_MODEL         : {LLM_MODEL}")
    print(f"LLM_TEMPERATURE   : {LLM_TEMPERATURE}")
    print(f"THINKING_BUDGET   : {GEMINI_THINKING_BUDGET}  (0 = thinking off, faster)")
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
        print("!  No Gemini keys. Add GEMINI_API_KEY_<n> to .env.")
        ready = False
    if not TAVILY_API_KEY:
        print("!  TAVILY_API_KEY missing. The Searcher needs it.")
        ready = False
    if ready:
        print("OK  Core keys present.")
        if LLM_PROVIDER == "google" and len(pool) > 1:
            print(f"    Gemini calls round-robin across {len(pool)} keys, with failover.")
    print("    (Tip: run `python config.py check-keys` to test the keys live.)")


def check_keys() -> None:
    """Make ONE real call to Gemini and Tavily to confirm the keys are live."""
    print("Live key check (uses a tiny bit of quota)")
    print("-" * 46)
    try:
        t = time.time()
        resp = get_llm().invoke("Reply with exactly one word: pong")
        text = getattr(resp, "content", resp)
        print(f"OK  Gemini replied: {str(text).strip()[:50]!r}  ({time.time()-t:.1f}s)")
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
