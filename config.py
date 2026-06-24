"""
config.py -one central place for all of Atlas's configuration.

Anything that depends on the environment (which LLM to use, API keys, behavior
knobs) is read HERE, once. The rest of the code imports from this file and never
touches os.environ directly -so there's a single source of truth.

    import config
    llm = config.get_llm()          # a ready-to-use chat model

Check your setup any time with:

    python config.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# Load variables from a local .env file into the environment.
# Real OS environment variables take precedence, which is what you want in
# deployment (set vars directly, no .env file needed).
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
# Hard cap on supervisor loops -the safety net that prevents infinite agent
# loops (a real failure mode in multi-agent systems).
MAX_ITERATIONS = int(os.getenv("ATLAS_MAX_ITERATIONS", "6"))
# How many results the Searcher asks Tavily for per query.
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))


# ----------------------------------------------------------------------
# LLM factory
# ----------------------------------------------------------------------
def get_llm(model: str | None = None, temperature: float | None = None):
    """Return a LangChain chat model for the configured provider.

    The provider-specific package is imported INSIDE this function (a "lazy
    import") so that merely importing config.py doesn't require every provider's
    SDK to be installed -only the one you actually use.

    Every Atlas agent gets its model through here, so switching the whole system
    from Gemini to GPT-4o-mini or Claude is a one-line change in .env.
    """
    provider = LLM_PROVIDER
    model = model or LLM_MODEL
    temperature = LLM_TEMPERATURE if temperature is None else temperature

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=GOOGLE_API_KEY,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature, api_key=OPENAI_API_KEY)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model, temperature=temperature, api_key=ANTHROPIC_API_KEY
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. Use one of: google | openai | anthropic."
    )


# ----------------------------------------------------------------------
# Setup self-check:  python config.py
# ----------------------------------------------------------------------
def _status(value: str | None) -> str:
    """Report whether a secret is present WITHOUT printing the secret itself."""
    return f"set ({len(value)} chars)" if value else "MISSING"


def check() -> None:
    print("Atlas configuration")
    print("-" * 44)
    print(f"LLM_PROVIDER      : {LLM_PROVIDER}")
    print(f"LLM_MODEL         : {LLM_MODEL}")
    print(f"LLM_TEMPERATURE   : {LLM_TEMPERATURE}")
    print(f"MAX_ITERATIONS    : {MAX_ITERATIONS}")
    print(f"TAVILY_MAX_RESULTS: {TAVILY_MAX_RESULTS}")
    print("-" * 44)
    print("API keys (values hidden):")
    print(f"  GOOGLE_API_KEY    : {_status(GOOGLE_API_KEY)}")
    print(f"  OPENAI_API_KEY    : {_status(OPENAI_API_KEY)}")
    print(f"  ANTHROPIC_API_KEY : {_status(ANTHROPIC_API_KEY)}")
    print(f"  TAVILY_API_KEY    : {_status(TAVILY_API_KEY)}")
    print(f"  LANGFUSE_PUBLIC   : {_status(LANGFUSE_PUBLIC_KEY)}")
    print(f"  LANGFUSE_SECRET   : {_status(LANGFUSE_SECRET_KEY)}")
    print("-" * 44)

    # Non-fatal guidance about what each upcoming phase needs.
    key_for = {
        "google": GOOGLE_API_KEY,
        "openai": OPENAI_API_KEY,
        "anthropic": ANTHROPIC_API_KEY,
    }
    if not key_for.get(LLM_PROVIDER):
        print(f"!  No API key for provider '{LLM_PROVIDER}'. Add it to .env before Phase 1.")
    if not TAVILY_API_KEY:
        print("!  TAVILY_API_KEY missing. The Searcher (Phase 1) needs it.")
    if key_for.get(LLM_PROVIDER) and TAVILY_API_KEY:
        print("OK  Core keys present -you're ready for Phase 1.")
    else:
        print("   (Phase 0 itself is fine without keys -you only need them from Phase 1.)")


if __name__ == "__main__":
    check()
