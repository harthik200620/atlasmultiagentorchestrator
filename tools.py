"""
tools.py — Atlas's external tools (the capabilities its agents call).

Right now that's web search and page extraction via Tavily. Each tool has a
STRICT contract (clear inputs and outputs) and is built so a network hiccup or a
dead link can NEVER crash a run: it retries transient failures and then degrades
gracefully (returns [] or "") instead of raising.
"""

from __future__ import annotations

import time

from tavily import TavilyClient

import config
from state import SearchResult

# One shared Tavily client, created on first use (so importing tools.py is cheap
# and doesn't require a key to be present yet).
_client: TavilyClient | None = None


def _tavily() -> TavilyClient:
    global _client
    if _client is None:
        if not config.TAVILY_API_KEY:
            raise RuntimeError("TAVILY_API_KEY missing — add it to your .env file.")
        _client = TavilyClient(api_key=config.TAVILY_API_KEY)
    return _client


def tavily_search(
    query: str,
    max_results: int | None = None,
    search_depth: str = "basic",
    retries: int = 2,
) -> list[SearchResult]:
    """Search the web for `query`; return a list of SearchResult dicts.

    - asks Tavily for raw page content too, so the Reader has text to work with
    - retries transient failures with a short backoff
    - on persistent failure returns [] (never raises), so the pipeline survives
    """
    max_results = max_results or config.TAVILY_MAX_RESULTS
    query = (query or "").strip()
    if not query:
        return []

    for attempt in range(retries + 1):
        try:
            raw = _tavily().search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_raw_content=True,
            )
            results: list[SearchResult] = []
            for item in raw.get("results", []):
                results.append(
                    {
                        "title": item.get("title") or "",
                        "url": item.get("url") or "",
                        "content": item.get("content") or "",
                        "raw_content": item.get("raw_content") or "",
                    }
                )
            return results
        except Exception as err:  # noqa: BLE001
            if attempt < retries:
                print(
                    f"[tools] search failed (try {attempt + 1}/{retries + 1}): "
                    f"{type(err).__name__} — retrying..."
                )
                time.sleep(1.5 * (attempt + 1))
            else:
                print(
                    f"[tools] search gave up after {retries + 1} tries: "
                    f"{type(err).__name__}: {err}"
                )
    return []


def tavily_extract(url: str, retries: int = 1) -> str:
    """Fetch the full text of one page. Returns "" if the link is dead or
    unfetchable — a dead link must never crash Atlas."""
    url = (url or "").strip()
    if not url:
        return ""

    for attempt in range(retries + 1):
        try:
            raw = _tavily().extract(urls=[url])
            items = raw.get("results", [])
            return (items[0].get("raw_content") or "") if items else ""
        except Exception as err:  # noqa: BLE001
            if attempt < retries:
                time.sleep(1.0)
            else:
                print(f"[tools] extract failed for {url[:60]}: {type(err).__name__}")
    return ""


if __name__ == "__main__":
    # Quick standalone test:  python tools.py
    print("Testing tavily_search...")
    hits = tavily_search("LangGraph supervisor multi-agent pattern", max_results=3)
    print(f"  got {len(hits)} result(s)")
    for i, h in enumerate(hits, 1):
        print(f"  {i}. {h['title'][:70]}")
        print(f"     {h['url']}")
        print(f"     snippet: {h['content'][:90]}...")
        print(f"     raw_content chars: {len(h['raw_content'])}")

    if hits:
        print("\nTesting tavily_extract on the first URL...")
        text = tavily_extract(hits[0]["url"])
        print(f"  extracted {len(text)} chars")
