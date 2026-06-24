"""
Searcher agent. Runs the plan's queries (or the goal) through the Tavily search
tool and hands back de-duplicated raw results for the Reader.

  reads : state["goal"], state["plan"] (optional)
  writes: state["search_results"], state["status"], state["log"]
"""

from __future__ import annotations

from state import AtlasState, new_state
from tools import tavily_search


def searcher(state: AtlasState, max_sources: int = 8) -> dict:
    goal = (state.get("goal") or "").strip()
    plan = state.get("plan") or []

    queries = [q for q in (plan if plan else [goal]) if q and q.strip()]

    # Cap total sources to keep the Reader's per-source LLM calls bounded.
    results = []
    seen_urls = set()
    for query in queries:
        if len(results) >= max_sources:
            break
        for hit in tavily_search(query):
            url = hit["url"]
            if url and url not in seen_urls:
                seen_urls.add(url)
                results.append(hit)
                if len(results) >= max_sources:
                    break

    # No results means failure, so the supervisor can react.
    status = "reading" if results else "failed"
    note = (
        f"Searcher: ran {len(queries)} query(ies), "
        f"found {len(results)} unique source(s)."
    )
    return {"search_results": results, "status": status, "log": [note]}


if __name__ == "__main__":
    state = new_state("What is LangGraph and how does its supervisor pattern work?")
    out = searcher(state)
    print(out["log"][0])
    for i, r in enumerate(out["search_results"], 1):
        print(f"{i}. {r['title'][:70]}")
        print(f"   {r['url']}")
        print(f"   raw_content chars: {len(r['raw_content'])}")
