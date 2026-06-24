"""
Searcher agent — turns the research goal (and the Planner's steps, when present)
into web searches, and hands back the raw results for the Reader to digest.

It's a 'tool-using' agent: no LLM reasoning here, just disciplined use of the
Tavily search tool, de-duplicating by URL so we don't read the same page twice.

CONTRACT
  reads : state["goal"], state["plan"] (optional)
  writes: state["search_results"], state["status"], state["log"]
"""

from __future__ import annotations

from state import AtlasState, new_state
from tools import tavily_search


def searcher(state: AtlasState) -> dict:
    goal = (state.get("goal") or "").strip()
    plan = state.get("plan") or []

    # Search each plan step if we have a plan; otherwise search the goal itself.
    queries = [q for q in (plan if plan else [goal]) if q and q.strip()]

    results = []
    seen_urls = set()
    for query in queries:
        for hit in tavily_search(query):
            url = hit["url"]
            if url and url not in seen_urls:
                seen_urls.add(url)
                results.append(hit)

    # If we found nothing at all, flag failure so the supervisor can react later.
    status = "reading" if results else "failed"
    note = (
        f"Searcher: ran {len(queries)} query(ies), "
        f"found {len(results)} unique source(s)."
    )
    return {"search_results": results, "status": status, "log": [note]}


if __name__ == "__main__":
    # Run the Searcher alone:  python searcher.py
    state = new_state("What is LangGraph and how does its supervisor pattern work?")
    out = searcher(state)
    print(out["log"][0])
    for i, r in enumerate(out["search_results"], 1):
        print(f"{i}. {r['title'][:70]}")
        print(f"   {r['url']}")
        print(f"   raw_content chars: {len(r['raw_content'])}")
