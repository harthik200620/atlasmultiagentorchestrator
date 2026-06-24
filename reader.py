"""
Reader agent — reads the Searcher's raw results and extracts the key FACTS that
are relevant to the goal, each tied to its source URL. That (fact + URL) pair is
our Evidence, and it's what lets the final report cite where every claim came from.

This is where the LLM earns its keep: it distills messy page text into clean,
sourced claims. If a result has no usable text, the Reader tries to fetch the page
directly (dead-link fallback) before giving up on that source.

CONTRACT
  reads : state["goal"], state["search_results"]
  writes: state["evidence"], state["status"], state["log"]
"""

from __future__ import annotations

import config
from state import AtlasState, Evidence, new_state
from tools import tavily_extract
from utils import extract_json, truncate

READER_PROMPT = """You are the Reader in a research pipeline.
Goal: {goal}

Below is the text of ONE web source. Extract up to {k} factual claims from it
that are RELEVANT to the goal. Ignore navigation, ads, and boilerplate.

Return ONLY a JSON array; each element looks like:
  {{"claim": "the fact in your own words, one sentence",
    "snippet": "a short supporting quote from the source, <=200 chars"}}
If the source has nothing relevant, return [].

SOURCE TITLE: {title}
SOURCE TEXT:
{body}
"""


def reader(state: AtlasState, max_facts_per_source: int = 3) -> dict:
    goal = state.get("goal") or ""
    results = state.get("search_results") or []
    llm = config.get_llm()

    evidence: list[Evidence] = []
    for r in results:
        body = r.get("raw_content") or r.get("content") or ""
        if len(body) < 200:
            # Thin or empty — try fetching the page directly (dead-link fallback).
            body = tavily_extract(r.get("url", "")) or body
        if len(body.strip()) < 80:
            continue  # nothing worth reading here; skip this source

        prompt = READER_PROMPT.format(
            goal=goal,
            k=max_facts_per_source,
            title=r.get("title", ""),
            body=truncate(body, 6000),
        )
        try:
            resp = llm.invoke(prompt)
            facts = extract_json(getattr(resp, "content", "")) or []
        except Exception as err:  # noqa: BLE001
            print(f"[reader] LLM failed on {r.get('url', '')[:50]}: {type(err).__name__}")
            facts = []

        if not isinstance(facts, list):
            continue
        for f in facts:
            if isinstance(f, dict) and str(f.get("claim", "")).strip():
                evidence.append(
                    {
                        "claim": str(f["claim"]).strip(),
                        "source_url": r.get("url", ""),
                        "snippet": str(f.get("snippet", "")).strip()[:200],
                    }
                )

    status = "writing" if evidence else "failed"
    note = (
        f"Reader: extracted {len(evidence)} sourced fact(s) "
        f"from {len(results)} source(s)."
    )
    return {"evidence": evidence, "status": status, "log": [note]}


if __name__ == "__main__":
    # Run the Reader alone:  python reader.py
    # (We do a tiny real search first so the Reader has something to read.)
    from tools import tavily_search

    goal = "What is LangGraph's supervisor multi-agent pattern?"
    state = new_state(goal)
    state["search_results"] = tavily_search(goal, max_results=3)
    out = reader(state)
    print(out["log"][0])
    for i, e in enumerate(out["evidence"], 1):
        print(f"{i}. {e['claim']}")
        print(f"   source : {e['source_url']}")
        print(f"   snippet: {e['snippet'][:90]}")
