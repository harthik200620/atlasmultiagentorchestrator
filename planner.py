"""
Planner agent — the first worker. It turns the research goal into a short list of
focused web-search queries (the 'plan'), so the Searcher fetches varied, relevant
sources instead of just echoing the goal back.

CONTRACT
  reads : state["goal"]
  writes: state["plan"], state["status"], state["log"]
"""

from __future__ import annotations

import config
from state import AtlasState, new_state
from utils import extract_json

PLANNER_PROMPT = """You are the Planner in a research pipeline.
Break the research goal into 3-5 focused web-search queries that TOGETHER gather
the evidence needed to answer it well. Make them specific and varied — different
angles, not reworded duplicates.

Return ONLY a JSON array of short query strings, e.g.
["query one", "query two", "query three"].

GOAL: {goal}
"""


def planner(state: AtlasState) -> dict:
    goal = (state.get("goal") or "").strip()

    plan: list[str] = []
    try:
        resp = config.get_llm().invoke(PLANNER_PROMPT.format(goal=goal))
        parsed = extract_json(getattr(resp, "content", "")) or []
        if isinstance(parsed, list):
            plan = [str(q).strip() for q in parsed if str(q).strip()][:5]
    except Exception as err:  # noqa: BLE001
        print(f"[planner] LLM failed: {type(err).__name__}: {err}")

    if not plan:
        plan = [goal]  # safe fallback: at least search the goal itself

    note = f"Planner: broke the goal into {len(plan)} search step(s)."
    return {"plan": plan, "status": "searching", "log": [note]}


if __name__ == "__main__":
    # Run the Planner alone:  python planner.py
    out = planner(new_state("How does LangGraph's supervisor pattern compare to CrewAI?"))
    print(out["log"][0])
    for i, q in enumerate(out["plan"], 1):
        print(f"  {i}. {q}")
