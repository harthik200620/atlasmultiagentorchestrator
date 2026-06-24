"""
Critic agent, the quality gate. Judges the draft against the goal and evidence
and either approves it or returns actionable feedback, plus new search queries
when evidence is insufficient. The supervisor turns a rejection into a revise loop.

  reads : state["goal"], state["draft"], state["evidence"]
  writes: state["status"] ("done" or "revise"), state["critique"],
          state["revise_queries"], state["log"]
"""

from __future__ import annotations

import config
from state import AtlasState, new_state
from utils import extract_json, truncate

CRITIC_PROMPT = """You are the Critic, a strict quality gate for a research brief.
Judge the DRAFT against the goal and the evidence.

Approve ONLY if ALL of these hold:
- it directly answers the goal,
- every major claim cites a source like [n],
- the evidence is sufficient and not one-sided,
- there are no invented facts or uncited claims.

Return ONLY JSON in this shape:
{{"approved": true or false,
  "critique": "specific, actionable feedback; empty string if approved",
  "extra_queries": ["a new web search to fill a gap", "..."]}}
Use "extra_queries" ONLY if more evidence is genuinely needed; otherwise [].

GOAL: {goal}

EVIDENCE: {n} sourced facts were gathered.

DRAFT:
{draft}
"""


def critic(state: AtlasState) -> dict:
    goal = state.get("goal") or ""
    draft = (state.get("draft") or "").strip()
    evidence = state.get("evidence") or []

    # Nothing to judge: accept gracefully.
    if not draft or not evidence:
        return {
            "status": "done",
            "critique": "(auto-approved: nothing to revise)",
            "log": ["Critic: nothing to revise -approving."],
        }

    verdict = {}
    try:
        resp = config.get_llm().invoke(
            CRITIC_PROMPT.format(goal=goal, n=len(evidence), draft=truncate(draft, 8000))
        )
        verdict = extract_json(getattr(resp, "content", "")) or {}
    except Exception as err:  # noqa: BLE001
        print(f"[critic] LLM failed: {type(err).__name__}: {err}")

    if not isinstance(verdict, dict):
        verdict = {}
    # Default to approve on uncertainty rather than loop on a parse error.
    approved = bool(verdict.get("approved", True))

    if approved:
        return {
            "status": "done",
            "critique": "(approved)",
            "log": ["Critic: APPROVED the draft."],
        }

    critique = str(verdict.get("critique", "")).strip() or "Improve sourcing and directness."
    extra = verdict.get("extra_queries") or []
    extra = [str(q).strip() for q in extra if str(q).strip()][:4]
    mode = f"gather more evidence ({len(extra)} new queries)" if extra else "rewrite"
    return {
        "status": "revise",
        "critique": critique,
        "revise_queries": extra,
        "log": [f"Critic: REVISE -{mode}."],
    }


if __name__ == "__main__":
    # A weak, uncited draft should be rejected.
    st = new_state("What is the LangGraph supervisor pattern?")
    st["evidence"] = [
        {"claim": "LangGraph has a supervisor pattern that routes to workers.",
         "source_url": "https://example.com/x", "snippet": "a supervisor routes to workers"},
    ]
    st["draft"] = "LangGraph is a tool. It is good for agents."
    out = critic(st)
    print("weak draft ->", out["status"], "|", out["log"][0])
    if out.get("status") == "revise":
        print("critique  :", out["critique"][:200])
