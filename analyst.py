"""
Analyst agent — reads the gathered Evidence and produces a SYNTHESIS (themes,
agreements, contradictions, and gaps) that the Writer uses as a scaffold.

Separating "understand the evidence" from "write it up" is a real multi-agent
design win: the Writer produces a tighter brief, and the Analyst's gap-spotting
feeds the Critic's decision about whether more research is needed.

CONTRACT
  reads : state["goal"], state["evidence"]
  writes: state["analysis"], state["status"], state["log"]
"""

from __future__ import annotations

import config
from state import AtlasState, new_state
from utils import numbered_evidence, truncate

ANALYST_PROMPT = """You are the Analyst in a research pipeline. Study the evidence
and produce a SYNTHESIS for the Writer — do NOT write the final brief.

Identify, as short bullet points (refer to facts by their [n] number):
- the main points that answer the goal,
- agreements and any CONTRADICTIONS between sources,
- notable GAPS (what is missing to answer the goal well).

GOAL: {goal}

EVIDENCE (numbered):
{evidence_block}
"""


def analyst(state: AtlasState) -> dict:
    goal = state.get("goal") or ""
    evidence = state.get("evidence") or []

    if not evidence:
        return {
            "analysis": "",
            "status": "writing",
            "log": ["Analyst: no evidence to analyze; passing through."],
        }

    analysis = ""
    try:
        resp = config.get_llm().invoke(
            ANALYST_PROMPT.format(
                goal=goal,
                evidence_block=truncate(numbered_evidence(evidence), 8000),
            )
        )
        analysis = getattr(resp, "content", "") or ""
    except Exception as err:  # noqa: BLE001
        print(f"[analyst] LLM failed: {type(err).__name__}: {err}")

    note = f"Analyst: synthesized {len(evidence)} evidence item(s)."
    return {"analysis": analysis, "status": "writing", "log": [note]}


if __name__ == "__main__":
    # Run the Analyst alone with hand-made evidence:  python analyst.py
    state = new_state("Compare LangGraph and CrewAI.")
    state["evidence"] = [
        {"claim": "LangGraph gives low-level graph control over agents.",
         "source_url": "https://example.com/a", "snippet": "..."},
        {"claim": "CrewAI is a low-code, role-based multi-agent framework.",
         "source_url": "https://example.com/b", "snippet": "..."},
    ]
    out = analyst(state)
    print(out["log"][0])
    print("-" * 60)
    print(out["analysis"])
