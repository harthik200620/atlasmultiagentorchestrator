"""
Writer agent — turns the gathered Evidence (and the Analyst's synthesis) into a
short, well-structured research brief in Markdown, where every claim cites its
source URL. On a revision pass it also addresses the Critic's feedback. If the
evidence is thin, it says so HONESTLY instead of padding with unsourced filler.

CONTRACT
  reads : state["goal"], state["evidence"], state["analysis"], state["critique"]
  writes: state["draft"], state["status"], state["log"]
"""

from __future__ import annotations

import config
from state import AtlasState, new_state
from utils import numbered_evidence, truncate

BASE_RULES = """You are the Writer in a research pipeline. Write a concise,
well-organized research brief in Markdown that answers the goal USING ONLY the
evidence provided.

Rules:
- Every factual claim cites its source as [n](url) using the numbered sources.
- Do NOT invent facts or URLs. If evidence is thin or one-sided, say so in a
  short "Caveats" line.
- Structure: a one-line summary, then 2-5 short sections, then a "Sources" list."""

# Critique placeholders that mean "approved", so we don't feed them back as feedback.
_APPROVED_MARKERS = {"(approved)", "(auto-approved: nothing to revise)"}


def _build_prompt(goal: str, evidence: list, analysis: str, critique: str) -> str:
    parts = [BASE_RULES]
    if analysis and analysis.strip():
        parts.append("Use this analysis from the Analyst as your scaffold:\n" + analysis.strip())
    if critique and critique.strip() and critique.strip() not in _APPROVED_MARKERS:
        parts.append(
            "A reviewer asked you to REVISE the previous draft. Address this "
            "feedback specifically:\n" + critique.strip()
        )
    parts.append(f"GOAL: {goal}")
    parts.append(
        "EVIDENCE (numbered; cite by number):\n"
        + truncate(numbered_evidence(evidence), 8000)
    )
    return "\n\n".join(parts)


def writer(state: AtlasState) -> dict:
    goal = state.get("goal") or ""
    evidence = state.get("evidence") or []
    analysis = state.get("analysis") or ""
    critique = state.get("critique") or ""

    # Honest fallback: no evidence -> say so, never fabricate. (Done, no Critic.)
    if not evidence:
        draft = (
            f"# {goal}\n\n"
            "**Insufficient evidence was gathered to answer this reliably.** "
            "No usable sources were found. Atlas will not fabricate claims or "
            "citations."
        )
        return {
            "draft": draft,
            "status": "done",
            "log": ["Writer: no evidence — wrote an honest 'insufficient' note."],
        }

    prompt = _build_prompt(goal, evidence, analysis, critique)
    try:
        resp = config.get_llm().invoke(prompt)
        draft = getattr(resp, "content", "") or ""
    except Exception as err:  # noqa: BLE001  — never crash the pipeline
        print(f"[writer] LLM failed: {type(err).__name__}: {err}")
        draft = ""

    if not draft.strip():
        # Fallback: still hand back the sourced facts we gathered.
        bullets = "\n".join(
            f"- {e['claim']} ([source]({e['source_url']}))" for e in evidence
        )
        draft = (
            f"# {goal}\n\n"
            "_(Auto-summary unavailable; listing the gathered evidence.)_\n\n"
            f"{bullets}"
        )

    note = f"Writer: drafted a brief from {len(evidence)} evidence item(s)."
    # Hand off to the Critic for quality-gating.
    return {"draft": draft, "status": "criticizing", "log": [note]}


if __name__ == "__main__":
    # Run the Writer alone with hand-made evidence:  python writer.py
    state = new_state("Is LangGraph a good fit for multi-agent systems?")
    state["evidence"] = [
        {"claim": "LangGraph models agent workflows as a graph of nodes sharing one state.",
         "source_url": "https://langchain-ai.github.io/langgraph/",
         "snippet": "LangGraph is a library for building stateful, multi-actor apps."},
        {"claim": "It supports a supervisor pattern where a router delegates to worker agents.",
         "source_url": "https://langchain-ai.github.io/langgraph/concepts/multi_agent/",
         "snippet": "A supervisor agent decides which worker to call next."},
    ]
    out = writer(state)
    print(out["log"][0], "(status ->", out["status"] + ")")
    print("-" * 60)
    print(out["draft"])
