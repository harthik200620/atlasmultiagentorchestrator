"""
Writer agent — turns the gathered Evidence into a short, well-structured research
brief in Markdown, where every claim cites its source URL. If the evidence is
thin, it says so HONESTLY instead of padding the brief with unsourced filler.

CONTRACT
  reads : state["goal"], state["evidence"]
  writes: state["draft"], state["status"], state["log"]
"""

from __future__ import annotations

import config
from state import AtlasState, new_state
from utils import truncate

WRITER_PROMPT = """You are the Writer in a research pipeline. Write a concise,
well-organized research brief in Markdown that answers the goal USING ONLY the
evidence provided below.

Rules:
- Every factual claim must cite its source with a Markdown link like [1](url),
  using the numbered sources below.
- Do NOT invent facts or URLs. If the evidence is thin or one-sided, say so
  honestly in a short "Caveats" line.
- Structure: a one-line summary, then 2-5 short sections, then a "Sources" list.

GOAL: {goal}

EVIDENCE (numbered; cite by number):
{evidence_block}
"""


def _evidence_block(evidence) -> str:
    lines = [
        f"[{i}] {e['claim']} (source: {e['source_url']})"
        for i, e in enumerate(evidence, 1)
    ]
    return "\n".join(lines) if lines else "(no evidence gathered)"


def writer(state: AtlasState) -> dict:
    goal = state.get("goal") or ""
    evidence = state.get("evidence") or []

    # Honest fallback: no evidence -> say so, never fabricate.
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

    llm = config.get_llm()
    prompt = WRITER_PROMPT.format(
        goal=goal, evidence_block=truncate(_evidence_block(evidence), 8000)
    )
    resp = llm.invoke(prompt)
    draft = getattr(resp, "content", "") or ""

    note = f"Writer: drafted a brief from {len(evidence)} evidence item(s)."
    return {"draft": draft, "status": "done", "log": [note]}


if __name__ == "__main__":
    # Run the Writer alone with hand-made evidence:  python writer.py
    state = new_state("Is LangGraph a good fit for multi-agent systems?")
    state["evidence"] = [
        {
            "claim": "LangGraph models agent workflows as a graph of nodes sharing one state.",
            "source_url": "https://langchain-ai.github.io/langgraph/",
            "snippet": "LangGraph is a library for building stateful, multi-actor applications.",
        },
        {
            "claim": "It supports a supervisor pattern where a router delegates to worker agents.",
            "source_url": "https://langchain-ai.github.io/langgraph/concepts/multi_agent/",
            "snippet": "A supervisor agent decides which worker to call next.",
        },
    ]
    out = writer(state)
    print(out["log"][0])
    print("-" * 60)
    print(out["draft"])
