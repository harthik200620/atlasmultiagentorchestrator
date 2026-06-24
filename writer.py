"""
Writer agent. Turns the evidence and the Analyst's synthesis into a short
Markdown research brief that cites its sources by number, addressing the Critic's
feedback on a revision pass. If evidence is thin it says so rather than padding
with unsourced filler.

  reads : state["goal"], state["evidence"], state["analysis"], state["critique"]
  writes: state["draft"], state["status"], state["log"]
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

import config
from state import AtlasState, new_state
from utils import numbered_evidence, source_order, truncate

BASE_RULES = """You are the Writer in a research pipeline. Write a concise,
well-organized research brief in Markdown that answers the goal USING ONLY the
evidence provided.

Rules:
- Support every factual claim with a citation: the bracketed source number from
  the EVIDENCE list, e.g. [1] or [2]. Combine related ones as [1, 3].
- Do NOT invent facts or sources. If evidence is thin or one-sided, add a short
  "Caveats" line saying so.
- Structure: a one-line summary, then 2-5 short sections. Do NOT add your own
  Sources/References list - it is generated automatically from the citations."""

# Critique placeholders that mean "approved"; not fed back as feedback.
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
        "EVIDENCE (cite claims by the bracketed source number):\n"
        + truncate(numbered_evidence(evidence), 8000)
    )
    return "\n\n".join(parts)


def _site(url: str) -> str:
    """Short, human-readable label for a source URL (its domain)."""
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return url


def _strip_sources_section(md: str) -> str:
    """Drop any Sources/References list the model wrote; we generate our own."""
    m = re.search(
        r"(?im)^\s*(?:#{1,6}\s*|\*\*)?(sources?|references?|citations?|bibliography)\b\s*:?\**\s*$",
        md,
    )
    return md[: m.start()].rstrip() if m else md.rstrip()


def _linkify_citations(md: str, order: list) -> str:
    """Turn bare [n] / [n, m] markers into clickable links to their source URL."""
    def repl(match: re.Match) -> str:
        nums = [int(x) for x in re.findall(r"\d+", match.group(1))]
        if not any(1 <= n <= len(order) for n in nums):
            return match.group(0)  # not a citation we recognize - leave as-is
        parts = [
            f"[{n}]({order[n - 1]})" if 1 <= n <= len(order) else str(n)
            for n in nums
        ]
        return "\\[" + ", ".join(parts) + "\\]"

    # [n], [n, m], [n,m] -- but skip an existing [text](link).
    return re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\](?!\()", repl, md)


def _sources_section(order: list) -> str:
    if not order:
        return ""
    lines = ["## Sources", ""]
    lines += [f"{i}. [{_site(url)}]({url})" for i, url in enumerate(order, 1)]
    return "\n".join(lines)


def _finalize(draft: str, order: list) -> str:
    """Make citations clickable and append a numbered Sources list."""
    draft = _linkify_citations(_strip_sources_section(draft), order)
    sources = _sources_section(order)
    return f"{draft.rstrip()}\n\n{sources}" if sources else draft.rstrip()


def writer(state: AtlasState) -> dict:
    goal = state.get("goal") or ""
    evidence = state.get("evidence") or []
    analysis = state.get("analysis") or ""
    critique = state.get("critique") or ""

    # No evidence: say so rather than fabricate, and skip the Critic.
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
            "log": ["Writer: no evidence - wrote an honest 'insufficient' note."],
        }

    order = source_order(evidence)
    prompt = _build_prompt(goal, evidence, analysis, critique)
    try:
        resp = config.get_llm().invoke(prompt)
        draft = getattr(resp, "content", "") or ""
    except Exception as err:  # noqa: BLE001
        print(f"[writer] LLM failed: {type(err).__name__}: {err}")
        draft = ""

    if not draft.strip():
        # Fallback: hand back the gathered facts, already cited by source number.
        num = {url: i for i, url in enumerate(order, 1)}
        bullets = "\n".join(
            f"- {e['claim']} \\[[{num[e['source_url']]}]({e['source_url']})\\]"
            for e in evidence
            if e.get("source_url") in num
        )
        draft = (
            f"# {goal}\n\n"
            "_(Auto-summary unavailable; listing the gathered evidence.)_\n\n"
            f"{bullets}"
        )

    draft = _finalize(draft, order)
    note = f"Writer: drafted a brief from {len(evidence)} evidence item(s)."
    return {"draft": draft, "status": "criticizing", "log": [note]}


if __name__ == "__main__":
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
