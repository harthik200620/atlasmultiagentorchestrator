"""
state.py — the shared "blackboard" that every Atlas agent reads and writes.

THE KEY MULTI-AGENT IDEA: there is exactly ONE state object for a run. Each agent
(a "node" in the graph) receives the whole state, does its one job, and returns
ONLY the keys it changed. LangGraph merges those changes back in.

How a key merges is decided by its *reducer*:
  - No reducer   -> the new value REPLACES the old one        (e.g. `draft`)
  - `add` reducer-> the new value is APPENDED to the old list (e.g. `evidence`)

This file only DEFINES the shape of the state. The agents that fill it in and the
graph that passes it around arrive in later phases.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages

# The supervisor reads `status` to decide which worker runs next.
Status = Literal[
    "planning",     # Planner is about to break the goal into steps
    "searching",    # Searcher is querying Tavily
    "reading",      # Reader is extracting facts from results
    "analyzing",    # Analyst is synthesizing/comparing
    "writing",      # Writer is drafting the report
    "criticizing",  # Critic is quality-gating the draft
    "revise",       # Critic rejected — loop back and improve
    "done",         # Critic approved — we're finished
    "failed",       # unrecoverable error (e.g. all searches failed)
]


class SearchResult(TypedDict):
    """One raw hit returned by Tavily, before the Reader distills it."""

    title: str
    url: str
    content: str       # Tavily's short snippet for the hit
    raw_content: str   # full extracted page text, when available (else "")


class Evidence(TypedDict):
    """A single fact the Reader extracted, tied to where it came from.

    Every claim in Atlas's final report must trace back to one of these — that's
    how we guarantee the brief is *sourced*, not hallucinated.
    """

    claim: str         # the fact, in our own words
    source_url: str    # the page it came from
    snippet: str       # the supporting text we actually saw


class AtlasState(TypedDict, total=False):
    """The whole shared state.

    `total=False` means an agent may return just the keys it touched instead of
    the entire object — which is exactly how LangGraph nodes are meant to work.
    """

    # --- the task ---
    goal: str                                       # what the user asked for

    # --- Planner output ---
    plan: list[str]                                 # ordered research sub-steps

    # --- running log of agent messages (APPENDED via add_messages) ---
    messages: Annotated[list, add_messages]

    # --- material gathered as agents work (APPENDED via `add`) ---
    search_results: Annotated[list[SearchResult], add]
    evidence: Annotated[list[Evidence], add]

    # --- writing & review (REPLACED each time, no reducer) ---
    draft: str                                      # current report draft
    critique: str                                   # Critic's latest feedback

    # --- control flow ---
    status: Status                                  # where we are in the pipeline
    next_agent: str                                 # supervisor's routing decision
    iterations: int                                 # loop counter vs MAX_ITERATIONS


def new_state(goal: str) -> AtlasState:
    """Build a fresh, empty state for a new research goal."""
    return {
        "goal": goal,
        "plan": [],
        "messages": [],
        "search_results": [],
        "evidence": [],
        "draft": "",
        "critique": "",
        "status": "planning",
        "next_agent": "planner",
        "iterations": 0,
    }


if __name__ == "__main__":
    # Tiny smoke test:  python state.py
    s = new_state("Compare LangGraph and CrewAI for building multi-agent apps.")
    print("Created AtlasState with keys:")
    print("  " + ", ".join(s.keys()))
    print(f"goal       : {s['goal']}")
    print(f"status     : {s['status']}")
    print(f"next_agent : {s['next_agent']}")
    print(f"iterations : {s['iterations']}")
    print("OK  state.py imports and builds a fresh state correctly.")
