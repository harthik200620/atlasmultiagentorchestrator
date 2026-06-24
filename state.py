"""
state.py - the shared state that every Atlas agent reads and writes.

There is one state object per run. Each agent (a node in the graph) receives the
whole state and returns only the keys it changed; LangGraph merges them back in.
A key's reducer decides how it merges: no reducer replaces the old value, while
the `add` reducer appends to the existing list.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Literal, TypedDict

# The supervisor reads `status` to decide which worker runs next.
Status = Literal[
    "planning",     # Planner is about to break the goal into steps
    "searching",    # Searcher is querying Tavily
    "reading",      # Reader is extracting facts from results
    "analyzing",    # Analyst is synthesizing/comparing
    "writing",      # Writer is drafting the report
    "criticizing",  # Critic is quality-gating the draft
    "revise",       # Critic rejected, loop back and improve
    "done",         # finished
    "failed",       # unrecoverable (e.g. no sources found at all)
]


class SearchResult(TypedDict):
    """One raw hit returned by Tavily, before the Reader distills it."""

    title: str
    url: str
    content: str       # Tavily's short snippet for the hit
    raw_content: str   # full extracted page text, when available (else "")


class Evidence(TypedDict):
    """A single fact the Reader extracted, tied to its source.

    Every claim in the final report must trace back to one of these.
    """

    claim: str         # the fact, in our own words
    source_url: str    # the page it came from
    snippet: str       # the supporting text we actually saw


class AtlasState(TypedDict, total=False):
    """The whole shared state.

    `total=False` lets an agent return just the keys it touched rather than the
    entire object.
    """

    # --- the task ---
    goal: str                                       # what the user asked for

    # --- Planner output ---
    plan: list[str]                                 # ordered research sub-steps

    # --- human-readable activity log (APPENDED via `add`; shown in the UI) ---
    log: Annotated[list[str], add]

    # --- material gathered as agents work (APPENDED via `add`) ---
    search_results: Annotated[list[SearchResult], add]
    evidence: Annotated[list[Evidence], add]

    # --- synthesis, writing & review (REPLACED each time, no reducer) ---
    analysis: str                                   # Analyst's synthesis
    draft: str                                      # current report draft
    critique: str                                   # Critic's latest feedback

    # --- control flow ---
    status: Status                                  # where we are in the pipeline
    next_agent: str                                 # supervisor's routing decision
    iterations: int                                 # total supervisor visits
    revisions: int                                  # Critic-triggered revise rounds
    revise_queries: list[str]                       # extra searches the Critic asked for


def new_state(goal: str) -> AtlasState:
    """Build a fresh, empty state for a new research goal."""
    return {
        "goal": goal,
        "plan": [],
        "log": [],
        "search_results": [],
        "evidence": [],
        "analysis": "",
        "draft": "",
        "critique": "",
        "status": "planning",
        "next_agent": "planner",
        "iterations": 0,
        "revisions": 0,
        "revise_queries": [],
    }


if __name__ == "__main__":
    s = new_state("Compare LangGraph and CrewAI for building multi-agent apps.")
    print("Created AtlasState with keys:")
    print("  " + ", ".join(s.keys()))
    print(f"status     : {s['status']}")
    print(f"next_agent : {s['next_agent']}")
    print("OK  state.py imports and builds a fresh state correctly.")
