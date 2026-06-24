"""
supervisor.py -the ROUTER at the center of Atlas.

In the supervisor multi-agent pattern, workers don't call each other. Every
worker reports back to the supervisor, which looks at the shared `status` and
decides who runs next. That single decision point is what makes the system easy
to reason about.

Phase 3: the supervisor now also drives the Critic's REVISE-LOOP. When the Critic
rejects a draft, the supervisor either:
  - routes back to the Searcher with the Critic's new queries (gather more), or
  - routes back to the Writer to rewrite addressing the critique,
and it keeps looping until the Critic approves OR we hit MAX_ITERATIONS revisions.

CONTRACT
  reads : state["status"], state["iterations"], state["revisions"], state["revise_queries"]
  writes: state["next_agent"], state["iterations"], state["revisions"],
          state["plan"], state["revise_queries"], state["status"], state["log"]
"""

from __future__ import annotations

import config
from state import AtlasState

# Which worker handles each forward status. The worker that just ran sets the
# NEXT status; the supervisor turns that status into the next destination.
STATUS_TO_AGENT = {
    "planning": "planner",
    "searching": "searcher",
    "reading": "reader",
    "analyzing": "analyst",
    "writing": "writer",
    "criticizing": "critic",
    "failed": "writer",   # degrade gracefully -let the Writer say "insufficient"
    "done": "end",
}

# Absolute backstop on supervisor visits (belt-and-suspenders against a runaway
# loop). The MEANINGFUL cap is MAX_ITERATIONS revise-rounds, handled below.
MAX_SUPERVISOR_STEPS = 12 + 12 * max(1, config.MAX_ITERATIONS)


def supervisor(state: AtlasState) -> dict:
    iterations = state.get("iterations", 0) + 1
    revisions = state.get("revisions", 0)
    status = state.get("status", "planning")

    if iterations > MAX_SUPERVISOR_STEPS:
        return {
            "next_agent": "end",
            "iterations": iterations,
            "log": [f"Supervisor: absolute step backstop ({MAX_SUPERVISOR_STEPS}) reached -stopping."],
        }

    # --- the Critic asked for a revision ---
    if status == "revise":
        if revisions >= config.MAX_ITERATIONS:
            return {
                "next_agent": "end",
                "iterations": iterations,
                "status": "done",
                "log": [f"Supervisor: max revisions ({config.MAX_ITERATIONS}) reached -accepting current draft."],
            }
        revisions += 1
        extra = state.get("revise_queries") or []
        if extra:
            # Evidence gap -> gather MORE using the Critic's new queries.
            return {
                "next_agent": "searcher",
                "iterations": iterations,
                "revisions": revisions,
                "plan": extra,
                "revise_queries": [],
                "status": "searching",
                "log": [f"Supervisor: revision {revisions}/{config.MAX_ITERATIONS} -gathering more evidence ({len(extra)} new queries)."],
            }
        # Evidence is fine -> just rewrite, addressing the critique.
        return {
            "next_agent": "writer",
            "iterations": iterations,
            "revisions": revisions,
            "status": "writing",
            "log": [f"Supervisor: revision {revisions}/{config.MAX_ITERATIONS} -rewriting to address the critique."],
        }

    # --- normal forward routing ---
    nxt = STATUS_TO_AGENT.get(status, "end")
    return {
        "next_agent": nxt,
        "iterations": iterations,
        "log": [f"Supervisor (step {iterations}): status='{status}' -> {nxt}"],
    }


def route(state: AtlasState) -> str:
    """Conditional-edge function: tell LangGraph which node to visit next."""
    return state.get("next_agent", "end")


if __name__ == "__main__":
    # Test the routing logic alone (no graph, no API):  python supervisor.py
    print("Forward routing:")
    for st in ["planning", "searching", "reading", "analyzing", "writing", "criticizing", "done"]:
        print(f"  status={st:12s} -> {supervisor({'status': st, 'iterations': 0})['next_agent']}")

    print("\nRevise routing:")
    r1 = supervisor({"status": "revise", "iterations": 5, "revisions": 0, "revise_queries": ["q1", "q2"]})
    print(f"  revise + new queries -> {r1['next_agent']} (revisions={r1['revisions']})")
    r2 = supervisor({"status": "revise", "iterations": 5, "revisions": 0})
    print(f"  revise, no queries   -> {r2['next_agent']} (revisions={r2['revisions']})")
    r3 = supervisor({"status": "revise", "iterations": 5, "revisions": config.MAX_ITERATIONS})
    print(f"  revise at cap        -> {r3['next_agent']} (status={r3.get('status')}, graceful stop)")
