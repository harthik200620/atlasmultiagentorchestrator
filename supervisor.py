"""
supervisor.py — the ROUTER at the center of Atlas.

In the supervisor multi-agent pattern, workers don't call each other. Instead,
every worker reports back to the supervisor, and the supervisor looks at the
shared state (here, the `status` field) and decides who runs next. That single
decision point is what makes the whole system easy to reason about — and it's
exactly where the Critic's revise-loop will plug in (Phase 3).

It also enforces a hard safety cap so the graph can never loop forever.

CONTRACT
  reads : state["status"], state["iterations"]
  writes: state["next_agent"], state["iterations"], state["log"]
"""

from __future__ import annotations

import config
from state import AtlasState

# Which worker handles each status. The worker that just ran sets the NEXT
# status; the supervisor turns that status into the next destination.
STATUS_TO_AGENT = {
    "planning": "planner",
    "searching": "searcher",
    "reading": "reader",
    "writing": "writer",
    "failed": "writer",   # degrade gracefully — let the Writer say "insufficient"
    "done": "end",
}


def supervisor(state: AtlasState) -> dict:
    iterations = state.get("iterations", 0) + 1
    status = state.get("status", "planning")

    # Safety net: never exceed the configured step cap (anti-infinite-loop).
    if iterations > config.MAX_ITERATIONS:
        return {
            "next_agent": "end",
            "iterations": iterations,
            "log": [
                f"Supervisor: hit safety cap ({config.MAX_ITERATIONS} steps) — stopping."
            ],
        }

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
    for st in ["planning", "searching", "reading", "writing", "done", "failed"]:
        out = supervisor({"status": st, "iterations": 0})
        print(f"status={st:10s} -> next_agent={out['next_agent']}")
    capped = supervisor({"status": "reading", "iterations": 999})
    print(f"over-cap            -> next_agent={capped['next_agent']} (safety stop)")
