"""
graph.py — assembles the worker agents into a LangGraph SUPERVISOR graph and
gives you a CLI to run a research goal end to end.

Topology (hub-and-spoke — the supervisor pattern):

        START -> supervisor -> (planner | searcher | reader | writer | END)
                     ^                |
                     |________________|     every worker reports back

Run it:
    python graph.py "your research goal here"
    python graph.py --mermaid          # print the graph diagram (paste into mermaid.live)
"""

from __future__ import annotations

import sys

from langgraph.graph import END, START, StateGraph

from planner import planner
from reader import reader
from searcher import searcher
from state import AtlasState, new_state
from supervisor import route, supervisor
from writer import writer

# name -> node function, for the worker agents
WORKERS = {
    "planner": planner,
    "searcher": searcher,
    "reader": reader,
    "writer": writer,
}


def build_graph():
    """Wire the supervisor + workers into a compiled LangGraph."""
    builder = StateGraph(AtlasState)

    builder.add_node("supervisor", supervisor)
    for name, fn in WORKERS.items():
        builder.add_node(name, fn)

    # Enter at the supervisor; it decides the first move.
    builder.add_edge(START, "supervisor")

    # The supervisor routes to the chosen worker (or to END).
    builder.add_conditional_edges(
        "supervisor",
        route,
        {**{name: name for name in WORKERS}, "end": END},
    )

    # Every worker reports straight back to the supervisor.
    for name in WORKERS:
        builder.add_edge(name, "supervisor")

    return builder.compile()


def run(goal: str, verbose: bool = True) -> AtlasState:
    """Run Atlas on a goal, printing each agent's log line as it happens."""
    graph = build_graph()
    final_state: AtlasState = new_state(goal)
    printed = 0

    # stream_mode="values" yields the FULL accumulated state after each step,
    # so we can show progress live and keep the last snapshot as the result.
    for snapshot in graph.stream(
        final_state, {"recursion_limit": 50}, stream_mode="values"
    ):
        final_state = snapshot
        log = snapshot.get("log") or []
        for line in log[printed:]:
            if verbose:
                print(f"  - {line}")
        printed = len(log)

    return final_state


def _print_summary(state: AtlasState) -> None:
    print("\n" + "=" * 64)
    print("FINAL BRIEF")
    print("=" * 64)
    print(state.get("draft") or "(no draft produced)")
    print("\n" + "-" * 64)
    print(f"sources gathered : {len(state.get('search_results') or [])}")
    print(f"evidence items   : {len(state.get('evidence') or [])}")
    print(f"supervisor steps : {state.get('iterations', 0)}")
    print(f"final status     : {state.get('status')}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--mermaid":
        print(build_graph().get_graph().draw_mermaid())
        sys.exit(0)

    goal = " ".join(sys.argv[1:]).strip() or input("Research goal: ").strip()
    if not goal:
        print("No goal provided.")
        sys.exit(1)

    print(f"\nATLAS researching: {goal}")
    print("=" * 64)
    result = run(goal)
    _print_summary(result)
