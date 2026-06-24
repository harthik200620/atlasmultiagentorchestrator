"""
graph.py - assembles the worker agents into a LangGraph supervisor graph and
exposes a CLI to run a research goal end to end.

Topology is hub-and-spoke: START -> supervisor -> one of the workers (planner,
searcher, reader, analyst, writer, critic) -> back to the supervisor, repeating
until END. The Critic can send work back, so the supervisor loops on more
searching or a rewrite until the Critic approves or MAX_ITERATIONS revisions are
used. If Langfuse keys are set, the whole run is recorded as one trace tree.
"""

from __future__ import annotations

import sys

from langgraph.graph import END, START, StateGraph

import tracing
from analyst import analyst
from critic import critic
from planner import planner
from reader import reader
from searcher import searcher
from state import AtlasState, new_state
from supervisor import MAX_SUPERVISOR_STEPS, route, supervisor
from writer import writer

WORKERS = {
    "planner": planner,
    "searcher": searcher,
    "reader": reader,
    "analyst": analyst,
    "writer": writer,
    "critic": critic,
}

# Give LangGraph's own recursion guard plenty of room above our supervisor cap.
RECURSION_LIMIT = 2 * MAX_SUPERVISOR_STEPS + 20


def build_graph():
    """Wire the supervisor and workers into a compiled LangGraph."""
    builder = StateGraph(AtlasState)

    builder.add_node("supervisor", supervisor)
    for name, fn in WORKERS.items():
        builder.add_node(name, fn)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route,
        {**{name: name for name in WORKERS}, "end": END},
    )
    for name in WORKERS:
        builder.add_edge(name, "supervisor")

    return builder.compile()


def _stream(graph, init_state: AtlasState, cfg: dict, verbose: bool) -> AtlasState:
    """Run the graph, printing each new log line as it appears, and return the
    final accumulated state (stream_mode='values' yields full snapshots)."""
    final_state: AtlasState = init_state
    printed = 0
    for snapshot in graph.stream(init_state, cfg, stream_mode="values"):
        final_state = snapshot
        log = snapshot.get("log") or []
        for line in log[printed:]:
            if verbose:
                print(f"  - {line}")
        printed = len(log)
    return final_state


def run(goal: str, verbose: bool = True, trace: bool = True) -> AtlasState:
    """Run Atlas on a goal. If Langfuse is configured and trace=True, the whole
    run is captured as one named trace and the trace URL is printed. The eval
    passes trace=False, since it doesn't need traces and they add export
    overhead."""
    graph = build_graph()
    init_state = new_state(goal)
    cfg: dict = {"recursion_limit": RECURSION_LIMIT}

    handler = tracing.callback_handler() if trace else None
    if handler is not None:
        cfg["callbacks"] = [handler]

    client = tracing.langfuse_client() if trace else None
    if client is None:
        return _stream(graph, init_state, cfg, verbose)

    # Wrap the whole run in one named trace so every agent and LLM call nests
    # underneath it.
    with client.start_as_current_observation(
        name="atlas-research", as_type="chain", input=goal
    ) as span:
        final_state = _stream(graph, init_state, cfg, verbose)
        try:  # best-effort: label the trace with the result and some stats
            span.update_trace(
                input=goal,
                output=final_state.get("draft", ""),
                metadata={
                    "revisions": final_state.get("revisions", 0),
                    "evidence_items": len(final_state.get("evidence") or []),
                    "final_status": final_state.get("status"),
                },
            )
        except Exception:  # noqa: BLE001 - never let tracing break a run
            pass
        trace_id = client.get_current_trace_id()

    client.flush()
    url = client.get_trace_url(trace_id=trace_id) if trace_id else None
    if url:
        final_state["_trace_url"] = url
        if verbose:
            print(f"\n[langfuse] trace -> {url}")
    return final_state


def _print_summary(state: AtlasState) -> None:
    print("\n" + "=" * 64)
    print("FINAL BRIEF")
    print("=" * 64)
    print(state.get("draft") or "(no draft produced)")
    print("\n" + "-" * 64)
    print(f"sources gathered : {len(state.get('search_results') or [])}")
    print(f"evidence items   : {len(state.get('evidence') or [])}")
    print(f"revisions made   : {state.get('revisions', 0)}")
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
