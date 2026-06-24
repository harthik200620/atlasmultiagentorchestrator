# Atlas — a multi-agent research orchestrator

Atlas takes a research goal — *"Compare X and Y"*, *"What's the state of Z in 2026?"* —
and runs a **team of specialized AI agents** that plan, search the web, read sources,
synthesize findings, write a **cited** brief, and critique it until it's good enough.

Built with **LangGraph** using the **supervisor / multi-agent** pattern.

---

## The idea (multi-agent, in one picture)

```
            ┌─────────────┐
   goal ──▶ │ SUPERVISOR  │ ◀──────────  shared STATE  ──────────┐
            │  (router)   │     goal, plan, evidence, draft …     │
            └──────┬──────┘                                       │
                   │  picks the next worker based on `status`     │
   ┌───────┬───────┼────────┬─────────┬─────────┐                 │
   ▼       ▼       ▼        ▼         ▼         ▼                 │
 Planner Searcher Reader  Analyst   Writer    Critic ─────────────┘
 (steps) (Tavily) (facts) (synth)  (draft)  (approve / send back)
```

- **Shared state** — one object every agent reads and writes (`state.py`).
- **Supervisor** — inspects the state and routes to the next worker, with a hard
  iteration cap so it can never loop forever (`supervisor.py`, Phase 2+).
- **Workers** — small single-purpose agents: Planner, Searcher, Reader, Analyst,
  Writer, Critic.
- **Critic loop** — the Critic either approves the draft or sends it back with
  feedback; the supervisor revises until approved or capped.
- **Sourcing rule** — every claim in the final report cites a source URL; if
  evidence is thin, Atlas says so honestly.

---

## Tech stack

| Concern        | Tool                                                  |
|----------------|-------------------------------------------------------|
| Orchestration  | LangGraph (custom supervisor pattern)                 |
| LLM            | Gemini 2.5 Flash (swappable → gpt-4o-mini / Claude)   |
| Web search     | Tavily (search + page extraction)                     |
| Tracing        | Langfuse                                              |
| Evaluation     | 40-goal test set + LLM-as-judge                       |
| UI             | Streamlit                                             |

---

## Setup

1. **Python 3.11** + a virtual environment (already created in `.venv` if you used the build steps):
   ```powershell
   py -V:Astral/CPython3.11.15 -m venv .venv      # Windows (this machine)
   .\.venv\Scripts\Activate.ps1                    # activate (PowerShell)
   # source .venv/bin/activate                     # macOS / Linux
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Add your keys:** copy the template and fill it in.
   ```powershell
   Copy-Item .env.example .env
   ```
   You'll need a `TAVILY_API_KEY` and one LLM key (`GOOGLE_API_KEY` by default).
   Get them free at: [Tavily](https://tavily.com) · [Google AI Studio](https://aistudio.google.com/apikey).
4. **Verify your setup:**
   ```bash
   python config.py        # prints config + which keys are present
   python state.py         # smoke-tests the shared state
   ```

---

## Project layout

```
atlas/
├── config.py          # all env/config in one place + the LLM factory
├── state.py           # the shared AtlasState (the "blackboard")
├── requirements.txt
├── .env.example       # template — copy to .env (which is gitignored)
├── .gitignore
└── README.md
```
More files arrive each phase: `tools.py`, the worker agents, `supervisor.py`,
`graph.py`, `evaluate.py`, `app.py`.

---

## Build progress

- [x] **Phase 0** — repo, config, shared state, git. *(done)*
- [ ] **Phase 1** — tools + Searcher / Reader / Writer agents (each runnable alone).
- [ ] **Phase 2** — minimal supervisor graph (Planner → Searcher → Reader → Writer) + CLI.
- [ ] **Phase 3** — Analyst + Critic + revise-loop + retries/memory.
- [ ] **Phase 4** — Langfuse tracing.
- [ ] **Phase 5** — `evaluate.py` (40 goals, LLM-as-judge) + success rate.
- [ ] **Phase 6** — Streamlit UI + final writeup.

---

## My result

> Atlas scored **__ / 40  (= __%)** task-success on the LLM-as-judge eval.
> _(Filled in at Phase 5 — this is the number for your resume.)_
