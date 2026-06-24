# 🧭 Atlas — a multi-agent research orchestrator

Atlas turns a research goal — *"Compare X and Y"*, *"What's the state of Z in 2026?"* —
into a **cited research brief**, produced by a team of specialized AI agents that plan,
search the web, read sources, synthesize, write, and **critique their own work** until it
meets a quality bar.

Built with **LangGraph** (custom **supervisor** multi-agent pattern), a **FastAPI**
streaming backend, and a **Next.js** frontend.

> **Live API:** **https://atlasmultiagentorchestrator.onrender.com** — try
> [`/health`](https://atlasmultiagentorchestrator.onrender.com/health). The `web/`
> frontend deploys to Vercel with `NEXT_PUBLIC_API_URL` pointed at this API.

---

## Why it's interesting

- **A real multi-agent system**, not one big prompt — a supervisor routes between six
  specialized agents over a shared state.
- **Every claim is sourced.** The brief cites a URL for each fact; when evidence is thin,
  Atlas says so instead of bluffing.
- **Self-correcting.** A Critic agent reviews each draft and can send it back for more
  research or a rewrite, bounded by a revision cap.
- **Production-shaped.** Distributed tracing (Langfuse), an automated evaluation
  (40 goals, LLM-as-judge), API-key rotation with failover, and a deployable
  frontend/backend split.

---

## What you can use it for

Atlas is a general-purpose **question → sourced brief** engine. Anywhere you'd otherwise
burn an afternoon on open tabs, it runs the search → read → synthesize → write loop for you:

- **Tech & architecture decisions** — "LangGraph vs CrewAI", "Postgres vs MongoDB for a
  high-write app", "REST vs GraphQL" → a cited trade-off brief instead of 20 tabs.
- **Market & competitive scans** — survey a product, vendor, or space with clickable sources.
- **State-of-the-field reviews** — "What's the state of X in 2026?", each claim tied to a URL.
- **Decision & due-diligence memos** — build-vs-buy, framework picks, technology bets.
- **Grounded explainers** — "What is RAG and how does it work?" backed by real sources, not
  the model's memory.
- **A reusable agent backbone** — repoint the supervisor + workers, the sourcing rule, and
  the self-critique loop at any domain (legal, medical, finance) by swapping prompts and tools.

**Why it helps**

- **Every claim is cited** — auditable, and it says *"insufficient evidence"* instead of bluffing.
- **It checks its own work** — the Critic loops weak drafts back for more research or a rewrite
  before you ever see them.
- **Hours → seconds** — it automates the manual search-read-write grind.
- **Model- & budget-flexible** — swap Gemini / GPT / Claude with one env var; key rotation
  squeezes throughput out of free tiers.
- **Observable & deployable** — live streaming UI, Langfuse traces per run, clean
  frontend/backend split.

---

## Architecture

### The agent graph (supervisor pattern)

```
            +-------------+
   goal --> | SUPERVISOR  | <-------- shared STATE ---------+
            |  (router)   |   goal, plan, evidence, draft   |
            +------+------+                                 |
                   |  routes by `status` (+ revise-loop)    |
   +-------+-------+--------+---------+----------+           |
   v       v       v        v         v          v          |
 Planner Searcher Reader  Analyst   Writer     Critic ------+
(queries)(Tavily) (facts) (synth) (cited draft)(approve/revise)
```

The Critic can route back to the **Searcher** (gather more evidence) or the **Writer**
(rewrite); the supervisor loops until the Critic approves or `MAX_ITERATIONS` revisions
are used. Only the supervisor makes routing decisions — workers just do one job and report
back.

### Deployment (why two hosts)

```
  Browser --> Next.js (Vercel)  --SSE-->  FastAPI + agents (Render)  --> Gemini / Tavily
```

The agent backend runs 45 s–2 min per request with heavy dependencies — too long and too
large for Vercel's serverless functions — so it lives on a persistent host (Render).
Vercel serves the fast static frontend.

---

## Tech stack

| Concern        | Tool |
|----------------|------|
| Orchestration  | LangGraph (custom supervisor pattern) |
| LLM            | Gemini 2.5 Flash (swappable → GPT-4o-mini / Claude) |
| Web search     | Tavily (search + page extraction) |
| Backend API    | FastAPI + Uvicorn (SSE streaming) |
| Frontend       | Next.js (App Router) + react-markdown |
| Tracing        | Langfuse |
| Evaluation     | 40-goal test set + LLM-as-judge |

---

## How it works (the pipeline)

1. **Planner** — breaks the goal into 3–5 focused search queries.
2. **Searcher** — runs them through Tavily and de-dupes sources.
3. **Reader** — extracts sourced **facts** (claim + URL) from each page (LLM).
4. **Analyst** — synthesizes the evidence: themes, contradictions, gaps.
5. **Writer** — writes a cited Markdown brief from the analysis.
6. **Critic** — approves, or sends it back with feedback / new queries.

All six share one `AtlasState`; the **supervisor** is the single decision point.

---

## Repository layout

```
atlas/
├── config.py        # env/config + LLM factory + rotating Gemini key pool
├── state.py         # AtlasState (the shared blackboard) + reducers
├── tools.py         # Tavily search + extract (retries, graceful degrade)
├── utils.py         # JSON rescue, truncation, evidence formatting
├── planner.py  searcher.py  reader.py  analyst.py  writer.py  critic.py
├── supervisor.py    # the router + revise-loop + safety cap
├── graph.py         # LangGraph wiring + CLI:  python graph.py "your goal"
├── tracing.py       # optional Langfuse tracing
├── evaluate.py      # LLM-as-judge evaluation over...
├── eval_goals.py    # ...40 varied research goals
├── server.py        # FastAPI backend (SSE) for the web UI
├── render.yaml      # Render deploy config (backend)
├── requirements.txt
└── web/             # Next.js frontend (deploys to Vercel)
    └── app/page.js  # UI: live agent steps + the cited brief
```

---

## Run it locally

**Backend** (Python 3.11):

```powershell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env       # then fill in your keys
python config.py check-keys       # verify the keys are live
python -m uvicorn server:app --port 8000
```

Run a single research from the CLI instead:
`python graph.py "Compare LangGraph and CrewAI for multi-agent systems"`

**Frontend** (Node 18+):

```powershell
cd web
npm install
Copy-Item .env.local.example .env.local    # points at http://127.0.0.1:8000
npm run dev                                 # open http://localhost:3000
```

### Keys you need (all have free tiers)

`TAVILY_API_KEY` (search) and a Gemini key — `GOOGLE_API_KEY` or numbered
`GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`, … (multiple keys from different Google projects
**rotate** for higher free-tier throughput). `LANGFUSE_*` is optional (tracing).

---

## Evaluation

```powershell
python -u evaluate.py             # all 40 goals -> results/eval_results.json + a score
python -u evaluate.py --limit 5   # a quick subset first
```

An LLM-as-judge scores each brief 1–5 (does it answer the goal? is it well-sourced?);
`success = score >= 4`. Every verdict's reasoning is saved alongside the score.

**Result:** **8 / 10 (80%)** task-success · avg judge score **4.1 / 5** — measured on a
40-goal sample with **Gemini 2.5 flash ** as the backing model (real Tavily search + the same
agent/judge prompts; `success = score ≥ 4`). Run the full 40-goal set on any configured
provider with `python evaluate.py`.

---

## Deploy

**Backend → Render**
New ► **Blueprint** ► import this repo (`render.yaml` is auto-detected) ► in the service's
**Environment** tab add your secrets (`GEMINI_API_KEY_*`, `TAVILY_API_KEY`, `LANGFUSE_*`) ►
deploy ► copy the URL.

**Frontend → Vercel**
Add New ► **Project** ► import this repo ► **set Root Directory to `web`** ► add an env var
`NEXT_PUBLIC_API_URL` = your Render URL ► deploy.
_(`NEXT_PUBLIC_*` is baked in at build time, so set it before building or redeploy after.
For tighter CORS, set `ALLOWED_ORIGINS` on Render to your Vercel URL.)_

---

## Engineering notes (things that bit, and the fixes)

- **Key rotation + fast failover.** Free-tier Gemini limits are low, so `RotatingGemini`
  round-robins across N keys and fails over on HTTP 429. The key detail: set the client's
  `max_retries=0` so a throttled key fails *immediately* to the next one instead of backing
  off ~60 s on the same key first — this alone cut per-goal eval time roughly **13×**.
- **Thinking off for speed.** `thinking_budget=0` on Gemini 2.5 Flash dropped per-call
  latency from ~24 s to ~2 s.
- **Streaming + tracing across threads.** Starlette drives the SSE generator across
  threadpool threads, which broke Langfuse's thread-local span context. Fix: run the agent
  in one dedicated worker thread and pipe its log lines out through a queue.

---

_LangGraph supervisor pattern — six specialized agents, one cited brief._
