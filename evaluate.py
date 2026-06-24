"""
evaluate.py — measure how well Atlas does, with an LLM-as-judge.

For each goal in eval_goals.GOALS it:
  1. runs Atlas end to end,
  2. asks a judge LLM whether the brief CORRECTLY answers the goal AND is sourced,
  3. records the verdict.
Then it saves everything to results/ and prints the task-success rate — the
headline number for your resume.

Usage:
  python evaluate.py                 # ALL 40 goals (slow; hundreds of LLM calls)
  python evaluate.py --limit 5       # just the first 5 (use this to validate first)
  python evaluate.py --start 10 --limit 5
  python evaluate.py --verbose       # also show each agent's log lines

Tip: lower the revise cap for a faster/cheaper eval, e.g. (PowerShell)
  $env:ATLAS_MAX_ITERATIONS=1 ; python evaluate.py

NOTE: the judge uses the same model family as Atlas (Gemini), so there is some
self-evaluation bias. For a more independent score, set LLM_PROVIDER to a
different provider for the judge, or spot-check the saved reasoning by hand.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import config
from eval_goals import GOALS
from graph import run as run_atlas
from utils import extract_json, truncate

JUDGE_PROMPT = """You are a strict evaluator of research briefs.

Given a research GOAL and the BRIEF an automated researcher produced, decide how
well the brief answers the goal AND how well it is sourced.

Score 1-5:
  5 = fully answers the goal, accurate, key claims cited with real URLs
  4 = answers the goal well, mostly cited, only minor gaps
  3 = partially answers, or thin/uneven sourcing
  2 = largely fails to answer, or mostly uncited
  1 = wrong, empty, or fabricated

Return ONLY JSON:
{{"score": <1-5>, "sourced": true/false, "reasoning": "one or two sentences"}}

GOAL: {goal}

BRIEF:
{brief}
"""


def judge(goal: str, brief: str) -> dict:
    """Score one brief. success is derived from score (>=4) for consistency."""
    if not brief.strip():
        return {"score": 1, "success": False, "sourced": False, "reasoning": "Empty brief."}

    llm = config.get_llm(temperature=0.0)  # low temp -> consistent judging
    for attempt in range(2):
        try:
            resp = llm.invoke(JUDGE_PROMPT.format(goal=goal, brief=truncate(brief, 6000)))
            v = extract_json(getattr(resp, "content", "")) or {}
            if isinstance(v, dict) and "score" in v:
                score = int(v.get("score", 1))
                return {
                    "score": score,
                    "success": score >= 4,
                    "sourced": bool(v.get("sourced", False)),
                    "reasoning": str(v.get("reasoning", "")).strip(),
                }
        except Exception as err:  # noqa: BLE001
            print(f"  [judge] attempt {attempt + 1} failed: {type(err).__name__}")
            time.sleep(1)
    return {
        "score": 0, "success": False, "sourced": False,
        "reasoning": "judge could not score (parse/LLM error)", "judge_error": True,
    }


def evaluate(goals, verbose_runs=False):
    results = []
    for i, goal in enumerate(goals, 1):
        print(f"\n[{i}/{len(goals)}] {goal}")
        t0 = time.time()
        try:
            state = run_atlas(goal, verbose=verbose_runs, trace=False)
            brief = state.get("draft", "") or ""
            meta = {
                "sources": len(state.get("search_results") or []),
                "evidence": len(state.get("evidence") or []),
                "revisions": state.get("revisions", 0),
                "status": state.get("status"),
                "trace_url": state.get("_trace_url"),
            }
        except Exception as err:  # noqa: BLE001 — one bad goal must not abort the eval
            print(f"  RUN FAILED: {type(err).__name__}: {err}")
            brief, meta = "", {"error": f"{type(err).__name__}: {err}"}

        verdict = judge(goal, brief)
        elapsed = round(time.time() - t0, 1)
        mark = "PASS" if verdict["success"] else "FAIL"
        print(f"  -> {mark}  score={verdict['score']}/5  ({elapsed}s)  {verdict['reasoning'][:100]}")
        results.append({"goal": goal, **verdict, **meta, "seconds": elapsed, "brief": brief})
    return results


def summarize_and_save(results) -> float:
    graded = [r for r in results if not r.get("judge_error")]
    n = len(results)
    passed = sum(1 for r in results if r.get("success"))
    rate = (passed / n * 100) if n else 0.0
    avg_score = (sum(r.get("score", 0) for r in graded) / len(graded)) if graded else 0.0

    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", "eval_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total": n,
                "passed": passed,
                "rate_pct": round(rate, 1),
                "avg_score": round(avg_score, 2),
                "max_iterations": config.MAX_ITERATIONS,
                "model": f"{config.LLM_PROVIDER}:{config.LLM_MODEL}",
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 64)
    print("ATLAS EVALUATION")
    print("=" * 64)
    print(f"Task success : {passed}/{n}  =  {rate:.1f}%")
    print(f"Avg score    : {avg_score:.2f} / 5")
    print(f"Revise cap    : {config.MAX_ITERATIONS}   Model: {config.LLM_PROVIDER}:{config.LLM_MODEL}")
    print(f"Results saved: {path}")
    return rate


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate Atlas with an LLM-as-judge.")
    ap.add_argument("--limit", type=int, default=None, help="run only the first N goals")
    ap.add_argument("--start", type=int, default=0, help="skip the first N goals")
    ap.add_argument("--verbose", action="store_true", help="show each agent's log lines")
    args = ap.parse_args()

    goals = GOALS[args.start:]
    if args.limit:
        goals = goals[: args.limit]

    print(f"Evaluating Atlas on {len(goals)} goal(s)  (revise cap = {config.MAX_ITERATIONS})...")
    results = evaluate(goals, verbose_runs=args.verbose)
    summarize_and_save(results)
