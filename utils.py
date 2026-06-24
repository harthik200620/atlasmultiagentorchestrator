"""
utils.py — small shared helpers used across Atlas's agents.

Kept separate from tools.py (which talks to external services) because these are
pure, local helpers: clipping text, rescuing JSON out of an LLM reply, and
rendering evidence as a numbered list the LLM can cite by [n].
"""

from __future__ import annotations

import json
import re


def truncate(text: str, limit: int = 4000) -> str:
    """Clip long page text so we don't blow the LLM's context window (or budget)."""
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + " ...[truncated]"


def extract_json(text: str):
    """Best-effort parse of JSON that an LLM returned.

    LLMs often wrap JSON in ```json ... ``` fences or add a sentence around it.
    This strips fences, tries a direct parse, and as a last resort grabs the
    first [...] or {...} block. Returns the parsed object, or None if nothing
    parses (callers treat None as "no data" rather than crashing).
    """
    if not text:
        return None

    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()

    for candidate in (cleaned, text):
        try:
            return json.loads(candidate)
        except Exception:
            pass

    match = re.search(r"(\[.*\]|\{.*\})", cleaned, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            return None
    return None


def numbered_evidence(evidence) -> str:
    """Render evidence as a numbered list the LLM can cite by [n]."""
    lines = [
        f"[{i}] {e['claim']} (source: {e['source_url']})"
        for i, e in enumerate(evidence, 1)
    ]
    return "\n".join(lines) if lines else "(no evidence gathered)"
