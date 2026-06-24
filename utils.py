"""
utils.py - small, pure helpers shared across Atlas's agents: clipping text,
parsing JSON out of an LLM reply, and rendering evidence as a citable list.
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


def source_order(evidence) -> list:
    """Unique source URLs in first-seen order; their positions are the [n] citation numbers."""
    order, seen = [], set()
    for e in evidence or []:
        url = (e.get("source_url") or "").strip()
        if url and url not in seen:
            seen.add(url)
            order.append(url)
    return order


def numbered_evidence(evidence) -> str:
    """Group claims under their source's citation number, so the LLM cites by [n] = source."""
    order = source_order(evidence)
    num = {url: i for i, url in enumerate(order, 1)}
    grouped = {}
    for e in evidence or []:
        n = num.get((e.get("source_url") or "").strip())
        if n:
            grouped.setdefault(n, []).append(str(e.get("claim", "")).strip())
    if not grouped:
        return "(no evidence gathered)"
    lines = []
    for n in range(1, len(order) + 1):
        if n not in grouped:
            continue
        lines.append(f"[{n}] {order[n - 1]}")
        lines.extend(f"    - {claim}" for claim in grouped[n])
    return "\n".join(lines)
