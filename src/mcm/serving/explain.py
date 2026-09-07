"""LLM narrative generation.

Compute first, explain second. The scores are the verdict; this runs afterwards
on its own endpoint and must never gate them. If it fails, is unconfigured, or
times out, the item still has everything a moderator needs to decide.

The prompt is given the *computed* numbers and asked to explain them, never to
judge the content itself. An LLM asked to decide would be a second, unmeasured
classifier sitting in front of the one the report actually evaluates — and its
disagreements would be invisible.
"""

from __future__ import annotations

import os
import time
from typing import Any

from mcm.utils.logging import get_logger

log = get_logger(__name__)

HEAD_NAME = {"toxicity": "harassment/hate-speech", "misinformation": "misinformation"}

SYSTEM_PROMPT = """You explain the output of a multimodal content-moderation model to a human moderator.

You are given scores the model already computed. Your job is to explain what \
those numbers mean for this specific item, not to re-judge the content or \
substitute your own verdict.

Rules:
- Never state a conclusion the scores do not support. If the fused score is 0.6, \
that is uncertain, and your language must read as uncertain.
- When the item is emergent (both single-modality scores low, fused score high), \
say plainly what the image and the caption each contribute and why they matter \
together. That relationship is the finding.
- This item may be flagged on more than one independent ground — for instance \
both harassment and misinformation can each clear their own threshold on the \
same item. When more than one is listed below, address EACH one specifically, \
in proportion to how strongly it scored. Do not silently drop any of them or \
imply only the first is real: a moderator who reads your summary and misses a \
genuine harassment concern because you only discussed misinformation has been \
actively misled, not just given an incomplete answer.
- Do not moralise, and do not address the person who posted. You are writing for \
a moderator who will decide.
- Plain English. No headings, no bullet points. Three or four sentences if there \
is one concern; up to six if there is more than one — do not pad length when \
there is only one thing to say."""


def _user_prompt(payload: dict[str, Any]) -> str:
    heads: dict[str, dict[str, Any]] = payload["heads"]
    lines = [
        f"Caption: {payload['text'] or '(none)'}",
        f"Image present: {payload['has_image']}",
        f"Threshold: {payload['threshold']:.2f}",
        "",
    ]

    if len(heads) > 1:
        lines.append(
            f"This item is flagged on {len(heads)} independent grounds — "
            "address every one of them below, not just the strongest."
        )
        lines.append("")

    for task, h in heads.items():
        m = h["modality_scores"]
        lines.append(f"[{HEAD_NAME.get(task, task)}] predicted: {h['label']}")
        lines.append(f"  Vision-only score: {m['cv_only']:.2f}")
        lines.append(f"  Language-only score: {m['nlp_only']:.2f}")
        lines.append(f"  Fused score: {m['fusion']:.2f}")
        lines.append("")

    if payload.get("is_emergent"):
        lines.append(
            "This item is EMERGENT: neither modality alone crosses the "
            "threshold, but the fused score does."
        )
    if payload.get("top_tokens"):
        lines.append(
            "Most influential caption tokens: "
            + ", ".join(f"{t}({s:+.2f})" for t, s in payload["top_tokens"])
        )
    if payload.get("regions"):
        lines.append(
            "Most attended image regions: "
            + ", ".join(f"{r['label']}({r['score']:.2f})" for r in payload["regions"])
        )
    return "\n".join(lines)


def generate(payload: dict[str, Any], timeout: float = 20.0) -> dict[str, Any]:
    """Produce a narrative, or a structured failure.

    Returns the Explanation shape from docs/api.md. Never raises: a failure here
    must degrade the page, not break it.
    """
    started = time.perf_counter()
    prompt = _user_prompt(payload)

    for backend in (_try_gemini, _try_groq):
        try:
            result = backend(prompt, timeout)
        except Exception as e:  # noqa: BLE001
            log.warning("%s failed: %s", backend.__name__, e)
            continue
        if result:
            text, model = result
            return {
                "status": "ready",
                "narrative": text.strip(),
                "key_factors": _key_factors(payload),
                "model": model,
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }

    return {
        "status": "unavailable",
        "narrative": None,
        "key_factors": _key_factors(payload),
        "model": None,
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }


def _try_gemini(prompt: str, timeout: float) -> tuple[str, str] | None:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    from google import genai
    from google.genai import types

    # Without an explicit timeout this call has none: get_explanation is a sync
    # endpoint, and FastAPI runs sync handlers in a bounded thread pool, so a
    # single hung request can tie up a worker indefinitely and the "compute
    # first, explain second" contract stops holding under exactly the failure
    # it was meant to survive. Timeout is milliseconds in this SDK.
    client = genai.Client(
        api_key=key, http_options=types.HttpOptions(timeout=int(timeout * 1000))
    )
    # gemini-2.5-pro was retired for new API keys (see git history for the 404
    # that caught it). gemini-3.1-pro-preview was its confirmed replacement but
    # measured ~16s per explanation in production — well past this endpoint's
    # own "compute first, explain second" design intent, even though it stays
    # inside the 20s internal timeout. gemini-2.5-flash, checked live against
    # the same deployed service and the same prompt on 2026-09-06, is roughly
    # half the latency (~7-8s) with narrative quality that reads equally clear,
    # accurate and appropriately hedged on every case tried — a small, fixed
    # explanation-of-computed-scores task, not open-ended reasoning, so this is
    # exactly the kind of call a faster/cheaper tier should be adequate for.
    # gemini-3.1-flash does not exist under this name; do not reintroduce it
    # without checking the API's own error body for the current name first.
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    resp = client.models.generate_content(
        model=model,
        contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
    )
    return (resp.text or "", model)


def _try_groq(prompt: str, timeout: float) -> tuple[str, str] | None:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    from groq import Groq

    client = Groq(api_key=key, timeout=timeout)
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=320,
    )
    return (resp.choices[0].message.content or "", model)


def _key_factors(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Structured factors, derived from the computed scores rather than the LLM.

    These are the numbers themselves, so they stay correct and available even
    when no narrative could be generated. One triplet per active head — a
    two-head item gets two independent sets of factors, tagged so the UI can
    group them, rather than the second head's numbers being dropped the way
    they were before every head in payload["heads"] was included here.
    """
    factors: list[dict[str, Any]] = []
    for task, h in payload["heads"].items():
        m = h["modality_scores"]
        factors.append(
            {
                "modality": "image",
                "factor": "vision-only signal",
                "weight": round(m["cv_only"], 3),
                "head": task,
            }
        )
        factors.append(
            {
                "modality": "text",
                "factor": "language-only signal",
                "weight": round(m["nlp_only"], 3),
                "head": task,
            }
        )
        delta = m["fusion"] - max(m["cv_only"], m["nlp_only"])
        if delta > 0:
            factors.append(
                {
                    "modality": "cross",
                    "factor": "gain from modelling the pair jointly",
                    "weight": round(delta, 3),
                    "head": task,
                }
            )
    return factors
