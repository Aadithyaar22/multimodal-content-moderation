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
- Do not moralise, and do not address the person who posted. You are writing for \
a moderator who will decide.
- Three or four sentences. Plain English. No headings, no bullet points."""


def _user_prompt(payload: dict[str, Any]) -> str:
    m = payload["modality_scores"]
    task = payload["task"]
    lines = [
        f"Task: {task}",
        f"Caption: {payload['text'] or '(none)'}",
        f"Image present: {payload['has_image']}",
        "",
        f"Vision-only score: {m['cv_only']:.2f}",
        f"Language-only score: {m['nlp_only']:.2f}",
        f"Fused score: {m['fusion']:.2f}",
        f"Threshold: {payload['threshold']:.2f}",
    ]
    if payload.get("is_emergent"):
        lines.append(
            "\nThis item is EMERGENT: neither modality alone crosses the "
            "threshold, but the fused score does."
        )
    if payload.get("top_tokens"):
        lines.append(
            "\nMost influential caption tokens: "
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

    client = genai.Client(api_key=key)
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
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
    when no narrative could be generated.
    """
    m = payload["modality_scores"]
    factors = [
        {"modality": "image", "factor": "vision-only signal", "weight": round(m["cv_only"], 3)},
        {"modality": "text", "factor": "language-only signal", "weight": round(m["nlp_only"], 3)},
    ]
    delta = m["fusion"] - max(m["cv_only"], m["nlp_only"])
    if delta > 0:
        factors.append(
            {
                "modality": "cross",
                "factor": "gain from modelling the pair jointly",
                "weight": round(delta, 3),
            }
        )
    return factors
