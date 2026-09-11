"""Live, web-search-grounded claim checking.

Compute first, ground second — the same contract as explain.py, on its own
endpoint, never gating the verdict: unconfigured or failing degrades to
status="unavailable", and a moderator (or anyone else) still has the full
verdict and scores regardless of whether this ran.

What this adds beyond explain.py's narrative: explain.py explains the scores
the model *already computed*, using only what the model already knows —
useful, but it cannot tell anyone whether the post is actually true. This
module asks a different, genuinely independent question: does the caption's
own claim hold up against current, real search results, right now. The
misinformation head predicts a learned *pattern* (recycled image plus
urgency framing looks like past misinformation); this checks the specific
claim itself. They are meant to disagree sometimes — a caption can match no
known misinformation pattern and still be checkably false, or match the
pattern and still turn out to be true.

Gemini's search-grounding tool is the only backend wired up so far. Groq does
not expose the same built-in web-search tool on the models this project
otherwise calls, so there is no fallback here yet — an unset GEMINI_API_KEY,
or Gemini being unreachable, degrades straight to "unavailable" rather than
silently trying a model with no search capability and returning an
ungrounded guess dressed up as a checked claim.
"""

from __future__ import annotations

import os
import time
from typing import Any

from mcm.utils.logging import get_logger

log = get_logger(__name__)

_VALID_VERDICTS = {"supported", "contradicted", "unclear", "no_factual_claim"}

SYSTEM_PROMPT = """You check whether the factual claim in a social-media caption \
holds up against current, real information. You have a live web search tool — \
use it before answering; never answer from memory alone.

Your response MUST start with exactly one line in the form:
VERDICT: supported
VERDICT: contradicted
VERDICT: unclear
VERDICT: no_factual_claim

- no_factual_claim: the caption makes no checkable factual assertion (opinion, \
joke, generic content with nothing to verify).
- supported: multiple credible, current sources corroborate the specific claim.
- contradicted: credible sources contradict the claim, or a fact it depends on \
(for instance, the claimed date or location of an image).
- unclear: search did not turn up enough recent, credible material to decide \
either way — say so plainly rather than guessing.

After the VERDICT line, state in your first sentence exactly which claim you \
checked, then write 2-4 more sentences explaining what your sources actually \
say and why. Never state a conclusion your search results do not support, and \
never invent a source. This is an independent check — the moderation model's \
own prediction below is context only, not something to defer to."""


def _user_prompt(payload: dict[str, Any]) -> str:
    lines = [f"Caption: {payload.get('text') or '(none)'}"]
    if payload.get("ocr_text"):
        lines.append(f"Text visible in the image: {payload['ocr_text']}")
    if payload.get("misinformation_label"):
        lines.append(
            "For context only — the moderation model separately scored this "
            f"{payload['misinformation_label']} on misinformation "
            f"({payload.get('misinformation_score') or 0:.2f}); do not defer to it."
        )
    return "\n".join(lines)


def generate(payload: dict[str, Any], timeout: float = 25.0) -> dict[str, Any]:
    """Check the caption's claim against live search, or a structured failure.

    Returns the FactCheck shape from docs/api.md. Never raises: a failure here
    must degrade the page, not break it. Slower than explain.generate by
    design — a real search round-trip costs more than explaining a number the
    model already computed — so this lives on its own endpoint the client
    calls only when someone actually wants a claim checked, not on every
    analysis.
    """
    started = time.perf_counter()
    prompt = _user_prompt(payload)

    try:
        result = _try_gemini_grounded(prompt, timeout)
    except Exception as e:  # noqa: BLE001
        log.warning("grounded fact-check failed: %s", e)
        result = None

    if not result:
        return {
            "status": "unavailable",
            "verdict": None,
            "summary": None,
            "sources": [],
            "model": None,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }

    text, sources, model = result
    verdict, summary = _parse(text)
    return {
        "status": "ready",
        "verdict": verdict,
        "summary": summary,
        "sources": sources,
        "model": model,
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }


def _try_gemini_grounded(prompt: str, timeout: float) -> tuple[str, list[dict], str] | None:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    from google import genai
    from google.genai import types

    # Same "must have an explicit timeout" reasoning as explain.py's Gemini
    # call: a sync endpoint on FastAPI's bounded thread pool cannot afford a
    # hung request. A grounded call does a real search round-trip on top of
    # generation, so this endpoint's own timeout budget (see app.py) is
    # already set higher than explain.py's.
    client = genai.Client(
        api_key=key, http_options=types.HttpOptions(timeout=int(timeout * 1000))
    )
    model = os.getenv("GEMINI_FACTCHECK_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    resp = client.models.generate_content(
        model=model,
        contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            # The SDK enables automatic function calling by default whenever
            # any tool is present, including the built-in search tool this
            # call never declares a Python function for — there is nothing
            # for AFC to call back into, so disabling it silences a
            # misleading warning rather than changing any real behavior.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    return (resp.text or "", _sources(resp), model)


def _sources(resp: Any) -> list[dict]:
    """Citations the model actually used, or an empty list.

    grounding_metadata is absent entirely when the model answered without
    searching (a plausible, correct outcome for a caption with no claim to
    check) — that is not a failure, just nothing to cite.
    """
    try:
        candidates = resp.candidates or []
        if not candidates:
            return []
        gm = candidates[0].grounding_metadata
        if not gm or not gm.grounding_chunks:
            return []
        out = []
        for chunk in gm.grounding_chunks:
            web = getattr(chunk, "web", None)
            if web and web.uri:
                out.append(
                    {"title": web.title or web.domain or web.uri, "url": web.uri, "domain": web.domain}
                )
        return out
    except Exception:  # noqa: BLE001
        # Malformed or unexpected response shape must degrade the citation
        # list, not take the whole fact-check down with it — the summary text
        # is still useful without its sources.
        log.exception("could not extract grounding sources")
        return []


def _parse(text: str) -> tuple[str, str]:
    """Split the mandated VERDICT line from the explanation that follows.

    Defaults to "unclear" with the full raw text as the summary if the model
    didn't follow the format — a moderator seeing an odd-shaped but present
    answer is better than a 500, and "unclear" is the honest default when this
    module can't even tell what verdict was meant.
    """
    text = text.strip()
    first_line, _, rest = text.partition("\n")
    label, sep, value = first_line.partition(":")
    if sep and label.strip().lower() == "verdict":
        token = value.strip().lower()
        if token in _VALID_VERDICTS:
            return token, rest.strip() or text
    log.warning("fact-check response missing a valid VERDICT line; defaulting to unclear")
    return "unclear", text
