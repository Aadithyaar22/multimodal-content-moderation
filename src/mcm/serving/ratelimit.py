"""Minimal per-client rate limits for the service's costly endpoints.

/analyze and /fact-check are the two paths a stranger can hit without any
prior state that also cost real money: /analyze decodes an upload and runs
CLIP plus every head; /fact-check is a live web-search round-trip on top of
generation (~5-10s observed, genuinely billed per call). A public Cloud Run
URL with no limit on either is an open invitation to run up the bill or
degrade the demo for everyone else, so both get a limit even though this is
a decision-support tool rather than a security product.

/fact-check shipped with no limit at all — a real gap, not a deliberate
choice, closed here. It gets its own, tighter bucket rather than sharing
/analyze's: a single fact-check costs meaningfully more than a single
analyze, and a shared bucket would mean a burst of one silently starves the
other's budget for the same client, which reads as a bug to whoever hits it.

No new dependency: Cloud Run's `--max-instances` cap already bounds the
worst case, and slowapi or an external limiter would be one more moving part
for a problem a few dozen lines of Python fully solves at this scale. If this
service ever sits behind more than a handful of concurrent instances, replace
this with a shared store (Redis, or the same MongoDB already optionally
configured) — a per-process window under-counts once there is more than one
instance, which is a real limitation, not an oversight.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

# /analyze: a CLIP forward pass plus every head. Generous — this is the path
# a moderator's own normal workload runs through, not just a stranger's.
WINDOW_SECONDS = 60
MAX_REQUESTS_PER_WINDOW = 20

# /fact-check: a real search round-trip, billed per call, and slower per
# request than /analyze — five per minute is already more than a person
# checking claims by hand would issue, and leaves no realistic room for a
# script to run up the bill unnoticed.
FACT_CHECK_WINDOW_SECONDS = 60
FACT_CHECK_MAX_REQUESTS_PER_WINDOW = 5

# /auth/login and /auth/register: not billed like the two above, but the one
# place this service does credential verification, which makes it the one
# place a brute-force or account-enumeration script would actually aim at.
# 10 attempts per 5 minutes is generous for a person who fat-fingered their
# password twice, and hostile to a script trying passwords in a loop.
AUTH_WINDOW_SECONDS = 300
AUTH_MAX_REQUESTS_PER_WINDOW = 10

_lock = threading.Lock()
# Keyed by "{bucket}:{client}" so /analyze and /fact-check track independent
# budgets per caller in one shared structure — a single _hits.clear() (used
# by tests to reset state between cases) still clears both.
_hits: dict[str, deque[float]] = defaultdict(deque)


def _client_key(request: Request) -> str:
    # Cloud Run terminates TLS in front of the container and forwards the
    # original caller in X-Forwarded-For; request.client.host would otherwise
    # be the load balancer's address for every request, making the limit
    # global instead of per-caller.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _enforce(bucket: str, request: Request, max_requests: int, window_seconds: int) -> None:
    """Raise 429 once a client exceeds `bucket`'s own window."""
    key = f"{bucket}:{_client_key(request)}"
    now = time.monotonic()

    with _lock:
        hits = _hits[key]
        while hits and now - hits[0] > window_seconds:
            hits.popleft()

        if len(hits) >= max_requests:
            retry_after = max(1, int(window_seconds - (now - hits[0])))
            raise HTTPException(
                status_code=429,
                detail="too many requests; slow down",
                headers={"Retry-After": str(retry_after)},
            )

        hits.append(now)

        # Bounded memory: a long-running instance seeing many distinct
        # callers across both buckets must not accumulate one deque per
        # (bucket, IP) pair forever.
        if len(_hits) > 10_000:
            _hits.clear()


def enforce_rate_limit(request: Request) -> None:
    """FastAPI dependency for /analyze."""
    # Read the module-level constants at call time, not via a captured
    # default argument — tests monkeypatch MAX_REQUESTS_PER_WINDOW directly,
    # and a closure over the value at import time would not see that.
    _enforce("analyze", request, MAX_REQUESTS_PER_WINDOW, WINDOW_SECONDS)


def enforce_fact_check_rate_limit(request: Request) -> None:
    """FastAPI dependency for /fact-check."""
    _enforce(
        "fact_check", request, FACT_CHECK_MAX_REQUESTS_PER_WINDOW, FACT_CHECK_WINDOW_SECONDS
    )


def enforce_auth_rate_limit(request: Request) -> None:
    """FastAPI dependency for /auth/login and /auth/register."""
    _enforce("auth", request, AUTH_MAX_REQUESTS_PER_WINDOW, AUTH_WINDOW_SECONDS)
