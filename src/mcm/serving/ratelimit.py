"""A minimal per-client rate limit for /analyze.

/analyze is the expensive path — it decodes an upload, runs CLIP plus every
head, and (once an explanation is requested) a paid LLM call follows. It is
also the only endpoint a stranger can hit without any prior state. A public
Cloud Run URL with no limit on it is an open invitation to run up the bill or
degrade the demo for everyone else, so a limit belongs here even though this is
a decision-support tool rather than a security product.

No new dependency: Cloud Run's `--max-instances` cap already bounds the
worst case, and slowapi or an external limiter would be one more moving part
for a problem a dozen lines of Python fully solves at this scale. If this
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

WINDOW_SECONDS = 60
MAX_REQUESTS_PER_WINDOW = 20

_lock = threading.Lock()
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


def enforce_rate_limit(request: Request) -> None:
    """FastAPI dependency: raises 429 once a client exceeds the window."""
    key = _client_key(request)
    now = time.monotonic()

    with _lock:
        hits = _hits[key]
        while hits and now - hits[0] > WINDOW_SECONDS:
            hits.popleft()

        if len(hits) >= MAX_REQUESTS_PER_WINDOW:
            retry_after = max(1, int(WINDOW_SECONDS - (now - hits[0])))
            raise HTTPException(
                status_code=429,
                detail="too many requests; slow down",
                headers={"Retry-After": str(retry_after)},
            )

        hits.append(now)

        # Bounded memory: a long-running instance seeing many distinct callers
        # must not accumulate one deque per IP forever.
        if len(_hits) > 10_000:
            _hits.clear()
