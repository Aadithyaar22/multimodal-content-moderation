"""Google Sign-In verification for moderator-facing write actions.

Only POST /items/{id}/decision needs this — /analyze and /fact-check stay
public by design (the point of the rumour-checking pivot is that anyone can
check a claim with no account). A decision is different: it is a permanent
record of "a moderator looked at this and decided X", and until this it
trusted whatever `moderator_id` string a client happened to send in the
request body. Nothing verified it — anyone with the URL could submit a
decision as anyone, including as someone else's name.

This verifies a Google ID token — the credential Google Identity Services
hands the frontend after a real Google sign-in — against Google's own public
keys, using the same `google-auth` library `google-genai` already depends
on. No new client secret to manage, no session store, no password anywhere:
Google does the authentication, this only checks the token is genuinely
theirs, current, and issued for this app specifically.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import HTTPException, Request
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token

from mcm.utils.logging import get_logger

log = get_logger(__name__)

# One shared HTTP transport for fetching and caching Google's public signing
# keys across requests — the library refreshes it on its own schedule;
# constructing a fresh one per request would refetch the key set every time.
_transport = google_requests.Request()


def verify_google_token(token: str) -> dict[str, Any]:
    """Verify a Google ID token and return its claims, or raise ValueError.

    Checks the cryptographic signature against Google's current public keys,
    the issuer, and expiry (all handled by google-auth itself), plus that the
    token was issued for *this* app specifically once GOOGLE_CLIENT_ID is
    configured — without an audience check, any Google sign-in anywhere
    would verify, which defeats the point. An unconfigured GOOGLE_CLIENT_ID
    is treated as auth being unavailable, not as "accept anything": see
    docs/deployment.md for the one-time setup this requires in Google Cloud
    Console, which only the project owner can do.
    """
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    if not client_id:
        raise ValueError("GOOGLE_CLIENT_ID is not configured on this deployment")

    claims = google_id_token.verify_oauth2_token(token, _transport, audience=client_id)

    if not claims.get("email_verified"):
        raise ValueError("Google account email is not verified")

    return claims


def require_moderator(request: Request) -> dict[str, Any]:
    """FastAPI dependency: the verified moderator identity, or 401.

    Callers use the returned claims' "email" as the durable moderator id and
    "name" for display — never a client-supplied moderator_id, which
    DecisionRequest no longer even accepts as a field.
    """
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        raise HTTPException(401, "sign in required", headers={"WWW-Authenticate": "Bearer"})

    token = header[len("bearer ") :].strip()
    if not token:
        raise HTTPException(401, "sign in required", headers={"WWW-Authenticate": "Bearer"})

    try:
        return verify_google_token(token)
    except ValueError as e:
        log.warning("rejected decision: %s", e)
        raise HTTPException(
            401, "invalid or expired sign-in", headers={"WWW-Authenticate": "Bearer"}
        ) from e
