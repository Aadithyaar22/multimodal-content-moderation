"""Email/password accounts — an alternative to Google Sign-In, not a replacement.

Some people don't have, or don't want to use, a Google account. This gives
POST /items/{id}/decision a second, independent way to authenticate:
register with an email and password, get back a self-issued token that
require_moderator (mcm.serving.auth) accepts exactly like a Google ID
token — same downstream code, same DecisionResponse.moderator_id shape.

Two things this deliberately does NOT do, both real limitations rather than
oversights:

No email verification. Registering with an email nobody controls succeeds —
there is no confirmation link, because sending one needs a transactional
email provider (SendGrid, SES, ...) this project doesn't have configured.
The moderator_id this produces is "whoever holds this password", not "a
verified email address" the way Google's is.

No password reset. Losing a password currently means losing the account —
the same missing piece (no way to send mail) is why.

Passwords are hashed with Argon2id (via argon2-cffi's PasswordHasher, whose
defaults follow OWASP's current recommendation) — never stored, never
logged, never returned in any response. A verification failure and a
"no such account" both produce the identical error message and status code
end-to-end (see app.py's login endpoint), so a client can't use the error to
enumerate which emails are registered.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from jwt import PyJWTError
from jwt import decode as jwt_decode
from jwt import encode as jwt_encode

from mcm.utils.logging import get_logger

log = get_logger(__name__)

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LENGTH = 8

JWT_ALGORITHM = "HS256"
JWT_ISSUER = "vanguard-moderation-api"
JWT_EXPIRY_SECONDS = 7 * 24 * 3600  # a week — long enough a moderator isn't
# re-authenticating mid-shift, short enough a leaked token ages out on its own

_hasher = PasswordHasher()


def normalize_email(email: str) -> str:
    return email.strip().lower()


def valid_email(email: str) -> bool:
    return bool(EMAIL_RE.match(email))


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except VerifyMismatchError:
        return False
    except Exception:  # noqa: BLE001
        # A malformed stored hash (shouldn't happen, but must not 500 a
        # login attempt) reads the same as a wrong password.
        log.exception("password verification failed on a malformed hash")
        return False


def _secret() -> str:
    secret = os.getenv("JWT_SECRET")
    if not secret:
        raise ValueError("JWT_SECRET is not configured on this deployment")
    return secret


def issue_token(email: str, name: str) -> str:
    now = int(time.time())
    payload = {
        "sub": email,
        "email": email,
        "name": name,
        "iat": now,
        "exp": now + JWT_EXPIRY_SECONDS,
        "iss": JWT_ISSUER,
    }
    return jwt_encode(payload, _secret(), algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> dict[str, Any]:
    """Verify a self-issued token and return its claims, or raise ValueError.

    Mirrors mcm.serving.auth.verify_google_token's contract exactly (same
    exception type on failure) so require_moderator can try both
    verification paths uniformly.
    """
    try:
        return jwt_decode(token, _secret(), algorithms=[JWT_ALGORITHM], issuer=JWT_ISSUER)
    except PyJWTError as e:
        raise ValueError(str(e)) from e
