"""Google Sign-In verification.

The real cryptographic check (signature against Google's live public keys)
is google-auth's own job, not retested here — these tests pin this module's
own logic: the audience/config gate, the email_verified gate, and the
Authorization header parsing, none of which google-auth does for us.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException, Request

from mcm.serving import auth


def _fake_request(header: str | None) -> Request:
    headers = [(b"authorization", header.encode())] if header else []
    scope = {"type": "http", "headers": headers}
    return Request(scope)


class TestVerifyGoogleToken:
    def test_no_client_id_configured_raises(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
        with pytest.raises(ValueError, match="not configured"):
            auth.verify_google_token("whatever")

    def test_unverified_email_is_rejected(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id")
        monkeypatch.setattr(
            auth.google_id_token,
            "verify_oauth2_token",
            lambda token, transport, audience=None: {
                "email": "someone@example.com",
                "email_verified": False,
            },
        )
        with pytest.raises(ValueError, match="not verified"):
            auth.verify_google_token("t")

    def test_valid_token_returns_claims(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id")
        monkeypatch.setattr(
            auth.google_id_token,
            "verify_oauth2_token",
            lambda token, transport, audience=None: {
                "email": "real.mod@gmail.com",
                "email_verified": True,
                "name": "Real Mod",
                "aud": audience,
            },
        )
        claims = auth.verify_google_token("t")
        assert claims["email"] == "real.mod@gmail.com"
        # The configured client id must actually be passed through as the
        # audience — accepting a token verified against no audience (or the
        # wrong one) would mean any Google sign-in anywhere verifies here.
        assert claims["aud"] == "test-client-id"

    def test_bad_signature_propagates_as_value_error(self, monkeypatch):
        """google-auth itself raises on a forged/expired token; that must
        reach require_moderator as a 401, not surface as a 500."""
        monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id")

        def raises(token, transport, audience=None):
            raise ValueError("Token expired")

        monkeypatch.setattr(auth.google_id_token, "verify_oauth2_token", raises)
        with pytest.raises(ValueError):
            auth.verify_google_token("t")


class TestRequireModerator:
    def test_missing_header_is_401(self):
        with pytest.raises(HTTPException) as exc:
            auth.require_moderator(_fake_request(None))
        assert exc.value.status_code == 401

    def test_non_bearer_header_is_401(self):
        with pytest.raises(HTTPException) as exc:
            auth.require_moderator(_fake_request("Basic dXNlcjpwYXNz"))
        assert exc.value.status_code == 401

    def test_empty_bearer_token_is_401(self):
        with pytest.raises(HTTPException) as exc:
            auth.require_moderator(_fake_request("Bearer "))
        assert exc.value.status_code == 401

    def test_valid_token_returns_claims(self, monkeypatch):
        monkeypatch.setattr(
            auth, "verify_google_token", lambda token: {"email": "real.mod@gmail.com"}
        )
        claims = auth.require_moderator(_fake_request("Bearer good-token"))
        assert claims["email"] == "real.mod@gmail.com"

    def test_verification_failure_is_401_not_500(self, monkeypatch):
        def raises(token):
            raise ValueError("bad token")

        monkeypatch.setattr(auth, "verify_google_token", raises)
        with pytest.raises(HTTPException) as exc:
            auth.require_moderator(_fake_request("Bearer bad-token"))
        assert exc.value.status_code == 401

    def test_bearer_prefix_is_case_insensitive(self, monkeypatch):
        monkeypatch.setattr(auth, "verify_google_token", lambda token: {"email": "x@y.com"})
        claims = auth.require_moderator(_fake_request("bearer good-token"))
        assert claims["email"] == "x@y.com"
