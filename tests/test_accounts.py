"""Email/password accounts: hashing, token issuance, and the shape that
lets require_moderator treat a self-issued token identically to a Google
one. The real hashing algorithm (Argon2id) is argon2-cffi's own job to get
right — these tests pin this module's own logic: that a verified password
actually round-trips, that a wrong one is rejected without raising, and that
a malformed/unconfigured setup fails safely rather than crashing a request.
"""

from __future__ import annotations

import time

import pytest

from mcm.serving import accounts


class TestEmailValidation:
    def test_accepts_a_normal_address(self):
        assert accounts.valid_email("mod@example.com")

    def test_rejects_missing_at(self):
        assert not accounts.valid_email("mod.example.com")

    def test_rejects_missing_domain_dot(self):
        assert not accounts.valid_email("mod@example")

    def test_normalize_lowercases_and_strips(self):
        assert accounts.normalize_email("  Mod@Example.COM  ") == "mod@example.com"


class TestPasswordHashing:
    def test_correct_password_verifies(self):
        h = accounts.hash_password("correct horse battery staple")
        assert accounts.verify_password("correct horse battery staple", h)

    def test_wrong_password_does_not_verify(self):
        h = accounts.hash_password("correct horse battery staple")
        assert not accounts.verify_password("wrong password entirely", h)

    def test_hash_never_contains_the_plaintext(self):
        """A basic sanity check that this is actually hashing, not just
        storing or lightly encoding the password."""
        h = accounts.hash_password("my-actual-password-123")
        assert "my-actual-password-123" not in h

    def test_malformed_stored_hash_fails_closed_not_with_a_crash(self):
        """A stored value that isn't a real Argon2 hash (corrupted data, a
        future migration bug) must read as 'wrong password', never as a
        500 that takes the login endpoint down."""
        assert not accounts.verify_password("anything", "not-a-real-hash")

    def test_same_password_hashes_differently_each_time(self):
        """Argon2 salts automatically — two accounts with the same password
        must not produce comparably identical hashes in storage."""
        assert accounts.hash_password("shared-password") != accounts.hash_password(
            "shared-password"
        )


class TestTokens:
    def test_issued_token_verifies_and_round_trips_claims(self, monkeypatch):
        monkeypatch.setenv("JWT_SECRET", "test-secret")
        token = accounts.issue_token("mod@example.com", "Test Mod")
        claims = accounts.verify_token(token)
        assert claims["email"] == "mod@example.com"
        assert claims["name"] == "Test Mod"

    def test_no_secret_configured_raises_on_issue(self, monkeypatch):
        monkeypatch.delenv("JWT_SECRET", raising=False)
        with pytest.raises(ValueError, match="not configured"):
            accounts.issue_token("mod@example.com", "Test Mod")

    def test_no_secret_configured_raises_on_verify(self, monkeypatch):
        monkeypatch.delenv("JWT_SECRET", raising=False)
        with pytest.raises(ValueError):
            accounts.verify_token("whatever")

    def test_garbage_token_raises_value_error_not_a_jwt_specific_exception(self, monkeypatch):
        """require_moderator only knows to catch ValueError from both
        verify_google_token and verify_token — a raw PyJWTError leaking
        through here would bypass that and surface as a 500."""
        monkeypatch.setenv("JWT_SECRET", "test-secret")
        with pytest.raises(ValueError):
            accounts.verify_token("not.a.real.token")

    def test_token_signed_with_a_different_secret_is_rejected(self, monkeypatch):
        monkeypatch.setenv("JWT_SECRET", "secret-a")
        token = accounts.issue_token("mod@example.com", "Test Mod")

        monkeypatch.setenv("JWT_SECRET", "secret-b")
        with pytest.raises(ValueError):
            accounts.verify_token(token)

    def test_expired_token_is_rejected(self, monkeypatch):
        monkeypatch.setenv("JWT_SECRET", "test-secret")
        monkeypatch.setattr(accounts, "JWT_EXPIRY_SECONDS", -1)
        token = accounts.issue_token("mod@example.com", "Test Mod")
        time.sleep(0.01)
        with pytest.raises(ValueError):
            accounts.verify_token(token)

    def test_token_from_a_different_issuer_is_rejected(self, monkeypatch):
        """Pins that verify_token actually checks the issuer claim, not
        just the signature — a token forged for a different app entirely
        (but somehow signed with the same secret) must not verify."""
        import jwt as pyjwt

        monkeypatch.setenv("JWT_SECRET", "test-secret")
        now = int(time.time())
        forged = pyjwt.encode(
            {"email": "x@y.com", "name": "X", "iat": now, "exp": now + 60, "iss": "someone-else"},
            "test-secret",
            algorithm="HS256",
        )
        with pytest.raises(ValueError):
            accounts.verify_token(forged)
