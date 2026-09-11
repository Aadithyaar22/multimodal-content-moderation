"""Integration tests for the FastAPI application.

tests/test_serving.py already covers the pure logic (the emergent rule, the
verdict bands, the Store). Nothing until now exercised the app itself — the
routing, the status codes, the request-to-response wiring, or that a decision
recorded through the API actually changes what the queue returns. A regression
in any of that would previously have shipped with every other test green.

The model bundle is never really loaded here. ``run_arms`` is monkeypatched to
a deterministic stand-in, so these tests run in milliseconds and do not depend
on network access or trained checkpoints being present. ``TestClient(app)``
used without a ``with`` block does not invoke the lifespan handler (verified
separately), so the background model-loading thread never starts.
"""

from __future__ import annotations

import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from mcm.serving import app as app_module
from mcm.serving import ratelimit
from mcm.serving.inference import ArmOutputs
from mcm.serving.store import Store


class FakeBundle:
    """Just enough of ModelBundle for analyze()'s bookkeeping, not its models."""

    def __init__(self):
        self.tasks = ["toxicity"]
        self.arms = {"toxicity": {}}
        self.deepfake = None
        self.ocr = None
        self.clip = _FakeClipForReuse()
        self.device = torch.device("cpu")
        self.ready = True


@pytest.fixture
def client(monkeypatch):
    # A fresh store and rate-limit window per test, so tests cannot see each
    # other's items or trip each other's limits depending on run order.
    monkeypatch.setattr(app_module, "_store", Store(uri=""))
    monkeypatch.setattr(app_module, "_images", {})
    ratelimit._hits.clear()

    monkeypatch.setitem(app_module._state, "bundle", FakeBundle())
    monkeypatch.setitem(app_module._state, "loading", False)
    monkeypatch.setitem(app_module._state, "error", None)

    def fake_run_arms(bundle, task, image, text):
        # A fixed, unremarkable score: present in every response but not
        # engineered to be emergent, so tests that don't care about the
        # emergent flag aren't accidentally exercising it.
        arms = ArmOutputs(
            cv_only=0.3, nlp_only=0.3, fusion=0.3, fusion_probs={"benign": 0.7, "harmful": 0.3}
        )
        return arms, {"encode": 1, "unimodal": 1, "fusion": 1}

    monkeypatch.setattr(app_module, "run_arms", fake_run_arms)

    return TestClient(app_module.app)


def _png_bytes(size=(8, 8), color=(120, 40, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


class _FakeClipForReuse:
    """Deterministic per-image embeddings without loading the real ~600MB
    CLIP model. The embedding is the image's mean RGB, padded to a plausible
    width — a solid-colour test fixture is fully described by that, so two
    identical-coloured images produce (near-)identical embeddings and two
    differently-coloured ones produce genuinely different directions, which
    is all _check_image_reuse's cosine-similarity path needs to be exercised
    for real rather than through the AttributeError-and-degrade path a
    FakeBundle with no .clip at all was silently taking before this existed.
    """

    def preprocess_images(self, images):
        import numpy as np

        arrs = [np.asarray(img.convert("RGB")).reshape(-1, 3).mean(axis=0) for img in images]
        return torch.tensor(np.array(arrs), dtype=torch.float32)

    def encode_pooled(self, pixel_values=None, text_inputs=None):
        from types import SimpleNamespace

        padded = torch.nn.functional.pad(pixel_values, (0, 16 - pixel_values.shape[-1]))
        return SimpleNamespace(image_emb=padded)


class TestHealth:
    def test_reports_loading_not_error(self, client, monkeypatch):
        monkeypatch.setitem(app_module._state, "bundle", None)
        monkeypatch.setitem(app_module._state, "loading", True)
        body = client.get("/api/v1/health").json()
        assert body["status"] == "ok"
        assert body["models_loaded"] is False
        assert body["error"] is None

    def test_reports_error_distinctly_from_loading(self, client, monkeypatch):
        """Without this distinction a client polling on models_loaded alone
        cannot tell a slow cold start from a deployment that will never come up."""
        monkeypatch.setitem(app_module._state, "bundle", None)
        monkeypatch.setitem(app_module._state, "loading", False)
        monkeypatch.setitem(app_module._state, "error", "checkpoint missing")
        body = client.get("/api/v1/health").json()
        assert body["status"] == "error"
        assert body["error"] == "checkpoint missing"

    def test_ready_bundle_reports_loaded(self, client):
        body = client.get("/api/v1/health").json()
        assert body["models_loaded"] is True
        assert body["status"] == "ok"


class TestAnalyzeValidation:
    def test_rejects_neither_text_nor_image(self, client):
        r = client.post("/api/v1/analyze", data={})
        assert r.status_code == 422

    def test_rejects_oversized_upload(self, client):
        huge = b"0" * (app_module.MAX_UPLOAD_BYTES + 1)
        r = client.post(
            "/api/v1/analyze",
            data={"text": "hello"},
            files={"image": ("big.png", huge, "image/png")},
        )
        assert r.status_code == 413

    def test_rejects_undecodable_image(self, client):
        r = client.post(
            "/api/v1/analyze",
            data={"text": "hello"},
            files={"image": ("not-an-image.png", b"definitely not png bytes", "image/png")},
        )
        assert r.status_code == 415

    def test_503_when_models_not_ready(self, client, monkeypatch):
        monkeypatch.setitem(app_module._state, "bundle", None)
        r = client.post("/api/v1/analyze", data={"text": "hello"})
        assert r.status_code == 503
        assert "Retry-After" in r.headers


class TestAnalyzeResponse:
    def test_text_only_response_shape(self, client):
        r = client.post("/api/v1/analyze", data={"text": "hello world"})
        assert r.status_code == 200
        body = r.json()

        # The one line in the whole contract that must never be anything else.
        assert body["verdict"]["auto_action"] is None

        assert body["input"]["has_image"] is False
        assert body["input"]["modalities"] == ["text"]
        # No image was supplied, so a CV-only reading would be the head's bias
        # on a null vector, not evidence — it must not be reported.
        assert "toxicity" not in body["modality_scores"]["cv_only"]
        assert "toxicity" in body["modality_scores"]["nlp_only"]

    def test_image_and_text_reports_both_arms(self, client):
        r = client.post(
            "/api/v1/analyze",
            data={"text": "hello"},
            files={"image": ("a.png", _png_bytes(), "image/png")},
        )
        body = r.json()
        assert body["input"]["has_image"] is True
        assert "toxicity" in body["modality_scores"]["cv_only"]
        assert "toxicity" in body["modality_scores"]["nlp_only"]

    def test_item_is_retrievable_afterwards(self, client):
        created = client.post("/api/v1/analyze", data={"text": "hello"}).json()
        fetched = client.get(f"/api/v1/items/{created['item_id']}").json()
        assert fetched["item_id"] == created["item_id"]
        assert fetched["status"] == "pending"

    def test_unknown_item_is_404(self, client):
        assert client.get("/api/v1/items/does_not_exist").status_code == 404


class _FakeOcrReader:
    def readtext(self, image, detail=0):
        return ["GIFT", "INCOMING"]


class TestOcr:
    def test_extracts_text_when_image_present(self, client, monkeypatch):
        monkeypatch.setattr(app_module._state["bundle"], "ocr", _FakeOcrReader())
        r = client.post(
            "/api/v1/analyze",
            data={"text": "hello"},
            files={"image": ("a.png", _png_bytes(), "image/png")},
        )
        body = r.json()
        assert body["input"]["ocr_text"] == "GIFT INCOMING"
        assert "ocr" in body["latency_ms"]

    def test_no_image_means_no_ocr_text(self, client, monkeypatch):
        monkeypatch.setattr(app_module._state["bundle"], "ocr", _FakeOcrReader())
        r = client.post("/api/v1/analyze", data={"text": "hello"})
        assert r.json()["input"]["ocr_text"] is None

    def test_run_ocr_false_is_honoured_even_when_a_reader_is_loaded(self, client, monkeypatch):
        """The client's own opt-out must win over the server having a reader
        available — run_ocr=false means "do not run it", not "best effort"."""
        monkeypatch.setattr(app_module._state["bundle"], "ocr", _FakeOcrReader())
        r = client.post(
            "/api/v1/analyze",
            data={"text": "hello", "run_ocr": "false"},
            files={"image": ("a.png", _png_bytes(), "image/png")},
        )
        assert r.json()["input"]["ocr_text"] is None

    def test_ocr_text_never_reaches_the_classifier(self, client, monkeypatch):
        """The scoping decision this feature depends on: OCR text is shown to
        the moderator but must never be spliced into what the heads score —
        the models were trained on captions only, never on OCR'd meme text,
        so mixing them in would be an untested distribution shift disguised
        as a feature. Verified by asserting the exact text run_arms received."""
        monkeypatch.setattr(app_module._state["bundle"], "ocr", _FakeOcrReader())
        seen_text = {}

        def capturing_run_arms(bundle, task, image, text):
            seen_text["text"] = text
            arms = ArmOutputs(
                cv_only=0.3, nlp_only=0.3, fusion=0.3, fusion_probs={"benign": 0.7, "harmful": 0.3}
            )
            return arms, {"encode": 1}

        monkeypatch.setattr(app_module, "run_arms", capturing_run_arms)
        client.post(
            "/api/v1/analyze",
            data={"text": "the actual caption"},
            files={"image": ("a.png", _png_bytes(), "image/png")},
        )
        assert seen_text["text"] == "the actual caption"

    def test_no_reader_loaded_degrades_to_none(self, client):
        """bundle.ocr defaults to None in the fixture, matching a deployment
        where the reader failed to load — must not 500."""
        r = client.post(
            "/api/v1/analyze",
            data={"text": "hello"},
            files={"image": ("a.png", _png_bytes(), "image/png")},
        )
        assert r.status_code == 200
        assert r.json()["input"]["ocr_text"] is None


class TestImageReuse:
    """_check_image_reuse's real cosine-similarity path, exercised through
    _FakeClipForReuse rather than the AttributeError-and-degrade path the
    FakeBundle used to silently take (it had no .clip attribute at all)."""

    def test_first_sighting_is_not_reused(self, client):
        r = client.post(
            "/api/v1/analyze",
            data={"text": "first caption"},
            files={"image": ("a.png", _png_bytes(), "image/png")},
        )
        body = r.json()["image_reuse"]
        assert body["checked"] is True
        assert body["is_reused"] is False

    def test_second_sighting_of_the_same_image_is_flagged(self, client):
        first = client.post(
            "/api/v1/analyze",
            data={"text": "first caption"},
            files={"image": ("a.png", _png_bytes(), "image/png")},
        ).json()

        second = client.post(
            "/api/v1/analyze",
            data={"text": "second caption, different claim"},
            files={"image": ("b.png", _png_bytes(), "image/png")},
        ).json()

        reuse = second["image_reuse"]
        assert reuse["checked"] is True
        assert reuse["is_reused"] is True
        assert reuse["first_seen_item_id"] == first["item_id"]
        assert reuse["first_seen_text"] == "first caption"
        assert reuse["similarity"] > 0.9

    def test_a_different_image_is_not_flagged_as_reused(self, client):
        client.post(
            "/api/v1/analyze",
            data={"text": "first caption"},
            files={"image": ("a.png", _png_bytes(color=(120, 40, 40)), "image/png")},
        )
        r = client.post(
            "/api/v1/analyze",
            data={"text": "unrelated caption"},
            files={"image": ("b.png", _png_bytes(color=(10, 200, 10)), "image/png")},
        )
        assert r.json()["image_reuse"]["is_reused"] is False

    def test_no_image_means_not_checked(self, client):
        r = client.post("/api/v1/analyze", data={"text": "hello"})
        body = r.json()["image_reuse"]
        assert body["checked"] is False
        assert body["is_reused"] is False
        assert body["reason"] == "no image supplied"

    def test_reuse_never_influences_the_verdict(self, client):
        """Purely informational, the same guarantee OCR's isolation test pins:
        the verdict comes from run_arms alone, so a reused image must score
        identically to a first-sighting one."""
        first = client.post(
            "/api/v1/analyze",
            data={"text": "first caption"},
            files={"image": ("a.png", _png_bytes(), "image/png")},
        ).json()
        second = client.post(
            "/api/v1/analyze",
            data={"text": "second caption"},
            files={"image": ("b.png", _png_bytes(), "image/png")},
        ).json()
        assert second["image_reuse"]["is_reused"] is True
        assert second["verdict"] == first["verdict"]


class TestFactCheck:
    def test_degrades_to_unavailable_without_crashing(self, client, monkeypatch):
        """No API key configured (the fixture's default state) must return a
        structured unavailable result, matching every other optional branch
        in this service, not a 500."""
        item_id = client.post("/api/v1/analyze", data={"text": "hello"}).json()["item_id"]
        r = client.get(f"/api/v1/items/{item_id}/fact-check")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "unavailable"
        assert body["verdict"] is None
        assert body["sources"] == []

    def test_ready_result_is_cached_across_calls(self, client, monkeypatch):
        """A real search round-trip costs more than explaining a score the
        model already computed — a moderator reopening an item must not pay
        for a second search."""
        calls = {"n": 0}

        def fake_generate(payload):
            calls["n"] += 1
            return {
                "status": "ready",
                "verdict": "contradicted",
                "summary": "Sources place this image years before the claimed event.",
                "sources": [{"title": "Example", "url": "https://example.org", "domain": "example.org"}],
                "model": "gemini-2.5-flash",
                "latency_ms": 4200,
            }

        monkeypatch.setattr(app_module.factcheck_mod, "generate", fake_generate)
        item_id = client.post("/api/v1/analyze", data={"text": "share before they delete this"}).json()[
            "item_id"
        ]

        first = client.get(f"/api/v1/items/{item_id}/fact-check").json()
        second = client.get(f"/api/v1/items/{item_id}/fact-check").json()

        assert first["verdict"] == "contradicted"
        assert first == second
        assert calls["n"] == 1

    def test_unknown_item_is_404(self, client):
        assert client.get("/api/v1/items/does_not_exist/fact-check").status_code == 404

    def test_misinformation_context_is_passed_without_deferring(self, client, monkeypatch):
        """The context passed to factcheck.generate must carry the model's
        own misinformation reading, but the prompt built from it must say not
        to defer to it (see TestUserPrompt in test_factcheck.py) — this pins
        that app.py actually wires the score through, not just that the
        module knows what to do with one."""
        seen = {}

        def capturing_generate(payload):
            seen.update(payload)
            return {
                "status": "unavailable",
                "verdict": None,
                "summary": None,
                "sources": [],
                "model": None,
                "latency_ms": 1,
            }

        monkeypatch.setattr(app_module.factcheck_mod, "generate", capturing_generate)
        item_id = client.post(
            "/api/v1/analyze", data={"text": "share before they delete this"}
        ).json()["item_id"]
        client.get(f"/api/v1/items/{item_id}/fact-check")

        assert seen["text"] == "share before they delete this"
        assert "misinformation_label" in seen


class TestQueueAndDecision:
    def test_total_pending_is_not_the_filtered_count(self, client):
        """The bug this pins: total_pending used to report the count matching
        whatever filter was applied, so status=resolved on an all-pending store
        reported total_pending=0 even though items were, in fact, pending."""
        client.post("/api/v1/analyze", data={"text": "one"})
        client.post("/api/v1/analyze", data={"text": "two"})

        resolved_view = client.get("/api/v1/queue", params={"status": "resolved"}).json()
        assert resolved_view["total_matching"] == 0  # nothing resolved yet
        assert resolved_view["total_pending"] == 2  # but two items are waiting

    def test_decision_moves_item_from_pending_to_resolved(self, client):
        item = client.post("/api/v1/analyze", data={"text": "hello"}).json()
        item_id = item["item_id"]

        pending_before = client.get("/api/v1/queue", params={"status": "pending"}).json()
        assert item_id in [i["item_id"] for i in pending_before["items"]]

        r = client.post(
            f"/api/v1/items/{item_id}/decision",
            json={"action": "approve", "moderator_id": "mod_test"},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "resolved"

        pending_after = client.get("/api/v1/queue", params={"status": "pending"}).json()
        assert item_id not in [i["item_id"] for i in pending_after["items"]]
        assert pending_after["total_pending"] == 0

        resolved_after = client.get("/api/v1/queue", params={"status": "resolved"}).json()
        assert item_id in [i["item_id"] for i in resolved_after["items"]]

    def test_decision_on_unknown_item_is_404(self, client):
        r = client.post(
            "/api/v1/items/does_not_exist/decision",
            json={"action": "approve", "moderator_id": "mod_test"},
        )
        assert r.status_code == 404

    def test_queue_ranks_by_priority_not_arrival(self, client, monkeypatch):
        # Second item scores higher, so it must lead the queue despite arriving second.
        calls = iter(
            [
                ArmOutputs(cv_only=0.2, nlp_only=0.2, fusion=0.2, fusion_probs={"benign": 0.8, "harmful": 0.2}),
                ArmOutputs(cv_only=0.9, nlp_only=0.9, fusion=0.9, fusion_probs={"benign": 0.1, "harmful": 0.9}),
            ]
        )
        monkeypatch.setattr(
            app_module, "run_arms", lambda *a, **k: (next(calls), {"encode": 1})
        )
        client.post("/api/v1/analyze", data={"text": "low priority"})
        client.post("/api/v1/analyze", data={"text": "high priority"})

        items = client.get("/api/v1/queue").json()["items"]
        assert items[0]["text_preview"] == "high priority"


class TestActiveHeads:
    """End-to-end version of the queue-filter blind spot: a real Hateful Memes
    example scored toxicity=0.76 (correctly harmful) but top_head came out
    "misinformation" because that head — a 3-way classifier, not comparable to
    a 2-way one on raw score alone — happened to read 0.94. A queue filter
    keyed on top_head made that item invisible under head=toxicity."""

    def test_both_heads_crossing_threshold_are_both_active(self, client, monkeypatch):
        two_task_bundle = FakeBundle()
        two_task_bundle.tasks = ["misinformation", "toxicity"]
        two_task_bundle.arms = {"misinformation": {}, "toxicity": {}}
        monkeypatch.setitem(app_module._state, "bundle", two_task_bundle)

        def scored(bundle, task, image, text):
            # Mirrors the real example: toxicity clears threshold but
            # misinformation reads higher, so misinformation is lead_task.
            fusion = {"toxicity": 0.76, "misinformation": 0.94}[task]
            probs = (
                {"benign": 1 - fusion, "harmful": fusion}
                if task == "toxicity"
                else {"true": 1 - fusion, "satire": fusion, "misleading": 0.0}
            )
            arms = ArmOutputs(cv_only=fusion, nlp_only=fusion, fusion=fusion, fusion_probs=probs)
            return arms, {"encode": 1}

        monkeypatch.setattr(app_module, "run_arms", scored)

        body = client.post("/api/v1/analyze", data={"text": "hi"}).json()
        assert body["heads"]["toxicity"]["label"] == "harmful"
        assert set(body["active_heads"]) == {"toxicity", "misinformation"}

        # The point of the fix: findable under the head that did NOT win.
        by_toxicity = client.get("/api/v1/queue", params={"head": "toxicity"}).json()
        assert body["item_id"] in [i["item_id"] for i in by_toxicity["items"]]

        by_misinfo = client.get("/api/v1/queue", params={"head": "misinformation"}).json()
        assert body["item_id"] in [i["item_id"] for i in by_misinfo["items"]]

    def test_only_the_crossing_head_is_active(self, client, monkeypatch):
        two_task_bundle = FakeBundle()
        two_task_bundle.tasks = ["misinformation", "toxicity"]
        two_task_bundle.arms = {"misinformation": {}, "toxicity": {}}
        monkeypatch.setitem(app_module._state, "bundle", two_task_bundle)

        def scored(bundle, task, image, text):
            fusion = {"toxicity": 0.2, "misinformation": 0.9}[task]
            probs = (
                {"benign": 1 - fusion, "harmful": fusion}
                if task == "toxicity"
                else {"true": 1 - fusion, "satire": fusion, "misleading": 0.0}
            )
            arms = ArmOutputs(cv_only=fusion, nlp_only=fusion, fusion=fusion, fusion_probs=probs)
            return arms, {"encode": 1}

        monkeypatch.setattr(app_module, "run_arms", scored)

        body = client.post("/api/v1/analyze", data={"text": "hi"}).json()
        assert body["active_heads"] == ["misinformation"]

        by_toxicity = client.get("/api/v1/queue", params={"head": "toxicity"}).json()
        assert body["item_id"] not in [i["item_id"] for i in by_toxicity["items"]]

    def test_explanation_is_built_from_every_active_head_not_just_top_head(
        self, client, monkeypatch
    ):
        """get_explanation used to build its prompt from record["top_head"]
        alone. Reproduces the real example end to end: toxicity=0.76 (harmful)
        clears threshold but misinformation=0.94 is higher and would be
        top_head — the explanation payload must still describe both."""
        two_task_bundle = FakeBundle()
        two_task_bundle.tasks = ["misinformation", "toxicity"]
        two_task_bundle.arms = {"misinformation": {}, "toxicity": {}}
        monkeypatch.setitem(app_module._state, "bundle", two_task_bundle)

        def scored(bundle, task, image, text):
            fusion = {"toxicity": 0.76, "misinformation": 0.94}[task]
            probs = (
                {"benign": 1 - fusion, "harmful": fusion}
                if task == "toxicity"
                else {"true": 1 - fusion, "satire": fusion, "misleading": 0.0}
            )
            arms = ArmOutputs(cv_only=fusion, nlp_only=fusion, fusion=fusion, fusion_probs=probs)
            return arms, {"encode": 1}

        monkeypatch.setattr(app_module, "run_arms", scored)

        captured = {}

        def fake_generate(payload, timeout=20.0):
            captured.update(payload)
            return {
                "status": "ready",
                "narrative": "stub",
                "key_factors": [],
                "model": "stub-model",
                "latency_ms": 1,
            }

        monkeypatch.setattr(app_module.explain_mod, "generate", fake_generate)

        item_id = client.post("/api/v1/analyze", data={"text": "hi"}).json()["item_id"]
        client.get(f"/api/v1/items/{item_id}/explanation")

        assert set(captured["heads"]) == {"toxicity", "misinformation"}
        assert captured["heads"]["toxicity"]["modality_scores"]["fusion"] == 0.76
        assert captured["heads"]["misinformation"]["modality_scores"]["fusion"] == 0.94


class TestRateLimit:
    def test_blocks_after_the_configured_window(self, client, monkeypatch):
        monkeypatch.setattr(ratelimit, "MAX_REQUESTS_PER_WINDOW", 3)

        for _ in range(3):
            assert client.post("/api/v1/analyze", data={"text": "hi"}).status_code == 200

        blocked = client.post("/api/v1/analyze", data={"text": "hi"})
        assert blocked.status_code == 429
        assert "Retry-After" in blocked.headers

    def test_only_analyze_is_rate_limited(self, client, monkeypatch):
        """/health and /queue must stay reachable during a burst on /analyze —
        a moderator's queue should not go dark because of upload traffic."""
        monkeypatch.setattr(ratelimit, "MAX_REQUESTS_PER_WINDOW", 1)
        client.post("/api/v1/analyze", data={"text": "hi"})
        client.post("/api/v1/analyze", data={"text": "hi"})  # now over the limit

        assert client.get("/api/v1/health").status_code == 200
        assert client.get("/api/v1/queue").status_code == 200


class TestCors:
    def test_allowed_origin_gets_the_header(self, client):
        r = client.get(
            "/api/v1/health", headers={"Origin": "http://localhost:3000"}
        )
        assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"

    def test_disallowed_origin_gets_no_header(self, client):
        r = client.get(
            "/api/v1/health", headers={"Origin": "https://evil.example"}
        )
        assert "access-control-allow-origin" not in r.headers
