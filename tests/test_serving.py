"""Serving logic.

The emergent rule gets the most attention here. It decides the flag the whole
product is built around, it is computed per request rather than trained, and a
mistake in it would silently inflate or erase the project's headline number
without any test failing elsewhere.
"""

from __future__ import annotations

import pytest

from mcm.serving.app import _run_ocr
from mcm.serving.inference import (
    EMERGENT_MARGIN,
    THRESHOLD,
    ArmOutputs,
    emergent_signal,
    priority_score,
    verdict_for,
)
from mcm.serving.store import Store


def arms(cv: float, nlp: float, fusion: float) -> ArmOutputs:
    return ArmOutputs(cv_only=cv, nlp_only=nlp, fusion=fusion, fusion_probs={})


class TestEmergentSignal:
    def test_flags_the_canonical_case(self):
        """Both modalities quiet, fused loud — the case the project exists for."""
        is_emergent, delta = emergent_signal(arms(0.22, 0.31, 0.71))
        assert is_emergent
        assert delta == pytest.approx(0.40, abs=1e-6)

    def test_not_emergent_when_a_modality_already_crosses(self):
        # Language alone already flags it, so nothing emerged from the pair.
        is_emergent, _ = emergent_signal(arms(0.20, 0.85, 0.90))
        assert not is_emergent

    def test_not_emergent_when_fusion_stays_below_threshold(self):
        """A gain that does not change the decision is not a finding.

        Counting this would inflate the headline rate with items no moderator
        would ever be shown.
        """
        is_emergent, delta = emergent_signal(arms(0.05, 0.08, 0.45))
        assert not is_emergent
        assert delta > EMERGENT_MARGIN  # large gain, still under threshold

    def test_not_emergent_on_a_marginal_gain(self):
        # Just over threshold by a hair; within noise, so not called emergent.
        is_emergent, _ = emergent_signal(arms(0.44, 0.46, 0.52))
        assert not is_emergent

    def test_margin_boundary_is_inclusive(self):
        # Chosen so the delta lands exactly on the margin *and* the fused score
        # clears the threshold — both conditions must hold, so the unimodal arms
        # have to sit close enough below threshold for the margin to carry it over.
        unimodal = 0.40
        fusion = unimodal + EMERGENT_MARGIN
        assert fusion >= THRESHOLD, "test values must clear the threshold"
        is_emergent, _ = emergent_signal(arms(unimodal, unimodal, fusion))
        assert is_emergent

    def test_margin_is_required_even_when_threshold_is_crossed(self):
        # Fused clears threshold but only just outruns the better arm.
        is_emergent, _ = emergent_signal(arms(0.42, 0.45, 0.55))
        assert not is_emergent

    def test_delta_is_measured_against_the_better_arm(self):
        _, delta = emergent_signal(arms(0.10, 0.40, 0.70))
        assert delta == pytest.approx(0.30, abs=1e-6)


class TestVerdict:
    def test_three_bands(self):
        assert verdict_for(0.95)[0] == "harmful"
        assert verdict_for(0.60)[0] == "review"
        assert verdict_for(0.20)[0] == "benign"

    def test_threshold_is_review_not_harmful(self):
        """Borderline items go to a person rather than into a confident bucket."""
        assert verdict_for(THRESHOLD)[0] == "review"

    def test_nothing_is_ever_auto_actioned(self):
        for score in (0.1, 0.5, 0.99):
            _, action = verdict_for(score)
            assert action in {"queue_for_review", "no_action"}
            assert "remove" not in action


class TestPriority:
    def test_emergent_items_are_lifted(self):
        assert priority_score(0.6, True) > priority_score(0.6, False)

    def test_never_exceeds_one(self):
        assert priority_score(0.99, True) <= 1.0


class TestStore:
    def test_falls_back_to_memory_without_a_uri(self):
        assert Store(uri="").backend == "memory"

    def test_round_trip(self):
        s = Store(uri="")
        s.put({"item_id": "a", "status": "pending", "priority_score": 0.5})
        assert s.get("a")["priority_score"] == 0.5

    def test_queue_ranks_by_priority_not_arrival(self):
        s = Store(uri="")
        for i, p in enumerate([0.1, 0.9, 0.5]):
            s.put({"item_id": str(i), "status": "pending", "priority_score": p})
        items, total = s.query()
        assert [i["item_id"] for i in items] == ["1", "2", "0"]
        assert total == 3

    def test_emergent_filter(self):
        s = Store(uri="")
        s.put({"item_id": "a", "status": "pending", "priority_score": 0.5, "is_emergent": True})
        s.put({"item_id": "b", "status": "pending", "priority_score": 0.9, "is_emergent": False})
        items, _ = s.query(emergent_only=True)
        assert [i["item_id"] for i in items] == ["a"]

    def test_head_filter_matches_active_heads_not_just_top_head(self):
        """The bug this pins: a meme scoring toxicity=0.76 (correctly harmful)
        had top_head="misinformation" because that head read 0.94 — a 3-way
        score, not comparable to a 2-way one. Filtering on top_head alone made
        it invisible to a moderator asking for harassment cases specifically."""
        s = Store(uri="")
        s.put(
            {
                "item_id": "both",
                "status": "pending",
                "priority_score": 0.9,
                "top_head": "misinformation",
                "active_heads": ["misinformation", "toxicity"],
            }
        )
        s.put(
            {
                "item_id": "misinfo_only",
                "status": "pending",
                "priority_score": 0.8,
                "top_head": "misinformation",
                "active_heads": ["misinformation"],
            }
        )
        items, _ = s.query(head="toxicity")
        assert [i["item_id"] for i in items] == ["both"]

    def test_head_filter_never_matches_everything(self):
        """Regression for a real bug caught before it shipped: `head in
        active_heads or [top_head]` parses as `(head in active_heads) or
        [top_head]`, and a non-empty fallback list is truthy regardless of
        `head` — so the filter matched every row, not just the requested one."""
        s = Store(uri="")
        s.put(
            {
                "item_id": "a",
                "status": "pending",
                "priority_score": 0.5,
                "top_head": "toxicity",
                "active_heads": ["toxicity"],
            }
        )
        items, _ = s.query(head="misinformation")
        assert items == []

    def test_head_filter_falls_back_to_top_head_for_old_records(self):
        """A record predating active_heads (a pre-existing MongoDB document,
        in practice) must stay findable under the one head it does know."""
        s = Store(uri="")
        s.put({"item_id": "old", "status": "pending", "priority_score": 0.5, "top_head": "toxicity"})
        items, _ = s.query(head="toxicity")
        assert [i["item_id"] for i in items] == ["old"]

    def test_memory_store_is_bounded(self):
        """An unbounded dict in a long-running container is a slow leak."""
        from mcm.serving.store import MAX_MEMORY_ITEMS

        s = Store(uri="")
        for i in range(MAX_MEMORY_ITEMS + 50):
            s.put({"item_id": str(i), "status": "pending", "priority_score": 0.5})
        _, total = s.query(limit=1)
        assert total == MAX_MEMORY_ITEMS

    def test_rates_are_over_responders_not_all_decisions(self):
        """Feedback fields are optional; a blank must not count as a 'no'."""
        s = Store(uri="")
        s.put({"item_id": "a", "status": "resolved", "agreed_with_model": True})
        s.put({"item_id": "b", "status": "resolved"})  # no answer given
        assert s.aggregate_stats()["model"]["agreement_rate"] == 1.0

    def test_stats_on_an_empty_store_do_not_divide_by_zero(self):
        stats = Store(uri="").aggregate_stats()
        assert stats["queue"]["pending"] == 0
        assert stats["model"]["agreement_rate"] == 0.0


class _FakeCursor:
    """Just enough of pymongo's Cursor for Store's own call sites: chained
    .sort().skip().limit(), and plain iteration when none of those are
    called (as in find_similar_image's `list(collection.find(...))`)."""

    def __init__(self, docs: list[dict]):
        self._docs = docs

    def sort(self, field, direction):
        self._docs = sorted(self._docs, key=lambda d: d.get(field, 0), reverse=direction == -1)
        return self

    def skip(self, n):
        self._docs = self._docs[n:]
        return self

    def limit(self, n):
        self._docs = self._docs[:n]
        return self

    def __iter__(self):
        return iter(self._docs)


class FakeMongoCollection:
    """Minimal in-memory stand-in for the pymongo Collection surface Store
    actually calls. A real MongoClient needs a live server and has no place
    in a unit test; this mirrors the same deterministic-fake approach as
    FakeOcrReader below and _FakeClipForReuse in test_serving_app.py — it
    tests that Store drives pymongo the way it means to, not pymongo itself,
    which is someone else's well-tested library.
    """

    def __init__(self):
        self.docs: dict[str, dict] = {}

    def create_index(self, *args, **kwargs):
        pass

    def replace_one(self, filt, doc, upsert=False):
        self.docs[filt["item_id"]] = dict(doc)

    def find_one(self, filt, projection=None):
        doc = self.docs.get(filt["item_id"])
        return dict(doc) if doc is not None else None

    def update_one(self, filt, update):
        key = filt["item_id"]
        if key in self.docs:
            self.docs[key].update(update.get("$set", {}))

    @staticmethod
    def _matches(doc: dict, query: dict) -> bool:
        for key, want in query.items():
            have = doc.get(key)
            if isinstance(want, dict) and "$gte" in want:
                if have is None or have < want["$gte"]:
                    return False
            elif isinstance(have, list):
                # Mongo's equality match against an array field is an
                # implicit "any element equals this" — the same semantics
                # Store.query relies on for the active_heads filter.
                if want not in have:
                    return False
            elif have != want:
                return False
        return True

    def find(self, query=None, projection=None):
        query = query or {}
        return _FakeCursor([dict(d) for d in self.docs.values() if self._matches(d, query)])

    def count_documents(self, query=None):
        query = query or {}
        return sum(1 for d in self.docs.values() if self._matches(d, query))


def _mongo_store() -> Store:
    """A Store wired to fake Mongo collections instead of a real MongoClient.

    Bypasses __init__'s connection attempt entirely (uri="") and substitutes
    the private collection attributes directly — this is testing Store's own
    branch logic (`if self._collection is not None`), not pymongo's wire
    protocol, so there is nothing to gain from a real network round trip.
    """
    s = Store(uri="")
    s._collection = FakeMongoCollection()
    s._image_collection = FakeMongoCollection()
    return s


class TestStoreMongoBackend:
    """Everything in TestStore above only ever exercised the in-memory
    fallback (Store(uri="")'s default). Nothing had run the `if
    self._collection is not None` branch of a single Store method until now
    — a real MongoDB-backed deployment was flying on zero test coverage of
    its own code path."""

    def test_backend_reports_mongodb(self):
        assert _mongo_store().backend == "mongodb"

    def test_round_trip(self):
        s = _mongo_store()
        s.put({"item_id": "a", "status": "pending", "priority_score": 0.5})
        assert s.get("a")["priority_score"] == 0.5

    def test_update_patches_in_place(self):
        s = _mongo_store()
        s.put({"item_id": "a", "status": "pending", "priority_score": 0.5})
        s.update("a", {"status": "resolved"})
        assert s.get("a")["status"] == "resolved"

    def test_queue_ranks_by_priority(self):
        s = _mongo_store()
        for i, p in enumerate([0.1, 0.9, 0.5]):
            s.put({"item_id": str(i), "status": "pending", "priority_score": p})
        items, total = s.query()
        assert [i["item_id"] for i in items] == ["1", "2", "0"]
        assert total == 3

    def test_head_filter_matches_active_heads(self):
        s = _mongo_store()
        s.put(
            {
                "item_id": "both",
                "status": "pending",
                "priority_score": 0.9,
                "active_heads": ["misinformation", "toxicity"],
            }
        )
        s.put(
            {
                "item_id": "misinfo_only",
                "status": "pending",
                "priority_score": 0.8,
                "active_heads": ["misinformation"],
            }
        )
        items, _ = s.query(head="toxicity")
        assert [i["item_id"] for i in items] == ["both"]

    def test_count_pending_ignores_query_filters(self):
        """The same split that test_serving_app.py's queue tests pin for the
        in-memory backend — the Mongo branch computes this with its own
        count_documents({"status": "pending"}) call, independently."""
        s = _mongo_store()
        s.put({"item_id": "a", "status": "pending", "priority_score": 0.5})
        s.put({"item_id": "b", "status": "resolved", "priority_score": 0.5})
        assert s.count_pending() == 1

    def test_image_reuse_round_trip(self):
        s = _mongo_store()
        s.add_image_embedding("first", [1.0, 0.0, 0.0], "2026-01-01T00:00:00Z", text="original")
        match = s.find_similar_image([0.99, 0.01, 0.0], threshold=0.9)
        assert match is not None
        assert match["item_id"] == "first"
        assert match["text"] == "original"

    def test_image_reuse_below_threshold_is_no_match(self):
        s = _mongo_store()
        s.add_image_embedding("first", [1.0, 0.0, 0.0], "2026-01-01T00:00:00Z", text="original")
        assert s.find_similar_image([0.0, 1.0, 0.0], threshold=0.9) is None

    def test_aggregate_stats_reads_from_mongo(self):
        s = _mongo_store()
        s.put({"item_id": "a", "status": "resolved", "agreed_with_model": True})
        assert s.aggregate_stats()["model"]["agreement_rate"] == 1.0


class FakeOcrReader:
    """Minimal stand-in for easyocr.Reader — a real one costs a 94MB model
    download and ~2s init, which has no place in a unit test."""

    def __init__(self, texts: list[str] | None = None, raises: bool = False):
        self._texts = texts if texts is not None else ["hello", "world"]
        self._raises = raises

    def readtext(self, image, detail=0):
        if self._raises:
            raise RuntimeError("simulated OCR failure")
        return list(self._texts)


class FakeBundleForOcr:
    """Just enough of ModelBundle for _run_ocr's bookkeeping."""

    def __init__(self, ocr=None):
        self.ocr = ocr


class TestRunOcr:
    def test_returns_none_when_reader_not_loaded(self):
        assert _run_ocr(FakeBundleForOcr(ocr=None), object()) is None

    def test_returns_none_when_no_image(self):
        assert _run_ocr(FakeBundleForOcr(ocr=FakeOcrReader()), None) is None

    def test_joins_detected_lines(self):
        bundle = FakeBundleForOcr(ocr=FakeOcrReader(["GIFT", "INCOMING"]))
        assert _run_ocr(bundle, object()) == "GIFT INCOMING"

    def test_empty_detection_is_none_not_empty_string(self):
        """An image with no text must read as 'nothing detected', not as an
        empty-but-present string a client would render as a blank line."""
        bundle = FakeBundleForOcr(ocr=FakeOcrReader([]))
        assert _run_ocr(bundle, object()) is None

    def test_degrades_to_none_on_failure_rather_than_raising(self):
        """OCR is auxiliary — the same contract as the deepfake branch. An
        error here must never take the whole /analyze request down with it."""
        bundle = FakeBundleForOcr(ocr=FakeOcrReader(raises=True))
        assert _run_ocr(bundle, object()) is None
