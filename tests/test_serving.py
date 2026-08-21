"""Serving logic.

The emergent rule gets the most attention here. It decides the flag the whole
product is built around, it is computed per request rather than trained, and a
mistake in it would silently inflate or erase the project's headline number
without any test failing elsewhere.
"""

from __future__ import annotations

import pytest

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
