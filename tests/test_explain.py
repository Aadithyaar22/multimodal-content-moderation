"""LLM explanation prompt and key-factor construction.

No test touched this module before. That mattered concretely: the original
_user_prompt and _key_factors were built around a single task's scores, and
get_explanation fed them only record["top_head"] — so an item independently
flagged on two grounds (toxicity=0.76 harmful, misinformation=0.94 misleading,
the real example that caught this) got a narrative that discussed only
whichever head happened to score higher. A moderator reading the summary would
never learn the item was also flagging as harassment. These tests pin the fix:
every active head must appear in the prompt and in the structured factors, not
just the one the verdict headline names.
"""

from __future__ import annotations

from mcm.serving.explain import _key_factors, _user_prompt, generate


def _heads(**by_task: dict) -> dict:
    """Build a payload["heads"] dict from {task: (label, cv, nlp, fusion)}."""
    return {
        task: {
            "label": vals[0],
            "modality_scores": {"cv_only": vals[1], "nlp_only": vals[2], "fusion": vals[3]},
        }
        for task, vals in by_task.items()
    }


class TestUserPrompt:
    def test_single_head_does_not_claim_multiple_grounds(self):
        payload = {
            "heads": _heads(toxicity=("harmful", 0.2, 0.3, 0.71)),
            "text": "hi",
            "has_image": True,
            "threshold": 0.5,
            "is_emergent": False,
        }
        prompt = _user_prompt(payload)
        assert "independent grounds" not in prompt
        assert "[harassment/hate-speech]" in prompt
        assert "0.71" in prompt

    def test_two_heads_are_both_present_and_flagged_as_multiple(self):
        """The exact real case: toxicity clears threshold, misinformation
        scores higher — both must appear, not just the higher one."""
        payload = {
            "heads": _heads(
                toxicity=("harmful", 0.73, 0.81, 0.76),
                misinformation=("misleading", 0.62, 0.55, 0.94),
            ),
            "text": "go back to where you came from, nobody wants you here",
            "has_image": True,
            "threshold": 0.5,
            "is_emergent": False,
        }
        prompt = _user_prompt(payload)
        assert "flagged on 2 independent grounds" in prompt
        assert "[harassment/hate-speech]" in prompt
        assert "[misinformation]" in prompt
        assert "0.76" in prompt  # toxicity's fused score
        assert "0.94" in prompt  # misinformation's fused score, numerically higher
        assert "predicted: harmful" in prompt
        assert "predicted: misleading" in prompt

    def test_emergent_note_still_appended_after_head_blocks(self):
        payload = {
            "heads": _heads(toxicity=("harmful", 0.2, 0.3, 0.71)),
            "text": "hi",
            "has_image": True,
            "threshold": 0.5,
            "is_emergent": True,
        }
        assert "EMERGENT" in _user_prompt(payload)


class TestKeyFactors:
    def test_single_head_factors_are_tagged_with_that_head(self):
        payload = {"heads": _heads(toxicity=("harmful", 0.2, 0.3, 0.71))}
        factors = _key_factors(payload)
        assert all(f["head"] == "toxicity" for f in factors)
        assert {f["factor"] for f in factors} >= {"vision-only signal", "language-only signal"}

    def test_two_heads_produce_factors_for_both_not_just_the_stronger(self):
        """This is the bug in structured form: before every head flowed into
        _key_factors, a second active head's numbers were simply absent from
        this list, even though the narrative claimed to describe the item."""
        payload = {
            "heads": _heads(
                toxicity=("harmful", 0.73, 0.81, 0.76),
                misinformation=("misleading", 0.62, 0.55, 0.94),
            )
        }
        factors = _key_factors(payload)
        heads_present = {f["head"] for f in factors}
        assert heads_present == {"toxicity", "misinformation"}
        # Both heads' vision/language signals must survive, not just one.
        tox = [f for f in factors if f["head"] == "toxicity"]
        mis = [f for f in factors if f["head"] == "misinformation"]
        assert len(tox) >= 2
        assert len(mis) >= 2

    def test_cross_gain_factor_only_when_fusion_exceeds_best_unimodal(self):
        payload = {"heads": _heads(toxicity=("benign", 0.9, 0.9, 0.2))}
        factors = _key_factors(payload)
        assert not any(f["factor"] == "gain from modelling the pair jointly" for f in factors)

        payload2 = {"heads": _heads(toxicity=("harmful", 0.1, 0.2, 0.9))}
        factors2 = _key_factors(payload2)
        cross = [f for f in factors2 if f["factor"] == "gain from modelling the pair jointly"]
        assert len(cross) == 1
        assert cross[0]["head"] == "toxicity"


class TestGenerateFallback:
    def test_unavailable_still_carries_every_active_heads_factors(self, monkeypatch):
        """The bug survives even the no-key fallback path if key_factors is
        built from the same narrowed payload — checked here with no API keys
        configured, so this never makes a network call."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        payload = {
            "heads": _heads(
                toxicity=("harmful", 0.73, 0.81, 0.76),
                misinformation=("misleading", 0.62, 0.55, 0.94),
            ),
            "text": "hi",
            "has_image": True,
            "threshold": 0.5,
            "is_emergent": False,
        }
        result = generate(payload)
        assert result["status"] == "unavailable"
        assert result["narrative"] is None
        heads_present = {f["head"] for f in result["key_factors"]}
        assert heads_present == {"toxicity", "misinformation"}
