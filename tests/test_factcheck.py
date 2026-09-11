"""Claim grounding.

Unlike explain.py's narrative, this module's whole point is to answer a
question the model's own scores cannot: does the caption's claim actually
hold up. The parsing logic here is what turns a free-text LLM response into a
verdict a client can render as a badge, so a malformed response degrading to
"unclear" (not a crash, not a silently wrong verdict) is the behavior worth
pinning most carefully.
"""

from __future__ import annotations

from types import SimpleNamespace

from mcm.serving.factcheck import _parse, _sources, _user_prompt, generate


class TestParse:
    def test_well_formed_response(self):
        text = (
            "VERDICT: contradicted\n"
            "The claim that this photo shows last week's flooding is false; "
            "reverse search places the same image in a 2019 report."
        )
        verdict, summary = _parse(text)
        assert verdict == "contradicted"
        assert summary.startswith("The claim that this photo")

    def test_is_case_and_whitespace_tolerant(self):
        verdict, _ = _parse("  verdict:   Supported  \nMultiple outlets confirm this.")
        assert verdict == "supported"

    def test_missing_verdict_line_defaults_to_unclear(self):
        """A moderator seeing an odd-shaped but present answer is better than
        a crash, and 'unclear' is the honest default when the parser can't
        even tell what verdict was meant."""
        text = "This claim appears to be about a policy change from last year."
        verdict, summary = _parse(text)
        assert verdict == "unclear"
        assert summary == text

    def test_invalid_verdict_token_defaults_to_unclear(self):
        verdict, summary = _parse("VERDICT: probably-true\nSome explanation.")
        assert verdict == "unclear"
        # The full text is kept, not just the (invalid) label, since the
        # summary is still the most useful thing to show.
        assert "probably-true" in summary

    def test_no_factual_claim(self):
        verdict, _ = _parse("VERDICT: no_factual_claim\nThis is a joke post with nothing to check.")
        assert verdict == "no_factual_claim"


class TestUserPrompt:
    def test_includes_ocr_text_when_present(self):
        prompt = _user_prompt({"text": "share before they delete this", "ocr_text": "URGENT"})
        assert "URGENT" in prompt

    def test_omits_ocr_line_when_absent(self):
        prompt = _user_prompt({"text": "hello", "ocr_text": None})
        assert "visible in the image" not in prompt

    def test_includes_misinformation_context_without_deferring_to_it(self):
        prompt = _user_prompt(
            {"text": "hello", "misinformation_label": "misleading", "misinformation_score": 0.94}
        )
        assert "misleading" in prompt
        assert "do not defer to it" in prompt


class TestSources:
    def test_extracts_web_chunks(self):
        resp = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    grounding_metadata=SimpleNamespace(
                        grounding_chunks=[
                            SimpleNamespace(
                                web=SimpleNamespace(
                                    uri="https://example.org/report", title="Example report", domain="example.org"
                                )
                            )
                        ]
                    )
                )
            ]
        )
        sources = _sources(resp)
        assert sources == [
            {"title": "Example report", "url": "https://example.org/report", "domain": "example.org"}
        ]

    def test_no_grounding_metadata_is_not_an_error(self):
        """A model that answered without searching (a correct outcome for a
        caption with nothing to check) has nothing to cite — that must
        degrade to an empty list, not raise."""
        resp = SimpleNamespace(candidates=[SimpleNamespace(grounding_metadata=None)])
        assert _sources(resp) == []

    def test_malformed_response_degrades_to_empty_list(self):
        assert _sources(SimpleNamespace()) == []
        assert _sources(None) == []


class TestGenerateDegradesWithoutAKey:
    def test_no_gemini_key_is_unavailable_not_a_crash(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        result = generate({"text": "hello"})
        assert result["status"] == "unavailable"
        assert result["verdict"] is None
        assert result["sources"] == []
