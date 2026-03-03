"""Tests for llm/gemini.py — JSON parsing and completion mocking."""

from __future__ import annotations

import pytest

from trustandverify.llm.gemini import GeminiBackend, _parse_json_robust


class TestParseJsonRobust:
    """Unit tests for the robust JSON parser — no LLM calls needed."""

    def test_direct_json_object(self):
        raw = '{"evidence": "Studies show X", "supports": true, "confidence": 0.8}'
        result = _parse_json_robust(raw)
        assert result["evidence"] == "Studies show X"
        assert result["supports"] is True
        assert result["confidence"] == 0.8

    def test_strips_json_markdown_fence(self):
        raw = '```json\n{"evidence": "X", "supports": true}\n```'
        result = _parse_json_robust(raw)
        assert result["evidence"] == "X"

    def test_strips_plain_markdown_fence(self):
        raw = '```\n{"evidence": "X", "supports": false}\n```'
        result = _parse_json_robust(raw)
        assert result["supports"] is False

    def test_extracts_json_from_surrounding_text(self):
        raw = 'Here is the evidence: {"evidence": "X", "confidence": 0.7} hope that helps'
        result = _parse_json_robust(raw)
        assert result["evidence"] == "X"

    def test_handles_unicode_whitespace(self):
        raw = '{"evidence":\xa0"Non-breaking space", "supports": true}'
        result = _parse_json_robust(raw)
        assert result["evidence"] == "Non-breaking space"

    def test_array_response_wrapped_as_items(self):
        raw = '["Claim one", "Claim two", "Claim three"]'
        result = _parse_json_robust(raw)
        assert "items" in result
        assert result["items"] == ["Claim one", "Claim two", "Claim three"]

    def test_array_in_markdown_fence(self):
        raw = '```json\n["Claim A", "Claim B"]\n```'
        result = _parse_json_robust(raw)
        assert result["items"] == ["Claim A", "Claim B"]

    def test_returns_defaults_on_unparseable_input(self):
        raw = "Sorry, I cannot help with that."
        defaults = {"evidence": raw, "supports": True, "confidence": 0.5}
        result = _parse_json_robust(raw, defaults=defaults)
        assert result == defaults

    def test_empty_string_returns_defaults(self):
        result = _parse_json_robust("", defaults={"x": 1})
        assert result == {"x": 1}


class TestGeminiBackend:
    def test_is_available_with_key(self):
        backend = GeminiBackend(api_key="fake-key")
        assert backend.is_available() is True

    def test_is_available_without_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        backend = GeminiBackend(api_key="")
        assert backend.is_available() is False

    def test_default_model(self):
        backend = GeminiBackend(api_key="x")
        assert backend.model == "gemini/gemini-2.5-flash"

    async def test_complete_returns_string(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock

        mock_choice = MagicMock()
        mock_choice.message.content = "Remote workers are more productive."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_acompletion = AsyncMock(return_value=mock_response)
        monkeypatch.setattr("trustandverify.llm.gemini.GeminiBackend.complete",
                            AsyncMock(return_value="Remote workers are more productive."))

        backend = GeminiBackend(api_key="fake")
        result = await backend.complete("Is remote work productive?")
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_complete_json_parses_result(self, monkeypatch):
        from unittest.mock import AsyncMock

        raw = '{"evidence": "13% more productive", "supports": true, "confidence": 0.85}'
        monkeypatch.setattr(
            "trustandverify.llm.gemini.GeminiBackend.complete",
            AsyncMock(return_value=raw),
        )

        backend = GeminiBackend(api_key="fake")
        result = await backend.complete_json("Extract evidence...")
        assert result["supports"] is True
        assert result["confidence"] == 0.85
