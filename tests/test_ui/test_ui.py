"""Tests for ui/app.py — helper functions (testable without Streamlit)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from trustandverify.core.models import Report
from trustandverify.ui.app import _check_env, _opinion_bar, _run_agent, _verdict_emoji


class TestOpinionBar:
    def test_returns_html(self):
        html = _opinion_bar(0.7, 0.1, 0.2)
        assert "opinion-bar" in html
        assert "Belief" in html
        assert "Disbelief" in html
        assert "Uncertainty" in html

    def test_values_in_output(self):
        html = _opinion_bar(0.5, 0.3, 0.2)
        assert "0.500" in html
        assert "0.300" in html
        assert "0.200" in html


class TestVerdictEmoji:
    def test_supported(self):
        assert _verdict_emoji("supported") == "✅"

    def test_contested(self):
        assert _verdict_emoji("contested") == "⚠️"

    def test_refuted(self):
        assert _verdict_emoji("refuted") == "❌"

    def test_unknown(self):
        assert _verdict_emoji("unknown") == "❓"


class TestCheckEnv:
    def test_all_keys_set(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "fake")
        monkeypatch.setenv("GEMINI_API_KEY", "fake")
        ok, missing = _check_env()
        assert ok is True
        assert missing == []

    def test_missing_tavily(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "fake")
        ok, missing = _check_env()
        assert ok is False
        assert "TAVILY_API_KEY" in missing

    def test_missing_both(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        ok, missing = _check_env()
        assert ok is False
        assert len(missing) == 2


class TestRunAgent:
    def test_calls_trust_agent(self):
        mock_report = MagicMock(spec=Report)

        with patch("trustandverify.ui.app.asyncio.run", return_value=mock_report) as mock_run:
            result = _run_agent("Is coffee healthy?", 3)
            assert result is mock_report
            mock_run.assert_called_once()
