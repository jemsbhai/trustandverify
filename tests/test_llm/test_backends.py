"""Tests for OpenAI, Anthropic, and Ollama backends — mocked, no real API calls."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trustandverify.llm.anthropic import AnthropicBackend
from trustandverify.llm.ollama import OllamaBackend
from trustandverify.llm.openai import OpenAIBackend


# ── OpenAIBackend ──────────────────────────────────────────────────────────────

class TestOpenAIBackend:
    def test_is_available_with_key(self):
        assert OpenAIBackend(api_key="sk-fake").is_available() is True

    def test_is_available_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert OpenAIBackend(api_key="").is_available() is False

    def test_default_model(self):
        assert OpenAIBackend(api_key="x").model == "gpt-4o"

    async def test_complete_returns_string(self, monkeypatch):
        monkeypatch.setattr(
            "trustandverify.llm.openai.OpenAIBackend.complete",
            AsyncMock(return_value="Remote work is productive."),
        )
        result = await OpenAIBackend(api_key="fake").complete("Is remote work productive?")
        assert isinstance(result, str)

    async def test_complete_json_parses_response(self, monkeypatch):
        raw = '{"evidence": "13% gain", "supports": true, "confidence": 0.85}'
        monkeypatch.setattr(
            "trustandverify.llm.openai.OpenAIBackend.complete",
            AsyncMock(return_value=raw),
        )
        result = await OpenAIBackend(api_key="fake").complete_json("extract evidence")
        assert result["supports"] is True
        assert result["confidence"] == 0.85

    async def test_complete_json_handles_markdown_fence(self, monkeypatch):
        raw = '```json\n{"supports": false, "confidence": 0.3}\n```'
        monkeypatch.setattr(
            "trustandverify.llm.openai.OpenAIBackend.complete",
            AsyncMock(return_value=raw),
        )
        result = await OpenAIBackend(api_key="fake").complete_json("extract")
        assert result["supports"] is False


# ── AnthropicBackend ───────────────────────────────────────────────────────────

class TestAnthropicBackend:
    def test_is_available_with_key(self):
        assert AnthropicBackend(api_key="sk-ant-fake").is_available() is True

    def test_is_available_without_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert AnthropicBackend(api_key="").is_available() is False

    async def test_complete_returns_string(self, monkeypatch):
        monkeypatch.setattr(
            "trustandverify.llm.anthropic.AnthropicBackend.complete",
            AsyncMock(return_value="Coffee has health benefits."),
        )
        result = await AnthropicBackend(api_key="fake").complete("Is coffee healthy?")
        assert isinstance(result, str)

    async def test_complete_json_parses_response(self, monkeypatch):
        raw = '{"evidence": "antioxidants found", "supports": true, "confidence": 0.75}'
        monkeypatch.setattr(
            "trustandverify.llm.anthropic.AnthropicBackend.complete",
            AsyncMock(return_value=raw),
        )
        result = await AnthropicBackend(api_key="fake").complete_json("extract")
        assert result["confidence"] == 0.75


# ── OllamaBackend ──────────────────────────────────────────────────────────────

class TestOllamaBackend:
    def test_is_available_always_true(self):
        assert OllamaBackend().is_available() is True

    def test_default_model(self):
        assert OllamaBackend().model == "llama3"

    def test_custom_model(self):
        assert OllamaBackend(model="mistral").model == "mistral"

    async def test_complete_returns_string(self, monkeypatch):
        monkeypatch.setattr(
            "trustandverify.llm.ollama.OllamaBackend.complete",
            AsyncMock(return_value="Some completion."),
        )
        result = await OllamaBackend().complete("test prompt")
        assert isinstance(result, str)

    async def test_complete_json_handles_fence(self, monkeypatch):
        raw = '```json\n{"items": ["Claim A", "Claim B"]}\n```'
        monkeypatch.setattr(
            "trustandverify.llm.ollama.OllamaBackend.complete",
            AsyncMock(return_value=raw),
        )
        result = await OllamaBackend().complete_json("decompose")
        assert "items" in result
