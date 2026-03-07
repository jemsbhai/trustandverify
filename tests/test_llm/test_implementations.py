"""Tests for LLM backend implementations — actual complete() and complete_json() bodies."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trustandverify.llm.anthropic import AnthropicBackend
from trustandverify.llm.gemini import GeminiBackend, _parse_json_robust
from trustandverify.llm.ollama import OllamaBackend
from trustandverify.llm.openai import OpenAIBackend

# ── _parse_json_robust (comprehensive) ────────────────────────────────────────


class TestParseJsonRobust:
    def test_plain_dict(self):
        assert _parse_json_robust('{"a": 1}') == {"a": 1}

    def test_plain_list_wrapped(self):
        result = _parse_json_robust('["a", "b"]')
        assert result == {"items": ["a", "b"]}

    def test_markdown_fence_dict(self):
        raw = '```json\n{"key": "value"}\n```'
        assert _parse_json_robust(raw) == {"key": "value"}

    def test_markdown_fence_list(self):
        raw = '```json\n["a", "b"]\n```'
        assert _parse_json_robust(raw) == {"items": ["a", "b"]}

    def test_embedded_json_in_text(self):
        raw = 'Here is the result: {"score": 0.9} and more text'
        assert _parse_json_robust(raw) == {"score": 0.9}

    def test_embedded_array_in_text(self):
        raw = 'Claims: ["Claim A", "Claim B"] done.'
        assert _parse_json_robust(raw) == {"items": ["Claim A", "Claim B"]}

    def test_unicode_whitespace_cleaned(self):
        raw = '{"key":\xa0"value"}'
        assert _parse_json_robust(raw) == {"key": "value"}

    def test_unparseable_returns_defaults(self):
        assert _parse_json_robust("totally not json", defaults={"fallback": True}) == {
            "fallback": True
        }

    def test_unparseable_returns_empty_dict_by_default(self):
        assert _parse_json_robust("garbage") == {}


# ── GeminiBackend ─────────────────────────────────────────────────────────────


class TestGeminiBackendImpl:
    def test_is_available_with_key(self):
        assert GeminiBackend(api_key="fake").is_available() is True

    def test_is_available_without_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        assert GeminiBackend(api_key="").is_available() is False

    async def test_complete_calls_litellm(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "LLM response text"

        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(return_value=mock_response)

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            backend = GeminiBackend(api_key="fake")
            result = await backend.complete("test prompt")
            assert result == "LLM response text"
            mock_litellm.acompletion.assert_called_once()

    async def test_complete_with_system_prompt(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "response"

        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(return_value=mock_response)

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            backend = GeminiBackend(api_key="fake")
            await backend.complete("prompt", system="You are helpful")
            call_kwargs = mock_litellm.acompletion.call_args
            messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
            assert any(m["role"] == "system" for m in messages)

    async def test_complete_without_system_prompt(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "response"

        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(return_value=mock_response)

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            backend = GeminiBackend(api_key="fake")
            await backend.complete("prompt")
            call_kwargs = mock_litellm.acompletion.call_args
            messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
            assert not any(m["role"] == "system" for m in messages)

    async def test_complete_raises_on_missing_litellm(self):
        with patch.dict("sys.modules", {"litellm": None}):
            backend = GeminiBackend(api_key="fake")
            with pytest.raises(ImportError):
                await backend.complete("test")

    async def test_complete_json_parses_response(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"score": 0.9}'

        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(return_value=mock_response)

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            backend = GeminiBackend(api_key="fake")
            result = await backend.complete_json("extract data")
            assert result == {"score": 0.9}

    async def test_complete_json_returns_defaults_on_garbage(self):
        """defaults kwarg must propagate to _parse_json_robust."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "totally not json"

        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(return_value=mock_response)

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            backend = GeminiBackend(api_key="fake")
            result = await backend.complete_json(
                "extract", defaults={"evidence": "", "supports": True}
            )
            assert result == {"evidence": "", "supports": True}


# ── OpenAIBackend ─────────────────────────────────────────────────────────────


class TestOpenAIBackendImpl:
    async def test_complete_calls_openai_sdk(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OpenAI response"

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"openai": mock_openai}):
            backend = OpenAIBackend(api_key="sk-fake")
            result = await backend.complete("test prompt")
            assert result == "OpenAI response"

    async def test_complete_with_system(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "response"

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"openai": mock_openai}):
            backend = OpenAIBackend(api_key="sk-fake")
            await backend.complete("prompt", system="Be helpful")
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
            assert any(m["role"] == "system" for m in messages)

    async def test_complete_raises_on_missing_sdk(self):
        with patch.dict("sys.modules", {"openai": None}):
            backend = OpenAIBackend(api_key="sk-fake")
            with pytest.raises(ImportError, match="openai"):
                await backend.complete("test")

    async def test_complete_json(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"key": "val"}'

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"openai": mock_openai}):
            backend = OpenAIBackend(api_key="sk-fake")
            result = await backend.complete_json("extract")
            assert result == {"key": "val"}


# ── AnthropicBackend ──────────────────────────────────────────────────────────


class TestAnthropicBackendImpl:
    async def test_complete_calls_anthropic_sdk(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Anthropic response"

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        mock_module = MagicMock()
        mock_module.AsyncAnthropic = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"anthropic": mock_module}):
            backend = AnthropicBackend(api_key="sk-ant-fake")
            result = await backend.complete("test")
            assert result == "Anthropic response"

    async def test_complete_with_system(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "response"

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        mock_module = MagicMock()
        mock_module.AsyncAnthropic = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"anthropic": mock_module}):
            backend = AnthropicBackend(api_key="sk-ant-fake")
            await backend.complete("prompt", system="Be helpful")
            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert "system" in call_kwargs

    async def test_complete_raises_on_missing_sdk(self):
        with patch.dict("sys.modules", {"anthropic": None}):
            backend = AnthropicBackend(api_key="sk-ant-fake")
            with pytest.raises(ImportError, match="anthropic"):
                await backend.complete("test")

    async def test_complete_json(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = '{"supports": true}'

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        mock_module = MagicMock()
        mock_module.AsyncAnthropic = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"anthropic": mock_module}):
            backend = AnthropicBackend(api_key="sk-ant-fake")
            result = await backend.complete_json("extract")
            assert result["supports"] is True


# ── OllamaBackend ─────────────────────────────────────────────────────────────


class TestOllamaBackendImpl:
    async def test_complete_calls_ollama_sdk(self):
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value={"message": {"content": "Ollama response"}})

        mock_module = MagicMock()
        mock_module.AsyncClient = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"ollama": mock_module}):
            backend = OllamaBackend()
            result = await backend.complete("test")
            assert result == "Ollama response"

    async def test_complete_with_system(self):
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value={"message": {"content": "r"}})

        mock_module = MagicMock()
        mock_module.AsyncClient = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"ollama": mock_module}):
            backend = OllamaBackend()
            await backend.complete("prompt", system="Be helpful")
            call_kwargs = mock_client.chat.call_args.kwargs
            messages = call_kwargs.get("messages")
            assert any(m["role"] == "system" for m in messages)

    async def test_complete_raises_on_missing_sdk(self):
        with patch.dict("sys.modules", {"ollama": None}):
            backend = OllamaBackend()
            with pytest.raises(ImportError, match="ollama"):
                await backend.complete("test")

    async def test_complete_json(self):
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value={"message": {"content": '{"items": ["a"]}'}})

        mock_module = MagicMock()
        mock_module.AsyncClient = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"ollama": mock_module}):
            backend = OllamaBackend()
            result = await backend.complete_json("decompose")
            assert result == {"items": ["a"]}
