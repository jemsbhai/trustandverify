"""Tests for search/tavily.py — mock HTTP, parse response, error handling."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trustandverify.core.models import SearchResult
from trustandverify.search.tavily import TavilySearch


@pytest.fixture
def tavily_response() -> dict:
    """Canned Tavily API response."""
    return {
        "results": [
            {
                "title": "Remote Work Productivity Study",
                "url": "https://www.nber.org/papers/w123",
                "content": "Stanford study found remote workers are 13% more productive.",
                "score": 0.92,
            },
            {
                "title": "Office vs Remote: Meta-analysis",
                "url": "https://harvard.edu/research/remote",
                "content": "Meta-analysis of 50 studies shows mixed results.",
                "score": 0.78,
            },
        ]
    }


class TestTavilySearch:
    def test_is_available_true_when_key_set(self):
        backend = TavilySearch(api_key="tvly-fake-key")
        assert backend.is_available() is True

    def test_is_available_false_when_no_key(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        backend = TavilySearch(api_key="")
        assert backend.is_available() is False

    async def test_search_returns_empty_when_no_key(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        backend = TavilySearch(api_key="")
        results = await backend.search("remote work productivity")
        assert results == []

    async def test_search_parses_results_correctly(self, tavily_response):
        mock_response = MagicMock()
        mock_response.json.return_value = tavily_response
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("trustandverify.search.tavily.httpx.AsyncClient", return_value=mock_client):
            backend = TavilySearch(api_key="tvly-fake")
            results = await backend.search("remote work productivity", max_results=5)

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].title == "Remote Work Productivity Study"
        assert results[0].url == "https://www.nber.org/papers/w123"
        assert results[0].score == 0.92

    async def test_search_returns_empty_on_http_error(self):
        import httpx

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(
            side_effect=httpx.RequestError("connection failed", request=MagicMock())
        )

        with patch("trustandverify.search.tavily.httpx.AsyncClient", return_value=mock_client):
            backend = TavilySearch(api_key="tvly-fake")
            results = await backend.search("remote work productivity")

        assert results == []

    async def test_search_handles_empty_results(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("trustandverify.search.tavily.httpx.AsyncClient", return_value=mock_client):
            backend = TavilySearch(api_key="tvly-fake")
            results = await backend.search("obscure query")

        assert results == []
