"""Additional search backend tests for full coverage — error paths and edge cases."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from trustandverify.core.models import SearchResult
from trustandverify.search.tavily import TavilySearch
from trustandverify.search.bing import BingSearch
from trustandverify.search.serpapi import SerpAPISearch


# ── TavilySearch (full coverage) ──────────────────────────────────────────────


class TestTavilySearchFull:
    def test_is_available_with_key(self):
        assert TavilySearch(api_key="fake").is_available() is True

    def test_is_available_without_key(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        assert TavilySearch(api_key="").is_available() is False

    async def test_returns_empty_without_key(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        results = await TavilySearch(api_key="").search("test")
        assert results == []

    async def test_parses_results(self):
        payload = {
            "results": [
                {"title": "T1", "url": "https://example.com", "content": "Text", "score": 0.9},
                {"title": "T2", "url": "https://example2.com", "content": "More", "score": 0.7},
            ]
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("trustandverify.search.tavily.httpx.AsyncClient", return_value=mock_client):
            results = await TavilySearch(api_key="fake").search("test", max_results=5)

        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].title == "T1"
        assert results[1].score == 0.7

    async def test_empty_results_field(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": []}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("trustandverify.search.tavily.httpx.AsyncClient", return_value=mock_client):
            results = await TavilySearch(api_key="fake").search("test")
        assert results == []

    async def test_http_status_error(self):
        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 429

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError("rate limited", request=mock_request, response=mock_response)
        )

        with patch("trustandverify.search.tavily.httpx.AsyncClient", return_value=mock_client):
            results = await TavilySearch(api_key="fake").search("test")
        assert results == []

    async def test_request_error(self):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(
            side_effect=httpx.RequestError("timeout", request=MagicMock())
        )

        with patch("trustandverify.search.tavily.httpx.AsyncClient", return_value=mock_client):
            results = await TavilySearch(api_key="fake").search("test")
        assert results == []

    async def test_unexpected_error(self):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=ValueError("unexpected"))

        with patch("trustandverify.search.tavily.httpx.AsyncClient", return_value=mock_client):
            results = await TavilySearch(api_key="fake").search("test")
        assert results == []


# ── BingSearch error paths ────────────────────────────────────────────────────


class TestBingSearchErrors:
    async def test_http_status_error(self):
        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 403

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError("forbidden", request=mock_request, response=mock_response)
        )

        with patch("trustandverify.search.bing.httpx.AsyncClient", return_value=mock_client):
            results = await BingSearch(api_key="fake").search("test")
        assert results == []

    async def test_generic_error(self):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=RuntimeError("boom"))

        with patch("trustandverify.search.bing.httpx.AsyncClient", return_value=mock_client):
            results = await BingSearch(api_key="fake").search("test")
        assert results == []


# ── SerpAPISearch error paths ─────────────────────────────────────────────────


class TestSerpAPISearchErrors:
    async def test_returns_empty_without_key(self, monkeypatch):
        monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
        results = await SerpAPISearch(api_key="").search("test")
        assert results == []

    async def test_http_status_error(self):
        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError("server error", request=mock_request, response=mock_response)
        )

        with patch("trustandverify.search.serpapi.httpx.AsyncClient", return_value=mock_client):
            results = await SerpAPISearch(api_key="fake").search("test")
        assert results == []

    async def test_generic_error(self):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=RuntimeError("boom"))

        with patch("trustandverify.search.serpapi.httpx.AsyncClient", return_value=mock_client):
            results = await SerpAPISearch(api_key="fake").search("test")
        assert results == []
