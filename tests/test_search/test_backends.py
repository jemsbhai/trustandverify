"""Tests for search backends — Brave, Bing, SerpAPI, and MultiSearch."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trustandverify.core.models import SearchResult
from trustandverify.search.bing import BingSearch
from trustandverify.search.brave import BraveSearch
from trustandverify.search.multi import MultiSearch
from trustandverify.search.serpapi import SerpAPISearch


# ── BraveSearch ────────────────────────────────────────────────────────────────

class TestBraveSearch:
    def test_is_available_with_key(self):
        assert BraveSearch(api_key="fake").is_available() is True

    def test_is_available_without_key(self, monkeypatch):
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        assert BraveSearch(api_key="").is_available() is False

    async def test_returns_empty_without_key(self, monkeypatch):
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        results = await BraveSearch(api_key="").search("test")
        assert results == []

    async def test_parses_results(self):
        payload = {"web": {"results": [
            {"title": "T1", "url": "https://example.com", "description": "Desc", "score": 0.8},
        ]}}
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        mock_resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("trustandverify.search.brave.httpx.AsyncClient", return_value=mock_client):
            results = await BraveSearch(api_key="fake").search("test")

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].title == "T1"

    async def test_returns_empty_on_error(self):
        import httpx
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(
            side_effect=httpx.RequestError("fail", request=MagicMock())
        )
        with patch("trustandverify.search.brave.httpx.AsyncClient", return_value=mock_client):
            results = await BraveSearch(api_key="fake").search("test")
        assert results == []


# ── BingSearch ─────────────────────────────────────────────────────────────────

class TestBingSearch:
    def test_is_available_with_key(self):
        assert BingSearch(api_key="fake").is_available() is True

    def test_is_available_without_key(self, monkeypatch):
        monkeypatch.delenv("BING_API_KEY", raising=False)
        assert BingSearch(api_key="").is_available() is False

    async def test_returns_empty_without_key(self, monkeypatch):
        monkeypatch.delenv("BING_API_KEY", raising=False)
        results = await BingSearch(api_key="").search("test")
        assert results == []

    async def test_parses_results(self):
        payload = {"webPages": {"value": [
            {"name": "T1", "url": "https://example.com", "snippet": "Snip"},
        ]}}
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        mock_resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("trustandverify.search.bing.httpx.AsyncClient", return_value=mock_client):
            results = await BingSearch(api_key="fake").search("test")

        assert len(results) == 1
        assert results[0].url == "https://example.com"


# ── SerpAPISearch ──────────────────────────────────────────────────────────────

class TestSerpAPISearch:
    def test_is_available_with_key(self):
        assert SerpAPISearch(api_key="fake").is_available() is True

    def test_is_available_without_key(self, monkeypatch):
        monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
        assert SerpAPISearch(api_key="").is_available() is False

    async def test_parses_organic_results(self):
        payload = {"organic_results": [
            {"title": "T1", "link": "https://example.com", "snippet": "Snip"},
        ]}
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        mock_resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("trustandverify.search.serpapi.httpx.AsyncClient", return_value=mock_client):
            results = await SerpAPISearch(api_key="fake").search("test")

        assert len(results) == 1
        assert results[0].title == "T1"


# ── MultiSearch ────────────────────────────────────────────────────────────────

def _mock_backend(name: str, results: list[SearchResult], available: bool = True):
    b = MagicMock()
    b.name = name
    b.is_available = MagicMock(return_value=available)
    b.search = AsyncMock(return_value=results)
    return b


class TestMultiSearch:
    def test_raises_with_no_backends(self):
        with pytest.raises(ValueError):
            MultiSearch([])

    def test_is_available_if_any_backend_available(self):
        a = _mock_backend("a", [], available=True)
        b = _mock_backend("b", [], available=False)
        assert MultiSearch([a, b]).is_available() is True

    def test_is_unavailable_if_all_unavailable(self):
        a = _mock_backend("a", [], available=False)
        assert MultiSearch([a]).is_available() is False

    async def test_deduplicates_by_url(self):
        r1 = SearchResult(title="T1", url="https://dup.com", content="A", score=0.9)
        r2 = SearchResult(title="T1 copy", url="https://dup.com", content="B", score=0.8)
        r3 = SearchResult(title="T2", url="https://unique.com", content="C", score=0.7)

        a = _mock_backend("a", [r1, r3])
        b = _mock_backend("b", [r2])

        results = await MultiSearch([a, b]).search("test", max_results=10)
        urls = [r.url for r in results]
        assert urls.count("https://dup.com") == 1
        assert "https://unique.com" in urls

    async def test_respects_max_results(self):
        results_a = [SearchResult(title=f"T{i}", url=f"https://a.com/{i}", content="x", score=0.5)
                     for i in range(5)]
        results_b = [SearchResult(title=f"U{i}", url=f"https://b.com/{i}", content="y", score=0.5)
                     for i in range(5)]
        a = _mock_backend("a", results_a)
        b = _mock_backend("b", results_b)

        results = await MultiSearch([a, b]).search("test", max_results=3)
        assert len(results) <= 3

    async def test_skips_unavailable_backends(self):
        r = SearchResult(title="T", url="https://x.com", content="c", score=0.9)
        good = _mock_backend("good", [r], available=True)
        bad = _mock_backend("bad", [], available=False)

        results = await MultiSearch([good, bad]).search("test")
        assert len(results) == 1
        bad.search.assert_not_called()

    async def test_returns_empty_if_all_unavailable(self):
        a = _mock_backend("a", [], available=False)
        results = await MultiSearch([a]).search("test")
        assert results == []

    async def test_interleaves_results_for_diversity(self):
        """Results from both backends should appear before any are cut off."""
        ra = [SearchResult(title="A", url="https://a.com/1", content="", score=0.9)]
        rb = [SearchResult(title="B", url="https://b.com/1", content="", score=0.8)]
        a = _mock_backend("a", ra)
        b = _mock_backend("b", rb)

        results = await MultiSearch([a, b]).search("test", max_results=5)
        urls = {r.url for r in results}
        assert "https://a.com/1" in urls
        assert "https://b.com/1" in urls
