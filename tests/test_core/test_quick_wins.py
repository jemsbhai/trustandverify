"""Quick-win coverage tests — small gaps across many modules."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from jsonld_ex.confidence_algebra import Opinion

from trustandverify.core.agent import TrustAgent
from trustandverify.core.config import TrustConfig
from trustandverify.core.models import (
    Claim, Conflict, Evidence, Report, Source, Verdict,
)
from trustandverify.export.html import HtmlExporter
from trustandverify.llm.gemini import _parse_json_robust
from trustandverify.llm.prompts import decompose_query
from trustandverify.search.brave import BraveSearch
from trustandverify.storage.memory import InMemoryStorage
from trustandverify.storage.sqlite import SQLiteStorage


# ── __init__.py: verify() one-liner ──────────────────────────────────────────


class TestVerifyOneLiner:
    async def test_verify_calls_agent(self):
        """verify() in __init__ creates an agent and calls .verify()."""
        mock_report = MagicMock(spec=Report)

        with (
            patch("trustandverify.TrustAgent") as MockAgent,
        ):
            instance = MockAgent.return_value
            instance.verify = AsyncMock(return_value=mock_report)

            from trustandverify import verify
            result = await verify("Is coffee healthy?", num_claims=3, verbose=True)

            assert result is mock_report
            instance.verify.assert_called_once_with("Is coffee healthy?", verbose=True)

    async def test_verify_propagates_num_claims(self):
        """verify(num_claims=N) must pass TrustConfig(num_claims=N) to TrustAgent."""
        mock_report = MagicMock(spec=Report)

        with patch("trustandverify.TrustAgent") as MockAgent:
            instance = MockAgent.return_value
            instance.verify = AsyncMock(return_value=mock_report)

            from trustandverify import verify
            await verify("q", num_claims=7)

            # Verify TrustAgent was constructed with the right config
            call_kwargs = MockAgent.call_args
            config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
            assert config is not None, "TrustConfig not passed to TrustAgent"
            assert config.num_claims == 7, f"Expected num_claims=7, got {config.num_claims}"


# ── core/agent.py: cache=None + enable_cache=True default branch ─────────────


class TestAgentCacheDefault:
    def test_cache_enabled_creates_file_cache(self):
        from trustandverify.cache.file_cache import FileCache
        agent = TrustAgent(
            config=TrustConfig(enable_cache=True),
            search=MagicMock(),
            llm=MagicMock(),
        )
        assert isinstance(agent.cache, FileCache)

    def test_cache_explicit(self):
        custom_cache = MagicMock()
        agent = TrustAgent(
            config=TrustConfig(enable_cache=True),
            search=MagicMock(),
            llm=MagicMock(),
            cache=custom_cache,
        )
        assert agent.cache is custom_cache

    def test_cache_disabled(self):
        agent = TrustAgent(
            config=TrustConfig(enable_cache=False),
            search=MagicMock(),
            llm=MagicMock(),
        )
        assert agent.cache is None


# ── export/html.py: empty summary + no-opinion claim ─────────────────────────


class TestHtmlExporterGaps:
    def _make_report(self, summary="", opinion=None, evidence=None) -> Report:
        from datetime import datetime, timezone
        claim = Claim(
            text="Test claim",
            verdict=Verdict.NO_EVIDENCE,
            assessment="",
            opinion=opinion,
            evidence=evidence or [],
        )
        return Report(
            id="test-id",
            query="test query",
            claims=[claim],
            conflicts=[],
            summary=summary,
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

    def test_empty_summary_omits_block(self):
        report = self._make_report(summary="")
        html = HtmlExporter().render(report)
        assert "Executive Summary" not in html

    def test_claim_without_opinion_shows_dash(self):
        report = self._make_report(opinion=None)
        html = HtmlExporter().render(report)
        assert "—" in html  # em dash for missing probability

    def test_claim_with_opinion(self):
        op = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5)
        report = self._make_report(opinion=op)
        html = HtmlExporter().render(report)
        assert "Belief" in html
        assert "opinion-bar" in html


# ── llm/gemini.py: array extraction fallback paths ───────────────────────────


class TestGeminiParserGaps:
    def test_fence_with_no_newline(self):
        """Fence that starts with ``` but has no newline should still work."""
        raw = '```{"key": "val"}```'
        result = _parse_json_robust(raw)
        # Falls through fence stripping → extracts { } block
        assert result == {"key": "val"}

    def test_array_in_text_fallback(self):
        """Array extraction when direct and fence parsing both fail."""
        raw = 'Some preamble text ["Claim A", "Claim B"] and trailing text'
        result = _parse_json_robust(raw)
        assert result == {"items": ["Claim A", "Claim B"]}

    def test_invalid_json_in_braces(self):
        """Step 3: { } extraction with invalid JSON inside."""
        raw = 'Some text {not: valid json, missing quotes} end'
        result = _parse_json_robust(raw)
        # Falls through brace extraction (fails parse) to step 4 or unparseable
        assert isinstance(result, dict)

    def test_invalid_json_in_brackets(self):
        """Step 4: [ ] extraction with invalid JSON inside."""
        raw = 'Some text [not valid, json array] end'
        result = _parse_json_robust(raw)
        # Falls through bracket extraction (fails parse) to unparseable
        assert isinstance(result, dict)

    def test_completely_unparseable_prints_warning(self, capsys):
        result = _parse_json_robust("not json at all")
        assert result == {}
        captured = capsys.readouterr()
        assert "Could not parse" in captured.out


# ── llm/prompts.py: num_claims=0 branch ──────────────────────────────────────


class TestPromptsGap:
    def test_decompose_zero_claims(self):
        prompt = decompose_query("Is coffee healthy?", num_claims=0)
        assert "3-5" in prompt
        assert "Is coffee healthy?" in prompt

    def test_decompose_specific_claims(self):
        prompt = decompose_query("Is coffee healthy?", num_claims=4)
        assert "exactly 4" in prompt


# ── search/brave.py: generic error path ───────────────────────────────────────


class TestBraveGenericError:
    async def test_generic_exception(self):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=RuntimeError("unexpected"))

        with patch("trustandverify.search.brave.httpx.AsyncClient", return_value=mock_client):
            results = await BraveSearch(api_key="fake").search("test")
        assert results == []

    async def test_http_status_error(self):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "rate limited",
                request=MagicMock(),
                response=MagicMock(status_code=429),
            )
        )

        with patch("trustandverify.search.brave.httpx.AsyncClient", return_value=mock_client):
            results = await BraveSearch(api_key="fake").search("test")
        assert results == []


# ── storage/memory.py: save_claim + get_claims_for_query ─────────────────────


class TestMemoryStorageGaps:
    async def test_save_claim(self):
        storage = InMemoryStorage()
        claim = Claim(text="Test claim")
        result = await storage.save_claim(claim, "query-1")
        assert result == "Test claim"

    async def test_get_claims_for_query(self):
        storage = InMemoryStorage()
        claims = await storage.get_claims_for_query("nonexistent")
        assert claims == []

    async def test_save_and_retrieve_claims(self):
        storage = InMemoryStorage()
        c1 = Claim(text="Claim A")
        c2 = Claim(text="Claim B")
        await storage.save_claim(c1, "q1")
        await storage.save_claim(c2, "q1")
        claims = await storage.get_claims_for_query("q1")
        assert len(claims) == 2
        assert claims[0].text == "Claim A"
        assert claims[1].text == "Claim B"


# ── storage/sqlite.py: save_claim + get_claims_for_query ─────────────────────


class TestSqliteStorageGaps:
    async def test_save_claim(self):
        storage = SQLiteStorage(path=":memory:")
        claim = Claim(text="Test claim")
        result = await storage.save_claim(claim, "query-1")
        assert result == "Test claim"

    async def test_get_claims_for_query(self):
        storage = SQLiteStorage(path=":memory:")
        claims = await storage.get_claims_for_query("nonexistent")
        assert claims == []
