"""Tests for jac_interop.py — Jac bridge functions (no Jac runtime needed)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from jsonld_ex.confidence_algebra import Opinion

from trustandverify.core.agent import TrustAgent
from trustandverify.core.config import TrustConfig
from trustandverify.core.models import (
    Claim,
    Conflict,
    Evidence,
    Report,
    SearchResult,
    Source,
    Verdict,
)
from trustandverify.jac_interop import (
    _dict_to_report,
    _make_exporter,
    _make_llm,
    _make_search,
    _make_storage,
    _report_to_dict,
    _run_async,
    jac_configure_agent,
    jac_export,
    jac_verify,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_report() -> Report:
    return Report(
        id="test-001",
        query="Is coffee healthy?",
        claims=[
            Claim(
                text="Coffee has antioxidants",
                verdict=Verdict.SUPPORTED,
                assessment="Well supported.",
                opinion=Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5),
                evidence=[
                    Evidence(
                        text="Rich in antioxidants",
                        supports_claim=True,
                        relevance=0.9,
                        confidence_raw=0.85,
                        source=Source(
                            url="https://example.com",
                            title="Study",
                            content_snippet="Coffee contains...",
                            trust_score=0.8,
                        ),
                    ),
                ],
            ),
        ],
        conflicts=[
            Conflict(
                claim_text="Coffee insomnia",
                conflict_degree=0.35,
                num_supporting=2,
                num_contradicting=1,
            ),
        ],
        summary="Coffee has benefits.",
        created_at=datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc),
    )


# ── _make_search ──────────────────────────────────────────────────────────────


class TestMakeSearch:
    def test_tavily(self):
        from trustandverify.search.tavily import TavilySearch

        s = _make_search("tavily")
        assert isinstance(s, TavilySearch)

    def test_brave(self):
        from trustandverify.search.brave import BraveSearch

        s = _make_search("brave")
        assert isinstance(s, BraveSearch)

    def test_bing(self):
        from trustandverify.search.bing import BingSearch

        s = _make_search("bing")
        assert isinstance(s, BingSearch)

    def test_serpapi(self):
        from trustandverify.search.serpapi import SerpAPISearch

        s = _make_search("serpapi")
        assert isinstance(s, SerpAPISearch)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown search backend"):
            _make_search("google")

    def test_case_insensitive(self):
        from trustandverify.search.tavily import TavilySearch

        assert isinstance(_make_search("TAVILY"), TavilySearch)
        assert isinstance(_make_search(" Tavily "), TavilySearch)


# ── _make_llm ─────────────────────────────────────────────────────────────────


class TestMakeLlm:
    def test_gemini(self):
        from trustandverify.llm.gemini import GeminiBackend

        assert isinstance(_make_llm("gemini"), GeminiBackend)

    def test_openai(self):
        from trustandverify.llm.openai import OpenAIBackend

        assert isinstance(_make_llm("openai"), OpenAIBackend)

    def test_anthropic(self):
        from trustandverify.llm.anthropic import AnthropicBackend

        assert isinstance(_make_llm("anthropic"), AnthropicBackend)

    def test_ollama(self):
        from trustandverify.llm.ollama import OllamaBackend

        assert isinstance(_make_llm("ollama"), OllamaBackend)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM backend"):
            _make_llm("gpt5")


# ── _make_storage ─────────────────────────────────────────────────────────────


class TestMakeStorage:
    def test_memory(self):
        from trustandverify.storage.memory import InMemoryStorage

        assert isinstance(_make_storage("memory", ""), InMemoryStorage)

    def test_sqlite(self, tmp_path):
        from trustandverify.storage.sqlite import SQLiteStorage

        s = _make_storage("sqlite", str(tmp_path / "test.db"))
        assert isinstance(s, SQLiteStorage)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown storage backend"):
            _make_storage("postgres", "")


# ── _make_exporter ────────────────────────────────────────────────────────────


class TestMakeExporter:
    def test_jsonld(self):
        from trustandverify.export.jsonld import JsonLdExporter

        assert isinstance(_make_exporter("jsonld"), JsonLdExporter)

    def test_markdown(self):
        from trustandverify.export.markdown import MarkdownExporter

        assert isinstance(_make_exporter("markdown"), MarkdownExporter)
        assert isinstance(_make_exporter("md"), MarkdownExporter)

    def test_html(self):
        from trustandverify.export.html import HtmlExporter

        assert isinstance(_make_exporter("html"), HtmlExporter)

    def test_pdf(self):
        from trustandverify.export.pdf import PdfExporter

        assert isinstance(_make_exporter("pdf"), PdfExporter)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown export format"):
            _make_exporter("docx")


# ── _report_to_dict / _dict_to_report round-trip ─────────────────────────────


class TestReportSerialization:
    def test_round_trip(self, sample_report):
        d = _report_to_dict(sample_report)
        restored = _dict_to_report(d)
        assert restored.id == sample_report.id
        assert restored.query == sample_report.query
        assert restored.summary == sample_report.summary
        assert len(restored.claims) == len(sample_report.claims)
        assert len(restored.conflicts) == len(sample_report.conflicts)

    def test_claim_round_trip(self, sample_report):
        d = _report_to_dict(sample_report)
        restored = _dict_to_report(d)
        c = restored.claims[0]
        assert c.text == "Coffee has antioxidants"
        assert c.verdict == Verdict.SUPPORTED
        assert c.assessment == "Well supported."

    def test_opinion_round_trip(self, sample_report):
        d = _report_to_dict(sample_report)
        restored = _dict_to_report(d)
        op = restored.claims[0].opinion
        assert op is not None
        assert abs(op.belief - 0.7) < 1e-6
        assert abs(op.disbelief - 0.1) < 1e-6
        assert abs(op.uncertainty - 0.2) < 1e-6

    def test_opinion_additivity_preserved(self, sample_report):
        d = _report_to_dict(sample_report)
        restored = _dict_to_report(d)
        op = restored.claims[0].opinion
        total = op.belief + op.disbelief + op.uncertainty
        assert abs(total - 1.0) < 1e-6

    def test_evidence_round_trip(self, sample_report):
        d = _report_to_dict(sample_report)
        restored = _dict_to_report(d)
        ev = restored.claims[0].evidence[0]
        assert ev.text == "Rich in antioxidants"
        assert ev.supports_claim is True
        assert ev.source.url == "https://example.com"

    def test_conflict_round_trip(self, sample_report):
        d = _report_to_dict(sample_report)
        restored = _dict_to_report(d)
        c = restored.conflicts[0]
        assert c.claim_text == "Coffee insomnia"
        assert c.conflict_degree == 0.35

    def test_projected_probability_in_dict(self, sample_report):
        d = _report_to_dict(sample_report)
        op_dict = d["claims"][0]["opinion"]
        assert "projected_probability" in op_dict
        expected = 0.7 + 0.5 * 0.2  # belief + base_rate * uncertainty
        assert abs(op_dict["projected_probability"] - expected) < 1e-6

    def test_none_opinion_claim(self):
        report = Report(
            id="x",
            query="q",
            claims=[Claim(text="No opinion", opinion=None)],
            conflicts=[],
            summary="s",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        d = _report_to_dict(report)
        assert d["claims"][0]["opinion"] is None
        restored = _dict_to_report(d)
        assert restored.claims[0].opinion is None


# ── jac_configure_agent ───────────────────────────────────────────────────────


class TestJacConfigureAgent:
    def test_returns_agent(self):
        from trustandverify.core.agent import TrustAgent

        agent = jac_configure_agent()
        assert isinstance(agent, TrustAgent)

    def test_default_backends(self):
        agent = jac_configure_agent()
        assert agent.search.__class__.__name__ == "TavilySearch"
        assert agent.llm.__class__.__name__ == "GeminiBackend"

    def test_custom_backends(self):
        agent = jac_configure_agent(search_backend="brave", llm_backend="openai")
        assert agent.search.__class__.__name__ == "BraveSearch"
        assert agent.llm.__class__.__name__ == "OpenAIBackend"

    def test_config_passed(self):
        agent = jac_configure_agent(num_claims=5, enable_byzantine=True)
        assert agent.config.num_claims == 5
        assert agent.config.enable_byzantine is True

    def test_sqlite_storage(self, tmp_path):
        agent = jac_configure_agent(storage_backend="sqlite", db_path=str(tmp_path / "test.db"))
        assert agent.storage.__class__.__name__ == "SQLiteStorage"


# ── jac_verify ────────────────────────────────────────────────────────────────


class TestJacVerify:
    def test_returns_dict(self, sample_report):
        with patch("trustandverify.jac_interop._run_async", return_value=sample_report):
            result = jac_verify("Is coffee healthy?", enable_cache=False)
        assert isinstance(result, dict)
        assert result["query"] == "Is coffee healthy?"
        assert len(result["claims"]) == 1

    def test_dict_has_expected_keys(self, sample_report):
        with patch("trustandverify.jac_interop._run_async", return_value=sample_report):
            result = jac_verify("q", enable_cache=False)
        assert set(result.keys()) == {"id", "query", "claims", "conflicts", "summary", "created_at"}

    def test_claim_dict_structure(self, sample_report):
        with patch("trustandverify.jac_interop._run_async", return_value=sample_report):
            result = jac_verify("q", enable_cache=False)
        claim = result["claims"][0]
        assert set(claim.keys()) == {"text", "verdict", "assessment", "opinion", "evidence"}

    def test_opinion_dict_structure(self, sample_report):
        with patch("trustandverify.jac_interop._run_async", return_value=sample_report):
            result = jac_verify("q", enable_cache=False)
        op = result["claims"][0]["opinion"]
        assert set(op.keys()) == {
            "belief",
            "disbelief",
            "uncertainty",
            "base_rate",
            "projected_probability",
        }


# ── jac_export ────────────────────────────────────────────────────────────────


class TestJacExport:
    def test_jsonld_returns_string(self, sample_report):
        d = _report_to_dict(sample_report)
        result = jac_export(d, format="jsonld")
        assert isinstance(result, str)
        assert "TrustGraphReport" in result

    def test_markdown_returns_string(self, sample_report):
        d = _report_to_dict(sample_report)
        result = jac_export(d, format="markdown")
        assert isinstance(result, str)
        assert "TrustGraph" in result

    def test_html_returns_string(self, sample_report):
        d = _report_to_dict(sample_report)
        result = jac_export(d, format="html")
        assert isinstance(result, str)
        assert "<!DOCTYPE html>" in result

    def test_writes_to_file(self, sample_report, tmp_path):
        d = _report_to_dict(sample_report)
        path = str(tmp_path / "report.md")
        jac_export(d, format="markdown", output_path=path)
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "TrustGraph" in content

    def test_pdf_writes_binary(self, sample_report, tmp_path):
        """PDF export must write bytes via 'wb' mode, not 'w'."""
        d = _report_to_dict(sample_report)
        path = str(tmp_path / "report.pdf")

        mock_html_instance = MagicMock()
        mock_html_instance.write_pdf.return_value = b"%PDF-1.4 fake content"
        mock_html_class = MagicMock(return_value=mock_html_instance)

        with patch.dict("sys.modules", {"weasyprint": MagicMock(HTML=mock_html_class)}):
            result = jac_export(d, format="pdf", output_path=path)

        assert isinstance(result, bytes)
        with open(path, "rb") as fh:
            content = fh.read()
        assert content == b"%PDF-1.4 fake content"

    def test_unknown_format_raises(self, sample_report):
        d = _report_to_dict(sample_report)
        with pytest.raises(ValueError, match="Unknown export format"):
            jac_export(d, format="docx")


# ── _run_async() ────────────────────────────────────────────────────────────


class TestRunAsync:
    """Test both branches of _run_async()."""

    def test_no_running_loop_uses_asyncio_run(self):
        """When no event loop is running, _run_async uses asyncio.run()."""

        async def coro():
            return 42

        result = _run_async(coro())
        assert result == 42

    def test_running_loop_uses_nest_asyncio(self):
        """When an event loop IS running, _run_async uses nest_asyncio."""
        import nest_asyncio

        nest_asyncio.apply()

        async def inner():
            return 99

        async def outer():
            # We're inside a running loop here
            return _run_async(inner())

        result = asyncio.run(outer())
        assert result == 99


# ── _make_search("multi") ───────────────────────────────────────────────


class TestMakeSearchMulti:
    """All 3 branches of _make_search('multi')."""

    def test_multi_multiple_backends(self, monkeypatch):
        """When 2+ backends have keys, returns MultiSearch."""
        monkeypatch.setenv("TAVILY_API_KEY", "fake-tavily")
        monkeypatch.setenv("BRAVE_API_KEY", "fake-brave")
        monkeypatch.delenv("BING_API_KEY", raising=False)

        from trustandverify.search.multi import MultiSearch

        result = _make_search("multi")
        assert isinstance(result, MultiSearch)

    def test_multi_single_backend(self, monkeypatch):
        """When only 1 backend has a key, returns that single backend."""
        monkeypatch.setenv("TAVILY_API_KEY", "fake-tavily")
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        monkeypatch.delenv("BING_API_KEY", raising=False)

        from trustandverify.search.tavily import TavilySearch

        result = _make_search("multi")
        assert isinstance(result, TavilySearch)

    def test_multi_no_backends_raises(self, monkeypatch):
        """When no backends have keys, raises RuntimeError."""
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        monkeypatch.delenv("BING_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="No search backends available"):
            _make_search("multi")


# ── Equivalence: jac_verify() vs TrustAgent.verify() ─────────────────────
#
# These tests are SYNC (not async def) so that jac_verify() can call
# _run_async() → asyncio.run() without hitting a running event loop.
# The direct TrustAgent path also uses asyncio.run().
# Fresh mocks are created for each path to avoid shared state.


def _fresh_mock_llm():
    """Create a fresh canned mock LLM. Deterministic responses."""
    llm = MagicMock()

    async def complete(prompt, system=""):
        if "search query" in prompt.lower() or "web search" in prompt.lower():
            return "coffee health research"
        if "assessment" in prompt.lower() or "assess" in prompt.lower():
            return "Evidence supports the claim."
        if "summary" in prompt.lower() or "executive" in prompt.lower():
            return "Coffee appears to be healthy overall."
        return "generic response"

    async def complete_json(prompt, system="", defaults=None):
        if "decompose" in prompt.lower() or "verifiable" in prompt.lower():
            return {"items": ["Coffee contains antioxidants.", "Coffee may cause insomnia."]}
        return {
            "evidence": "Studies show health benefits.",
            "supports": True,
            "relevance": 0.85,
            "confidence": 0.8,
        }

    llm.complete = complete
    llm.complete_json = complete_json
    llm.is_available = MagicMock(return_value=True)
    llm.name = "mock"
    llm.model = "mock-1"
    return llm


def _fresh_mock_search():
    """Create a fresh canned mock search."""
    search = MagicMock()
    search.search = AsyncMock(
        return_value=[
            SearchResult(
                title="Study", url="https://example.com", content="Coffee is healthy.", score=0.9
            ),
        ]
    )
    search.is_available = MagicMock(return_value=True)
    search.name = "mock"
    return search


class TestJacVerifyEquivalence:
    """Verify jac_verify() produces identical results to TrustAgent.verify().

    IMPORTANT: These tests are sync so jac_verify() can use asyncio.run().
    Both paths use identically-configured canned mocks. The jac_verify()
    call is REAL — it goes through jac_configure_agent(), _run_async(),
    agent.verify(), and _report_to_dict(). Nothing is patched out except
    the backend factories.
    """

    def test_jac_verify_produces_claims(self):
        """jac_verify() must actually run the pipeline and return claims."""
        with (
            patch("trustandverify.jac_interop._make_search", return_value=_fresh_mock_search()),
            patch("trustandverify.jac_interop._make_llm", return_value=_fresh_mock_llm()),
        ):
            jac_result = jac_verify("Is coffee healthy?", num_claims=2, enable_cache=False)

        assert isinstance(jac_result, dict)
        assert jac_result["query"] == "Is coffee healthy?"
        assert len(jac_result["claims"]) == 2
        assert jac_result["claims"][0]["text"] == "Coffee contains antioxidants."
        assert jac_result["claims"][1]["text"] == "Coffee may cause insomnia."

    def test_jac_verify_claims_have_opinions(self):
        """Every claim from jac_verify() must have a fused opinion."""
        with (
            patch("trustandverify.jac_interop._make_search", return_value=_fresh_mock_search()),
            patch("trustandverify.jac_interop._make_llm", return_value=_fresh_mock_llm()),
        ):
            jac_result = jac_verify("Is coffee healthy?", num_claims=2, enable_cache=False)

        for claim in jac_result["claims"]:
            op = claim["opinion"]
            assert op is not None, f"No opinion for claim: {claim['text']}"
            assert 0.0 <= op["belief"] <= 1.0
            assert 0.0 <= op["disbelief"] <= 1.0
            assert 0.0 <= op["uncertainty"] <= 1.0
            total = op["belief"] + op["disbelief"] + op["uncertainty"]
            assert abs(total - 1.0) < 1e-6, f"b+d+u = {total} != 1.0 for claim: {claim['text']}"

    def test_jac_verify_matches_direct_agent(self):
        """jac_verify() and TrustAgent.verify() must produce identical results.

        This is the critical equivalence test. Both paths use identically-
        behaving mocks. The outputs must match: same claims, same verdicts,
        same opinion values (to machine precision), same summary.
        """
        # Path A: jac_verify() — the full interop codepath
        with (
            patch("trustandverify.jac_interop._make_search", return_value=_fresh_mock_search()),
            patch("trustandverify.jac_interop._make_llm", return_value=_fresh_mock_llm()),
        ):
            jac_result = jac_verify("Is coffee healthy?", num_claims=2, enable_cache=False)

        # Path B: TrustAgent.verify() directly
        agent = TrustAgent(
            config=TrustConfig(num_claims=2, enable_cache=False),
            search=_fresh_mock_search(),
            llm=_fresh_mock_llm(),
        )
        direct_report = asyncio.run(agent.verify("Is coffee healthy?"))

        # ── Structural equivalence ──
        assert len(jac_result["claims"]) == len(direct_report.claims)

        for jc, dc in zip(jac_result["claims"], direct_report.claims, strict=True):
            # Same text
            assert jc["text"] == dc.text
            # Same verdict
            assert jc["verdict"] == dc.verdict.value
            # Same opinion values
            if dc.opinion is not None:
                jop = jc["opinion"]
                assert abs(jop["belief"] - dc.opinion.belief) < 1e-10
                assert abs(jop["disbelief"] - dc.opinion.disbelief) < 1e-10
                assert abs(jop["uncertainty"] - dc.opinion.uncertainty) < 1e-10
                assert (
                    abs(jop["projected_probability"] - dc.opinion.projected_probability()) < 1e-10
                )

        # Same summary
        assert jac_result["summary"] == direct_report.summary

    def test_jac_verify_dict_round_trips_cleanly(self):
        """The dict from jac_verify() must survive _dict_to_report() and back."""
        with (
            patch("trustandverify.jac_interop._make_search", return_value=_fresh_mock_search()),
            patch("trustandverify.jac_interop._make_llm", return_value=_fresh_mock_llm()),
        ):
            jac_result = jac_verify("Is coffee healthy?", num_claims=2, enable_cache=False)

        # Round-trip: dict → Report → dict
        restored_report = _dict_to_report(jac_result)
        re_serialised = _report_to_dict(restored_report)

        # Claims must survive
        assert len(re_serialised["claims"]) == len(jac_result["claims"])
        for orig, rest in zip(jac_result["claims"], re_serialised["claims"], strict=True):
            assert orig["text"] == rest["text"]
            assert orig["verdict"] == rest["verdict"]
            if orig["opinion"] is not None:
                for key in ("belief", "disbelief", "uncertainty"):
                    assert abs(orig["opinion"][key] - rest["opinion"][key]) < 1e-10
                # Additivity preserved
                total = (
                    rest["opinion"]["belief"]
                    + rest["opinion"]["disbelief"]
                    + rest["opinion"]["uncertainty"]
                )
                assert abs(total - 1.0) < 1e-6
