"""Tests for jac_interop.py — Jac bridge functions (no Jac runtime needed)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from jsonld_ex.confidence_algebra import Opinion

from trustandverify.core.config import TrustConfig
from trustandverify.core.models import Claim, Conflict, Evidence, Report, Source, Verdict
from trustandverify.jac_interop import (
    _dict_to_report,
    _make_exporter,
    _make_llm,
    _make_search,
    _make_storage,
    _report_to_dict,
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
            id="x", query="q",
            claims=[Claim(text="No opinion", opinion=None)],
            conflicts=[], summary="s",
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
        agent = jac_configure_agent(
            storage_backend="sqlite", db_path=str(tmp_path / "test.db")
        )
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
        assert set(op.keys()) == {"belief", "disbelief", "uncertainty", "base_rate", "projected_probability"}


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

    def test_unknown_format_raises(self, sample_report):
        d = _report_to_dict(sample_report)
        with pytest.raises(ValueError, match="Unknown export format"):
            jac_export(d, format="docx")
