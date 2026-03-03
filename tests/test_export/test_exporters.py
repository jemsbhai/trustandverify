"""Tests for Markdown, HTML, and PDF exporters — snapshot structure tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from jsonld_ex.confidence_algebra import Opinion

from trustandverify.core.models import Claim, Conflict, Evidence, Report, Source, Verdict
from trustandverify.export.html import HtmlExporter
from trustandverify.export.markdown import MarkdownExporter
from trustandverify.export.pdf import PdfExporter


@pytest.fixture
def sample_report() -> Report:
    source = Source(
        url="https://nber.org/w1",
        title="NBER Study",
        content_snippet="Remote workers 13% more productive.",
        trust_score=0.85,
    )
    evidence = Evidence(
        text="Remote workers showed 13% productivity gain.",
        supports_claim=True,
        relevance=0.9,
        confidence_raw=0.8,
        source=source,
        opinion=Opinion(belief=0.567, disbelief=0.1, uncertainty=0.333, base_rate=0.5),
    )
    contra_source = Source(
        url="https://hbr.org/collab",
        title="HBR Collaboration Study",
        content_snippet="Collaboration suffers remotely.",
        trust_score=0.75,
    )
    contra_evidence = Evidence(
        text="Collaboration metrics declined 20%.",
        supports_claim=False,
        relevance=0.7,
        confidence_raw=0.65,
        source=contra_source,
        opinion=Opinion(belief=0.1, disbelief=0.567, uncertainty=0.333, base_rate=0.5),
    )
    claim = Claim(
        text="Remote workers are more productive than office workers.",
        evidence=[evidence, contra_evidence],
        opinion=Opinion(belief=0.733, disbelief=0.1, uncertainty=0.167, base_rate=0.5),
        verdict=Verdict.SUPPORTED,
        assessment="Evidence broadly supports the claim with some caveats.",
    )
    conflict = Conflict(
        claim_text="Remote workers are more productive",
        conflict_degree=0.31,
        num_supporting=1,
        num_contradicting=1,
    )
    return Report(
        id="test-001",
        query="Is remote work more productive than office work?",
        claims=[claim],
        conflicts=[conflict],
        summary="The evidence broadly supports increased remote work productivity.",
        created_at=datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc),
    )


# ── MarkdownExporter ───────────────────────────────────────────────────────────

class TestMarkdownExporter:
    def test_render_returns_string(self, sample_report):
        result = MarkdownExporter().render(sample_report)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_query(self, sample_report):
        result = MarkdownExporter().render(sample_report)
        assert sample_report.query in result

    def test_contains_summary(self, sample_report):
        result = MarkdownExporter().render(sample_report)
        assert sample_report.summary in result

    def test_contains_claim_text(self, sample_report):
        result = MarkdownExporter().render(sample_report)
        assert sample_report.claims[0].text in result

    def test_contains_verdict(self, sample_report):
        result = MarkdownExporter().render(sample_report)
        assert "SUPPORTED" in result

    def test_contains_projected_probability(self, sample_report):
        result = MarkdownExporter().render(sample_report)
        p = sample_report.claims[0].opinion.projected_probability()
        assert f"{p:.3f}" in result

    def test_contains_source_url(self, sample_report):
        result = MarkdownExporter().render(sample_report)
        assert "https://nber.org/w1" in result

    def test_contains_conflict_info(self, sample_report):
        result = MarkdownExporter().render(sample_report)
        assert "0.310" in result or "0.31" in result

    def test_render_to_file(self, sample_report, tmp_path):
        path = str(tmp_path / "report.md")
        MarkdownExporter().render_to_file(sample_report, path)
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert sample_report.query in content

    def test_no_evidence_claim(self, sample_report):
        sample_report.claims[0].verdict = Verdict.NO_EVIDENCE
        sample_report.claims[0].opinion = None
        result = MarkdownExporter().render(sample_report)
        assert "NO_EVIDENCE" in result
        assert "—" in result


# ── HtmlExporter ───────────────────────────────────────────────────────────────

class TestHtmlExporter:
    def test_render_returns_string(self, sample_report):
        result = HtmlExporter().render(sample_report)
        assert isinstance(result, str)

    def test_is_valid_html_structure(self, sample_report):
        result = HtmlExporter().render(sample_report)
        assert "<!DOCTYPE html>" in result
        assert "<html" in result
        assert "</html>" in result

    def test_contains_query(self, sample_report):
        result = HtmlExporter().render(sample_report)
        assert sample_report.query in result

    def test_contains_summary(self, sample_report):
        result = HtmlExporter().render(sample_report)
        assert sample_report.summary in result

    def test_contains_verdict_text(self, sample_report):
        result = HtmlExporter().render(sample_report)
        assert "SUPPORTED" in result

    def test_contains_opinion_bar(self, sample_report):
        result = HtmlExporter().render(sample_report)
        assert "opinion-bar" in result

    def test_contains_source_link(self, sample_report):
        result = HtmlExporter().render(sample_report)
        assert "https://nber.org/w1" in result

    def test_contains_conflict_section(self, sample_report):
        result = HtmlExporter().render(sample_report)
        assert "Conflicts" in result
        assert "0.310" in result or "0.31" in result

    def test_no_conflict_section_when_empty(self, sample_report):
        sample_report.conflicts = []
        result = HtmlExporter().render(sample_report)
        assert "Evidence Conflicts" not in result

    def test_escapes_html_in_query(self):
        from trustandverify.core.models import Report
        from datetime import datetime, timezone
        report = Report(
            id="x", query="<script>alert('xss')</script>",
            claims=[], conflicts=[], summary="safe",
            created_at=datetime.now(timezone.utc),
        )
        result = HtmlExporter().render(report)
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_render_to_file(self, sample_report, tmp_path):
        path = str(tmp_path / "report.html")
        HtmlExporter().render_to_file(sample_report, path)
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "<!DOCTYPE html>" in content


# ── PdfExporter ────────────────────────────────────────────────────────────────

class TestPdfExporter:
    def test_raises_import_error_without_weasyprint(self, sample_report, monkeypatch):
        """PdfExporter raises a clear ImportError when weasyprint is not installed."""
        import sys
        monkeypatch.setitem(sys.modules, "weasyprint", None)
        with pytest.raises((ImportError, TypeError)):
            PdfExporter().render(sample_report)

    def test_render_calls_weasyprint(self, sample_report):
        """PdfExporter calls WeasyPrint HTML().write_pdf() with the HTML string."""
        mock_html_instance = MagicMock()
        mock_html_instance.write_pdf.return_value = b"%PDF-1.4 fake"
        mock_html_class = MagicMock(return_value=mock_html_instance)

        with patch.dict("sys.modules", {"weasyprint": MagicMock(HTML=mock_html_class)}):
            result = PdfExporter().render(sample_report)

        mock_html_class.assert_called_once()
        call_kwargs = mock_html_class.call_args
        html_arg = call_kwargs[1].get("string") or call_kwargs[0][0]
        assert "<!DOCTYPE html>" in html_arg
        assert result == b"%PDF-1.4 fake"

    def test_render_to_file(self, sample_report, tmp_path):
        path = str(tmp_path / "report.pdf")
        mock_html_instance = MagicMock()
        mock_html_instance.write_pdf.return_value = b"%PDF-1.4 fake"
        mock_html_class = MagicMock(return_value=mock_html_instance)

        with patch.dict("sys.modules", {"weasyprint": MagicMock(HTML=mock_html_class)}):
            PdfExporter().render_to_file(sample_report, path)

        with open(path, "rb") as fh:
            content = fh.read()
        assert content == b"%PDF-1.4 fake"
