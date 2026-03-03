"""Tests for export/jsonld.py — snapshot tests against expected output structure."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from jsonld_ex.confidence_algebra import Opinion

from trustandverify.core.models import Claim, Conflict, Evidence, Report, Source, Verdict
from trustandverify.export.jsonld import JsonLdExporter


@pytest.fixture
def sample_report() -> Report:
    source = Source(
        url="https://www.nber.org/papers/w123",
        title="NBER Study",
        content_snippet="Remote workers showed 13% productivity gain.",
        trust_score=0.85,
    )
    evidence = Evidence(
        text="Remote workers showed 13% productivity gain.",
        supports_claim=True,
        relevance=0.9,
        confidence_raw=0.8,
        source=source,
        opinion=Opinion(belief=0.567, disbelief=0.100, uncertainty=0.333, base_rate=0.5),
    )
    claim = Claim(
        text="Remote workers are more productive than office workers.",
        evidence=[evidence],
        opinion=Opinion(belief=0.733, disbelief=0.100, uncertainty=0.167, base_rate=0.5),
        verdict=Verdict.SUPPORTED,
        assessment="Evidence broadly supports the claim.",
    )
    conflict = Conflict(
        claim_text="Remote workers are more productive",
        conflict_degree=0.25,
        num_supporting=2,
        num_contradicting=1,
    )
    return Report(
        id="test-report-001",
        query="Is remote work more productive than office work?",
        claims=[claim],
        conflicts=[conflict],
        summary="The evidence broadly supports increased remote work productivity.",
        created_at=datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc),
    )


class TestJsonLdExporter:
    def test_render_returns_string(self, sample_report):
        result = JsonLdExporter().render(sample_report)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_render_is_valid_json(self, sample_report):
        result = JsonLdExporter().render(sample_report)
        doc = json.loads(result)
        assert isinstance(doc, dict)

    def test_context_structure(self, sample_report):
        doc = json.loads(JsonLdExporter().render(sample_report))
        ctx = doc["@context"]
        assert ctx["ex"] == "https://jsonld-ex.org/vocab#"
        assert ctx["prov"] == "http://www.w3.org/ns/prov#"

    def test_report_type(self, sample_report):
        doc = json.loads(JsonLdExporter().render(sample_report))
        assert doc["@type"] == "ex:TrustGraphReport"

    def test_query_preserved(self, sample_report):
        doc = json.loads(JsonLdExporter().render(sample_report))
        assert doc["ex:query"] == sample_report.query

    def test_claims_rendered(self, sample_report):
        doc = json.loads(JsonLdExporter().render(sample_report))
        claims = doc["ex:claims"]
        assert len(claims) == 1
        assert claims[0]["@type"] == "ex:VerifiedClaim"
        assert claims[0]["ex:claimText"] == sample_report.claims[0].text

    def test_confidence_opinion_fields(self, sample_report):
        doc = json.loads(JsonLdExporter().render(sample_report))
        conf = doc["ex:claims"][0]["ex:confidence"]
        assert conf["@type"] == "ex:SubjectiveOpinion"
        for field in ("ex:belief", "ex:disbelief", "ex:uncertainty", "ex:baseRate", "ex:projectedProbability"):
            assert field in conf
            assert isinstance(conf[field], float)

    def test_projected_probability_matches_opinion(self, sample_report):
        doc = json.loads(JsonLdExporter().render(sample_report))
        conf = doc["ex:claims"][0]["ex:confidence"]
        op = sample_report.claims[0].opinion
        expected = round(op.projected_probability(), 4)
        assert conf["ex:projectedProbability"] == expected

    def test_conflicts_rendered(self, sample_report):
        doc = json.loads(JsonLdExporter().render(sample_report))
        conflicts = doc["ex:conflicts"]
        assert len(conflicts) == 1
        assert conflicts[0]["conflict_degree"] == 0.25
        assert conflicts[0]["num_supporting"] == 2

    def test_summary_preserved(self, sample_report):
        doc = json.loads(JsonLdExporter().render(sample_report))
        assert doc["ex:summary"] == sample_report.summary

    def test_sources_in_claim(self, sample_report):
        doc = json.loads(JsonLdExporter().render(sample_report))
        sources = doc["ex:claims"][0]["ex:sources"]
        assert len(sources) == 1
        assert sources[0]["url"] == "https://www.nber.org/papers/w123"
        assert sources[0]["supports"] is True

    def test_render_to_file(self, sample_report, tmp_path):
        path = str(tmp_path / "report.jsonld")
        JsonLdExporter().render_to_file(sample_report, path)
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        assert doc["@type"] == "ex:TrustGraphReport"
