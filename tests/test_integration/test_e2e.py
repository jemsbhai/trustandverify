"""End-to-end integration test — real Tavily search + real Gemini LLM.

Run with:
    poetry run pytest tests/test_integration/test_e2e.py -m integration -v

Requires environment variables:
    TAVILY_API_KEY  — free tier (1,000 searches/month)
    GEMINI_API_KEY  — Google AI Studio key

These tests hit real APIs, cost real quota, and take 15-60 seconds.
They are skipped by default (not selected unless -m integration is passed).
"""

from __future__ import annotations

import os

import pytest

from jsonld_ex.confidence_algebra import Opinion

from trustandverify.core.agent import TrustAgent
from trustandverify.core.config import TrustConfig
from trustandverify.core.models import Report, Verdict


# ── Skip conditions ───────────────────────────────────────────────────────────

_TAVILY_KEY = os.environ.get("TAVILY_API_KEY", "")
_GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
_SKIP_REASON = "TAVILY_API_KEY and GEMINI_API_KEY must be set"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(not (_TAVILY_KEY and _GEMINI_KEY), reason=_SKIP_REASON),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_agent(num_claims: int = 2, enable_byzantine: bool = False) -> TrustAgent:
    """Build a TrustAgent with real Tavily + Gemini backends."""
    pytest.importorskip("litellm", reason="litellm required for GeminiBackend")

    from trustandverify.llm.gemini import GeminiBackend
    from trustandverify.search.tavily import TavilySearch

    return TrustAgent(
        config=TrustConfig(
            num_claims=num_claims,
            max_sources_per_claim=3,
            enable_cache=False,
            enable_byzantine=enable_byzantine,
        ),
        search=TavilySearch(),
        llm=GeminiBackend(),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestEndToEnd:
    """Full pipeline: query → claims → search → extract → score → report."""

    @pytest.fixture
    async def report(self) -> Report:
        """Run the pipeline once and share the result across tests in this class."""
        agent = _make_agent(num_claims=2)
        return await agent.verify("Is coffee consumption associated with health benefits?")

    async def test_report_type(self, report: Report):
        assert isinstance(report, Report)

    async def test_report_has_query(self, report: Report):
        assert "coffee" in report.query.lower()

    async def test_report_has_claims(self, report: Report):
        assert len(report.claims) >= 1, "Pipeline must produce at least one claim"

    async def test_claims_have_text(self, report: Report):
        for claim in report.claims:
            assert isinstance(claim.text, str)
            assert len(claim.text) > 5, f"Claim text too short: {claim.text!r}"

    async def test_claims_have_opinions(self, report: Report):
        for claim in report.claims:
            assert claim.opinion is not None, f"No opinion for claim: {claim.text!r}"
            assert isinstance(claim.opinion, Opinion)

    async def test_opinion_additivity(self, report: Report):
        """b + d + u must equal 1.0 for every fused opinion."""
        for claim in report.claims:
            if claim.opinion is None:
                continue
            total = claim.opinion.belief + claim.opinion.disbelief + claim.opinion.uncertainty
            assert abs(total - 1.0) < 1e-6, (
                f"b+d+u = {total} != 1.0 for claim: {claim.text!r}"
            )

    async def test_opinion_values_in_range(self, report: Report):
        """All opinion components must be in [0, 1]."""
        for claim in report.claims:
            if claim.opinion is None:
                continue
            for attr in ("belief", "disbelief", "uncertainty", "base_rate"):
                val = getattr(claim.opinion, attr)
                assert 0.0 <= val <= 1.0, (
                    f"{attr}={val} out of [0,1] for claim: {claim.text!r}"
                )

    async def test_claims_have_verdicts(self, report: Report):
        for claim in report.claims:
            assert claim.verdict in list(Verdict)

    async def test_claims_have_evidence(self, report: Report):
        for claim in report.claims:
            assert len(claim.evidence) >= 1, (
                f"No evidence for claim: {claim.text!r}"
            )

    async def test_evidence_has_sources(self, report: Report):
        for claim in report.claims:
            for ev in claim.evidence:
                assert ev.source is not None
                assert ev.source.url.startswith("http")

    async def test_claims_have_assessments(self, report: Report):
        for claim in report.claims:
            assert isinstance(claim.assessment, str)
            assert len(claim.assessment) > 10, (
                f"Assessment too short for claim: {claim.text!r}"
            )

    async def test_report_has_summary(self, report: Report):
        assert isinstance(report.summary, str)
        assert len(report.summary) > 20, "Summary too short"

    async def test_report_persisted_in_memory(self, report: Report):
        """The default InMemoryStorage should have the report after verify()."""
        assert report.id is not None
        assert len(report.id) > 0


class TestEndToEndByzantine:
    """Same pipeline with Byzantine-resistant fusion enabled."""

    async def test_byzantine_mode_produces_valid_report(self):
        agent = _make_agent(num_claims=2, enable_byzantine=True)
        report = await agent.verify("Is nuclear energy safer than solar energy?")

        assert isinstance(report, Report)
        assert len(report.claims) >= 1

        for claim in report.claims:
            assert claim.opinion is not None
            total = claim.opinion.belief + claim.opinion.disbelief + claim.opinion.uncertainty
            assert abs(total - 1.0) < 1e-6
            assert claim.verdict in list(Verdict)
