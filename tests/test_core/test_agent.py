"""Tests for core/agent.py and core/pipeline.py — full pipeline with mocked backends."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from jsonld_ex.confidence_algebra import Opinion

from trustandverify.core.agent import TrustAgent
from trustandverify.core.config import TrustConfig
from trustandverify.core.models import Report, SearchResult, Verdict


# ── Canned mock data ──────────────────────────────────────────────────────────

CANNED_CLAIMS = ["Remote workers report higher individual output.",
                 "Collaboration suffers in remote settings."]

CANNED_SEARCH = [
    SearchResult(
        title="NBER Study",
        url="https://nber.org/papers/w1",
        content="Stanford study: remote workers 13% more productive.",
        score=0.9,
    ),
    SearchResult(
        title="Harvard Review",
        url="https://harvard.edu/remote",
        content="Collaboration and creativity may suffer remotely.",
        score=0.75,
    ),
]

CANNED_EVIDENCE = {
    "evidence": "Remote workers showed 13% productivity gain.",
    "supports": True,
    "relevance": 0.85,
    "confidence": 0.80,
}

CANNED_ASSESSMENT = "Evidence broadly supports productivity gains for remote workers."
CANNED_SUMMARY = "Overall, remote work appears to increase individual productivity."


def make_mock_llm() -> MagicMock:
    """Mock LLM that returns canned responses."""
    llm = MagicMock()

    async def complete(prompt: str, system: str = "") -> str:
        if "search query" in prompt.lower() or "web search" in prompt.lower():
            return "remote work productivity research"
        if "assessment" in prompt.lower() or "assess" in prompt.lower():
            return CANNED_ASSESSMENT
        if "summary" in prompt.lower() or "executive" in prompt.lower():
            return CANNED_SUMMARY
        return "remote work productivity"

    async def complete_json(prompt: str, system: str = "", defaults: dict | None = None) -> dict:
        if "decompose" in prompt.lower() or "verifiable" in prompt.lower():
            return {"items": CANNED_CLAIMS}
        return CANNED_EVIDENCE

    llm.complete = complete
    llm.complete_json = complete_json
    llm.is_available = MagicMock(return_value=True)
    return llm


def make_mock_search() -> MagicMock:
    """Mock search backend that returns canned results."""
    search = MagicMock()
    search.search = AsyncMock(return_value=CANNED_SEARCH)
    search.is_available = MagicMock(return_value=True)
    return search


# ── TrustAgent tests ──────────────────────────────────────────────────────────

class TestTrustAgent:
    async def test_verify_returns_report(self):
        agent = TrustAgent(
            config=TrustConfig(num_claims=2, enable_cache=False),
            search=make_mock_search(),
            llm=make_mock_llm(),
        )
        report = await agent.verify("Is remote work more productive?")
        assert isinstance(report, Report)

    async def test_report_has_claims(self):
        agent = TrustAgent(
            config=TrustConfig(num_claims=2, enable_cache=False),
            search=make_mock_search(),
            llm=make_mock_llm(),
        )
        report = await agent.verify("Is remote work more productive?")
        assert len(report.claims) > 0

    async def test_claims_have_opinions(self):
        agent = TrustAgent(
            config=TrustConfig(num_claims=2, enable_cache=False),
            search=make_mock_search(),
            llm=make_mock_llm(),
        )
        report = await agent.verify("Is remote work more productive?")
        for claim in report.claims:
            assert claim.opinion is not None
            assert isinstance(claim.opinion, Opinion)

    async def test_claims_have_verdicts(self):
        agent = TrustAgent(
            config=TrustConfig(num_claims=2, enable_cache=False),
            search=make_mock_search(),
            llm=make_mock_llm(),
        )
        report = await agent.verify("Is remote work more productive?")
        for claim in report.claims:
            assert claim.verdict in list(Verdict)

    async def test_report_has_summary(self):
        agent = TrustAgent(
            config=TrustConfig(num_claims=2, enable_cache=False),
            search=make_mock_search(),
            llm=make_mock_llm(),
        )
        report = await agent.verify("Is remote work more productive?")
        assert isinstance(report.summary, str)
        assert len(report.summary) > 0

    async def test_report_has_query(self):
        query = "Is remote work more productive?"
        agent = TrustAgent(
            config=TrustConfig(num_claims=2, enable_cache=False),
            search=make_mock_search(),
            llm=make_mock_llm(),
        )
        report = await agent.verify(query)
        assert report.query == query

    async def test_raises_without_search(self):
        agent = TrustAgent(llm=make_mock_llm())
        with pytest.raises(RuntimeError, match="search backend"):
            await agent.verify("test query")

    async def test_raises_without_llm(self):
        agent = TrustAgent(search=make_mock_search())
        with pytest.raises(RuntimeError, match="LLM backend"):
            await agent.verify("test query")

    async def test_report_saved_to_storage(self):
        from trustandverify.storage.memory import InMemoryStorage
        storage = InMemoryStorage()
        agent = TrustAgent(
            config=TrustConfig(num_claims=2, enable_cache=False),
            search=make_mock_search(),
            llm=make_mock_llm(),
            storage=storage,
        )
        report = await agent.verify("Is remote work more productive?")
        retrieved = await storage.get_report(report.id)
        assert retrieved is not None
        assert retrieved.id == report.id

    async def test_opinion_additivity_constraint(self):
        """All fused opinions must satisfy b + d + u = 1."""
        agent = TrustAgent(
            config=TrustConfig(num_claims=2, enable_cache=False),
            search=make_mock_search(),
            llm=make_mock_llm(),
        )
        report = await agent.verify("Is remote work more productive?")
        for claim in report.claims:
            if claim.opinion:
                total = claim.opinion.belief + claim.opinion.disbelief + claim.opinion.uncertainty
                assert abs(total - 1.0) < 1e-6, f"b+d+u != 1 for claim: {claim.text}"
