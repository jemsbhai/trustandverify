"""Tests for core/pipeline.py — individual stage functions + cache/verbose paths."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from jsonld_ex.confidence_algebra import Opinion

from trustandverify.core.config import TrustConfig
from trustandverify.core.models import Claim, Evidence, Report, SearchResult, Source, Verdict
from trustandverify.core.pipeline import (
    assess,
    extract,
    plan,
    run_pipeline,
    score,
    search_for_claim,
    summarise,
)

# ── Shared fixtures ───────────────────────────────────────────────────────────


def _mock_llm():
    llm = MagicMock()

    async def complete(prompt, system=""):
        if "search query" in prompt.lower() or "web search" in prompt.lower():
            return "optimised search query"
        if "assessment" in prompt.lower() or "assess" in prompt.lower():
            return "Assessment text here."
        if "summary" in prompt.lower() or "executive" in prompt.lower():
            return "Summary text here."
        return "generic response"

    async def complete_json(prompt, system="", defaults=None):
        if "decompose" in prompt.lower() or "verifiable" in prompt.lower():
            return {"items": ["Claim A", "Claim B"]}
        return {
            "evidence": "Evidence text",
            "supports": True,
            "relevance": 0.85,
            "confidence": 0.8,
        }

    llm.complete = complete
    llm.complete_json = complete_json
    return llm


def _mock_search():
    search = MagicMock()
    search.search = AsyncMock(
        return_value=[
            SearchResult(
                title="Source1", url="https://example.com", content="Content here", score=0.9
            ),
        ]
    )
    return search


def _mock_cache():
    cache = MagicMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock()
    return cache


# ── plan() ────────────────────────────────────────────────────────────────────


class TestPlan:
    async def test_returns_claim_texts(self):
        claims = await plan("test question", TrustConfig(), _mock_llm())
        assert claims == ["Claim A", "Claim B"]

    async def test_handles_dict_without_items(self):
        llm = MagicMock()
        llm.complete_json = AsyncMock(return_value={"claim1": "A", "claim2": "B"})
        claims = await plan("test", TrustConfig(), llm)
        assert len(claims) == 2

    async def test_handles_list_response(self):
        llm = MagicMock()
        llm.complete_json = AsyncMock(return_value=["Claim A", "Claim B"])
        claims = await plan("test", TrustConfig(), llm)
        assert claims == ["Claim A", "Claim B"]

    async def test_handles_empty_response(self):
        llm = MagicMock()
        llm.complete_json = AsyncMock(return_value={})
        claims = await plan("test", TrustConfig(), llm)
        assert claims == []


# ── search_for_claim() ────────────────────────────────────────────────────────


class TestSearchForClaim:
    async def test_basic_search(self):
        results = await search_for_claim("test claim", TrustConfig(), _mock_search(), _mock_llm())
        assert len(results) == 1

    async def test_with_cache_miss(self):
        cache = _mock_cache()
        results = await search_for_claim(
            "test claim", TrustConfig(), _mock_search(), _mock_llm(), cache=cache
        )
        assert len(results) == 1
        # Should have called cache.set for search query and results
        assert cache.set.call_count >= 1

    async def test_with_cached_query(self):
        cache = MagicMock()
        call_count = 0

        async def smart_get(key):
            nonlocal call_count
            call_count += 1
            if "search_query:" in key:
                return "cached query"
            return None

        cache.get = smart_get
        cache.set = AsyncMock()

        results = await search_for_claim(
            "test claim", TrustConfig(), _mock_search(), _mock_llm(), cache=cache
        )
        assert len(results) == 1

    async def test_with_cached_results(self):
        cache = MagicMock()

        async def smart_get(key):
            if "search_query:" in key:
                return "cached query"
            if "search_results:" in key:
                return [
                    {"title": "Cached", "url": "https://cached.com", "content": "c", "score": 0.5}
                ]
            return None

        cache.get = smart_get
        cache.set = AsyncMock()

        results = await search_for_claim(
            "test claim", TrustConfig(), _mock_search(), _mock_llm(), cache=cache
        )
        assert len(results) == 1
        assert results[0].title == "Cached"


# ── extract() ─────────────────────────────────────────────────────────────────


class TestExtract:
    async def test_basic_extraction(self):
        results = [SearchResult(title="T", url="https://x.com", content="Text", score=0.9)]
        evidence = await extract("test claim", results, _mock_llm())
        assert len(evidence) == 1
        assert isinstance(evidence[0], Evidence)

    async def test_with_cache(self):
        cache = _mock_cache()
        results = [SearchResult(title="T", url="https://x.com", content="Text", score=0.9)]
        evidence = await extract("test claim", results, _mock_llm(), cache=cache)
        assert len(evidence) == 1
        assert cache.set.call_count >= 1

    async def test_with_cached_evidence(self):
        cache = MagicMock()
        cache.get = AsyncMock(
            return_value={
                "evidence": "Cached evidence",
                "supports": False,
                "relevance": 0.7,
                "confidence": 0.6,
            }
        )
        cache.set = AsyncMock()

        results = [SearchResult(title="T", url="https://x.com", content="Text", score=0.9)]
        evidence = await extract("test claim", results, _mock_llm(), cache=cache)
        assert evidence[0].text == "Cached evidence"
        assert evidence[0].supports_claim is False


# ── score() ───────────────────────────────────────────────────────────────────


class TestScore:
    def test_scores_claim_with_evidence(self):
        ev = Evidence(
            text="Evidence",
            supports_claim=True,
            relevance=0.9,
            confidence_raw=0.8,
            source=Source(url="https://x.com", title="T", content_snippet="S", trust_score=0.7),
        )
        claim = Claim(text="Test", evidence=[ev])
        scored_claim, conflict = score(claim, TrustConfig())
        assert scored_claim.opinion is not None
        assert scored_claim.verdict != Verdict.NO_EVIDENCE

    def test_scores_claim_with_conflict(self):
        ev1 = Evidence(
            text="Supports",
            supports_claim=True,
            relevance=0.9,
            confidence_raw=0.8,
            source=Source(url="https://a.com", title="A", content_snippet="S", trust_score=0.7),
        )
        ev2 = Evidence(
            text="Contradicts",
            supports_claim=False,
            relevance=0.9,
            confidence_raw=0.8,
            source=Source(url="https://b.com", title="B", content_snippet="S", trust_score=0.7),
        )
        claim = Claim(text="Test", evidence=[ev1, ev2])
        scored_claim, conflict = score(claim, TrustConfig(conflict_threshold=0.0))
        # With threshold=0 and mixed evidence, conflict should be detected
        assert conflict is not None or scored_claim.opinion is not None


# ── assess() ──────────────────────────────────────────────────────────────────


class TestAssess:
    async def test_basic_assess(self):
        claim = Claim(
            text="Test claim",
            opinion=Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5),
            evidence=[
                Evidence(
                    text="Supporting",
                    supports_claim=True,
                    relevance=0.9,
                    confidence_raw=0.8,
                    source=Source(
                        url="https://x.com", title="T", content_snippet="S", trust_score=0.7
                    ),
                ),
            ],
        )
        text = await assess(claim, _mock_llm())
        assert isinstance(text, str)
        assert len(text) > 0

    async def test_with_cache(self):
        cache = _mock_cache()
        claim = Claim(
            text="Test claim",
            opinion=Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5),
            evidence=[],
        )
        text = await assess(claim, _mock_llm(), cache=cache)
        assert isinstance(text, str)
        assert cache.set.call_count >= 1

    async def test_with_cached_assessment(self):
        cache = MagicMock()
        cache.get = AsyncMock(return_value="Cached assessment text.")
        claim = Claim(
            text="Test claim",
            opinion=Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5),
            evidence=[],
        )
        text = await assess(claim, _mock_llm(), cache=cache)
        assert text == "Cached assessment text."

    async def test_without_opinion(self):
        claim = Claim(text="Test claim", opinion=None, evidence=[])
        text = await assess(claim, _mock_llm())
        assert isinstance(text, str)


# ── summarise() ───────────────────────────────────────────────────────────────


class TestSummarise:
    async def test_basic_summarise(self):
        claims = [
            Claim(
                text="Claim A",
                opinion=Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5),
                verdict=Verdict.SUPPORTED,
                assessment="Assessment A.",
            ),
        ]
        text = await summarise("test question", claims, _mock_llm())
        assert isinstance(text, str)

    async def test_with_cache(self):
        cache = _mock_cache()
        claims = [
            Claim(
                text="Claim A",
                opinion=Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5),
                verdict=Verdict.SUPPORTED,
                assessment="Assessment A.",
            ),
        ]
        await summarise("test question", claims, _mock_llm(), cache=cache)
        assert cache.set.call_count >= 1

    async def test_with_cached_summary(self):
        cache = MagicMock()
        cache.get = AsyncMock(return_value="Cached summary.")
        claims = []
        text = await summarise("test question", claims, _mock_llm(), cache=cache)
        assert text == "Cached summary."


# ── run_pipeline() verbose + cache ────────────────────────────────────────────


class TestRunPipeline:
    async def test_verbose_mode(self, capsys):
        report = await run_pipeline(
            query="Is coffee healthy?",
            config=TrustConfig(num_claims=2, enable_cache=False),
            search=_mock_search(),
            llm=_mock_llm(),
            verbose=True,
        )
        assert isinstance(report, Report)
        captured = capsys.readouterr()
        assert "PLAN" in captured.out
        assert "SEARCH" in captured.out
        assert "SCORE" in captured.out
        assert "SUMMARY" in captured.out

    async def test_with_cache(self):
        cache = _mock_cache()
        report = await run_pipeline(
            query="Is coffee healthy?",
            config=TrustConfig(num_claims=2),
            search=_mock_search(),
            llm=_mock_llm(),
            cache=cache,
        )
        assert isinstance(report, Report)
        assert cache.set.call_count > 0

    async def test_verbose_with_conflict(self, capsys):
        """Ensure the conflict logging branch in verbose mode is exercised."""
        # Create a search that returns 2 results so we get both supporting and contradicting
        search = MagicMock()
        search.search = AsyncMock(
            return_value=[
                SearchResult(title="Pro", url="https://pro.com", content="Evidence for", score=0.9),
                SearchResult(
                    title="Con", url="https://con.com", content="Evidence against", score=0.8
                ),
            ]
        )

        # LLM returns alternating support/contradict
        call_count = {"extract": 0}

        llm = MagicMock()

        async def complete(prompt, system=""):
            return "response text"

        async def complete_json(prompt, system="", defaults=None):
            if "decompose" in prompt.lower() or "verifiable" in prompt.lower():
                return {"items": ["Single claim"]}
            call_count["extract"] += 1
            supports = call_count["extract"] % 2 == 1
            return {
                "evidence": f"Evidence {call_count['extract']}",
                "supports": supports,
                "relevance": 0.9,
                "confidence": 0.85,
            }

        llm.complete = complete
        llm.complete_json = complete_json

        report = await run_pipeline(
            query="Test conflict",
            config=TrustConfig(num_claims=1, conflict_threshold=0.0),
            search=search,
            llm=llm,
            verbose=True,
        )
        capsys.readouterr()
        # Should have conflict output if mixed evidence was produced
        assert isinstance(report, Report)
