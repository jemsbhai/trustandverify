"""Tests for scoring/trust.py — estimate_source_trust and apply_trust_discount."""

from __future__ import annotations

import pytest
from jsonld_ex.confidence_algebra import Opinion

from trustandverify.scoring.trust import apply_trust_discount, estimate_source_trust


class TestEstimateSourceTrust:
    def test_gov_domain(self):
        assert estimate_source_trust("https://www.cdc.gov/page") >= 0.85

    def test_edu_domain(self):
        assert estimate_source_trust("https://harvard.edu/study") >= 0.85

    def test_arxiv(self):
        assert estimate_source_trust("https://arxiv.org/abs/1234") >= 0.80

    def test_pubmed(self):
        assert estimate_source_trust("https://pubmed.ncbi.nlm.nih.gov/1234") >= 0.80

    def test_reuters(self):
        score = estimate_source_trust("https://www.reuters.com/article")
        assert 0.70 <= score <= 0.85

    def test_wikipedia(self):
        score = estimate_source_trust("https://en.wikipedia.org/wiki/Example")
        assert 0.55 <= score <= 0.70

    def test_reddit(self):
        assert estimate_source_trust("https://www.reddit.com/r/science/") <= 0.40

    def test_unknown_domain(self):
        score = estimate_source_trust("https://www.someblog.example.com/post")
        assert score == 0.5

    def test_returns_float_in_range(self):
        for url in [
            "https://www.bbc.com/news/article",
            "https://www.nature.com/articles/s1234",
            "https://www.quora.com/question",
        ]:
            score = estimate_source_trust(url)
            assert 0.0 <= score <= 1.0


class TestApplyTrustDiscount:
    _base_opinion = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5)

    def test_full_trust_preserves_opinion(self):
        """trust=1.0 should return the opinion (approximately) unchanged."""
        discounted = apply_trust_discount(self._base_opinion, source_trust=1.0)
        assert abs(discounted.belief - self._base_opinion.belief) < 1e-6
        assert abs(discounted.disbelief - self._base_opinion.disbelief) < 1e-6

    def test_zero_trust_collapses_to_uncertainty(self):
        """trust=0.0 should make the opinion purely uncertain."""
        discounted = apply_trust_discount(self._base_opinion, source_trust=0.0)
        assert discounted.belief == 0.0
        assert discounted.disbelief == 0.0
        assert abs(discounted.uncertainty - 1.0) < 1e-9

    def test_partial_trust_reduces_belief(self):
        high = apply_trust_discount(self._base_opinion, source_trust=0.9)
        low = apply_trust_discount(self._base_opinion, source_trust=0.4)
        assert high.belief > low.belief
        assert high.uncertainty < low.uncertainty

    def test_additivity_preserved(self):
        for trust in [0.0, 0.3, 0.5, 0.9, 1.0]:
            discounted = apply_trust_discount(self._base_opinion, trust)
            total = discounted.belief + discounted.disbelief + discounted.uncertainty
            assert abs(total - 1.0) < 1e-9, f"b+d+u != 1 at trust={trust}"

    def test_invalid_trust_raises(self):
        with pytest.raises(ValueError):
            apply_trust_discount(self._base_opinion, source_trust=-0.1)
        with pytest.raises(ValueError):
            apply_trust_discount(self._base_opinion, source_trust=1.1)
