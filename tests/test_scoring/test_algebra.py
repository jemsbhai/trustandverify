"""Tests for scoring/algebra.py — score_claim() and build_evidence_opinion()."""

from __future__ import annotations

from jsonld_ex.confidence_algebra import Opinion

from trustandverify.core.models import Evidence, Source, Verdict
from trustandverify.scoring.algebra import build_evidence_opinion, score_claim


def _make_evidence(supports: bool, confidence: float = 0.8, trust: float = 0.7) -> Evidence:
    return Evidence(
        text="Test evidence",
        supports_claim=supports,
        relevance=0.9,
        confidence_raw=confidence,
        source=Source(
            url="https://example.com",
            title="Test Source",
            content_snippet="Snippet",
            trust_score=trust,
        ),
    )


class TestBuildEvidenceOpinion:
    def test_supporting_evidence(self):
        ev = _make_evidence(supports=True, confidence=0.8, trust=0.7)
        op = build_evidence_opinion(ev)
        assert isinstance(op, Opinion)
        assert op.belief > op.disbelief

    def test_contradicting_evidence_flips(self):
        ev = _make_evidence(supports=False, confidence=0.8, trust=0.7)
        op = build_evidence_opinion(ev)
        # Contradicting evidence should have disbelief > belief
        assert op.disbelief > op.belief


class TestScoreClaim:
    def test_empty_evidence_returns_vacuous(self):
        opinion, verdict, conflict, meta = score_claim([])
        assert verdict == Verdict.NO_EVIDENCE
        assert opinion.belief == 0.0
        assert opinion.disbelief == 0.0
        assert opinion.uncertainty == 1.0
        assert conflict is None

    def test_supporting_evidence_gives_supported(self):
        evidence = [
            _make_evidence(supports=True, confidence=0.85),
            _make_evidence(supports=True, confidence=0.80),
        ]
        opinion, verdict, conflict, meta = score_claim(evidence)
        assert verdict == Verdict.SUPPORTED
        assert opinion.belief > 0.5

    def test_contradicting_evidence_tracked(self):
        evidence = [
            _make_evidence(supports=True, confidence=0.85),
            _make_evidence(supports=False, confidence=0.80),
        ]
        opinion, verdict, conflict, meta = score_claim(evidence, conflict_threshold=0.0)
        # Should detect conflict when threshold is 0
        assert isinstance(opinion, Opinion)

    def test_all_contradicting(self):
        evidence = [
            _make_evidence(supports=False, confidence=0.85),
            _make_evidence(supports=False, confidence=0.80),
        ]
        opinion, verdict, conflict, meta = score_claim(evidence)
        assert verdict in (Verdict.REFUTED, Verdict.CONTESTED)

    def test_returns_cohesion(self):
        """score_claim always returns cohesion in its result."""
        evidence = [
            _make_evidence(supports=True, confidence=0.85),
            _make_evidence(supports=True, confidence=0.80),
        ]
        opinion, verdict, conflict, meta = score_claim(evidence)
        assert 0.0 <= meta["cohesion"] <= 1.0

    def test_default_mode_includes_byzantine_diagnostic(self):
        """Default mode (no Byzantine) should include a recommendation flag."""
        evidence = [
            _make_evidence(supports=True, confidence=0.85),
            _make_evidence(supports=True, confidence=0.80),
            _make_evidence(supports=False, confidence=0.90, trust=0.2),
        ]
        opinion, verdict, conflict, meta = score_claim(evidence)
        assert "byzantine_recommended" in meta
        assert isinstance(meta["byzantine_recommended"], bool)

    def test_outlier_triggers_byzantine_recommendation(self):
        """A strong outlier that retains signal after trust discount should trigger.

        The outlier needs enough trust (0.6) to retain its contradicting
        signal after apply_trust_discount + flip_opinion.  With trust=0.2
        the opinion collapses to near-vacuous uncertainty and won't be
        discordant.
        """
        evidence = [
            _make_evidence(supports=True, confidence=0.85, trust=0.9),
            _make_evidence(supports=True, confidence=0.80, trust=0.85),
            _make_evidence(supports=False, confidence=0.90, trust=0.6),
        ]
        opinion, verdict, conflict, meta = score_claim(evidence)
        assert meta["byzantine_recommended"] is True

    def test_byzantine_enabled_filters_outlier(self):
        """With enable_byzantine=True, the outlier should be filtered."""
        evidence = [
            _make_evidence(supports=True, confidence=0.85, trust=0.9),
            _make_evidence(supports=True, confidence=0.80, trust=0.85),
            _make_evidence(supports=False, confidence=0.90, trust=0.6),
        ]
        opinion, verdict, conflict, meta = score_claim(evidence, enable_byzantine=True)
        assert meta["used_byzantine"] is True
        assert len(meta["filtered"]) > 0

    def test_byzantine_enabled_no_recommendation(self):
        """When Byzantine is already on, no recommendation is needed."""
        evidence = [
            _make_evidence(supports=True, confidence=0.85, trust=0.9),
            _make_evidence(supports=True, confidence=0.80, trust=0.85),
            _make_evidence(supports=False, confidence=0.90, trust=0.6),
        ]
        opinion, verdict, conflict, meta = score_claim(evidence, enable_byzantine=True)
        assert "byzantine_recommended" not in meta

    def test_empty_evidence_meta(self):
        """Empty evidence should still return valid meta."""
        opinion, verdict, conflict, meta = score_claim([])
        assert verdict == Verdict.NO_EVIDENCE
        assert meta["cohesion"] == 1.0
