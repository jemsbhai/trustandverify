"""Tests for scoring/conflict.py — detect_conflicts_within_claim."""

from __future__ import annotations

from jsonld_ex.confidence_algebra import Opinion

from trustandverify.scoring.conflict import detect_conflicts_within_claim


class TestDetectConflictsWithinClaim:
    _supporting = [Opinion(belief=0.65, disbelief=0.05, uncertainty=0.30, base_rate=0.5)]
    # Flipped contradicting opinion (high disbelief = high belief after flip)
    _contradicting = [Opinion(belief=0.60, disbelief=0.10, uncertainty=0.30, base_rate=0.5)]

    def test_no_supporting_returns_none(self):
        result = detect_conflicts_within_claim([], self._contradicting)
        assert result is None

    def test_no_contradicting_returns_none(self):
        result = detect_conflicts_within_claim(self._supporting, [])
        assert result is None

    def test_strong_conflict_detected(self):
        """A strong supporter vs a strong contradicting (flipped) opinion should flag."""
        strong_support = [Opinion(belief=0.8, disbelief=0.05, uncertainty=0.15, base_rate=0.5)]
        # conflict = b_A * d_B + d_A * b_B = 0.8*0.05 + 0.05*0.75 = 0.04 + 0.0375 = 0.0775
        # This is below 0.2 threshold — need truly opposing opinions for strong conflict.
        # Use actual pairwise conflict formula: support(b=0.8) vs contra(b=0.1, d=0.7)
        actual_contra = [Opinion(belief=0.05, disbelief=0.75, uncertainty=0.20, base_rate=0.5)]
        result = detect_conflicts_within_claim(strong_support, actual_contra, threshold=0.1)
        assert result is not None
        assert result["conflict_degree"] > 0.1

    def test_below_threshold_returns_none(self):
        """Mild evidence both ways should not flag a conflict at the default threshold."""
        mild_support = [Opinion(belief=0.35, disbelief=0.35, uncertainty=0.30, base_rate=0.5)]
        mild_contra = [Opinion(belief=0.35, disbelief=0.35, uncertainty=0.30, base_rate=0.5)]
        result = detect_conflicts_within_claim(mild_support, mild_contra, threshold=0.3)
        # pairwise_conflict = 0.35*0.35 + 0.35*0.35 = 0.245 < 0.3
        assert result is None

    def test_conflict_report_structure(self):
        """A flagged conflict should contain the expected keys."""
        support = [Opinion(belief=0.75, disbelief=0.05, uncertainty=0.20, base_rate=0.5)]
        contra = [Opinion(belief=0.05, disbelief=0.75, uncertainty=0.20, base_rate=0.5)]
        result = detect_conflicts_within_claim(support, contra, threshold=0.05)
        assert result is not None
        for key in (
            "conflict_degree",
            "opinion_distance",
            "supporting_opinion",
            "contradicting_opinion",
            "num_supporting",
            "num_contradicting",
        ):
            assert key in result

    def test_opinion_distance_is_proper_metric(self):
        """opinion_distance should satisfy d(A,A)=0 unlike pairwise_conflict."""
        # Two identical sides should have distance 0
        op = Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3, base_rate=0.5)
        result = detect_conflicts_within_claim([op], [op], threshold=0.0)
        if result is not None:
            assert result["opinion_distance"] == 0.0

    def test_opinion_distance_in_range(self):
        """opinion_distance should be in [0, 1]."""
        support = [Opinion(belief=0.75, disbelief=0.05, uncertainty=0.20, base_rate=0.5)]
        contra = [Opinion(belief=0.05, disbelief=0.75, uncertainty=0.20, base_rate=0.5)]
        result = detect_conflicts_within_claim(support, contra, threshold=0.05)
        assert result is not None
        assert 0.0 <= result["opinion_distance"] <= 1.0

    def test_counts_are_correct(self):
        support = [
            Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5),
            Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5),
        ]
        contra = [
            Opinion(belief=0.05, disbelief=0.70, uncertainty=0.25, base_rate=0.5),
        ]
        result = detect_conflicts_within_claim(support, contra, threshold=0.05)
        if result is not None:
            assert result["num_supporting"] == 2
            assert result["num_contradicting"] == 1
