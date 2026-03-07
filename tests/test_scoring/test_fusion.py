"""Tests for scoring/fusion.py — fuse_evidence, Byzantine fusion, cohesion, distance."""

from __future__ import annotations

from jsonld_ex.confidence_algebra import Opinion

from trustandverify.scoring.fusion import (
    cohesion_score,
    diagnose_byzantine,
    fuse_evidence,
    fuse_evidence_byzantine,
    opinion_distance,
)


class TestFuseEvidence:
    def test_empty_list_returns_vacuous(self):
        result = fuse_evidence([])
        assert result.uncertainty == 1.0
        assert result.belief == 0.0
        assert result.disbelief == 0.0

    def test_single_opinion_returned_unchanged(self):
        op = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5)
        result = fuse_evidence([op])
        assert result == op

    def test_two_agreeing_sources_reduce_uncertainty(self):
        """Cumulative fusion of two supporting opinions should lower uncertainty."""
        op1 = Opinion(belief=0.567, disbelief=0.100, uncertainty=0.333, base_rate=0.5)
        op2 = Opinion(belief=0.675, disbelief=0.075, uncertainty=0.250, base_rate=0.5)
        fused = fuse_evidence([op1, op2])
        assert fused.uncertainty < min(op1.uncertainty, op2.uncertainty)

    def test_additivity_is_preserved(self):
        """b + d + u must equal 1 for any fused result."""
        opinions = [
            Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3, base_rate=0.5),
            Opinion(belief=0.4, disbelief=0.3, uncertainty=0.3, base_rate=0.5),
            Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5),
        ]
        result = fuse_evidence(opinions)
        total = result.belief + result.disbelief + result.uncertainty
        assert abs(total - 1.0) < 1e-9

    def test_opposing_sources_increase_disbelief(self):
        """When one source supports and one contradicts (flipped), belief should be balanced."""
        supporting = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5)
        # Simulate a flipped contradicting opinion
        contradicting = Opinion(belief=0.1, disbelief=0.6, uncertainty=0.3, base_rate=0.5)
        fused = fuse_evidence([supporting, contradicting])
        # Fused projected probability should be around 0.5 (contested)
        assert 0.3 < fused.projected_probability() < 0.7

    def test_many_agreeing_sources_push_probability_up(self):
        """Five strongly supporting sources should yield P well above a single source.

        A single opinion with b=0.65, u=0.30 gives P = 0.65 + 0.5*0.30 = 0.80.
        After fusing five identical independent sources, uncertainty collapses
        and P should be noticeably higher (empirically ~0.89).
        """
        single = Opinion(belief=0.65, disbelief=0.05, uncertainty=0.30, base_rate=0.5)
        opinions = [single] * 5
        fused = fuse_evidence(opinions)
        assert fused.projected_probability() > single.projected_probability()
        assert fused.uncertainty < single.uncertainty


class TestFuseEvidenceByzantine:
    """Tests for fuse_evidence_byzantine() — wraps jsonld-ex byzantine_fuse."""

    def test_fallback_for_fewer_than_3(self):
        """With < 3 opinions, falls back to regular fusion."""
        ops = [
            Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5),
            Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3, base_rate=0.5),
        ]
        result = fuse_evidence_byzantine(ops, trust_weights=[0.9, 0.8])
        assert result["used_byzantine"] is False
        assert result["filtered"] == []
        assert result["surviving_indices"] == [0, 1]

    def test_fallback_when_no_trust_weights(self):
        """Without trust_weights, falls back to regular fusion."""
        ops = [
            Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5),
            Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3, base_rate=0.5),
            Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5),
        ]
        result = fuse_evidence_byzantine(ops)
        assert result["used_byzantine"] is False

    def test_filters_outlier_with_low_trust(self):
        """An outlier opinion from an untrusted source should be removed."""
        agreeing_1 = Opinion(belief=0.7, disbelief=0.05, uncertainty=0.25, base_rate=0.5)
        agreeing_2 = Opinion(belief=0.65, disbelief=0.05, uncertainty=0.30, base_rate=0.5)
        outlier = Opinion(belief=0.05, disbelief=0.75, uncertainty=0.20, base_rate=0.5)
        ops = [agreeing_1, agreeing_2, outlier]
        trusts = [0.9, 0.85, 0.2]
        result = fuse_evidence_byzantine(ops, trust_weights=trusts)
        assert result["used_byzantine"] is True
        assert len(result["filtered"]) > 0
        # The outlier (index 2) should be the one removed
        removed_indices = [f["index"] for f in result["filtered"]]
        assert 2 in removed_indices

    def test_cohesion_returned(self):
        """Result always includes a cohesion score in [0, 1]."""
        ops = [
            Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5),
            Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5),
            Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5),
        ]
        result = fuse_evidence_byzantine(ops, trust_weights=[0.9, 0.9, 0.9])
        assert 0.0 <= result["cohesion"] <= 1.0

    def test_identical_opinions_perfect_cohesion(self):
        """Identical opinions should have cohesion = 1.0."""
        op = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5)
        ops = [op, op, op]
        result = fuse_evidence_byzantine(ops, trust_weights=[0.9, 0.9, 0.9])
        assert result["cohesion"] == 1.0

    def test_surviving_indices_correct(self):
        """Surviving indices should reflect which opinions were kept."""
        ops = [
            Opinion(belief=0.7, disbelief=0.05, uncertainty=0.25, base_rate=0.5),
            Opinion(belief=0.65, disbelief=0.05, uncertainty=0.30, base_rate=0.5),
            Opinion(belief=0.05, disbelief=0.75, uncertainty=0.20, base_rate=0.5),
        ]
        result = fuse_evidence_byzantine(ops, trust_weights=[0.9, 0.85, 0.2])
        # Surviving + filtered should cover all original indices
        removed = {f["index"] for f in result["filtered"]}
        surviving = set(result["surviving_indices"])
        assert removed | surviving == {0, 1, 2}
        assert removed & surviving == set()  # no overlap

    def test_fused_is_opinion(self):
        """The fused result should be a valid Opinion."""
        ops = [
            Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5),
            Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3, base_rate=0.5),
            Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5),
        ]
        result = fuse_evidence_byzantine(ops, trust_weights=[0.9, 0.8, 0.7])
        fused = result["fused"]
        assert isinstance(fused, Opinion)
        assert abs(fused.belief + fused.disbelief + fused.uncertainty - 1.0) < 1e-9


class TestCohesionScore:
    """Tests for cohesion_score re-export."""

    def test_identical_opinions(self):
        op = Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3, base_rate=0.5)
        assert cohesion_score([op, op, op]) == 1.0

    def test_single_opinion(self):
        op = Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3, base_rate=0.5)
        assert cohesion_score([op]) == 1.0

    def test_diverse_opinions_lower_cohesion(self):
        ops = [
            Opinion(belief=0.8, disbelief=0.05, uncertainty=0.15, base_rate=0.5),
            Opinion(belief=0.05, disbelief=0.80, uncertainty=0.15, base_rate=0.5),
        ]
        assert cohesion_score(ops) < 0.5


class TestOpinionDistance:
    """Tests for opinion_distance re-export."""

    def test_identical_opinions_zero(self):
        op = Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3, base_rate=0.5)
        assert opinion_distance(op, op) == 0.0

    def test_symmetry(self):
        a = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5)
        b = Opinion(belief=0.3, disbelief=0.4, uncertainty=0.3, base_rate=0.5)
        assert abs(opinion_distance(a, b) - opinion_distance(b, a)) < 1e-12

    def test_range_zero_to_one(self):
        a = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0, base_rate=0.5)
        b = Opinion(belief=0.0, disbelief=1.0, uncertainty=0.0, base_rate=0.5)
        d = opinion_distance(a, b)
        assert 0.0 <= d <= 1.0


class TestDiagnoseByzantine:
    """Tests for diagnose_byzantine() — lightweight discord check."""

    def test_agreeing_sources_not_recommended(self):
        """Harmonious group should not recommend Byzantine fusion."""
        ops = [
            Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5),
            Opinion(belief=0.65, disbelief=0.05, uncertainty=0.30, base_rate=0.5),
            Opinion(belief=0.55, disbelief=0.15, uncertainty=0.30, base_rate=0.5),
        ]
        diag = diagnose_byzantine(ops)
        assert diag["recommended"] is False

    def test_outlier_triggers_recommendation(self):
        """One strongly opposing opinion should trigger the recommendation."""
        ops = [
            Opinion(belief=0.7, disbelief=0.05, uncertainty=0.25, base_rate=0.5),
            Opinion(belief=0.65, disbelief=0.05, uncertainty=0.30, base_rate=0.5),
            Opinion(belief=0.05, disbelief=0.75, uncertainty=0.20, base_rate=0.5),
        ]
        diag = diagnose_byzantine(ops)
        assert diag["recommended"] is True
        assert diag["num_discordant"] >= 1
        assert len(diag["reason"]) > 0

    def test_fewer_than_3_not_recommended(self):
        """With < 3 opinions Byzantine filtering can't help."""
        ops = [
            Opinion(belief=0.7, disbelief=0.05, uncertainty=0.25, base_rate=0.5),
            Opinion(belief=0.05, disbelief=0.75, uncertainty=0.20, base_rate=0.5),
        ]
        diag = diagnose_byzantine(ops)
        assert diag["recommended"] is False

    def test_custom_threshold(self):
        """Higher threshold should be harder to trigger."""
        ops = [
            Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5),
            Opinion(belief=0.55, disbelief=0.15, uncertainty=0.30, base_rate=0.5),
            Opinion(belief=0.3, disbelief=0.4, uncertainty=0.30, base_rate=0.5),
        ]
        # Should recommend at low threshold
        low = diagnose_byzantine(ops, threshold=0.05)
        assert low["recommended"] is True
        # Should not recommend at very high threshold
        high = diagnose_byzantine(ops, threshold=0.9)
        assert high["recommended"] is False

    def test_cohesion_always_returned(self):
        """Diagnostic always includes cohesion."""
        ops = [
            Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5),
            Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5),
        ]
        diag = diagnose_byzantine(ops)
        assert 0.0 <= diag["cohesion"] <= 1.0
