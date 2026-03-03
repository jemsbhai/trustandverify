"""Tests for scoring/fusion.py — fuse_evidence and cumulative_fuse behaviour."""

from __future__ import annotations

import pytest
from jsonld_ex.confidence_algebra import Opinion

from trustandverify.scoring.fusion import fuse_evidence


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
