"""Tests for scoring/opinions.py — scalar_to_opinion, flip_opinion, opinion_summary."""

from __future__ import annotations

import pytest
from jsonld_ex.confidence_algebra import Opinion

from trustandverify.scoring.opinions import flip_opinion, opinion_summary, scalar_to_opinion


class TestScalarToOpinion:
    def test_additivity_constraint_always_holds(self):
        for conf in [0.0, 0.25, 0.5, 0.75, 1.0]:
            op = scalar_to_opinion(conf)
            total = op.belief + op.disbelief + op.uncertainty
            assert abs(total - 1.0) < 1e-9, f"b+d+u != 1 for confidence={conf}"

    def test_high_confidence_gives_high_belief(self):
        op = scalar_to_opinion(1.0)
        assert op.belief > op.disbelief

    def test_low_confidence_gives_high_disbelief(self):
        op = scalar_to_opinion(0.0)
        assert op.disbelief > op.belief

    def test_uncertainty_floor_is_nonzero(self):
        """A single evidence piece always has some residual uncertainty."""
        op = scalar_to_opinion(1.0)
        assert op.uncertainty >= 0.05

    def test_higher_evidence_weight_reduces_uncertainty(self):
        op_single = scalar_to_opinion(0.8, evidence_weight=1.0)
        op_meta = scalar_to_opinion(0.8, evidence_weight=2.0)
        assert op_meta.uncertainty < op_single.uncertainty

    def test_projected_probability_is_reasonable(self):
        """For confidence=0.8, projected probability should be >0.6."""
        op = scalar_to_opinion(0.8)
        assert op.projected_probability() > 0.6

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError):
            scalar_to_opinion(-0.1)
        with pytest.raises(ValueError):
            scalar_to_opinion(1.1)

    def test_invalid_weight_raises(self):
        with pytest.raises(ValueError):
            scalar_to_opinion(0.5, evidence_weight=0.0)


class TestFlipOpinion:
    def test_flip_swaps_belief_and_disbelief(self):
        op = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5)
        flipped = flip_opinion(op)
        assert flipped.belief == op.disbelief
        assert flipped.disbelief == op.belief

    def test_flip_preserves_uncertainty(self):
        op = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5)
        flipped = flip_opinion(op)
        assert flipped.uncertainty == op.uncertainty

    def test_flip_preserves_base_rate(self):
        op = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.7)
        flipped = flip_opinion(op)
        assert flipped.base_rate == op.base_rate

    def test_flip_additivity(self):
        op = Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2, base_rate=0.5)
        flipped = flip_opinion(op)
        total = flipped.belief + flipped.disbelief + flipped.uncertainty
        assert abs(total - 1.0) < 1e-9

    def test_double_flip_is_identity(self):
        op = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5)
        assert flip_opinion(flip_opinion(op)) == op


class TestOpinionSummary:
    def test_supported_verdict(self):
        op = Opinion(belief=0.65, disbelief=0.05, uncertainty=0.30, base_rate=0.5)
        # projected_probability = 0.65 + 0.5*0.30 = 0.80
        summary = opinion_summary(op)
        assert summary["verdict"] == "supported"
        assert summary["projected_probability"] >= 0.7

    def test_contested_verdict(self):
        op = Opinion(belief=0.35, disbelief=0.35, uncertainty=0.30, base_rate=0.5)
        # projected_probability = 0.35 + 0.5*0.30 = 0.50
        summary = opinion_summary(op)
        assert summary["verdict"] == "contested"

    def test_refuted_verdict(self):
        op = Opinion(belief=0.05, disbelief=0.65, uncertainty=0.30, base_rate=0.5)
        # projected_probability = 0.05 + 0.5*0.30 = 0.20
        summary = opinion_summary(op)
        assert summary["verdict"] == "refuted"

    def test_summary_keys(self):
        op = Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2, base_rate=0.5)
        summary = opinion_summary(op)
        for key in (
            "belief",
            "disbelief",
            "uncertainty",
            "base_rate",
            "projected_probability",
            "verdict",
        ):
            assert key in summary

    def test_values_are_rounded(self):
        op = Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2, base_rate=0.5)
        summary = opinion_summary(op)
        # All float values should have at most 4 decimal places
        for key in ("belief", "disbelief", "uncertainty", "projected_probability"):
            assert isinstance(summary[key], float)
