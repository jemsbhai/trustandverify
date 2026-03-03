"""Opinion helpers — scalar-to-opinion mapping and summary utilities.

We import Opinion directly from jsonld-ex and do NOT reimplement the
Subjective Logic math.  This module provides the application-level
helpers that sit on top of the formal algebra.
"""

from __future__ import annotations

from jsonld_ex.confidence_algebra import Opinion

__all__ = ["scalar_to_opinion", "flip_opinion", "opinion_summary", "Opinion"]


def scalar_to_opinion(confidence: float, evidence_weight: float = 1.0) -> Opinion:
    """Convert a scalar LLM confidence score in [0, 1] to a Subjective Logic opinion.

    A single evidence piece still carries meaningful uncertainty — we don't
    collapse it to a dogmatic opinion.  The base uncertainty of 0.3 means
    even a fully confident source (confidence=1.0) yields P ≈ 0.85, which
    appropriately reflects that one web search result is not ground truth.

    Uncertainty decreases as ``evidence_weight`` increases (e.g. a
    meta-analysis covering many studies could use weight=2.0).  Individual
    evidence pieces should always use the default weight=1.0; fusion handles
    combining multiple pieces mathematically.

    Args:
        confidence:      LLM-assessed confidence in [0, 1].
        evidence_weight: Relative weight of this piece of evidence.
                         Higher values reduce uncertainty.

    Returns:
        Opinion with b + d + u = 1.
    """
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")
    if evidence_weight <= 0:
        raise ValueError(f"evidence_weight must be > 0, got {evidence_weight}")

    # Uncertainty: decreases with evidence weight, floor at 0.05
    u = max(0.05, 0.3 / evidence_weight)
    remaining = 1.0 - u

    b = remaining * confidence
    d = remaining * (1.0 - confidence)

    return Opinion(belief=b, disbelief=d, uncertainty=u, base_rate=0.5)


def flip_opinion(op: Opinion) -> Opinion:
    """Flip an opinion by swapping belief and disbelief.

    Use this when evidence *contradicts* a claim.  Without flipping,
    both supporting and contradicting evidence would have high belief,
    making pairwise_conflict see no disagreement even when sources
    strongly oppose each other.

    Args:
        op: The opinion to flip.

    Returns:
        New Opinion with belief and disbelief swapped; u and base_rate unchanged.
    """
    return Opinion(
        belief=op.disbelief,
        disbelief=op.belief,
        uncertainty=op.uncertainty,
        base_rate=op.base_rate,
    )


def opinion_summary(op: Opinion) -> dict:
    """Return a human-readable summary dict for an opinion.

    Returns:
        Dict with keys: belief, disbelief, uncertainty, base_rate,
        projected_probability, verdict.
    """
    proj = op.projected_probability()
    verdict: str
    if proj >= 0.7:
        verdict = "supported"
    elif proj > 0.3:
        verdict = "contested"
    else:
        verdict = "refuted"

    return {
        "belief": round(op.belief, 4),
        "disbelief": round(op.disbelief, 4),
        "uncertainty": round(op.uncertainty, 4),
        "base_rate": round(op.base_rate, 4),
        "projected_probability": round(proj, 4),
        "verdict": verdict,
    }
