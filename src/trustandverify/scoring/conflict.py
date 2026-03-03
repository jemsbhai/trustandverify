"""Within-claim conflict detection.

Re-exports jsonld-ex conflict primitives and provides the pipeline-level
detect_conflicts_within_claim() function ported from the velrichack bridge.
"""

from __future__ import annotations

from jsonld_ex.confidence_algebra import (
    Opinion,
    conflict_metric,
    pairwise_conflict,
)

from trustandverify.scoring.fusion import fuse_evidence

__all__ = ["detect_conflicts_within_claim", "pairwise_conflict", "conflict_metric"]


def detect_conflicts_within_claim(
    supporting_opinions: list[Opinion],
    contradicting_opinions: list[Opinion],
    threshold: float = 0.2,
) -> dict | None:
    """Detect conflict *within a single claim* between supporting and contradicting evidence.

    This is semantically correct: we compare evidence FOR a claim against
    evidence AGAINST the *same* claim.  Cross-claim comparison is meaningless
    because different claims are about different propositions.

    Algorithm:
        1. Fuse all supporting opinions into one via cumulative fusion.
        2. Fuse all contradicting opinions into one.
        3. Compute pairwise_conflict between the two fused opinions.
        4. Return a conflict report if the score exceeds *threshold*.

    Args:
        supporting_opinions:   Evidence opinions that support the claim.
        contradicting_opinions: Evidence opinions that contradict the claim
                                (should already be flipped via flip_opinion()).
        threshold:             Conflict score above which a conflict is flagged.

    Returns:
        A conflict report dict, or None if conflict is below threshold or
        either side has no evidence.
    """
    if not supporting_opinions or not contradicting_opinions:
        return None

    fused_for = fuse_evidence(supporting_opinions)
    fused_against = fuse_evidence(contradicting_opinions)

    score = pairwise_conflict(fused_for, fused_against)

    if score <= threshold:
        return None

    return {
        "conflict_degree": round(float(score), 4),
        "supporting_opinion": {
            "belief": round(float(fused_for.belief), 4),
            "disbelief": round(float(fused_for.disbelief), 4),
            "uncertainty": round(float(fused_for.uncertainty), 4),
        },
        "contradicting_opinion": {
            "belief": round(float(fused_against.belief), 4),
            "disbelief": round(float(fused_against.disbelief), 4),
            "uncertainty": round(float(fused_against.uncertainty), 4),
        },
        "num_supporting": len(supporting_opinions),
        "num_contradicting": len(contradicting_opinions),
    }
