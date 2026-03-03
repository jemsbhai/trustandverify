"""Evidence fusion — wrappers around jsonld-ex cumulative/averaging fusion.

We re-export the jsonld-ex operators and add fuse_evidence(), which is the
main entry point used by the pipeline.
"""

from __future__ import annotations

from jsonld_ex.confidence_algebra import (
    Opinion,
    averaging_fuse,
    cumulative_fuse,
)

__all__ = ["fuse_evidence", "cumulative_fuse", "averaging_fuse"]

# Vacuous opinion: total ignorance — returned when there is no evidence.
_VACUOUS = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0, base_rate=0.5)


def fuse_evidence(opinions: list[Opinion]) -> Opinion:
    """Fuse a list of evidence opinions into a single opinion.

    Uses cumulative fusion (Jøsang 2016 §12.3) — the correct operator
    for *independent* sources observing the same proposition.  Each
    additional agreeing source reduces uncertainty; disagreeing sources
    balance belief against disbelief.

    An empty list returns the vacuous opinion (full uncertainty).
    A single opinion is returned unchanged.

    Args:
        opinions: Evidence opinions (already trust-discounted if desired).

    Returns:
        Fused Opinion.
    """
    if not opinions:
        return _VACUOUS
    if len(opinions) == 1:
        return opinions[0]
    return cumulative_fuse(*opinions)
