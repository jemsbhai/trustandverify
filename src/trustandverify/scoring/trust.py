"""Source trust scoring — heuristics and trust discount operator.

We re-export trust_discount from jsonld-ex and add the domain-based
source trust heuristic ported from the velrichack bridge.
"""

from __future__ import annotations

from jsonld_ex.confidence_algebra import Opinion, trust_discount

__all__ = ["estimate_source_trust", "apply_trust_discount", "trust_discount"]


def estimate_source_trust(url: str, title: str = "") -> float:  # noqa: ARG001
    """Heuristic trust score for a web source based on its domain.

    Returns a score in [0, 1].  These values are deliberately conservative —
    even .gov sources are not scored at 1.0 because domain alone cannot
    guarantee the specific page is reliable.

    Args:
        url:   The source URL.
        title: The page title (unused currently, reserved for future use).

    Returns:
        Trust score in [0, 1].
    """
    u = url.lower()

    # Government and academic institutions
    if ".gov" in u or ".edu" in u:
        return 0.9

    # Major research platforms
    if any(d in u for d in ("nature.com", "sciencedirect", "pubmed", "arxiv.org", "nber.org")):
        return 0.85

    # Established news organisations
    if any(d in u for d in ("reuters.com", "bbc.com", "nytimes.com", "wsj.com", "economist.com")):
        return 0.75

    # Wikipedia — useful but crowd-edited
    if "wikipedia.org" in u:
        return 0.60

    # Social / opinion platforms
    if any(d in u for d in ("reddit.com", "quora.com", "twitter.com", "x.com")):
        return 0.35

    return 0.5  # default for unrecognised domains


def apply_trust_discount(opinion: Opinion, source_trust: float) -> Opinion:
    """Discount an evidence opinion by the trustworthiness of its source.

    Implements Jøsang's trust discount operator (§14.3):
        - trust=1.0 → opinion is unchanged
        - trust=0.0 → opinion collapses to pure uncertainty

    Args:
        opinion:      The evidence opinion to discount.
        source_trust: Trust score for the source, in [0, 1].

    Returns:
        Discounted Opinion.
    """
    if not (0.0 <= source_trust <= 1.0):
        raise ValueError(f"source_trust must be in [0, 1], got {source_trust}")

    trust_op = Opinion(
        belief=source_trust,
        disbelief=1.0 - source_trust,
        uncertainty=0.0,
        base_rate=0.5,
    )
    return trust_discount(trust_op, opinion)
