"""Evidence fusion — wrappers around jsonld-ex cumulative/averaging fusion.

We re-export the jsonld-ex operators and add fuse_evidence(), which is the
main entry point used by the pipeline.  Byzantine-resistant fusion and
cohesion/distance utilities are also exposed here.
"""

from __future__ import annotations

from jsonld_ex.confidence_algebra import (
    Opinion,
    averaging_fuse,
    cumulative_fuse,
    pairwise_conflict,
)
from jsonld_ex.confidence_byzantine import (
    ByzantineConfig,
    ByzantineFusionReport,
    byzantine_fuse,
    cohesion_score,
    opinion_distance,
)

__all__ = [
    "fuse_evidence",
    "fuse_evidence_byzantine",
    "diagnose_byzantine",
    "cumulative_fuse",
    "averaging_fuse",
    "cohesion_score",
    "opinion_distance",
]

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


def fuse_evidence_byzantine(
    opinions: list[Opinion],
    trust_weights: list[float] | None = None,
    threshold: float = 0.15,
    min_agents: int = 2,
) -> dict:
    """Fuse evidence opinions with Byzantine-resistant filtering.

    Uses the ``combined`` strategy: discord × (1 − trust), which
    prioritises removal of sources that are both highly discordant
    AND lowly trusted.

    Falls back to regular cumulative fusion when fewer than 3 opinions
    are provided or when ``trust_weights`` is None.

    Args:
        opinions:      Per-evidence opinions (already trust-discounted
                       and flipped for contradicting evidence).
        trust_weights: Per-opinion source trust scores in [0, 1].
        threshold:     Discord score above which an agent may be removed.
        min_agents:    Never reduce below this many evidence pieces.

    Returns:
        Dict with keys: fused, filtered, cohesion, surviving_indices,
        used_byzantine.
    """
    if len(opinions) < 3 or trust_weights is None:
        fused = fuse_evidence(opinions)
        return {
            "fused": fused,
            "filtered": [],
            "cohesion": cohesion_score(opinions) if len(opinions) > 1 else 1.0,
            "surviving_indices": list(range(len(opinions))),
            "used_byzantine": False,
        }

    config = ByzantineConfig(
        strategy="combined",
        trust_weights=trust_weights,
        threshold=threshold,
        min_agents=min_agents,
    )

    report: ByzantineFusionReport = byzantine_fuse(opinions, config=config)

    filtered = [
        {
            "index": removal.index,
            "opinion": {
                "belief": round(float(removal.opinion.belief), 4),
                "disbelief": round(float(removal.opinion.disbelief), 4),
                "uncertainty": round(float(removal.opinion.uncertainty), 4),
            },
            "discord_score": round(float(removal.discord_score), 4),
            "reason": removal.reason,
        }
        for removal in report.removed
    ]

    return {
        "fused": report.fused,
        "filtered": filtered,
        "cohesion": round(float(report.cohesion_score), 4),
        "surviving_indices": report.surviving_indices,
        "used_byzantine": True,
    }


def diagnose_byzantine(
    opinions: list[Opinion],
    threshold: float = 0.15,
) -> dict:
    """Lightweight diagnostic: should Byzantine fusion be enabled?

    Computes mean pairwise discord for each opinion and checks whether
    any exceed the threshold.  This is the same O(n²) computation that
    Byzantine fusion does in its first pass, but stops at diagnosis —
    no removal, no re-fusion.

    Always returns cohesion so the caller can display it regardless
    of whether Byzantine fusion is recommended.

    Args:
        opinions:  Evidence opinions to diagnose.
        threshold: Discord score above which an agent is considered
                   discordant.

    Returns:
        Dict with keys: recommended (bool), num_discordant (int),
        reason (str), cohesion (float).
    """
    n = len(opinions)

    if n < 3:
        return {
            "recommended": False,
            "num_discordant": 0,
            "reason": "fewer than 3 opinions" if n > 0 else "no opinions",
            "cohesion": cohesion_score(opinions) if n > 1 else 1.0,
        }

    # Compute mean pairwise discord per agent
    discord = [0.0] * n
    for i in range(n):
        for j in range(i + 1, n):
            c = pairwise_conflict(opinions[i], opinions[j])
            discord[i] += c
            discord[j] += c
    for i in range(n):
        discord[i] /= n - 1

    discordant = [i for i in range(n) if discord[i] >= threshold]
    coh = cohesion_score(opinions)

    if discordant:
        return {
            "recommended": True,
            "num_discordant": len(discordant),
            "reason": (f"{len(discordant)} of {n} sources have discord >= {threshold}"),
            "cohesion": round(float(coh), 4),
        }

    return {
        "recommended": False,
        "num_discordant": 0,
        "reason": "all sources below discord threshold",
        "cohesion": round(float(coh), 4),
    }
