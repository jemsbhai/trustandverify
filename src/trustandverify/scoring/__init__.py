"""trustandverify.scoring — public API."""

from jsonld_ex.confidence_algebra import Opinion

from trustandverify.scoring.algebra import build_evidence_opinion, score_claim
from trustandverify.scoring.conflict import (
    conflict_metric,
    detect_conflicts_within_claim,
    pairwise_conflict,
)
from trustandverify.scoring.fusion import (
    averaging_fuse,
    cohesion_score,
    cumulative_fuse,
    diagnose_byzantine,
    fuse_evidence,
    fuse_evidence_byzantine,
    opinion_distance,
)
from trustandverify.scoring.opinions import flip_opinion, opinion_summary, scalar_to_opinion
from trustandverify.scoring.trust import apply_trust_discount, estimate_source_trust, trust_discount

__all__ = [
    # Opinion type (from jsonld-ex)
    "Opinion",
    # opinions.py
    "scalar_to_opinion",
    "flip_opinion",
    "opinion_summary",
    # fusion.py
    "fuse_evidence",
    "fuse_evidence_byzantine",
    "diagnose_byzantine",
    "cumulative_fuse",
    "averaging_fuse",
    "cohesion_score",
    "opinion_distance",
    # trust.py
    "estimate_source_trust",
    "apply_trust_discount",
    "trust_discount",
    # conflict.py
    "detect_conflicts_within_claim",
    "pairwise_conflict",
    "conflict_metric",
    # algebra.py
    "score_claim",
    "build_evidence_opinion",
]
