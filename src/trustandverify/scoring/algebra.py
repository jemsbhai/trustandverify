"""High-level scoring algebra — score_claim() and fuse_evidence_list().

This module is the main entry point for the pipeline.  It combines
opinions, trust, and conflict into a single score_claim() call.
"""

from __future__ import annotations

from jsonld_ex.confidence_algebra import Opinion

from trustandverify.core.models import Evidence, Verdict
from trustandverify.scoring.conflict import detect_conflicts_within_claim
from trustandverify.scoring.fusion import (
    diagnose_byzantine,
    fuse_evidence,
    fuse_evidence_byzantine,
)
from trustandverify.scoring.opinions import flip_opinion, opinion_summary, scalar_to_opinion
from trustandverify.scoring.trust import apply_trust_discount

__all__ = ["score_claim", "build_evidence_opinion"]


def build_evidence_opinion(ev: Evidence) -> Opinion:
    """Convert a single Evidence object to a trust-discounted Opinion.

    Steps:
        1. Map ev.confidence_raw → Opinion via scalar_to_opinion.
        2. Apply trust discount from ev.source.trust_score.
        3. If the evidence contradicts the claim, flip the opinion so
           that belief ↔ disbelief (required for correct conflict detection).

    Args:
        ev: Evidence object with confidence_raw, source, and supports_claim.

    Returns:
        Trust-discounted (and possibly flipped) Opinion.
    """
    raw_op = scalar_to_opinion(ev.confidence_raw)
    discounted = apply_trust_discount(raw_op, ev.source.trust_score)
    if not ev.supports_claim:
        discounted = flip_opinion(discounted)
    return discounted


def score_claim(
    evidence_list: list[Evidence],
    conflict_threshold: float = 0.2,
    enable_byzantine: bool = False,
    byzantine_threshold: float = 0.15,
) -> tuple[Opinion, Verdict, dict | None, dict]:
    """Score a claim from its list of Evidence objects.

    This is the pipeline's main scoring entry point.  It:
        1. Converts each Evidence to a trust-discounted opinion.
        2. Fuses all opinions via cumulative fusion (or Byzantine fusion
           if ``enable_byzantine`` is True).
        3. Derives the Verdict from the projected probability.
        4. Runs within-claim conflict detection.
        5. Returns a meta dict with cohesion, diagnostic, and (if
           Byzantine is enabled) filtering details.

    Args:
        evidence_list:        All evidence collected for this claim.
        conflict_threshold:   Conflict score above which a conflict is flagged.
        enable_byzantine:     If True, use Byzantine-resistant fusion.
        byzantine_threshold:  Discord threshold for Byzantine filtering.

    Returns:
        Tuple of (fused Opinion, Verdict, conflict dict or None, meta dict).

        The meta dict always contains:
            - cohesion (float): Source agreement score in [0, 1].

        When ``enable_byzantine`` is False (default):
            - byzantine_recommended (bool): True if diagnostic suggests filtering.
            - byzantine_reason (str): Human-readable reason.
            - num_discordant (int): How many sources exceed discord threshold.

        When ``enable_byzantine`` is True:
            - used_byzantine (bool): True.
            - filtered (list[dict]): Removed evidence details.
            - surviving_indices (list[int]): Which evidence was kept.
    """
    if not evidence_list:
        vacuous = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0, base_rate=0.5)
        return vacuous, Verdict.NO_EVIDENCE, None, {"cohesion": 1.0}

    supporting_ops: list[Opinion] = []
    contradicting_ops: list[Opinion] = []
    all_ops: list[Opinion] = []
    trust_weights: list[float] = []

    for ev in evidence_list:
        op = build_evidence_opinion(ev)
        all_ops.append(op)
        trust_weights.append(ev.source.trust_score)
        if ev.supports_claim:
            supporting_ops.append(op)
        else:
            contradicting_ops.append(op)

    # ── Fusion ──
    meta: dict
    if enable_byzantine:
        byz = fuse_evidence_byzantine(
            all_ops,
            trust_weights=trust_weights,
            threshold=byzantine_threshold,
        )
        fused = byz["fused"]
        meta = {
            "cohesion": byz["cohesion"],
            "used_byzantine": byz["used_byzantine"],
            "filtered": byz["filtered"],
            "surviving_indices": byz["surviving_indices"],
        }
    else:
        fused = fuse_evidence(all_ops)
        diag = diagnose_byzantine(all_ops, threshold=byzantine_threshold)
        meta = {
            "cohesion": diag["cohesion"],
            "byzantine_recommended": diag["recommended"],
            "byzantine_reason": diag["reason"],
            "num_discordant": diag["num_discordant"],
        }

    summary = opinion_summary(fused)

    verdict_map = {
        "supported": Verdict.SUPPORTED,
        "contested": Verdict.CONTESTED,
        "refuted": Verdict.REFUTED,
    }
    verdict = verdict_map.get(summary["verdict"], Verdict.NO_EVIDENCE)

    conflict = detect_conflicts_within_claim(supporting_ops, contradicting_ops, conflict_threshold)

    return fused, verdict, conflict, meta
