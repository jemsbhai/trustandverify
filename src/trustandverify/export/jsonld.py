"""JSON-LD exporter — ported from build_jsonld_claim() in velrichack bridge/confidence.py.

Produces reports conforming to Schema.org + jsonld-ex vocab + PROV-O,
matching the output format of the velrichack hackathon system exactly.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from trustandverify.core.models import Claim, Conflict, Opinion, Report

_CONTEXT = {
    "@vocab": "https://schema.org/",
    "ex": "https://jsonld-ex.org/vocab#",
    "prov": "http://www.w3.org/ns/prov#",
}


class JsonLdExporter:
    """Render a Report to JSON-LD format."""

    format_name = "jsonld"
    file_extension = ".jsonld"

    def render(self, report: Report) -> str:
        """Render report to a JSON-LD string."""
        return json.dumps(self._build_doc(report), indent=2)

    def render_to_file(self, report: Report, path: str) -> None:
        """Render report and write to file."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.render(report))

    # ── Internal builders ──────────────────────────────────────────

    def _build_doc(self, report: Report) -> dict:
        return {
            "@context": _CONTEXT,
            "@type": "ex:TrustGraphReport",
            "ex:query": report.query,
            "ex:generatedAt": report.created_at.isoformat(),
            "ex:claims": [self._build_claim(c) for c in report.claims],
            "ex:conflicts": [self._build_conflict(c) for c in report.conflicts],
            "ex:summary": report.summary,
        }

    def _build_claim(self, claim: Claim) -> dict:
        doc: dict = {
            "@type": "ex:VerifiedClaim",
            "ex:claimText": claim.text,
        }

        if claim.opinion is not None:
            doc["ex:confidence"] = _opinion_to_jsonld(claim.opinion)

        doc["prov:wasGeneratedBy"] = {
            "@type": "prov:Activity",
            "prov:wasAssociatedWith": "trustandverify",
            "prov:endedAtTime": datetime.now(timezone.utc).isoformat(),
        }

        # Sources from evidence
        sources = []
        for ev in claim.evidence:
            sources.append({
                "title": ev.source.title,
                "url": ev.source.url,
                "trust_score": ev.source.trust_score,
                "evidence": ev.text[:150],
                "supports": ev.supports_claim,
            })
        if sources:
            doc["ex:sources"] = sources

        if claim.assessment:
            doc["ex:assessment"] = claim.assessment

        return doc

    def _build_conflict(self, conflict: Conflict) -> dict:
        return {
            "claim": conflict.claim_text,
            "conflict_degree": round(conflict.conflict_degree, 4),
            "num_supporting": conflict.num_supporting,
            "num_contradicting": conflict.num_contradicting,
        }


def _opinion_to_jsonld(op: Opinion) -> dict:
    proj = op.projected_probability()
    return {
        "@type": "ex:SubjectiveOpinion",
        "ex:belief": round(op.belief, 4),
        "ex:disbelief": round(op.disbelief, 4),
        "ex:uncertainty": round(op.uncertainty, 4),
        "ex:baseRate": round(op.base_rate, 4),
        "ex:projectedProbability": round(proj, 4),
    }
