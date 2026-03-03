"""Core data models for trustandverify.

Opinion is imported from jsonld-ex (do not redefine it here).
All code that needs an opinion tuple should import Opinion from
trustandverify.scoring or directly from jsonld_ex.confidence_algebra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

# Re-export Opinion from jsonld-ex so the rest of the codebase can import it
# from here without knowing the underlying package.
from jsonld_ex.confidence_algebra import Opinion  # noqa: F401

__all__ = [
    "Opinion",
    "Verdict",
    "SearchResult",
    "Source",
    "Evidence",
    "Conflict",
    "Claim",
    "ReportSummary",
    "Report",
]


class Verdict(str, Enum):
    """Verdict for a claim based on aggregated evidence."""

    SUPPORTED = "supported"
    CONTESTED = "contested"
    REFUTED = "refuted"
    NO_EVIDENCE = "no_evidence"


@dataclass
class SearchResult:
    """Raw result from a search backend."""

    title: str
    url: str
    content: str
    score: float = 0.0


@dataclass
class Source:
    """A source of evidence with trust metadata."""

    url: str
    title: str
    content_snippet: str
    trust_score: float
    source_type: str = "web"


@dataclass
class Evidence:
    """A single piece of evidence linked to a source."""

    text: str
    supports_claim: bool
    relevance: float
    confidence_raw: float
    source: Source
    opinion: Opinion | None = None


@dataclass
class Conflict:
    """Within-claim conflict summary."""

    claim_text: str
    conflict_degree: float
    num_supporting: int
    num_contradicting: int


@dataclass
class Claim:
    """A verifiable claim decomposed from the original query."""

    text: str
    evidence: list[Evidence] = field(default_factory=list)
    opinion: Opinion | None = None
    verdict: Verdict = Verdict.NO_EVIDENCE
    assessment: str = ""


@dataclass
class ReportSummary:
    """Lightweight report listing entry."""

    id: str
    query: str
    created_at: datetime
    num_claims: int


@dataclass
class Report:
    """Full verification report."""

    id: str
    query: str
    claims: list[Claim]
    conflicts: list[Conflict]
    summary: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)
