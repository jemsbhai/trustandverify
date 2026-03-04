"""TrustConfig — all agent settings in one place."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrustConfig:
    """Configuration for the TrustAgent pipeline.

    All fields have sensible defaults so you can do ``TrustConfig()``
    for a minimal working setup and override only what you need.
    """

    # ── Claim decomposition ────────────────────────────────────────
    num_claims: int = 0
    """Number of claims to decompose the query into.
    0 = let the LLM decide (typically 3-5)."""

    # ── Evidence gathering ─────────────────────────────────────────
    max_sources_per_claim: int = 3
    """Maximum search results to fetch per claim."""

    # ── Confidence scoring ─────────────────────────────────────────
    base_uncertainty: float = 0.3
    """Starting uncertainty per individual evidence piece.
    Decreases as more corroborating sources are fused."""

    conflict_threshold: float = 0.2
    """Pairwise conflict score above which a within-claim conflict
    is flagged in the report."""

    # ── Source trust ───────────────────────────────────────────────
    default_source_trust: float = 0.5
    """Fallback trust score for unrecognised domains."""

    # ── Caching ────────────────────────────────────────────────────
    cache_ttl: int = 3600
    """Time-to-live for cached search/LLM results in seconds."""

    enable_cache: bool = True
    """Whether to use the cache layer at all."""

    # ── Byzantine fusion ─────────────────────────────────────────
    enable_byzantine: bool = False
    """Whether to use Byzantine-resistant fusion (default: off).
    When off, a lightweight diagnostic still flags whether
    Byzantine filtering would be recommended."""

    byzantine_threshold: float = 0.15
    """Discord score above which a source may be removed (Byzantine)
    or flagged (diagnostic)."""

    byzantine_min_agents: int = 2
    """Never reduce below this many evidence pieces during
    Byzantine filtering."""

    # ── Output ─────────────────────────────────────────────────────
    export_formats: list[str] = field(default_factory=lambda: ["jsonld"])
    """Default export formats when calling agent.verify()."""
