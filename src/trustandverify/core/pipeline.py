"""Pipeline stages: Plan → Search → Extract → Score → Report.

Each stage is a standalone async function so it can be tested and
composed independently.  TrustAgent in agent.py wires them together.

Ported directly from the TrustGraphAgent walker in trustgraph.jac.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from trustandverify.core.config import TrustConfig
from trustandverify.core.models import (
    Claim,
    Conflict,
    Evidence,
    Report,
    SearchResult,
    Source,
    Verdict,
)
from trustandverify.llm import prompts as P
from trustandverify.scoring.algebra import build_evidence_opinion, score_claim
from trustandverify.scoring.conflict import detect_conflicts_within_claim
from trustandverify.scoring.fusion import fuse_evidence
from trustandverify.scoring.opinions import flip_opinion, opinion_summary, scalar_to_opinion
from trustandverify.scoring.trust import apply_trust_discount, estimate_source_trust


# ── Stage 1: PLAN ─────────────────────────────────────────────────────────────

async def plan(query: str, config: TrustConfig, llm: object) -> list[str]:
    """Decompose the query into verifiable claim texts.

    Args:
        query:  The research question.
        config: TrustConfig (uses num_claims).
        llm:    LLMBackend instance.

    Returns:
        List of claim text strings.
    """
    prompt = P.decompose_query(query, num_claims=config.num_claims)
    result = await llm.complete_json(prompt)  # type: ignore[union-attr]

    # complete_json wraps list responses as {"items": [...]}
    if isinstance(result, dict) and "items" in result:
        claims = result["items"]
    elif isinstance(result, list):
        claims = result
    else:
        claims = list(result.values()) if result else []

    return [str(c) for c in claims if c]


# ── Stage 2: SEARCH ───────────────────────────────────────────────────────────

async def search_for_claim(
    claim_text: str,
    config: TrustConfig,
    search: object,
    llm: object,
    cache: object | None = None,
) -> list[SearchResult]:
    """Generate a search query for a claim and fetch results.

    Args:
        claim_text: The claim to find evidence for.
        config:     TrustConfig (uses max_sources_per_claim).
        search:     SearchBackend instance.
        llm:        LLMBackend instance.
        cache:      Optional CacheBackend.

    Returns:
        List of SearchResult objects.
    """
    # Generate an optimised search query
    query_prompt = P.claim_to_search_query(claim_text)
    cache_key = f"search_query:{claim_text}"

    if cache is not None:
        cached_query = await cache.get(cache_key)  # type: ignore[union-attr]
    else:
        cached_query = None

    if cached_query:
        search_query = cached_query
    else:
        search_query = (await llm.complete(query_prompt)).strip()  # type: ignore[union-attr]
        if cache is not None:
            await cache.set(cache_key, search_query)  # type: ignore[union-attr]

    # Fetch results, with optional cache
    results_key = f"search_results:{search_query}:{config.max_sources_per_claim}"
    if cache is not None:
        cached_results = await cache.get(results_key)  # type: ignore[union-attr]
        if cached_results:
            return [SearchResult(**r) for r in cached_results]

    results = await search.search(search_query, config.max_sources_per_claim)  # type: ignore[union-attr]

    if cache is not None and results:
        await cache.set(  # type: ignore[union-attr]
            results_key,
            [{"title": r.title, "url": r.url, "content": r.content, "score": r.score}
             for r in results],
        )

    return results


# ── Stage 3: EXTRACT ──────────────────────────────────────────────────────────

async def extract(
    claim_text: str,
    results: list[SearchResult],
    llm: object,
    cache: object | None = None,
) -> list[Evidence]:
    """Extract structured evidence from search results for a claim.

    Args:
        claim_text: The claim being verified.
        results:    Search results to analyse.
        llm:        LLMBackend instance.
        cache:      Optional CacheBackend.

    Returns:
        List of Evidence objects, one per search result.
    """
    evidence_list: list[Evidence] = []

    for result in results:
        cache_key = f"evidence:{claim_text}:{result.url}"

        if cache is not None:
            cached = await cache.get(cache_key)  # type: ignore[union-attr]
        else:
            cached = None

        if cached:
            ev_data = cached
        else:
            prompt = P.extract_evidence(claim_text, result.content[:1500])
            ev_data = await llm.complete_json(  # type: ignore[union-attr]
                prompt,
                defaults={
                    "evidence": result.content[:300],
                    "supports": True,
                    "relevance": 0.5,
                    "confidence": 0.5,
                },
            )
            if cache is not None:
                await cache.set(cache_key, ev_data)  # type: ignore[union-attr]

        trust = estimate_source_trust(result.url, result.title)
        source = Source(
            url=result.url,
            title=result.title,
            content_snippet=result.content[:500],
            trust_score=trust,
        )

        evidence_list.append(
            Evidence(
                text=str(ev_data.get("evidence", result.content[:300]))[:300],
                supports_claim=bool(ev_data.get("supports", True)),
                relevance=float(ev_data.get("relevance", 0.5)),
                confidence_raw=float(ev_data.get("confidence", 0.5)),
                source=source,
            )
        )

    return evidence_list


# ── Stage 4: SCORE ────────────────────────────────────────────────────────────

def score(claim: Claim, config: TrustConfig) -> tuple[Claim, Conflict | None]:
    """Score a claim from its evidence list using Subjective Logic.

    Mutates claim.opinion and claim.verdict in place.

    Returns:
        (updated Claim, Conflict | None)
    """
    fused_opinion, verdict, conflict_data = score_claim(
        claim.evidence, conflict_threshold=config.conflict_threshold
    )
    claim.opinion = fused_opinion
    claim.verdict = verdict

    conflict: Conflict | None = None
    if conflict_data is not None:
        conflict = Conflict(
            claim_text=claim.text[:80],
            conflict_degree=conflict_data["conflict_degree"],
            num_supporting=conflict_data["num_supporting"],
            num_contradicting=conflict_data["num_contradicting"],
        )

    return claim, conflict


# ── Stage 5: REPORT ───────────────────────────────────────────────────────────

async def assess(claim: Claim, llm: object, cache: object | None = None) -> str:
    """Write a 2-3 sentence assessment for a single scored claim.

    Args:
        claim: A scored Claim with opinion and verdict set.
        llm:   LLMBackend instance.
        cache: Optional CacheBackend.

    Returns:
        Assessment text string.
    """
    supporting = [e.text for e in claim.evidence if e.supports_claim]
    contradicting = [e.text for e in claim.evidence if not e.supports_claim]
    confidence = claim.opinion.projected_probability() if claim.opinion else 0.5

    cache_key = f"assess:{claim.text}:{confidence:.3f}"
    if cache is not None:
        cached = await cache.get(cache_key)  # type: ignore[union-attr]
        if cached:
            return str(cached)

    prompt = P.assess_claim(claim.text, supporting, contradicting, confidence)
    text = (await llm.complete(prompt)).strip()  # type: ignore[union-attr]

    if cache is not None:
        await cache.set(cache_key, text)  # type: ignore[union-attr]

    return text


async def summarise(
    query: str,
    claims: list[Claim],
    llm: object,
    cache: object | None = None,
) -> str:
    """Write an executive summary for the full report.

    Args:
        query:  The original research question.
        claims: All scored and assessed claims.
        llm:    LLMBackend instance.
        cache:  Optional CacheBackend.

    Returns:
        Summary text string.
    """
    assessed_strs = [
        f"[{c.verdict.value.upper()} "
        f"P={c.opinion.projected_probability():.2f}] {c.text}\n"
        f"  Assessment: {c.assessment}"
        for c in claims
        if c.opinion is not None
    ]

    cache_key = f"summary:{query}:{len(claims)}"
    if cache is not None:
        cached = await cache.get(cache_key)  # type: ignore[union-attr]
        if cached:
            return str(cached)

    prompt = P.write_summary(query, assessed_strs)
    text = (await llm.complete(prompt)).strip()  # type: ignore[union-attr]

    if cache is not None:
        await cache.set(cache_key, text)  # type: ignore[union-attr]

    return text


async def run_pipeline(
    query: str,
    config: TrustConfig,
    search: object,
    llm: object,
    cache: object | None = None,
    verbose: bool = False,
) -> Report:
    """Run the full 5-stage pipeline and return a Report.

    Args:
        query:   The research question.
        config:  TrustConfig.
        search:  SearchBackend instance.
        llm:     LLMBackend instance.
        cache:   Optional CacheBackend.
        verbose: Print progress to stdout.

    Returns:
        A fully populated Report.
    """

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    _log(f"\n{'='*60}")
    _log("  trustandverify — Agentic Knowledge Verification")
    _log(f"{'='*60}")
    _log(f"\n  Query: {query}\n")

    # ── Stage 1: PLAN ──
    _log("[1/5] PLAN — Decomposing query into verifiable claims...")
    claim_texts = await plan(query, config, llm)
    _log(f"      Found {len(claim_texts)} claims.\n")

    claims: list[Claim] = [Claim(text=t) for t in claim_texts]
    conflicts: list[Conflict] = []

    # ── Stages 2-4: per claim ──
    for i, claim in enumerate(claims):
        _log(f"[2/5] SEARCH — Claim {i+1}: {claim.text[:70]}...")
        results = await search_for_claim(claim.text, config, search, llm, cache)
        _log(f"      Found {len(results)} sources.")

        _log("[3/5] EXTRACT — Analysing evidence...")
        claim.evidence = await extract(claim.text, results, llm, cache)

        _log("[4/5] SCORE — Fusing evidence with Subjective Logic...")
        claim, conflict = score(claim, config)

        if claim.opinion is not None:
            summary = opinion_summary(claim.opinion)
            _log(f"      Result: {summary['verdict']} (P={summary['projected_probability']})")
            _log(f"      Opinion: b={summary['belief']} d={summary['disbelief']} u={summary['uncertainty']}")

        if conflict is not None:
            conflicts.append(conflict)
            _log(
                f"      CONFLICT: {conflict.num_supporting} support vs "
                f"{conflict.num_contradicting} contradict "
                f"(degree={conflict.conflict_degree})"
            )

        _log("[4/5] ASSESS — Writing claim assessment...")
        claim.assessment = await assess(claim, llm, cache)
        _log(f"      {claim.assessment[:100]}...\n")

    # ── Stage 5: REPORT ──
    _log("[5/5] REPORT — Generating summary...")
    summary_text = await summarise(query, claims, llm, cache)

    report = Report(
        id=str(uuid.uuid4()),
        query=query,
        claims=claims,
        conflicts=conflicts,
        summary=summary_text,
        created_at=datetime.now(timezone.utc),
    )

    if verbose:
        _log(f"\n{'='*60}")
        _log("  SUMMARY:")
        _log(f"  {summary_text}")
        _log(f"{'='*60}\n")

    return report
