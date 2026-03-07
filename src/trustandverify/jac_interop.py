"""Jac interop — synchronous bridge for calling trustandverify from Jac walkers.

Provides sync wrappers around TrustAgent so Jac walkers can call
the full verification pipeline without async boilerplate.

Usage from Jac::

    import from trustandverify.jac_interop {
        jac_verify, jac_export, jac_configure_agent
    }

    result = jac_verify("Is coffee healthy?", num_claims=3)
    jac_export(result, format="markdown", output_path="report.md")
"""

from __future__ import annotations

import asyncio
from typing import Any

from trustandverify.core.agent import TrustAgent
from trustandverify.core.config import TrustConfig
from trustandverify.core.models import Report


def _run_async(coro: Any) -> Any:
    """Run an async coroutine synchronously — safe to call from Jac walkers."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Already inside an event loop (e.g. Jupyter, some Jac runtimes)
    import nest_asyncio  # type: ignore[import]

    nest_asyncio.apply()
    return loop.run_until_complete(coro)


def jac_configure_agent(
    search_backend: str = "tavily",
    llm_backend: str = "gemini",
    num_claims: int = 0,
    max_sources: int = 3,
    enable_byzantine: bool = False,
    enable_cache: bool = True,
    storage_backend: str = "memory",
    db_path: str = "trustandverify.db",
) -> TrustAgent:
    """Build a fully configured TrustAgent from simple string parameters.

    This is the main configuration entry point for Jac walkers — it
    translates simple string backend names into the correct Python objects.

    Args:
        search_backend:   "tavily", "brave", "bing", "serpapi", or "multi"
        llm_backend:      "gemini", "openai", "anthropic", or "ollama"
        num_claims:       Number of claims (0 = LLM decides)
        max_sources:      Max search results per claim
        enable_byzantine: Enable Byzantine-resistant fusion
        enable_cache:     Enable file-based caching
        storage_backend:  "memory" or "sqlite"
        db_path:          Path for SQLite DB (only used if storage_backend="sqlite")

    Returns:
        A configured TrustAgent ready to call .verify()
    """
    config = TrustConfig(
        num_claims=num_claims,
        max_sources_per_claim=max_sources,
        enable_byzantine=enable_byzantine,
        enable_cache=enable_cache,
    )

    search = _make_search(search_backend)
    llm = _make_llm(llm_backend)
    storage = _make_storage(storage_backend, db_path)

    return TrustAgent(config=config, search=search, llm=llm, storage=storage)


def jac_verify(
    query: str,
    *,
    search_backend: str = "tavily",
    llm_backend: str = "gemini",
    num_claims: int = 0,
    max_sources: int = 3,
    enable_byzantine: bool = False,
    enable_cache: bool = True,
    storage_backend: str = "memory",
    db_path: str = "trustandverify.db",
    verbose: bool = False,
) -> dict:
    """Run the full verification pipeline and return a JSON-serialisable dict.

    This is the one-liner entry point for Jac walkers. Returns a plain
    dict (not a Report dataclass) so Jac can work with it directly.

    Args:
        query:            The research question or claim to verify.
        search_backend:   "tavily", "brave", "bing", "serpapi", or "multi"
        llm_backend:      "gemini", "openai", "anthropic", or "ollama"
        num_claims:       Number of claims (0 = LLM decides)
        max_sources:      Max search results per claim
        enable_byzantine: Enable Byzantine-resistant fusion
        enable_cache:     Enable file-based caching
        storage_backend:  "memory" or "sqlite"
        db_path:          Path for SQLite DB
        verbose:          Print step-by-step progress

    Returns:
        Dict with keys: id, query, claims, conflicts, summary, created_at.
        Each claim has: text, verdict, assessment, opinion (dict), evidence (list).
    """
    agent = jac_configure_agent(
        search_backend=search_backend,
        llm_backend=llm_backend,
        num_claims=num_claims,
        max_sources=max_sources,
        enable_byzantine=enable_byzantine,
        enable_cache=enable_cache,
        storage_backend=storage_backend,
        db_path=db_path,
    )
    report = _run_async(agent.verify(query, verbose=verbose))
    return _report_to_dict(report)


def jac_export(
    report_dict: dict,
    format: str = "jsonld",
    output_path: str | None = None,
) -> str:
    """Export a report dict to the requested format.

    Args:
        report_dict: The dict returned by jac_verify().
        format:      "jsonld", "markdown", "md", "html"
        output_path: Optional file path to write the output.

    Returns:
        The rendered string.
    """
    report = _dict_to_report(report_dict)
    exporter = _make_exporter(format)
    rendered = exporter.render(report)

    if output_path is not None:
        mode = "wb" if isinstance(rendered, bytes) else "w"
        kwargs = {} if isinstance(rendered, bytes) else {"encoding": "utf-8"}
        with open(output_path, mode, **kwargs) as fh:
            fh.write(rendered)

    return rendered


# ── Factory helpers ────────────────────────────────────────────────────────────


def _make_search(backend: str) -> object:
    """Resolve a search backend name to an instance."""
    backend = backend.lower().strip()
    if backend == "tavily":
        from trustandverify.search.tavily import TavilySearch

        return TavilySearch()
    elif backend == "brave":
        from trustandverify.search.brave import BraveSearch

        return BraveSearch()
    elif backend == "bing":
        from trustandverify.search.bing import BingSearch

        return BingSearch()
    elif backend == "serpapi":
        from trustandverify.search.serpapi import SerpAPISearch

        return SerpAPISearch()
    elif backend == "multi":
        from trustandverify.search.bing import BingSearch
        from trustandverify.search.brave import BraveSearch
        from trustandverify.search.multi import MultiSearch
        from trustandverify.search.tavily import TavilySearch

        backends = [b for b in [TavilySearch(), BraveSearch(), BingSearch()] if b.is_available()]
        if not backends:
            raise RuntimeError("No search backends available. Set at least one API key.")
        if len(backends) == 1:
            return backends[0]
        return MultiSearch(backends)
    else:
        raise ValueError(
            f"Unknown search backend: {backend!r}. Choose from: tavily, brave, bing, serpapi, multi"
        )


def _make_llm(backend: str) -> object:
    """Resolve an LLM backend name to an instance."""
    backend = backend.lower().strip()
    if backend == "gemini":
        from trustandverify.llm.gemini import GeminiBackend

        return GeminiBackend()
    elif backend == "openai":
        from trustandverify.llm.openai import OpenAIBackend

        return OpenAIBackend()
    elif backend == "anthropic":
        from trustandverify.llm.anthropic import AnthropicBackend

        return AnthropicBackend()
    elif backend == "ollama":
        from trustandverify.llm.ollama import OllamaBackend

        return OllamaBackend()
    else:
        raise ValueError(
            f"Unknown LLM backend: {backend!r}. Choose from: gemini, openai, anthropic, ollama"
        )


def _make_storage(backend: str, db_path: str) -> object:
    """Resolve a storage backend name to an instance."""
    backend = backend.lower().strip()
    if backend == "memory":
        from trustandverify.storage.memory import InMemoryStorage

        return InMemoryStorage()
    elif backend == "sqlite":
        from trustandverify.storage.sqlite import SQLiteStorage

        return SQLiteStorage(db_path)
    else:
        raise ValueError(f"Unknown storage backend: {backend!r}. Choose from: memory, sqlite")


def _make_exporter(format: str) -> object:
    """Resolve a format name to an exporter instance."""
    format = format.lower().strip()
    if format == "jsonld":
        from trustandverify.export.jsonld import JsonLdExporter

        return JsonLdExporter()
    elif format in ("markdown", "md"):
        from trustandverify.export.markdown import MarkdownExporter

        return MarkdownExporter()
    elif format == "html":
        from trustandverify.export.html import HtmlExporter

        return HtmlExporter()
    elif format == "pdf":
        from trustandverify.export.pdf import PdfExporter

        return PdfExporter()
    else:
        raise ValueError(
            f"Unknown export format: {format!r}. Choose from: jsonld, markdown, html, pdf"
        )


# ── Serialisation helpers ──────────────────────────────────────────────────────


def _report_to_dict(report: Report) -> dict:
    """Convert a Report to a plain dict for Jac consumption."""
    return {
        "id": report.id,
        "query": report.query,
        "summary": report.summary,
        "created_at": report.created_at.isoformat(),
        "claims": [
            {
                "text": c.text,
                "verdict": c.verdict.value,
                "assessment": c.assessment,
                "opinion": {
                    "belief": c.opinion.belief,
                    "disbelief": c.opinion.disbelief,
                    "uncertainty": c.opinion.uncertainty,
                    "base_rate": c.opinion.base_rate,
                    "projected_probability": c.opinion.projected_probability(),
                }
                if c.opinion
                else None,
                "evidence": [
                    {
                        "text": e.text,
                        "supports_claim": e.supports_claim,
                        "relevance": e.relevance,
                        "confidence_raw": e.confidence_raw,
                        "source": {
                            "url": e.source.url,
                            "title": e.source.title,
                            "trust_score": e.source.trust_score,
                        },
                    }
                    for e in c.evidence
                ],
            }
            for c in report.claims
        ],
        "conflicts": [
            {
                "claim_text": c.claim_text,
                "conflict_degree": c.conflict_degree,
                "num_supporting": c.num_supporting,
                "num_contradicting": c.num_contradicting,
            }
            for c in report.conflicts
        ],
    }


def _dict_to_report(data: dict) -> Report:
    """Convert a plain dict back to a Report for export."""
    from datetime import datetime

    from jsonld_ex.confidence_algebra import Opinion

    from trustandverify.core.models import Claim, Conflict, Evidence, Source, Verdict

    claims = []
    for cd in data.get("claims", []):
        op = None
        if cd.get("opinion"):
            od = cd["opinion"]
            op = Opinion(
                belief=od["belief"],
                disbelief=od["disbelief"],
                uncertainty=od["uncertainty"],
                base_rate=od.get("base_rate", 0.5),
            )
        evidence = []
        for ed in cd.get("evidence", []):
            sd = ed.get("source", {})
            evidence.append(
                Evidence(
                    text=ed["text"],
                    supports_claim=ed["supports_claim"],
                    relevance=ed["relevance"],
                    confidence_raw=ed["confidence_raw"],
                    source=Source(
                        url=sd.get("url", ""),
                        title=sd.get("title", ""),
                        content_snippet="",
                        trust_score=sd.get("trust_score", 0.5),
                    ),
                )
            )
        claims.append(
            Claim(
                text=cd["text"],
                verdict=Verdict(cd.get("verdict", "no_evidence")),
                assessment=cd.get("assessment", ""),
                opinion=op,
                evidence=evidence,
            )
        )

    conflicts = [
        Conflict(
            claim_text=c.get("claim_text", ""),
            conflict_degree=c.get("conflict_degree", 0.0),
            num_supporting=c.get("num_supporting", 0),
            num_contradicting=c.get("num_contradicting", 0),
        )
        for c in data.get("conflicts", [])
    ]

    return Report(
        id=data.get("id", ""),
        query=data.get("query", ""),
        claims=claims,
        conflicts=conflicts,
        summary=data.get("summary", ""),
        created_at=datetime.fromisoformat(data["created_at"])
        if "created_at" in data
        else datetime.now(),
    )
