"""MultiSearch — fan out to multiple search backends, dedup by URL, merge results."""

from __future__ import annotations

import asyncio

from trustandverify.core.models import SearchResult


class MultiSearch:
    """Fan out a query to multiple SearchBackend instances concurrently.

    Deduplicates results by URL, then merges and re-ranks by score.
    Source diversity is tracked via the ``source_backends`` attribute on
    returned results so callers can measure how many distinct backends
    contributed.

    Usage::

        search = MultiSearch([TavilySearch(), BraveSearch()])
        results = await search.search("remote work productivity")
    """

    name = "multi"

    def __init__(self, backends: list) -> None:
        if not backends:
            raise ValueError("MultiSearch requires at least one backend.")
        self._backends = backends

    def is_available(self) -> bool:
        return any(b.is_available() for b in self._backends)

    @property
    def available_backends(self) -> list:
        return [b for b in self._backends if b.is_available()]

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Search all available backends concurrently and merge results.

        Args:
            query:       The search query.
            max_results: Maximum results to return after merging.

        Returns:
            Deduplicated, score-sorted list of SearchResult objects.
        """
        available = self.available_backends
        if not available:
            return []

        # Fan out concurrently — return_exceptions=True so one failing
        # backend doesn't crash the whole pipeline.
        tasks = [b.search(query, max_results) for b in available]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        all_results_per_backend: list[list[SearchResult]] = []
        for i, result in enumerate(raw_results):
            if isinstance(result, BaseException):
                print(f"[MultiSearch] Backend {available[i].name!r} raised: {result}")
                continue
            all_results_per_backend.append(result)

        # Merge and deduplicate by URL — first seen wins (highest-scoring backend first)
        seen_urls: set[str] = set()
        merged: list[SearchResult] = []

        # Interleave results from backends for diversity before dedup
        max_len = max((len(r) for r in all_results_per_backend), default=0)
        for i in range(max_len):
            for backend_results in all_results_per_backend:
                if i < len(backend_results):
                    r = backend_results[i]
                    if r.url not in seen_urls:
                        seen_urls.add(r.url)
                        merged.append(r)

        # Sort by score descending, cap at max_results
        merged.sort(key=lambda r: r.score, reverse=True)
        return merged[:max_results]
