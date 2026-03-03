"""BraveSearch — Brave Search API using httpx."""

from __future__ import annotations

import os

import httpx

from trustandverify.core.models import SearchResult

BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"


class BraveSearch:
    """Search backend using the Brave Search API.

    Requires the ``BRAVE_API_KEY`` environment variable.
    Free tier: 2,000 queries/month.
    """

    name = "brave"

    def __init__(self, api_key: str | None = None, timeout: float = 15.0) -> None:
        self._api_key = api_key or os.environ.get("BRAVE_API_KEY", "")
        self._timeout = timeout

    def is_available(self) -> bool:
        return bool(self._api_key)

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        if not self._api_key:
            return []

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self._api_key,
        }
        params = {"q": query, "count": min(max_results, 20)}

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(BRAVE_URL, headers=headers, params=params)
                resp.raise_for_status()
                data = resp.json()

            results = []
            for r in data.get("web", {}).get("results", [])[:max_results]:
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    content=r.get("description", ""),
                    score=float(r.get("score", 0.7)),
                ))
            return results

        except httpx.HTTPStatusError as e:
            print(f"[BraveSearch] HTTP error {e.response.status_code}: {e}")
            return []
        except Exception as e:  # noqa: BLE001
            print(f"[BraveSearch] Error: {e}")
            return []
