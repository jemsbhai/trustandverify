"""SerpAPISearch — Google search via SerpAPI using httpx."""

from __future__ import annotations

import os

import httpx

from trustandverify.core.models import SearchResult

SERPAPI_URL = "https://serpapi.com/search"


class SerpAPISearch:
    """Search backend using SerpAPI (Google Search).

    Requires the ``SERPAPI_API_KEY`` environment variable.
    """

    name = "serpapi"

    def __init__(self, api_key: str | None = None, timeout: float = 15.0) -> None:
        self._api_key = api_key or os.environ.get("SERPAPI_API_KEY", "")
        self._timeout = timeout

    def is_available(self) -> bool:
        return bool(self._api_key)

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        if not self._api_key:
            return []

        params = {
            "q": query,
            "api_key": self._api_key,
            "num": max_results,
            "engine": "google",
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(SERPAPI_URL, params=params)
                resp.raise_for_status()
                data = resp.json()

            results = []
            for r in data.get("organic_results", [])[:max_results]:
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("link", ""),
                    content=r.get("snippet", ""),
                    score=0.7,  # SerpAPI doesn't provide relevance scores
                ))
            return results

        except httpx.HTTPStatusError as e:
            print(f"[SerpAPISearch] HTTP error {e.response.status_code}: {e}")
            return []
        except Exception as e:  # noqa: BLE001
            print(f"[SerpAPISearch] Error: {e}")
            return []
