"""BingSearch — Bing Web Search API using httpx."""

from __future__ import annotations

import os

import httpx

from trustandverify.core.models import SearchResult

BING_URL = "https://api.bing.microsoft.com/v7.0/search"


class BingSearch:
    """Search backend using the Bing Web Search API.

    Requires the ``BING_API_KEY`` environment variable.
    """

    name = "bing"

    def __init__(self, api_key: str | None = None, timeout: float = 15.0) -> None:
        self._api_key = api_key or os.environ.get("BING_API_KEY", "")
        self._timeout = timeout

    def is_available(self) -> bool:
        return bool(self._api_key)

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        if not self._api_key:
            return []

        headers = {"Ocp-Apim-Subscription-Key": self._api_key}
        params = {"q": query, "count": min(max_results, 50), "responseFilter": "Webpages"}

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(BING_URL, headers=headers, params=params)
                resp.raise_for_status()
                data = resp.json()

            results = []
            for r in data.get("webPages", {}).get("value", [])[:max_results]:
                results.append(SearchResult(
                    title=r.get("name", ""),
                    url=r.get("url", ""),
                    content=r.get("snippet", ""),
                    score=0.7,
                ))
            return results

        except httpx.HTTPStatusError as e:
            print(f"[BingSearch] HTTP error {e.response.status_code}: {e}")
            return []
        except Exception as e:  # noqa: BLE001
            print(f"[BingSearch] Error: {e}")
            return []
