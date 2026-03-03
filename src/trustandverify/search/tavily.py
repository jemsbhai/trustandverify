"""TavilySearch — async httpx port of velrichack tools/search.py."""

from __future__ import annotations

import os

import httpx

from trustandverify.core.models import SearchResult

TAVILY_URL = "https://api.tavily.com/search"


class TavilySearch:
    """Search backend using the Tavily API.

    Requires the ``TAVILY_API_KEY`` environment variable.
    Free tier: 1,000 searches/month.
    """

    name = "tavily"

    def __init__(self, api_key: str | None = None, timeout: float = 15.0) -> None:
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        self._timeout = timeout

    def is_available(self) -> bool:
        return bool(self._api_key)

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Search via Tavily and return structured SearchResult objects.

        Returns an empty list (rather than raising) on any network or
        API error, so the pipeline degrades gracefully.
        """
        if not self._api_key:
            return []

        payload = {
            "query": query,
            "max_results": max_results,
            "include_answer": False,
            "search_depth": "basic",
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(TAVILY_URL, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            return [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    content=r.get("content", ""),
                    score=float(r.get("score", 0.0)),
                )
                for r in data.get("results", [])
            ]

        except httpx.HTTPStatusError as e:
            print(f"[TavilySearch] HTTP error {e.response.status_code}: {e}")
            return []
        except httpx.RequestError as e:
            print(f"[TavilySearch] Request error: {e}")
            return []
        except Exception as e:  # noqa: BLE001
            print(f"[TavilySearch] Unexpected error: {e}")
            return []
