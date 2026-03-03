"""CacheBackend protocol — caching layer for search results and LLM responses."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CacheBackend(Protocol):
    """Structural protocol for cache providers.

    The cache sits between the agent and the search/LLM backends.
    Same query + same backend = cached result. Configurable TTL.
    Saves API quota and speeds up repeated demos.
    """

    async def get(self, key: str) -> Any | None:
        """Retrieve a cached value, or None if missing/expired.

        Args:
            key: Cache key.

        Returns:
            The cached value, or None.
        """
        ...

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store a value in the cache.

        Args:
            key:   Cache key.
            value: Value to store (must be JSON-serialisable for
                   file/Redis backends).
            ttl:   Time-to-live in seconds.  None = use backend default.
        """
        ...

    async def invalidate(self, key: str) -> None:
        """Remove a specific key from the cache."""
        ...
