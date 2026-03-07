"""Redis-backed cache backend — async redis-py with TTL support.

Install with: pip install trustandverify[redis]
"""

from __future__ import annotations

import json
from typing import Any

_KEY_PREFIX = "tv:cache:"


class RedisCache:
    """Cache backend using Redis via redis-py (async client).

    Values are JSON-serialised and stored with a configurable TTL.
    Keys are prefixed with ``tv:cache:`` to avoid collisions with
    RedisStorage (which uses ``tv:report:``).

    Args:
        url:         Redis connection URL. Falls back to ``REDIS_URL``
                     env var, then ``redis://localhost:6379``.
        default_ttl: Default TTL in seconds (0 = no expiry).
    """

    def __init__(
        self,
        url: str | None = None,
        default_ttl: int = 3600,
    ) -> None:
        import os

        self._url = url or os.environ.get("REDIS_URL", "redis://localhost:6379")
        self._default_ttl = default_ttl
        self._client = None

    async def _get_client(self):
        if self._client is None:
            try:
                from redis.asyncio import from_url  # type: ignore[import]
            except ImportError as e:
                raise ImportError(
                    "RedisCache requires redis. Install with: pip install trustandverify[redis]"
                ) from e
            self._client = await from_url(self._url, decode_responses=True)
        return self._client

    def _key(self, key: str) -> str:
        return f"{_KEY_PREFIX}{key}"

    async def get(self, key: str) -> Any | None:
        """Retrieve a cached value, or None if missing/expired."""
        client = await self._get_client()
        raw = await client.get(self._key(key))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store a value in the cache with optional TTL."""
        client = await self._get_client()
        effective_ttl = ttl if ttl is not None else self._default_ttl
        payload = json.dumps(value, default=str)
        kwargs: dict[str, Any] = {}
        if effective_ttl > 0:
            kwargs["ex"] = effective_ttl
        await client.set(self._key(key), payload, **kwargs)

    async def invalidate(self, key: str) -> None:
        """Remove a specific key from the cache."""
        client = await self._get_client()
        await client.delete(self._key(key))
