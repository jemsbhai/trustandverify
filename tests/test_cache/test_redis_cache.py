"""Tests for cache/redis_cache.py — Redis-backed cache with TTL."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trustandverify.cache.protocol import CacheBackend


class TestRedisCacheBackend:
    def _make_cache(self, mock_client=None):
        from trustandverify.cache.redis_cache import RedisCache

        cache = RedisCache(url="redis://localhost:6379", default_ttl=3600)
        if mock_client is not None:
            cache._client = mock_client
        return cache

    def _mock_client(self, store: dict | None = None):
        """Build an AsyncMock redis client backed by an optional dict."""
        if store is None:
            store = {}
        client = AsyncMock()
        client.get = AsyncMock(side_effect=lambda k: store.get(k))
        client.set = AsyncMock(side_effect=lambda k, v, **kw: store.update({k: v}))
        client.delete = AsyncMock(side_effect=lambda k: store.pop(k, None))
        return client, store

    # ── Protocol conformance ──────────────────────────────────────

    def test_satisfies_cache_protocol(self):
        from trustandverify.cache.redis_cache import RedisCache

        cache = RedisCache(url="redis://localhost:6379")
        assert isinstance(cache, CacheBackend)

    # ── get / set basics ──────────────────────────────────────────

    async def test_get_returns_none_for_missing_key(self):
        client, _ = self._mock_client()
        cache = self._make_cache(client)
        assert await cache.get("nonexistent") is None

    async def test_set_and_get_dict(self):
        client, store = self._mock_client()
        cache = self._make_cache(client)

        await cache.set("key1", {"data": "hello"})
        result = await cache.get("key1")
        assert result == {"data": "hello"}

    async def test_set_and_get_string(self):
        client, store = self._mock_client()
        cache = self._make_cache(client)

        await cache.set("key2", "simple string")
        assert await cache.get("key2") == "simple string"

    async def test_set_and_get_list(self):
        client, store = self._mock_client()
        cache = self._make_cache(client)

        await cache.set("key3", [1, 2, 3])
        assert await cache.get("key3") == [1, 2, 3]

    # ── TTL ───────────────────────────────────────────────────────

    async def test_set_passes_default_ttl(self):
        client, _ = self._mock_client()
        cache = self._make_cache(client)

        await cache.set("k", "v")
        # Should have called redis set with ex=3600 (the default_ttl)
        call_kwargs = client.set.call_args
        assert call_kwargs.kwargs.get("ex") == 3600 or call_kwargs[1].get("ex") == 3600

    async def test_set_passes_custom_ttl(self):
        client, _ = self._mock_client()
        cache = self._make_cache(client)

        await cache.set("k", "v", ttl=60)
        call_kwargs = client.set.call_args
        assert call_kwargs.kwargs.get("ex") == 60 or call_kwargs[1].get("ex") == 60

    # ── invalidate ────────────────────────────────────────────────

    async def test_invalidate(self):
        client, store = self._mock_client()
        cache = self._make_cache(client)

        await cache.set("to_delete", "value")
        await cache.invalidate("to_delete")
        client.delete.assert_called_once_with("tv:cache:to_delete")

    async def test_invalidate_nonexistent_key(self):
        client, _ = self._mock_client()
        cache = self._make_cache(client)
        await cache.invalidate("never_existed")  # should not raise

    # ── Key prefixing ─────────────────────────────────────────────

    async def test_keys_are_prefixed(self):
        client, _ = self._mock_client()
        cache = self._make_cache(client)

        await cache.set("mykey", "val")
        set_key = client.set.call_args[0][0]
        assert set_key.startswith("tv:cache:")

    # ── get_client / ImportError ──────────────────────────────────

    async def test_get_client_raises_without_redis(self):
        from trustandverify.cache.redis_cache import RedisCache

        with patch.dict("sys.modules", {"redis": None, "redis.asyncio": None}):
            cache = RedisCache(url="redis://localhost:6379")
            with pytest.raises(ImportError, match="redis"):
                await cache._get_client()

    # ── JSON round-trip edge cases ────────────────────────────────

    async def test_get_returns_none_on_non_json(self):
        """If redis returns something that isn't valid JSON, return None."""
        client, _ = self._mock_client()
        client.get = AsyncMock(return_value="not-valid-json{{{")
        cache = self._make_cache(client)
        assert await cache.get("bad") is None
