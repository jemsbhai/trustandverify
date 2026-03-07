"""Tests for cache/file_cache.py — JSON file cache with TTL."""

from __future__ import annotations

import json
import time

import pytest

from trustandverify.cache.file_cache import FileCache


@pytest.fixture
def cache(tmp_path):
    return FileCache(cache_dir=str(tmp_path / "cache"), default_ttl=3600)


class TestFileCache:
    async def test_get_returns_none_for_missing_key(self, cache):
        assert await cache.get("nonexistent") is None

    async def test_set_and_get(self, cache):
        await cache.set("key1", {"data": "hello"})
        result = await cache.get("key1")
        assert result == {"data": "hello"}

    async def test_set_string_value(self, cache):
        await cache.set("key2", "simple string")
        assert await cache.get("key2") == "simple string"

    async def test_set_list_value(self, cache):
        await cache.set("key3", [1, 2, 3])
        assert await cache.get("key3") == [1, 2, 3]

    async def test_invalidate(self, cache):
        await cache.set("to_delete", "value")
        assert await cache.get("to_delete") == "value"
        await cache.invalidate("to_delete")
        assert await cache.get("to_delete") is None

    async def test_invalidate_nonexistent_key(self, cache):
        """Should not raise on missing key."""
        await cache.invalidate("never_existed")

    async def test_ttl_expired(self, tmp_path):
        cache = FileCache(cache_dir=str(tmp_path / "cache"), default_ttl=1)
        await cache.set("expires", "soon", ttl=1)
        # Manually backdate the expiry
        path = cache._path("expires")
        data = json.loads(path.read_text())
        data["expires_at"] = time.time() - 10  # expired 10s ago
        path.write_text(json.dumps(data))
        assert await cache.get("expires") is None

    async def test_ttl_not_expired(self, cache):
        await cache.set("fresh", "data", ttl=9999)
        assert await cache.get("fresh") == "data"

    async def test_custom_ttl_overrides_default(self, tmp_path):
        cache = FileCache(cache_dir=str(tmp_path / "cache"), default_ttl=3600)
        await cache.set("custom", "val", ttl=1)
        path = cache._path("custom")
        data = json.loads(path.read_text())
        assert data["expires_at"] is not None
        # Should expire within ~1 second of now, not 3600
        assert data["expires_at"] < time.time() + 5

    async def test_corrupted_json_returns_none(self, cache):
        await cache.set("corrupt", "val")
        path = cache._path("corrupt")
        path.write_text("this is not json{{{", encoding="utf-8")
        assert await cache.get("corrupt") is None

    async def test_cache_dir_created(self, tmp_path):
        dir_path = tmp_path / "deep" / "nested" / "cache"
        FileCache(cache_dir=str(dir_path))
        assert dir_path.exists()

    async def test_set_write_error_does_not_raise(self, tmp_path, capsys):
        """If the cache dir becomes unwritable, set should print warning, not raise."""
        cache = FileCache(cache_dir=str(tmp_path / "cache"), default_ttl=3600)
        # Override _path to return an invalid path

        def bad_path(key):
            from pathlib import Path

            return Path("/nonexistent/dir/that/does/not/exist/file.json")

        cache._path = bad_path
        await cache.set("fail", "data")  # Should not raise
        captured = capsys.readouterr()
        assert "Could not write" in captured.out
