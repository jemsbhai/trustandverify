"""File-based cache backend — JSON files on disk with TTL support."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


class FileCache:
    """Cache backend that stores values as JSON files in a local directory.

    Structure: ``{cache_dir}/{sha256(key)}.json``
    Each file contains ``{"value": ..., "expires_at": float | null}``.

    This is the default cache for development and demos — no extra
    dependencies required beyond the stdlib.
    """

    def __init__(
        self,
        cache_dir: str = ".trustandverify_cache",
        default_ttl: int = 3600,
    ) -> None:
        self._dir = Path(cache_dir)
        self._default_ttl = default_ttl
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self._dir / f"{digest}.json"

    async def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            expires_at = data.get("expires_at")
            if expires_at is not None and time.time() > expires_at:
                path.unlink(missing_ok=True)
                return None
            return data.get("value")
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expires_at = time.time() + effective_ttl if effective_ttl > 0 else None
        payload = {"value": value, "expires_at": expires_at}
        try:
            self._path(key).write_text(
                json.dumps(payload, default=str), encoding="utf-8"
            )
        except OSError as e:
            print(f"[FileCache] Could not write cache entry: {e}")

    async def invalidate(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)
