"""RedisStorage — storage backend using redis-py (async)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from trustandverify.core.models import Claim, Report, ReportSummary
from trustandverify.storage.sqlite import _dict_to_report, _report_to_dict

_REPORT_KEY = "tv:report:{id}"
_INDEX_KEY = "tv:reports:index"          # Sorted set: score=timestamp, member=id


class RedisStorage:
    """Storage backend using Redis via redis-py (async client).

    Install with: pip install trustandverify[redis]

    Args:
        url: Redis connection URL. Falls back to ``REDIS_URL`` env var,
             then ``redis://localhost:6379``.
        ttl: Optional TTL in seconds for report keys. None = no expiry.
    """

    name = "redis"

    def __init__(self, url: str | None = None, ttl: int | None = None) -> None:
        import os
        self._url = url or os.environ.get("REDIS_URL", "redis://localhost:6379")
        self._ttl = ttl
        self._client = None

    async def _get_client(self):
        if self._client is None:
            try:
                from redis.asyncio import from_url  # type: ignore[import]
            except ImportError as e:
                raise ImportError(
                    "RedisStorage requires redis. "
                    "Install with: pip install trustandverify[redis]"
                ) from e
            self._client = await from_url(self._url, decode_responses=True)
        return self._client

    def is_available(self) -> bool:
        return bool(self._url)

    async def save_report(self, report: Report) -> str:
        client = await self._get_client()
        key = _REPORT_KEY.format(id=report.id)
        data = json.dumps(_report_to_dict(report))

        await client.set(key, data, ex=self._ttl)

        # Track in sorted set by timestamp for list_reports ordering
        score = report.created_at.timestamp()
        await client.zadd(_INDEX_KEY, {report.id: score})

        return report.id

    async def get_report(self, report_id: str) -> Report | None:
        client = await self._get_client()
        raw = await client.get(_REPORT_KEY.format(id=report_id))
        if not raw:
            return None
        return _dict_to_report(json.loads(raw))

    async def list_reports(self, limit: int = 50) -> list[ReportSummary]:
        client = await self._get_client()
        # Newest first: reverse range by score
        ids = await client.zrevrange(_INDEX_KEY, 0, limit - 1)
        summaries = []
        for report_id in ids:
            raw = await client.get(_REPORT_KEY.format(id=report_id))
            if not raw:
                continue
            data = json.loads(raw)
            summaries.append(ReportSummary(
                id=data["id"],
                query=data.get("query", ""),
                created_at=datetime.fromisoformat(data.get("created_at", datetime.now(timezone.utc).isoformat())),
                num_claims=len(data.get("claims", [])),
            ))
        return summaries

    async def save_claim(self, claim: Claim) -> str:
        return claim.text

    async def get_claims_for_query(self, query_id: str) -> list[Claim]:
        return []
