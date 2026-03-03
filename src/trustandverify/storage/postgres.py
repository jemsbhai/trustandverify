"""PostgresStorage — storage backend using asyncpg."""

from __future__ import annotations

from datetime import datetime
import json

from trustandverify.core.models import Claim, Report, ReportSummary
from trustandverify.storage.sqlite import _dict_to_report, _report_to_dict

_CREATE_REPORTS = """
CREATE TABLE IF NOT EXISTS tv_reports (
    id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    summary TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    data JSONB NOT NULL
)
"""

_CREATE_CLAIMS = """
CREATE TABLE IF NOT EXISTS tv_claims (
    id SERIAL PRIMARY KEY,
    query_id TEXT NOT NULL,
    text TEXT NOT NULL,
    verdict TEXT NOT NULL,
    data JSONB NOT NULL
)
"""


class PostgresStorage:
    """Storage backend using PostgreSQL via asyncpg.

    Install with: pip install trustandverify[postgres]

    Args:
        dsn: asyncpg-compatible connection string, e.g.
             ``postgresql://user:pass@localhost/dbname``
             Falls back to ``POSTGRES_DSN`` env var.
    """

    name = "postgres"

    def __init__(self, dsn: str | None = None) -> None:
        import os
        self._dsn = dsn or os.environ.get("POSTGRES_DSN", "")
        self._pool = None

    async def _get_pool(self):
        if self._pool is None:
            try:
                import asyncpg  # type: ignore[import]
            except ImportError as e:
                raise ImportError(
                    "PostgresStorage requires asyncpg. "
                    "Install with: pip install trustandverify[postgres]"
                ) from e
            self._pool = await asyncpg.create_pool(self._dsn)
            async with self._pool.acquire() as conn:
                await conn.execute(_CREATE_REPORTS)
                await conn.execute(_CREATE_CLAIMS)
        return self._pool

    def is_available(self) -> bool:
        return bool(self._dsn)

    async def save_report(self, report: Report) -> str:
        pool = await self._get_pool()
        data = json.dumps(_report_to_dict(report))
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO tv_reports (id, query, summary, created_at, data) "
                "VALUES ($1, $2, $3, $4, $5::jsonb) "
                "ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data",
                report.id, report.query, report.summary, report.created_at, data,
            )
        return report.id

    async def get_report(self, report_id: str) -> Report | None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM tv_reports WHERE id = $1", report_id
            )
        if not row:
            return None
        return _dict_to_report(json.loads(row["data"]))

    async def list_reports(self, limit: int = 50) -> list[ReportSummary]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, query, created_at, data FROM tv_reports "
                "ORDER BY created_at DESC LIMIT $1", limit
            )
        summaries = []
        for r in rows:
            data = json.loads(r["data"])
            summaries.append(ReportSummary(
                id=r["id"], query=r["query"], created_at=r["created_at"],
                num_claims=len(data.get("claims", [])),
            ))
        return summaries

    async def save_claim(self, claim: Claim) -> str:
        return claim.text

    async def get_claims_for_query(self, query_id: str) -> list[Claim]:
        return []
