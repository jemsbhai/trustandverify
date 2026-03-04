"""MongoStorage — storage backend using Motor (async MongoDB driver)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from trustandverify.core.models import Claim, Report, ReportSummary
from trustandverify.storage.sqlite import _dict_to_report, _report_to_dict


class MongoStorage:
    """Storage backend using MongoDB via Motor.

    Install with: pip install trustandverify[mongo]

    Args:
        uri:      MongoDB connection URI. Falls back to ``MONGO_URI`` env var.
        database: Database name (default: ``trustandverify``).
    """

    name = "mongo"

    def __init__(
        self,
        uri: str | None = None,
        database: str = "trustandverify",
    ) -> None:
        import os
        self._uri = uri or os.environ.get("MONGO_URI", "mongodb://localhost:27017")
        self._database = database
        self._client = None

    def _get_collection(self):
        if self._client is None:
            try:
                import motor.motor_asyncio as motor  # type: ignore[import]
            except ImportError as e:
                raise ImportError(
                    "MongoStorage requires motor. "
                    "Install with: pip install trustandverify[mongo]"
                ) from e
            self._client = motor.AsyncIOMotorClient(self._uri)
        return self._client[self._database]["reports"]

    def is_available(self) -> bool:
        return bool(self._uri)

    async def save_report(self, report: Report) -> str:
        col = self._get_collection()
        data = _report_to_dict(report)
        await col.replace_one({"_id": report.id}, {"_id": report.id, **data}, upsert=True)
        return report.id

    async def get_report(self, report_id: str) -> Report | None:
        col = self._get_collection()
        doc = await col.find_one({"_id": report_id})
        if not doc:
            return None
        doc.pop("_id", None)
        return _dict_to_report(doc)

    async def list_reports(self, limit: int = 50) -> list[ReportSummary]:
        col = self._get_collection()
        cursor = col.find({}, {"_id": 1, "query": 1, "created_at": 1, "claims": 1})
        cursor = cursor.sort("created_at", -1).limit(limit)
        summaries = []
        async for doc in cursor:
            summaries.append(ReportSummary(
                id=doc["_id"],
                query=doc.get("query", ""),
                created_at=datetime.fromisoformat(doc.get("created_at", datetime.now(timezone.utc).isoformat())),
                num_claims=len(doc.get("claims", [])),
            ))
        return summaries

    async def save_claim(self, claim: Claim) -> str:
        return claim.text

    async def get_claims_for_query(self, query_id: str) -> list[Claim]:
        return []
