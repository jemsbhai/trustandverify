"""Neo4jStorage — storage backend using the neo4j driver."""

from __future__ import annotations

import json
from datetime import datetime

from trustandverify.core.models import Claim, Report, ReportSummary
from trustandverify.storage.sqlite import (
    _claim_to_dict,
    _dict_to_claim,
    _dict_to_report,
    _report_to_dict,
)


class Neo4jStorage:
    """Storage backend using Neo4j graph database.

    Install with: pip install trustandverify[neo4j]

    Args:
        uri:      Bolt URI, e.g. ``bolt://localhost:7687``
        username: Neo4j username (default: ``neo4j``)
        password: Neo4j password. Falls back to ``NEO4J_PASSWORD`` env var.
    """

    name = "neo4j"

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str | None = None,
    ) -> None:
        import os

        self._uri = uri
        self._username = username
        self._password = password or os.environ.get("NEO4J_PASSWORD", "")
        self._driver = None

    def _get_driver(self):
        if self._driver is None:
            try:
                from neo4j import AsyncGraphDatabase  # type: ignore[import]
            except ImportError as e:
                raise ImportError(
                    "Neo4jStorage requires neo4j. Install with: pip install trustandverify[neo4j]"
                ) from e
            self._driver = AsyncGraphDatabase.driver(
                self._uri, auth=(self._username, self._password)
            )
        return self._driver

    def is_available(self) -> bool:
        return bool(self._password)

    async def save_report(self, report: Report) -> str:
        driver = self._get_driver()
        data = json.dumps(_report_to_dict(report))
        async with driver.session() as session:
            await session.run(
                "MERGE (r:Report {id: $id}) "
                "SET r.query = $query, r.summary = $summary, "
                "r.created_at = $created_at, r.data = $data",
                id=report.id,
                query=report.query,
                summary=report.summary,
                created_at=report.created_at.isoformat(),
                data=data,
            )
        return report.id

    async def get_report(self, report_id: str) -> Report | None:
        driver = self._get_driver()
        async with driver.session() as session:
            result = await session.run(
                "MATCH (r:Report {id: $id}) RETURN r.data AS data", id=report_id
            )
            record = await result.single()
        if not record:
            return None
        return _dict_to_report(json.loads(record["data"]))

    async def list_reports(self, limit: int = 50) -> list[ReportSummary]:
        driver = self._get_driver()
        async with driver.session() as session:
            result = await session.run(
                "MATCH (r:Report) RETURN r.id AS id, r.query AS query, "
                "r.created_at AS created_at, r.data AS data "
                "ORDER BY r.created_at DESC LIMIT $limit",
                limit=limit,
            )
            records = await result.data()
        summaries = []
        for r in records:
            data = json.loads(r["data"])
            summaries.append(
                ReportSummary(
                    id=r["id"],
                    query=r["query"],
                    created_at=datetime.fromisoformat(r["created_at"]),
                    num_claims=len(data.get("claims", [])),
                )
            )
        return summaries

    async def save_claim(self, claim: Claim, query_id: str) -> str:
        driver = self._get_driver()
        data = json.dumps(_claim_to_dict(claim))
        async with driver.session() as session:
            await session.run(
                "CREATE (c:Claim {query_id: $query_id, text: $text, "
                "verdict: $verdict, data: $data})",
                query_id=query_id,
                text=claim.text,
                verdict=claim.verdict.value,
                data=data,
            )
        return claim.text

    async def get_claims_for_query(self, query_id: str) -> list[Claim]:
        driver = self._get_driver()
        async with driver.session() as session:
            result = await session.run(
                "MATCH (c:Claim {query_id: $query_id}) RETURN c.data AS data ORDER BY id(c)",
                query_id=query_id,
            )
            records = await result.data()
        return [_dict_to_claim(json.loads(r["data"])) for r in records]
