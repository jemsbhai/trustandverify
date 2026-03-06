"""SQLiteStorage — persistent storage using stdlib sqlite3 + asyncio.to_thread."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from functools import partial
from typing import Any

from trustandverify.core.models import Claim, Conflict, Evidence, Opinion, Report, ReportSummary, Source, Verdict

_CREATE_REPORTS = """
CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    summary TEXT NOT NULL,
    created_at TEXT NOT NULL,
    data TEXT NOT NULL
)
"""

_CREATE_CLAIMS = """
CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id TEXT NOT NULL,
    text TEXT NOT NULL,
    verdict TEXT NOT NULL,
    data TEXT NOT NULL
)
"""


class SQLiteStorage:
    """Storage backend using a local SQLite database.

    No extra dependencies — uses stdlib sqlite3 with asyncio.to_thread
    for non-blocking operation.

    Args:
        path: Path to the SQLite database file. Use ``:memory:`` for tests.
    """

    name = "sqlite"

    def __init__(self, path: str = "trustandverify.db") -> None:
        self._path = path
        # For :memory: databases, reuse a single connection — each new
        # sqlite3.connect(':memory:') call creates an entirely separate DB.
        self._persistent_conn: sqlite3.Connection | None = None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        if self._path == ":memory:":
            if self._persistent_conn is None:
                self._persistent_conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._persistent_conn.row_factory = sqlite3.Row
            return self._persistent_conn
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute(_CREATE_REPORTS)
        conn.execute(_CREATE_CLAIMS)
        conn.commit()

    # ── Report operations ──────────────────────────────────────────

    async def save_report(self, report: Report) -> str:
        data = _report_to_dict(report)

        def _save() -> None:
            conn = self._connect()
            conn.execute(
                "INSERT OR REPLACE INTO reports (id, query, summary, created_at, data) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    report.id,
                    report.query,
                    report.summary,
                    report.created_at.isoformat(),
                    json.dumps(data),
                ),
            )
            conn.commit()

        await asyncio.to_thread(_save)
        return report.id

    async def get_report(self, report_id: str) -> Report | None:
        def _get() -> dict | None:
            conn = self._connect()
            row = conn.execute(
                "SELECT data FROM reports WHERE id = ?", (report_id,)
            ).fetchone()
            return json.loads(row["data"]) if row else None

        data = await asyncio.to_thread(_get)
        return _dict_to_report(data) if data else None

    async def list_reports(self, limit: int = 50) -> list[ReportSummary]:
        def _list() -> list[dict]:
            conn = self._connect()
            rows = conn.execute(
                "SELECT id, query, created_at, data FROM reports "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

        rows = await asyncio.to_thread(_list)
        summaries = []
        for r in rows:
            data = json.loads(r["data"])
            summaries.append(ReportSummary(
                id=r["id"],
                query=r["query"],
                created_at=datetime.fromisoformat(r["created_at"]),
                num_claims=len(data.get("claims", [])),
            ))
        return summaries

    async def save_claim(self, claim: Claim, query_id: str) -> str:
        data = json.dumps(_claim_to_dict(claim))

        def _save() -> None:
            conn = self._connect()
            conn.execute(
                "INSERT INTO claims (query_id, text, verdict, data) VALUES (?, ?, ?, ?)",
                (query_id, claim.text, claim.verdict.value, data),
            )
            conn.commit()

        await asyncio.to_thread(_save)
        return claim.text

    async def get_claims_for_query(self, query_id: str) -> list[Claim]:
        def _get() -> list[dict]:
            conn = self._connect()
            rows = conn.execute(
                "SELECT data FROM claims WHERE query_id = ? ORDER BY id",
                (query_id,),
            ).fetchall()
            return [json.loads(row["data"]) for row in rows]

        rows = await asyncio.to_thread(_get)
        return [_dict_to_claim(d) for d in rows]


# ── Serialisation helpers ──────────────────────────────────────────────────────

def _report_to_dict(report: Report) -> dict:
    return {
        "id": report.id,
        "query": report.query,
        "summary": report.summary,
        "created_at": report.created_at.isoformat(),
        "claims": [_claim_to_dict(c) for c in report.claims],
        "conflicts": [
            {
                "claim_text": c.claim_text,
                "conflict_degree": c.conflict_degree,
                "num_supporting": c.num_supporting,
                "num_contradicting": c.num_contradicting,
            }
            for c in report.conflicts
        ],
        "metadata": report.metadata,
    }


def _claim_to_dict(claim: Claim) -> dict:
    return {
        "text": claim.text,
        "verdict": claim.verdict.value,
        "assessment": claim.assessment,
        "opinion": _opinion_to_dict(claim.opinion) if claim.opinion else None,
        "evidence": [
            {
                "text": e.text,
                "supports_claim": e.supports_claim,
                "relevance": e.relevance,
                "confidence_raw": e.confidence_raw,
                "source": {
                    "url": e.source.url,
                    "title": e.source.title,
                    "content_snippet": e.source.content_snippet,
                    "trust_score": e.source.trust_score,
                    "source_type": e.source.source_type,
                },
            }
            for e in claim.evidence
        ],
    }


def _opinion_to_dict(op: Opinion) -> dict:
    return {
        "belief": op.belief,
        "disbelief": op.disbelief,
        "uncertainty": op.uncertainty,
        "base_rate": op.base_rate,
    }


def _dict_to_report(data: dict) -> Report:
    claims = [_dict_to_claim(c) for c in data.get("claims", [])]
    conflicts = [
        Conflict(
            claim_text=c["claim_text"],
            conflict_degree=c["conflict_degree"],
            num_supporting=c["num_supporting"],
            num_contradicting=c["num_contradicting"],
        )
        for c in data.get("conflicts", [])
    ]
    return Report(
        id=data["id"],
        query=data["query"],
        summary=data["summary"],
        created_at=datetime.fromisoformat(data["created_at"]),
        claims=claims,
        conflicts=conflicts,
        metadata=data.get("metadata", {}),
    )


def _dict_to_claim(data: dict) -> Claim:
    opinion = None
    if data.get("opinion"):
        op = data["opinion"]
        opinion = Opinion(
            belief=op["belief"],
            disbelief=op["disbelief"],
            uncertainty=op["uncertainty"],
            base_rate=op.get("base_rate", 0.5),
        )
    evidence = []
    for e in data.get("evidence", []):
        s = e["source"]
        evidence.append(Evidence(
            text=e["text"],
            supports_claim=e["supports_claim"],
            relevance=e["relevance"],
            confidence_raw=e["confidence_raw"],
            source=Source(
                url=s["url"],
                title=s["title"],
                content_snippet=s["content_snippet"],
                trust_score=s["trust_score"],
                source_type=s.get("source_type", "web"),
            ),
            opinion=opinion,
        ))
    return Claim(
        text=data["text"],
        verdict=Verdict(data.get("verdict", "no_evidence")),
        assessment=data.get("assessment", ""),
        opinion=opinion,
        evidence=evidence,
    )
