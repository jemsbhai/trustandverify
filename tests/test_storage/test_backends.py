"""Tests for all storage backends.

Unit tests mock the underlying driver and run in normal pytest.
Integration tests hit real infrastructure and are marked @pytest.mark.integration
— skipped by default, run with: pytest -m integration
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from jsonld_ex.confidence_algebra import Opinion

from trustandverify.core.models import Claim, Conflict, Evidence, Report, Source, Verdict
from trustandverify.storage.memory import InMemoryStorage
from trustandverify.storage.sqlite import SQLiteStorage

# ── Shared fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def sample_report() -> Report:
    source = Source(
        url="https://nber.org/w1",
        title="NBER Study",
        content_snippet="Remote workers 13% more productive.",
        trust_score=0.85,
    )
    evidence = Evidence(
        text="Remote workers 13% more productive.",
        supports_claim=True,
        relevance=0.9,
        confidence_raw=0.8,
        source=source,
        opinion=Opinion(belief=0.567, disbelief=0.1, uncertainty=0.333, base_rate=0.5),
    )
    claim = Claim(
        text="Remote workers are more productive.",
        evidence=[evidence],
        opinion=Opinion(belief=0.733, disbelief=0.1, uncertainty=0.167, base_rate=0.5),
        verdict=Verdict.SUPPORTED,
        assessment="Evidence supports the claim.",
    )
    conflict = Conflict(
        claim_text="Remote workers are more productive",
        conflict_degree=0.25,
        num_supporting=2,
        num_contradicting=1,
    )
    return Report(
        id=str(uuid.uuid4()),
        query="Is remote work more productive?",
        claims=[claim],
        conflicts=[conflict],
        summary="Remote work appears to increase productivity.",
        created_at=datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc),
    )


# ── Shared contract tests (run against any backend) ───────────────────────────


async def assert_storage_contract(storage, report: Report) -> None:
    """Assert that a storage backend satisfies the full CRUD contract."""
    # Save
    report_id = await storage.save_report(report)
    assert report_id == report.id

    # Get
    retrieved = await storage.get_report(report.id)
    assert retrieved is not None
    assert retrieved.id == report.id
    assert retrieved.query == report.query
    assert retrieved.summary == report.summary
    assert len(retrieved.claims) == len(report.claims)

    # Claims round-trip (via report)
    original_claim = report.claims[0]
    retrieved_claim = retrieved.claims[0]
    assert retrieved_claim.text == original_claim.text
    assert retrieved_claim.verdict == original_claim.verdict

    # Opinion round-trip
    assert retrieved_claim.opinion is not None
    assert abs(retrieved_claim.opinion.belief - original_claim.opinion.belief) < 1e-4
    assert abs(retrieved_claim.opinion.disbelief - original_claim.opinion.disbelief) < 1e-4
    assert abs(retrieved_claim.opinion.uncertainty - original_claim.opinion.uncertainty) < 1e-4

    # Get non-existent
    missing = await storage.get_report("does-not-exist")
    assert missing is None

    # List
    summaries = await storage.list_reports(limit=10)
    ids = [s.id for s in summaries]
    assert report.id in ids

    # ── save_claim / get_claims_for_query ──
    for claim in report.claims:
        await storage.save_claim(claim, report.id)

    claims = await storage.get_claims_for_query(report.id)
    assert len(claims) == len(report.claims)
    assert claims[0].text == report.claims[0].text
    assert claims[0].verdict == report.claims[0].verdict

    # Non-existent query_id returns empty
    assert await storage.get_claims_for_query("no-such-query") == []


# ── InMemoryStorage ────────────────────────────────────────────────────────────


class TestInMemoryStorage:
    async def test_full_contract(self, sample_report):
        await assert_storage_contract(InMemoryStorage(), sample_report)

    async def test_list_returns_newest_first(self):
        storage = InMemoryStorage()
        r1 = Report(
            id="r1",
            query="q1",
            claims=[],
            conflicts=[],
            summary="s1",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        r2 = Report(
            id="r2",
            query="q2",
            claims=[],
            conflicts=[],
            summary="s2",
            created_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
        )
        await storage.save_report(r1)
        await storage.save_report(r2)
        summaries = await storage.list_reports()
        assert summaries[0].id == "r2"

    async def test_overwrite_same_id(self, sample_report):
        storage = InMemoryStorage()
        await storage.save_report(sample_report)
        sample_report.summary = "Updated summary"
        await storage.save_report(sample_report)
        retrieved = await storage.get_report(sample_report.id)
        assert retrieved.summary == "Updated summary"

    async def test_list_limit(self):
        storage = InMemoryStorage()
        for i in range(5):
            r = Report(
                id=f"r{i}",
                query=f"q{i}",
                claims=[],
                conflicts=[],
                summary="s",
                created_at=datetime(2026, 1, i + 1, tzinfo=timezone.utc),
            )
            await storage.save_report(r)
        summaries = await storage.list_reports(limit=3)
        assert len(summaries) == 3


# ── SQLiteStorage ──────────────────────────────────────────────────────────────


class TestSQLiteStorage:
    async def test_full_contract(self, sample_report):
        storage = SQLiteStorage(":memory:")
        await assert_storage_contract(storage, sample_report)

    async def test_persists_across_reconnect(self, tmp_path, sample_report):
        db_path = str(tmp_path / "test.db")
        storage1 = SQLiteStorage(db_path)
        await storage1.save_report(sample_report)

        storage2 = SQLiteStorage(db_path)
        retrieved = await storage2.get_report(sample_report.id)
        assert retrieved is not None
        assert retrieved.query == sample_report.query

    async def test_list_returns_newest_first(self, tmp_path):
        storage = SQLiteStorage(":memory:")
        r1 = Report(
            id="r1",
            query="q1",
            claims=[],
            conflicts=[],
            summary="s",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        r2 = Report(
            id="r2",
            query="q2",
            claims=[],
            conflicts=[],
            summary="s",
            created_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
        )
        await storage.save_report(r1)
        await storage.save_report(r2)
        summaries = await storage.list_reports()
        assert summaries[0].id == "r2"

    async def test_conflicts_round_trip(self, sample_report):
        storage = SQLiteStorage(":memory:")
        await storage.save_report(sample_report)
        retrieved = await storage.get_report(sample_report.id)
        assert len(retrieved.conflicts) == 1
        assert retrieved.conflicts[0].conflict_degree == sample_report.conflicts[0].conflict_degree


# ── PostgresStorage (unit — mocked) ───────────────────────────────────────────


class TestPostgresStorageUnit:
    async def test_is_available_with_dsn(self):
        from trustandverify.storage.postgres import PostgresStorage

        s = PostgresStorage(dsn="postgresql://user:pass@localhost/db")
        assert s.is_available() is True

    async def test_is_available_without_dsn(self, monkeypatch):
        from trustandverify.storage.postgres import PostgresStorage

        monkeypatch.delenv("POSTGRES_DSN", raising=False)
        s = PostgresStorage(dsn="")
        assert s.is_available() is False


# ── Neo4jStorage (unit — mocked) ──────────────────────────────────────────────


class TestNeo4jStorageUnit:
    async def test_is_available_with_password(self):
        from trustandverify.storage.neo4j import Neo4jStorage

        s = Neo4jStorage(password="secret")
        assert s.is_available() is True

    async def test_is_available_without_password(self, monkeypatch):
        from trustandverify.storage.neo4j import Neo4jStorage

        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
        s = Neo4jStorage(password="")
        assert s.is_available() is False


# ── MongoStorage (unit — mocked) ──────────────────────────────────────────────


class TestMongoStorageUnit:
    async def test_is_available(self):
        from trustandverify.storage.mongo import MongoStorage

        s = MongoStorage(uri="mongodb://localhost:27017")
        assert s.is_available() is True

    async def test_save_and_get_mocked(self, sample_report):
        from trustandverify.storage.mongo import MongoStorage

        mock_col = AsyncMock()
        mock_col.replace_one = AsyncMock()
        mock_col.find_one = AsyncMock(
            return_value={
                **{"_id": sample_report.id},
                **_report_to_minimal_dict(sample_report),
            }
        )

        s = MongoStorage()
        s._get_collection = MagicMock(return_value=mock_col)

        await s.save_report(sample_report)
        result = await s.get_report(sample_report.id)
        assert result is not None
        assert result.query == sample_report.query


# ── RedisStorage (unit — mocked) ──────────────────────────────────────────────


class TestRedisStorageUnit:
    async def test_is_available(self):
        from trustandverify.storage.redis import RedisStorage

        s = RedisStorage(url="redis://localhost:6379")
        assert s.is_available() is True

    async def test_save_and_get_mocked(self, sample_report):
        from trustandverify.storage.redis import RedisStorage

        stored: dict = {}

        mock_client = AsyncMock()
        mock_client.set = AsyncMock(side_effect=lambda k, v, **kw: stored.update({k: v}))
        mock_client.zadd = AsyncMock()
        mock_client.get = AsyncMock(side_effect=lambda k: stored.get(k))

        s = RedisStorage()
        s._client = mock_client

        await s.save_report(sample_report)
        result = await s.get_report(sample_report.id)
        assert result is not None
        assert result.id == sample_report.id
        assert result.query == sample_report.query


# ── Integration tests (need real infrastructure) ───────────────────────────────


@pytest.mark.integration
class TestSQLiteIntegration:
    async def test_full_roundtrip_on_disk(self, tmp_path, sample_report):
        storage = SQLiteStorage(str(tmp_path / "integration.db"))
        await assert_storage_contract(storage, sample_report)


@pytest.mark.integration
class TestPostgresIntegration:
    async def test_full_roundtrip(self, sample_report):
        import os

        from trustandverify.storage.postgres import PostgresStorage

        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            pytest.skip("POSTGRES_DSN not set")
        storage = PostgresStorage(dsn=dsn)
        await assert_storage_contract(storage, sample_report)


@pytest.mark.integration
class TestNeo4jIntegration:
    async def test_full_roundtrip(self, sample_report):
        import os

        from trustandverify.storage.neo4j import Neo4jStorage

        password = os.environ.get("NEO4J_PASSWORD")
        if not password:
            pytest.skip("NEO4J_PASSWORD not set")
        storage = Neo4jStorage(password=password)
        await assert_storage_contract(storage, sample_report)


@pytest.mark.integration
class TestMongoIntegration:
    async def test_full_roundtrip(self, sample_report):
        import os

        from trustandverify.storage.mongo import MongoStorage

        uri = os.environ.get("MONGO_URI")
        if not uri:
            pytest.skip("MONGO_URI not set")
        storage = MongoStorage(uri=uri)
        await assert_storage_contract(storage, sample_report)


@pytest.mark.integration
class TestRedisIntegration:
    async def test_full_roundtrip(self, sample_report):
        import os

        from trustandverify.storage.redis import RedisStorage

        url = os.environ.get("REDIS_URL")
        if not url:
            pytest.skip("REDIS_URL not set")
        storage = RedisStorage(url=url)
        await assert_storage_contract(storage, sample_report)


# ── Helper ─────────────────────────────────────────────────────────────────────


def _report_to_minimal_dict(report: Report) -> dict:
    """Minimal dict for mocking MongoDB find_one responses."""
    from trustandverify.storage.sqlite import _report_to_dict

    return _report_to_dict(report)
