"""Tests for storage backends — Postgres, Neo4j, Mongo, Redis (all mocked)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from jsonld_ex.confidence_algebra import Opinion

from trustandverify.core.models import Claim, Evidence, Report, Source, Verdict
from trustandverify.storage.sqlite import _report_to_dict

# ── Shared fixtures ───────────────────────────────────────────────────────────


def _make_report() -> Report:
    return Report(
        id="test-report-1",
        query="Is coffee healthy?",
        claims=[
            Claim(
                text="Coffee has antioxidants",
                verdict=Verdict.SUPPORTED,
                assessment="Well supported by evidence.",
                opinion=Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5),
                evidence=[
                    Evidence(
                        text="Rich in antioxidants",
                        supports_claim=True,
                        relevance=0.9,
                        confidence_raw=0.85,
                        source=Source(
                            url="https://example.com",
                            title="Health Study",
                            content_snippet="Coffee contains...",
                            trust_score=0.8,
                        ),
                    ),
                ],
            ),
        ],
        conflicts=[],
        summary="Coffee appears to be healthy.",
        created_at=datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
    )


# ── PostgresStorage ──────────────────────────────────────────────────────────


class TestPostgresStorage:
    def test_is_available_with_dsn(self):
        from trustandverify.storage.postgres import PostgresStorage

        storage = PostgresStorage(dsn="postgresql://localhost/test")
        assert storage.is_available() is True

    def test_is_available_without_dsn(self, monkeypatch):
        monkeypatch.delenv("POSTGRES_DSN", raising=False)
        from trustandverify.storage.postgres import PostgresStorage

        storage = PostgresStorage(dsn="")
        assert storage.is_available() is False

    async def test_save_and_get_report(self):
        from trustandverify.storage.postgres import PostgresStorage

        report = _make_report()
        report_data = json.dumps(_report_to_dict(report))

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={"data": report_data})

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_ctx)

        mock_asyncpg = MagicMock()
        mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)

        with patch.dict("sys.modules", {"asyncpg": mock_asyncpg}):
            storage = PostgresStorage(dsn="postgresql://localhost/test")
            storage._pool = mock_pool

            result_id = await storage.save_report(report)
            assert result_id == "test-report-1"

            retrieved = await storage.get_report("test-report-1")
            assert retrieved is not None
            assert retrieved.id == "test-report-1"

    async def test_get_report_not_found(self):
        from trustandverify.storage.postgres import PostgresStorage

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_ctx)

        storage = PostgresStorage(dsn="postgresql://localhost/test")
        storage._pool = mock_pool

        result = await storage.get_report("nonexistent")
        assert result is None

    async def test_list_reports(self):
        from trustandverify.storage.postgres import PostgresStorage

        report = _make_report()
        report_data = json.dumps(_report_to_dict(report))

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "id": "test-report-1",
                    "query": "Is coffee healthy?",
                    "created_at": datetime(2025, 6, 1, tzinfo=timezone.utc),
                    "data": report_data,
                },
            ]
        )

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_ctx)

        storage = PostgresStorage(dsn="postgresql://localhost/test")
        storage._pool = mock_pool

        summaries = await storage.list_reports(limit=10)
        assert len(summaries) == 1
        assert summaries[0].id == "test-report-1"

    async def test_save_and_get_claims(self):
        from trustandverify.storage.postgres import PostgresStorage

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_ctx)

        storage = PostgresStorage(dsn="postgresql://localhost/test")
        storage._pool = mock_pool

        claim = Claim(text="Test claim")
        result = await storage.save_claim(claim, "query-1")
        assert result == "Test claim"
        mock_conn.execute.assert_called_once()

        claims = await storage.get_claims_for_query("query-1")
        assert claims == []  # mocked to return empty

    async def test_get_pool_raises_without_asyncpg(self):
        from trustandverify.storage.postgres import PostgresStorage

        with patch.dict("sys.modules", {"asyncpg": None}):
            storage = PostgresStorage(dsn="postgresql://localhost/test")
            with pytest.raises(ImportError, match="asyncpg"):
                await storage._get_pool()

    async def test_get_pool_creates_pool(self):
        from trustandverify.storage.postgres import PostgresStorage

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_ctx)

        mock_asyncpg = MagicMock()
        mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)

        with patch.dict("sys.modules", {"asyncpg": mock_asyncpg}):
            storage = PostgresStorage(dsn="postgresql://localhost/test")
            pool = await storage._get_pool()
            assert pool is mock_pool
            mock_asyncpg.create_pool.assert_called_once()


# ── Neo4jStorage ──────────────────────────────────────────────────────────────


class TestNeo4jStorage:
    def test_is_available_with_password(self):
        from trustandverify.storage.neo4j import Neo4jStorage

        storage = Neo4jStorage(password="secret")
        assert storage.is_available() is True

    def test_is_available_without_password(self, monkeypatch):
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
        from trustandverify.storage.neo4j import Neo4jStorage

        storage = Neo4jStorage(password="")
        assert storage.is_available() is False

    async def test_save_and_get_report(self):
        from trustandverify.storage.neo4j import Neo4jStorage

        report = _make_report()
        report_data = json.dumps(_report_to_dict(report))

        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"data": report_data})

        mock_session = AsyncMock()
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session = MagicMock(return_value=mock_session)

        mock_neo4j = MagicMock()
        mock_neo4j.AsyncGraphDatabase = MagicMock()
        mock_neo4j.AsyncGraphDatabase.driver = MagicMock(return_value=mock_driver)

        with patch.dict("sys.modules", {"neo4j": mock_neo4j}):
            storage = Neo4jStorage(password="secret")
            storage._driver = mock_driver

            result_id = await storage.save_report(report)
            assert result_id == "test-report-1"

            retrieved = await storage.get_report("test-report-1")
            assert retrieved is not None
            assert retrieved.id == "test-report-1"

    async def test_get_report_not_found(self):
        from trustandverify.storage.neo4j import Neo4jStorage

        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session = MagicMock(return_value=mock_session)

        storage = Neo4jStorage(password="secret")
        storage._driver = mock_driver

        result = await storage.get_report("nonexistent")
        assert result is None

    async def test_list_reports(self):
        from trustandverify.storage.neo4j import Neo4jStorage

        report = _make_report()
        report_data = json.dumps(_report_to_dict(report))

        mock_result = AsyncMock()
        mock_result.data = AsyncMock(
            return_value=[
                {
                    "id": "test-report-1",
                    "query": "Is coffee healthy?",
                    "created_at": "2025-06-01T12:00:00+00:00",
                    "data": report_data,
                },
            ]
        )

        mock_session = AsyncMock()
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session = MagicMock(return_value=mock_session)

        storage = Neo4jStorage(password="secret")
        storage._driver = mock_driver

        summaries = await storage.list_reports(limit=10)
        assert len(summaries) == 1

    async def test_save_and_get_claims(self):
        from trustandverify.storage.neo4j import Neo4jStorage

        mock_run_result = AsyncMock()
        mock_run_result.data = AsyncMock(return_value=[])

        mock_session = AsyncMock()
        mock_session.run = AsyncMock(return_value=mock_run_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session = MagicMock(return_value=mock_session)

        storage = Neo4jStorage(password="secret")
        storage._driver = mock_driver

        claim = Claim(text="Test")
        result = await storage.save_claim(claim, "query-1")
        assert result == "Test"
        assert mock_session.run.call_count == 1

        claims = await storage.get_claims_for_query("query-1")
        assert claims == []

    async def test_get_driver_raises_without_neo4j(self):
        from trustandverify.storage.neo4j import Neo4jStorage

        with patch.dict("sys.modules", {"neo4j": None}):
            storage = Neo4jStorage(password="secret")
            with pytest.raises(ImportError, match="neo4j"):
                storage._get_driver()

    async def test_get_driver_creates_driver(self):
        from trustandverify.storage.neo4j import Neo4jStorage

        mock_driver = MagicMock()
        mock_neo4j = MagicMock()
        mock_neo4j.AsyncGraphDatabase = MagicMock()
        mock_neo4j.AsyncGraphDatabase.driver = MagicMock(return_value=mock_driver)

        with patch.dict("sys.modules", {"neo4j": mock_neo4j}):
            storage = Neo4jStorage(password="secret")
            driver = storage._get_driver()
            assert driver is mock_driver


# ── MongoStorage ──────────────────────────────────────────────────────────────


class TestMongoStorage:
    def test_is_available(self):
        from trustandverify.storage.mongo import MongoStorage

        storage = MongoStorage(uri="mongodb://localhost:27017")
        assert storage.is_available() is True

    async def test_save_report(self):
        from trustandverify.storage.mongo import MongoStorage

        mock_col = AsyncMock()
        mock_col.replace_one = AsyncMock()

        storage = MongoStorage()
        storage._get_collection = MagicMock(return_value=mock_col)

        report = _make_report()
        result_id = await storage.save_report(report)
        assert result_id == "test-report-1"
        mock_col.replace_one.assert_called_once()

    async def test_get_report(self):
        from trustandverify.storage.mongo import MongoStorage

        report = _make_report()
        doc = _report_to_dict(report)
        doc["_id"] = report.id

        mock_col = AsyncMock()
        mock_col.find_one = AsyncMock(return_value=doc)

        storage = MongoStorage()
        storage._get_collection = MagicMock(return_value=mock_col)

        retrieved = await storage.get_report("test-report-1")
        assert retrieved is not None
        assert retrieved.id == "test-report-1"

    async def test_get_report_not_found(self):
        from trustandverify.storage.mongo import MongoStorage

        mock_col = AsyncMock()
        mock_col.find_one = AsyncMock(return_value=None)

        storage = MongoStorage()
        storage._get_collection = MagicMock(return_value=mock_col)

        assert await storage.get_report("nonexistent") is None

    async def test_list_reports(self):
        from trustandverify.storage.mongo import MongoStorage

        report = _make_report()
        doc = _report_to_dict(report)
        doc["_id"] = report.id

        class AsyncDocIter:
            def __init__(self, docs):
                self._docs = list(docs)
                self._index = 0

            def sort(self, *a, **kw):
                return self

            def limit(self, *a, **kw):
                return self

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._index >= len(self._docs):
                    raise StopAsyncIteration
                val = self._docs[self._index]
                self._index += 1
                return val

        mock_col = MagicMock()
        mock_col.find = MagicMock(return_value=AsyncDocIter([doc]))

        storage = MongoStorage()
        storage._get_collection = MagicMock(return_value=mock_col)

        summaries = await storage.list_reports(limit=10)
        assert len(summaries) == 1

    async def test_save_and_get_claims_empty(self):
        """get_claims_for_query returns empty list when no docs match."""
        from trustandverify.storage.mongo import MongoStorage

        mock_claims_col = AsyncMock()
        mock_claims_col.insert_one = AsyncMock()

        class EmptyAsyncIter:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        mock_claims_col.find = MagicMock(return_value=EmptyAsyncIter())

        # Let _get_claims_collection run for real through a mocked _client
        mock_client = MagicMock()
        mock_client.__getitem__ = MagicMock(
            return_value=MagicMock(__getitem__=MagicMock(return_value=mock_claims_col))
        )

        storage = MongoStorage()
        storage._client = mock_client
        # _get_collection needs to not re-create _client
        storage._get_collection = MagicMock()

        claim = Claim(text="Test")
        result = await storage.save_claim(claim, "query-1")
        assert result == "Test"
        mock_claims_col.insert_one.assert_called_once()

        claims = await storage.get_claims_for_query("query-1")
        assert claims == []

    async def test_get_claims_with_documents(self):
        """get_claims_for_query must iterate docs and deserialise claims."""
        from trustandverify.storage.mongo import MongoStorage
        from trustandverify.storage.sqlite import _claim_to_dict

        # Build a real claim dict matching what save_claim stores
        original_claim = Claim(
            text="Coffee has antioxidants",
            verdict=Verdict.SUPPORTED,
            opinion=Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5),
        )
        doc = _claim_to_dict(original_claim)
        doc["_id"] = "fake-mongo-id"
        doc["query_id"] = "query-1"

        class DocAsyncIter:
            def __init__(self, docs):
                self._docs = list(docs)
                self._i = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._i >= len(self._docs):
                    raise StopAsyncIteration
                val = self._docs[self._i]
                self._i += 1
                return val

        mock_claims_col = MagicMock()
        mock_claims_col.find = MagicMock(return_value=DocAsyncIter([doc]))

        mock_client = MagicMock()
        mock_client.__getitem__ = MagicMock(
            return_value=MagicMock(__getitem__=MagicMock(return_value=mock_claims_col))
        )

        storage = MongoStorage()
        storage._client = mock_client
        storage._get_collection = MagicMock()

        claims = await storage.get_claims_for_query("query-1")
        assert len(claims) == 1
        assert claims[0].text == "Coffee has antioxidants"
        assert claims[0].verdict == Verdict.SUPPORTED
        assert abs(claims[0].opinion.belief - 0.7) < 1e-6

    async def test_get_collection_raises_without_motor(self):
        from trustandverify.storage.mongo import MongoStorage

        with patch.dict("sys.modules", {"motor": None, "motor.motor_asyncio": None}):
            storage = MongoStorage()
            with pytest.raises(ImportError, match="motor"):
                storage._get_collection()

    async def test_get_collection_creates_client(self):
        from trustandverify.storage.mongo import MongoStorage

        mock_client = MagicMock()
        mock_motor = MagicMock()
        mock_motor.AsyncIOMotorClient = MagicMock(return_value=mock_client)

        mock_module = MagicMock()
        mock_module.motor_asyncio = mock_motor

        with patch.dict("sys.modules", {"motor": mock_module, "motor.motor_asyncio": mock_motor}):
            storage = MongoStorage()
            storage._get_collection()
            mock_motor.AsyncIOMotorClient.assert_called_once()


# ── RedisStorage ──────────────────────────────────────────────────────────────


class TestRedisStorage:
    def test_is_available(self):
        from trustandverify.storage.redis import RedisStorage

        storage = RedisStorage(url="redis://localhost:6379")
        assert storage.is_available() is True

    async def test_save_report(self):
        from trustandverify.storage.redis import RedisStorage

        mock_client = AsyncMock()
        mock_client.set = AsyncMock()
        mock_client.zadd = AsyncMock()

        storage = RedisStorage()
        storage._client = mock_client

        report = _make_report()
        result_id = await storage.save_report(report)
        assert result_id == "test-report-1"
        mock_client.set.assert_called_once()
        mock_client.zadd.assert_called_once()

    async def test_get_report(self):
        from trustandverify.storage.redis import RedisStorage

        report = _make_report()
        report_data = json.dumps(_report_to_dict(report))

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=report_data)

        storage = RedisStorage()
        storage._client = mock_client

        retrieved = await storage.get_report("test-report-1")
        assert retrieved is not None
        assert retrieved.id == "test-report-1"

    async def test_get_report_not_found(self):
        from trustandverify.storage.redis import RedisStorage

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=None)

        storage = RedisStorage()
        storage._client = mock_client

        assert await storage.get_report("nonexistent") is None

    async def test_list_reports(self):
        from trustandverify.storage.redis import RedisStorage

        report = _make_report()
        report_data = json.dumps(_report_to_dict(report))

        mock_client = AsyncMock()
        mock_client.zrevrange = AsyncMock(return_value=["test-report-1"])
        mock_client.get = AsyncMock(return_value=report_data)

        storage = RedisStorage()
        storage._client = mock_client

        summaries = await storage.list_reports(limit=10)
        assert len(summaries) == 1
        assert summaries[0].id == "test-report-1"

    async def test_list_reports_missing_key(self):
        from trustandverify.storage.redis import RedisStorage

        mock_client = AsyncMock()
        mock_client.zrevrange = AsyncMock(return_value=["missing-id"])
        mock_client.get = AsyncMock(return_value=None)

        storage = RedisStorage()
        storage._client = mock_client

        summaries = await storage.list_reports(limit=10)
        assert len(summaries) == 0

    async def test_save_and_get_claims(self):
        from trustandverify.storage.redis import RedisStorage

        stored_lists: dict[str, list] = {}

        mock_client = AsyncMock()
        mock_client.rpush = AsyncMock(
            side_effect=lambda k, v: stored_lists.setdefault(k, []).append(v)
        )
        mock_client.lrange = AsyncMock(side_effect=lambda k, start, end: stored_lists.get(k, []))

        storage = RedisStorage()
        storage._client = mock_client

        claim = Claim(text="Test")
        result = await storage.save_claim(claim, "query-1")
        assert result == "Test"
        mock_client.rpush.assert_called_once()

        claims = await storage.get_claims_for_query("query-1")
        assert len(claims) == 1
        assert claims[0].text == "Test"

    async def test_get_client_raises_without_redis(self):
        from trustandverify.storage.redis import RedisStorage

        with patch.dict("sys.modules", {"redis": None, "redis.asyncio": None}):
            storage = RedisStorage()
            with pytest.raises(ImportError, match="redis"):
                await storage._get_client()

    async def test_get_client_creates_client(self):
        from trustandverify.storage.redis import RedisStorage

        mock_client = AsyncMock()
        mock_redis = MagicMock()
        mock_redis.from_url = AsyncMock(return_value=mock_client)

        mock_module = MagicMock()
        mock_module.asyncio = mock_redis

        with patch.dict("sys.modules", {"redis": mock_module, "redis.asyncio": mock_redis}):
            storage = RedisStorage()
            await storage._get_client()
            mock_redis.from_url.assert_called_once()
