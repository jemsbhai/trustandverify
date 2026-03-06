"""Protocol conformance tests — verify each backend satisfies its runtime_checkable Protocol.

These catch signature drift early. If a backend's method signature diverges
from the Protocol (renamed param, missing method, wrong attribute), the
isinstance() check will fail here.
"""

from __future__ import annotations

import pytest

from trustandverify.cache.file_cache import FileCache
from trustandverify.cache.protocol import CacheBackend
from trustandverify.cache.redis_cache import RedisCache
from trustandverify.export.html import HtmlExporter
from trustandverify.export.jsonld import JsonLdExporter
from trustandverify.export.markdown import MarkdownExporter
from trustandverify.export.pdf import PdfExporter
from trustandverify.export.protocol import ExportBackend
from trustandverify.llm.anthropic import AnthropicBackend
from trustandverify.llm.gemini import GeminiBackend
from trustandverify.llm.ollama import OllamaBackend
from trustandverify.llm.openai import OpenAIBackend
from trustandverify.llm.protocol import LLMBackend
from trustandverify.search.bing import BingSearch
from trustandverify.search.brave import BraveSearch
from trustandverify.search.multi import MultiSearch
from trustandverify.search.protocol import SearchBackend
from trustandverify.search.serpapi import SerpAPISearch
from trustandverify.search.tavily import TavilySearch
from trustandverify.storage.memory import InMemoryStorage
from trustandverify.storage.mongo import MongoStorage
from trustandverify.storage.neo4j import Neo4jStorage
from trustandverify.storage.postgres import PostgresStorage
from trustandverify.storage.protocol import StorageBackend
from trustandverify.storage.redis import RedisStorage
from trustandverify.storage.sqlite import SQLiteStorage


# ── SearchBackend ──────────────────────────────────────────────────────────────


class TestSearchBackendConformance:
    @pytest.mark.parametrize("cls,kwargs", [
        (TavilySearch, {"api_key": "fake"}),
        (BraveSearch, {"api_key": "fake"}),
        (BingSearch, {"api_key": "fake"}),
        (SerpAPISearch, {"api_key": "fake"}),
        (MultiSearch, {"backends": [TavilySearch(api_key="fake")]}),
    ])
    def test_satisfies_protocol(self, cls, kwargs):
        instance = cls(**kwargs)
        assert isinstance(instance, SearchBackend), (
            f"{cls.__name__} does not satisfy SearchBackend protocol"
        )


# ── LLMBackend ─────────────────────────────────────────────────────────────────


class TestLLMBackendConformance:
    @pytest.mark.parametrize("cls,kwargs", [
        (GeminiBackend, {"api_key": "fake"}),
        (OpenAIBackend, {"api_key": "fake"}),
        (AnthropicBackend, {"api_key": "fake"}),
        (OllamaBackend, {}),
    ])
    def test_satisfies_protocol(self, cls, kwargs):
        instance = cls(**kwargs)
        assert isinstance(instance, LLMBackend), (
            f"{cls.__name__} does not satisfy LLMBackend protocol"
        )


# ── StorageBackend ─────────────────────────────────────────────────────────────


class TestStorageBackendConformance:
    @pytest.mark.parametrize("cls,kwargs", [
        (InMemoryStorage, {}),
        (SQLiteStorage, {"path": ":memory:"}),
        (PostgresStorage, {"dsn": "postgresql://localhost/test"}),
        (Neo4jStorage, {"password": "fake"}),
        (MongoStorage, {"uri": "mongodb://localhost:27017"}),
        (RedisStorage, {"url": "redis://localhost:6379"}),
    ])
    def test_satisfies_protocol(self, cls, kwargs):
        instance = cls(**kwargs)
        assert isinstance(instance, StorageBackend), (
            f"{cls.__name__} does not satisfy StorageBackend protocol"
        )


# ── CacheBackend ───────────────────────────────────────────────────────────────


class TestCacheBackendConformance:
    def test_file_cache_satisfies_protocol(self, tmp_path):
        cache = FileCache(cache_dir=str(tmp_path / "cache"))
        assert isinstance(cache, CacheBackend)

    def test_redis_cache_satisfies_protocol(self):
        cache = RedisCache(url="redis://localhost:6379")
        assert isinstance(cache, CacheBackend)


# ── ExportBackend ──────────────────────────────────────────────────────────────


class TestExportBackendConformance:
    @pytest.mark.parametrize("cls", [
        JsonLdExporter,
        MarkdownExporter,
        HtmlExporter,
        PdfExporter,
    ])
    def test_satisfies_protocol(self, cls):
        instance = cls()
        assert isinstance(instance, ExportBackend), (
            f"{cls.__name__} does not satisfy ExportBackend protocol"
        )
