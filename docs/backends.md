# Backends Guide

trustandverify uses a **protocol-based plugin architecture**. Each backend type is defined as a `typing.Protocol` with `runtime_checkable`. Any object matching the protocol's method signatures works — no inheritance required.

## Search Backends

Search backends fetch web evidence for each claim. All implement the `SearchBackend` protocol:

```python
class SearchBackend(Protocol):
    name: str
    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]: ...
    def is_available(self) -> bool: ...
```

### TavilySearch

Default search backend. Free tier: 1,000 searches/month.

```python
from trustandverify.search import TavilySearch
search = TavilySearch()  # reads TAVILY_API_KEY from env
```

Env: `TAVILY_API_KEY`
Install: included in core (uses `httpx`)

### BraveSearch

Free tier: 2,000 queries/month.

```python
from trustandverify.search import BraveSearch
search = BraveSearch()  # reads BRAVE_API_KEY from env
```

Env: `BRAVE_API_KEY`
Install: included in core (uses `httpx`)

### BingSearch

Requires a Bing Web Search API subscription.

```python
from trustandverify.search import BingSearch
search = BingSearch()  # reads BING_API_KEY from env
```

Env: `BING_API_KEY`
Install: included in core (uses `httpx`)

### SerpAPISearch

Google search results via SerpAPI.

```python
from trustandverify.search import SerpAPISearch
search = SerpAPISearch()  # reads SERPAPI_API_KEY from env
```

Env: `SERPAPI_API_KEY`
Install: included in core (uses `httpx`)

### MultiSearch

Fans out a query to multiple backends concurrently, deduplicates by URL, and interleaves results for source diversity before sorting by score.

```python
from trustandverify.search import MultiSearch, TavilySearch, BraveSearch

search = MultiSearch([TavilySearch(), BraveSearch()])
```

MultiSearch uses `return_exceptions=True` on `asyncio.gather`, so one failing backend does not crash the pipeline — it is logged and skipped.

## LLM Backends

LLM backends handle claim decomposition, evidence extraction, assessment, and summary generation. All implement the `LLMBackend` protocol:

```python
class LLMBackend(Protocol):
    name: str
    model: str
    async def complete(self, prompt: str, system: str = "") -> str: ...
    async def complete_json(self, prompt: str, system: str = "", defaults: dict | None = None) -> dict: ...
    def is_available(self) -> bool: ...
```

The `complete_json` method includes robust fallback parsing: it strips markdown fences, extracts `{...}` or `[...]` blocks from mixed text, handles unicode whitespace, and returns `defaults` if all parsing fails.

### GeminiBackend

Default LLM. Uses Google's Gemini via LiteLLM.

```python
from trustandverify.llm import GeminiBackend
llm = GeminiBackend()               # default: gemini/gemini-2.5-flash
llm = GeminiBackend(model="gemini/gemini-2.0-flash")
```

Env: `GEMINI_API_KEY`
Install: `pip install trustandverify[gemini]`

### OpenAIBackend

```python
from trustandverify.llm import OpenAIBackend
llm = OpenAIBackend()               # default: gpt-4o
llm = OpenAIBackend(model="gpt-4o-mini")
```

Env: `OPENAI_API_KEY`
Install: `pip install trustandverify[openai]`

### AnthropicBackend

```python
from trustandverify.llm import AnthropicBackend
llm = AnthropicBackend()            # default: claude-sonnet-4-5
```

Env: `ANTHROPIC_API_KEY`
Install: `pip install trustandverify[anthropic]`

### OllamaBackend

Local LLM via Ollama. No API key needed — requires Ollama running at `localhost:11434`.

```python
from trustandverify.llm import OllamaBackend
llm = OllamaBackend()               # default: llama3
llm = OllamaBackend(model="mistral")
```

Install: `pip install trustandverify[ollama]`

## Storage Backends

Storage backends persist reports and claims. All implement the `StorageBackend` protocol:

```python
class StorageBackend(Protocol):
    name: str
    async def save_report(self, report: Report) -> str: ...
    async def get_report(self, report_id: str) -> Report | None: ...
    async def list_reports(self, limit: int = 50) -> list[ReportSummary]: ...
    async def save_claim(self, claim: Claim, query_id: str) -> str: ...
    async def get_claims_for_query(self, query_id: str) -> list[Claim]: ...
```

### InMemoryStorage (default)

Data lost when the process exits. No dependencies.

```python
from trustandverify.storage import InMemoryStorage
storage = InMemoryStorage()
```

### SQLiteStorage

Persistent local storage. No extra dependencies (uses stdlib `sqlite3`).

```python
from trustandverify.storage import SQLiteStorage
storage = SQLiteStorage("reports.db")
storage = SQLiteStorage(":memory:")   # for tests
```

### PostgresStorage

```python
from trustandverify.storage import PostgresStorage
storage = PostgresStorage(dsn="postgresql://user:pass@localhost/dbname")
```

Env: `POSTGRES_DSN`
Install: `pip install trustandverify[postgres]`

### Neo4jStorage

```python
from trustandverify.storage import Neo4jStorage
storage = Neo4jStorage(uri="bolt://localhost:7687", password="secret")
```

Env: `NEO4J_PASSWORD`
Install: `pip install trustandverify[neo4j]`

### MongoStorage

```python
from trustandverify.storage import MongoStorage
storage = MongoStorage(uri="mongodb://localhost:27017")
```

Env: `MONGO_URI`
Install: `pip install trustandverify[mongo]`

### RedisStorage

```python
from trustandverify.storage import RedisStorage
storage = RedisStorage(url="redis://localhost:6379")
```

Env: `REDIS_URL`
Install: `pip install trustandverify[redis]`

## Cache Backends

Cache backends reduce API calls by storing search results and LLM responses. All implement the `CacheBackend` protocol:

```python
class CacheBackend(Protocol):
    async def get(self, key: str) -> Any | None: ...
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None: ...
    async def invalidate(self, key: str) -> None: ...
```

### FileCache (default when `enable_cache=True`)

JSON files on disk. No extra dependencies.

```python
from trustandverify.cache import FileCache
cache = FileCache(cache_dir=".trustandverify_cache", default_ttl=3600)
```

### RedisCache

```python
from trustandverify.cache import RedisCache
cache = RedisCache(url="redis://localhost:6379", default_ttl=3600)
```

Keys are prefixed with `tv:cache:` to avoid collisions with `RedisStorage` (which uses `tv:report:`).

Install: `pip install trustandverify[redis]`

## Export Backends

Export backends render reports into output formats. All implement the `ExportBackend` protocol:

```python
class ExportBackend(Protocol):
    format_name: str
    file_extension: str
    def render(self, report: Report) -> str | bytes: ...
    def render_to_file(self, report: Report, path: str) -> None: ...
```

### JsonLdExporter

JSON-LD with Schema.org, PROV-O, and jsonld-ex vocabulary. SPARQL/RDF compatible.

```python
from trustandverify.export import JsonLdExporter
JsonLdExporter().render_to_file(report, "report.jsonld")
```

### MarkdownExporter

Human-readable Markdown with tables, verdict emojis, and source links.

```python
from trustandverify.export import MarkdownExporter
MarkdownExporter().render_to_file(report, "report.md")
```

### HtmlExporter

Self-contained HTML page with CSS, opinion bars, and XSS-escaped content.

```python
from trustandverify.export import HtmlExporter
HtmlExporter().render_to_file(report, "report.html")
```

### PdfExporter

PDF via WeasyPrint (wraps `HtmlExporter`). Returns `bytes`, not `str`.

```python
from trustandverify.export import PdfExporter
PdfExporter().render_to_file(report, "report.pdf")
```

Install: `pip install trustandverify[pdf]`

## Writing a Custom Backend

Any class matching the protocol shape works without inheritance:

```python
class MySearch:
    name = "my-search"

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        # Your implementation
        ...

    def is_available(self) -> bool:
        return True

# Use directly — no registration needed
agent = TrustAgent(config=config, search=MySearch(), llm=llm)
```

Verify protocol conformance in tests:

```python
from trustandverify.search.protocol import SearchBackend
assert isinstance(MySearch(), SearchBackend)
```
