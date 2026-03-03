"""TrustAgent — main orchestrator that wires config + backends + pipeline."""

from __future__ import annotations

from trustandverify.cache.file_cache import FileCache
from trustandverify.core.config import TrustConfig
from trustandverify.core.models import Report
from trustandverify.core.pipeline import run_pipeline
from trustandverify.storage.memory import InMemoryStorage


class TrustAgent:
    """Orchestrates the full verification pipeline.

    Usage::

        agent = TrustAgent(
            config=TrustConfig(num_claims=5),
            search=TavilySearch(),
            llm=GeminiBackend(),
        )
        report = await agent.verify("Is remote work more productive?")

    All backends are optional — sensible defaults are used if omitted:
        - storage: InMemoryStorage
        - cache:   FileCache (if config.enable_cache is True)
    """

    def __init__(
        self,
        config: TrustConfig | None = None,
        search: object = None,
        llm: object = None,
        storage: object | None = None,
        cache: object | None = None,
    ) -> None:
        self.config = config or TrustConfig()
        self.search = search
        self.llm = llm
        self.storage = storage or InMemoryStorage()

        if cache is not None:
            self.cache: object | None = cache
        elif self.config.enable_cache:
            self.cache = FileCache(default_ttl=self.config.cache_ttl)
        else:
            self.cache = None

    async def verify(self, query: str, verbose: bool = False) -> Report:
        """Run the full verification pipeline and return a Report.

        Args:
            query:   The research question or claim to verify.
            verbose: Print step-by-step progress to stdout.

        Returns:
            A fully populated Report with claims, opinions, conflicts,
            and summary.
        """
        if self.search is None:
            raise RuntimeError(
                "TrustAgent requires a search backend. "
                "Pass search=TavilySearch() or another SearchBackend."
            )
        if self.llm is None:
            raise RuntimeError(
                "TrustAgent requires an LLM backend. "
                "Pass llm=GeminiBackend() or another LLMBackend."
            )

        report = await run_pipeline(
            query=query,
            config=self.config,
            search=self.search,
            llm=self.llm,
            cache=self.cache,
            verbose=verbose,
        )

        # Persist to storage
        await self.storage.save_report(report)  # type: ignore[union-attr]

        return report
