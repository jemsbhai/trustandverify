"""SearchBackend protocol — any web search provider."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from trustandverify.core.models import SearchResult


@runtime_checkable
class SearchBackend(Protocol):
    """Structural protocol for web search providers.

    Any class with a matching ``search`` and ``is_available`` method
    satisfies this protocol without inheriting from it.
    Third-party backends can be dropped in without importing this class.
    """

    name: str

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Search the web and return structured results.

        Args:
            query:       The search query string.
            max_results: Maximum number of results to return.

        Returns:
            List of SearchResult objects, ordered by relevance.
        """
        ...

    def is_available(self) -> bool:
        """Return True if this backend is configured and ready to use.

        Typically checks that the required API key env var is set.
        """
        ...
