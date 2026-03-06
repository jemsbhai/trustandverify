"""StorageBackend protocol — persistent storage for reports and claims."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from trustandverify.core.models import Claim, Report, ReportSummary


@runtime_checkable
class StorageBackend(Protocol):
    """Structural protocol for storage providers.

    Default implementation is InMemoryStorage (no dependencies).
    SQLite, Postgres, Neo4j, MongoDB, and Redis backends are optional extras.
    """

    name: str

    async def save_report(self, report: Report) -> str:
        """Persist a full report.

        Args:
            report: The Report to save.

        Returns:
            The report ID (report.id).
        """
        ...

    async def get_report(self, report_id: str) -> Report | None:
        """Retrieve a report by ID, or None if not found."""
        ...

    async def list_reports(self, limit: int = 50) -> list[ReportSummary]:
        """List recent reports as lightweight summaries.

        Args:
            limit: Maximum number of summaries to return.

        Returns:
            List ordered by created_at descending.
        """
        ...

    async def save_claim(self, claim: Claim, query_id: str) -> str:
        """Persist a single claim associated with a query.

        Args:
            claim:    The Claim to save.
            query_id: The report/query ID this claim belongs to.

        Returns:
            The claim text (used as identifier).
        """
        ...

    async def get_claims_for_query(self, query_id: str) -> list[Claim]:
        """Retrieve all claims associated with a query ID."""
        ...
