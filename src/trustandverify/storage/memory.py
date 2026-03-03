"""InMemoryStorage — default storage backend, no dependencies required."""

from __future__ import annotations

from trustandverify.core.models import Claim, Report, ReportSummary


class InMemoryStorage:
    """In-memory storage backend backed by plain Python dicts.

    This is the default backend — it requires no external dependencies
    and works out of the box.  Data is lost when the process exits.
    Use SQLiteStorage or another persistent backend for production.
    """

    name = "memory"

    def __init__(self) -> None:
        self._reports: dict[str, Report] = {}
        self._claims: dict[str, list[Claim]] = {}  # query_id -> claims

    async def save_report(self, report: Report) -> str:
        self._reports[report.id] = report
        return report.id

    async def get_report(self, report_id: str) -> Report | None:
        return self._reports.get(report_id)

    async def list_reports(self, limit: int = 50) -> list[ReportSummary]:
        reports = sorted(
            self._reports.values(),
            key=lambda r: r.created_at,
            reverse=True,
        )
        return [
            ReportSummary(
                id=r.id,
                query=r.query,
                created_at=r.created_at,
                num_claims=len(r.claims),
            )
            for r in reports[:limit]
        ]

    async def save_claim(self, claim: Claim) -> str:
        # Claims are stored without a query association at this level.
        # Use save_report() for the full provenance chain.
        return claim.text

    async def get_claims_for_query(self, query_id: str) -> list[Claim]:
        return self._claims.get(query_id, [])
