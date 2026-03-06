"""ExportBackend protocol — render a Report to a specific output format."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from trustandverify.core.models import Report


@runtime_checkable
class ExportBackend(Protocol):
    """Structural protocol for report exporters.

    Implementations: JSON-LD (default), Markdown, HTML, PDF.
    """

    format_name: str
    file_extension: str

    def render(self, report: Report) -> str | bytes:
        """Render a report to a string or bytes.

        Args:
            report: The Report to render.

        Returns:
            Rendered string (JSON, Markdown, HTML) or bytes (PDF).
        """
        ...

    def render_to_file(self, report: Report, path: str) -> None:
        """Render a report and write it to a file.

        Args:
            report: The Report to render.
            path:   Destination file path.
        """
        ...
