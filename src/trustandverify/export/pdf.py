"""PDF exporter — renders a Report to PDF via WeasyPrint (wraps HtmlExporter).

Install with: pip install trustandverify[pdf]
"""

from __future__ import annotations

from trustandverify.core.models import Report
from trustandverify.export.html import HtmlExporter


class PdfExporter:
    """Render a Report to PDF using WeasyPrint.

    WeasyPrint converts the HTML report to PDF, preserving all
    colours, layout, and opinion bars.

    Install with: pip install trustandverify[pdf]
    """

    format_name = "pdf"
    file_extension = ".pdf"

    def render(self, report: Report) -> bytes:
        """Render report to raw PDF bytes."""
        try:
            from weasyprint import HTML  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "PdfExporter requires weasyprint. "
                "Install with: pip install trustandverify[pdf]"
            ) from e

        html_str = HtmlExporter().render(report)
        return HTML(string=html_str).write_pdf()

    def render_to_file(self, report: Report, path: str) -> None:
        """Render report and write PDF to file."""
        pdf_bytes = self.render(report)
        with open(path, "wb") as fh:
            fh.write(pdf_bytes)
