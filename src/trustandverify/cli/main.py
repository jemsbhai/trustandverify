"""trustandverify CLI — Typer app."""

import asyncio
import sys

try:
    import typer
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ImportError:  # pragma: no cover
    print(
        "CLI requires extras: pip install trustandverify[cli]",
        file=sys.stderr,
    )
    sys.exit(1)

from trustandverify._version import __version__

app = typer.Typer(
    name="trustandverify",
    help="Agentic knowledge verification using Subjective Logic confidence algebra.",
    add_completion=False,
)
console = Console()

_EXPORTERS = {
    "jsonld": "trustandverify.export.jsonld:JsonLdExporter",
    "markdown": "trustandverify.export.markdown:MarkdownExporter",
    "md": "trustandverify.export.markdown:MarkdownExporter",
    "html": "trustandverify.export.html:HtmlExporter",
    "pdf": "trustandverify.export.pdf:PdfExporter",
}


def _get_exporter(format_name: str):
    """Resolve a format name to an exporter instance."""
    key = format_name.lower().strip()
    if key not in _EXPORTERS:
        console.print(
            f"[bold red]Error:[/] Unknown format [bold]{format_name}[/]. "
            f"Choose from: {', '.join(sorted(set(_EXPORTERS.values())))}"
        )
        raise typer.Exit(1)
    module_path, cls_name = _EXPORTERS[key].rsplit(":", 1)
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)()


# ── verify command ─────────────────────────────────────────────────────────────


@app.command()
def verify(
    query: str = typer.Argument(..., help="The research question or claim to verify."),
    claims: int = typer.Option(
        0, "--claims", "-c", help="Number of claims to decompose into. 0 = auto (3-5)."
    ),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Write JSON-LD report to this file path."
    ),
    format: str = typer.Option(
        "jsonld", "--format", "-f", help="Output format: jsonld, markdown, html, pdf."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Print step-by-step agent progress."
    ),
) -> None:
    """Verify a research question or claim against live web evidence."""
    from trustandverify.core.agent import TrustAgent
    from trustandverify.core.config import TrustConfig
    from trustandverify.llm.gemini import GeminiBackend
    from trustandverify.search.tavily import TavilySearch

    search = TavilySearch()
    llm = GeminiBackend()

    if not search.is_available():
        console.print("[bold red]Error:[/] TAVILY_API_KEY environment variable not set.")
        raise typer.Exit(1)

    if not llm.is_available():
        console.print("[bold red]Error:[/] GEMINI_API_KEY environment variable not set.")
        raise typer.Exit(1)

    agent = TrustAgent(
        config=TrustConfig(num_claims=claims),
        search=search,
        llm=llm,
    )

    console.print(f"\n[bold cyan]🔍 trustandverify[/] v{__version__}")
    console.print(f"[dim]Query:[/] {query}\n")

    report = asyncio.run(agent.verify(query, verbose=verbose))

    # ── Results table ──
    table = Table(title="Verification Report", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Claim", style="white", max_width=55)
    table.add_column("Verdict", justify="center", width=12)
    table.add_column("P", justify="right", width=6)
    table.add_column("b / d / u", justify="right", width=18)

    verdict_colours = {
        "supported": "green",
        "contested": "yellow",
        "refuted": "red",
        "no_evidence": "dim",
    }

    for i, claim in enumerate(report.claims, 1):
        op = claim.opinion
        verdict_str = claim.verdict.value
        colour = verdict_colours.get(verdict_str, "white")

        if op:
            p_str = f"{op.projected_probability():.3f}"
            bdu_str = f"{op.belief:.2f} / {op.disbelief:.2f} / {op.uncertainty:.2f}"
        else:
            p_str = "—"
            bdu_str = "— / — / —"

        table.add_row(
            str(i),
            claim.text,
            f"[{colour}]{verdict_str.upper()}[/{colour}]",
            p_str,
            bdu_str,
        )

    console.print(table)

    if report.conflicts:
        console.print("\n[bold yellow]⚡ Conflicts detected:[/]")
        for c in report.conflicts:
            console.print(
                f"  [dim]{c.claim_text}[/] — "
                f"degree={c.conflict_degree:.3f}, "
                f"{c.num_supporting} support vs {c.num_contradicting} contradict"
            )

    console.print(Panel(report.summary, title="[bold]Summary[/]", border_style="cyan"))

    # ── Optional file output ──
    if output:
        exporter = _get_exporter(format)
        exporter.render_to_file(report, output)
        console.print(f"\n[dim]{exporter.format_name} report written to:[/] {output}")


# ── ui command ─────────────────────────────────────────────────────────────────


@app.command()
def ui() -> None:
    """Launch the Streamlit web dashboard."""
    import importlib.resources
    import subprocess

    try:
        import streamlit  # noqa: F401
    except ImportError:
        console.print(
            "[bold red]Error:[/] Streamlit not installed. Run: pip install trustandverify[ui]"
        )
        raise typer.Exit(1) from None

    # Locate the ui/app.py inside the installed package
    try:
        pkg_files = importlib.resources.files("trustandverify.ui")
        app_path = str(pkg_files.joinpath("app.py"))
    except Exception:
        console.print(
            "[bold red]Error:[/] Could not locate UI app. Is trustandverify installed correctly?"
        )
        raise typer.Exit(1) from None

    console.print("[bold cyan]🔍 Launching TrustGraph UI...[/]")
    subprocess.run(["streamlit", "run", app_path], check=False)


# ── version command ────────────────────────────────────────────────────────────


@app.command()
def version() -> None:
    """Show the installed version."""
    console.print(f"trustandverify {__version__}")


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
