"""Tests for cli/main.py — Typer CLI commands (all mocked, no real API calls)."""

from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from jsonld_ex.confidence_algebra import Opinion
from typer.testing import CliRunner

from trustandverify.cli.main import app
from trustandverify.core.models import Claim, Conflict, Evidence, Report, Source, Verdict

runner = CliRunner()


def _make_report() -> Report:
    return Report(
        id="test-id",
        query="Is coffee healthy?",
        claims=[
            Claim(
                text="Coffee has antioxidants",
                verdict=Verdict.SUPPORTED,
                assessment="Well supported.",
                opinion=Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5),
                evidence=[
                    Evidence(
                        text="Rich in antioxidants",
                        supports_claim=True,
                        relevance=0.9,
                        confidence_raw=0.85,
                        source=Source(
                            url="https://example.com",
                            title="Study",
                            content_snippet="Coffee...",
                            trust_score=0.8,
                        ),
                    ),
                ],
            ),
            Claim(
                text="Coffee causes insomnia",
                verdict=Verdict.CONTESTED,
                assessment="Mixed evidence.",
                opinion=Opinion(belief=0.4, disbelief=0.3, uncertainty=0.3, base_rate=0.5),
                evidence=[],
            ),
        ],
        conflicts=[
            Conflict(
                claim_text="Coffee causes insomnia",
                conflict_degree=0.35,
                num_supporting=2,
                num_contradicting=1,
            ),
        ],
        summary="Coffee appears to have both benefits and risks.",
        created_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
    )


def _cli_env(report):
    """Context manager that patches asyncio.run, agent.verify, and backend availability."""
    stack = ExitStack()
    stack.enter_context(patch("trustandverify.cli.main.asyncio.run", return_value=report))
    # Mock verify as a regular function (not async) so no unawaited coroutine is created
    stack.enter_context(patch("trustandverify.core.agent.TrustAgent.verify", return_value=report))
    stack.enter_context(
        patch("trustandverify.search.tavily.TavilySearch.is_available", return_value=True)
    )
    stack.enter_context(
        patch("trustandverify.llm.gemini.GeminiBackend.is_available", return_value=True)
    )
    return stack


class TestMainEntrypoint:
    def test_main_calls_app(self):
        from trustandverify.cli.main import main

        with patch("trustandverify.cli.main.app") as mock_app:
            main()
            mock_app.assert_called_once()


class TestVersionCommand:
    def test_prints_version(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "trustandverify" in result.output


class TestVerifyCommand:
    def test_missing_tavily_key(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        result = runner.invoke(app, ["verify", "Is coffee healthy?"])
        assert result.exit_code == 1
        assert "TAVILY_API_KEY" in result.output

    def test_missing_gemini_key(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "fake")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        result = runner.invoke(app, ["verify", "Is coffee healthy?"])
        assert result.exit_code == 1
        assert "GEMINI_API_KEY" in result.output

    def test_successful_verify(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "fake")
        monkeypatch.setenv("GEMINI_API_KEY", "fake")
        with _cli_env(_make_report()):
            result = runner.invoke(app, ["verify", "Is coffee healthy?"])
        assert result.exit_code == 0
        assert "SUPPORTED" in result.output
        assert "CONTESTED" in result.output
        assert "Conflicts" in result.output
        assert "Summary" in result.output

    def test_verify_with_output_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TAVILY_API_KEY", "fake")
        monkeypatch.setenv("GEMINI_API_KEY", "fake")
        output_file = str(tmp_path / "report.jsonld")
        with _cli_env(_make_report()):
            result = runner.invoke(app, ["verify", "Is coffee healthy?", "--output", output_file])
        assert result.exit_code == 0
        assert "report written to" in result.output
        assert (tmp_path / "report.jsonld").exists()

    def test_verify_markdown_format(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TAVILY_API_KEY", "fake")
        monkeypatch.setenv("GEMINI_API_KEY", "fake")
        output_file = str(tmp_path / "report.md")
        with _cli_env(_make_report()):
            result = runner.invoke(
                app,
                ["verify", "Is coffee healthy?", "--format", "markdown", "--output", output_file],
            )
        assert result.exit_code == 0
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "# TrustGraph" in content

    def test_verify_html_format(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TAVILY_API_KEY", "fake")
        monkeypatch.setenv("GEMINI_API_KEY", "fake")
        output_file = str(tmp_path / "report.html")
        with _cli_env(_make_report()):
            result = runner.invoke(
                app, ["verify", "Is coffee healthy?", "--format", "html", "--output", output_file]
            )
        assert result.exit_code == 0
        content = (tmp_path / "report.html").read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content

    def test_verify_unknown_format(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "fake")
        monkeypatch.setenv("GEMINI_API_KEY", "fake")
        with _cli_env(_make_report()):
            result = runner.invoke(
                app, ["verify", "Is coffee healthy?", "--format", "xyz", "--output", "out.txt"]
            )
        assert result.exit_code == 1

    def test_verify_with_claims_option(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "fake")
        monkeypatch.setenv("GEMINI_API_KEY", "fake")
        with _cli_env(_make_report()):
            result = runner.invoke(app, ["verify", "Is coffee healthy?", "--claims", "5"])
        assert result.exit_code == 0

    def test_verify_no_conflicts(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "fake")
        monkeypatch.setenv("GEMINI_API_KEY", "fake")
        report = _make_report()
        report.conflicts = []
        with _cli_env(report):
            result = runner.invoke(app, ["verify", "Is coffee healthy?"])
        assert result.exit_code == 0
        assert "Conflicts" not in result.output

    def test_verify_claim_without_opinion(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "fake")
        monkeypatch.setenv("GEMINI_API_KEY", "fake")
        report = _make_report()
        report.claims[0].opinion = None
        with _cli_env(report):
            result = runner.invoke(app, ["verify", "Is coffee healthy?"])
        assert result.exit_code == 0
        assert "\u2014" in result.output


class TestUiCommand:
    def test_streamlit_not_installed(self):
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "streamlit":
                raise ImportError("No module named 'streamlit'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = runner.invoke(app, ["ui"])
        assert result.exit_code == 1
        assert "Streamlit not installed" in result.output

    def test_streamlit_launches(self):
        mock_streamlit = MagicMock()
        mock_files = MagicMock()
        mock_files.joinpath.return_value = "/fake/path/app.py"

        with (
            patch.dict("sys.modules", {"streamlit": mock_streamlit}),
            patch("importlib.resources.files", return_value=mock_files),
            patch("subprocess.run"),
        ):
            result = runner.invoke(app, ["ui"])
        assert result.exit_code == 0

    def test_ui_app_path_not_found(self):
        mock_streamlit = MagicMock()

        with (
            patch.dict("sys.modules", {"streamlit": mock_streamlit}),
            patch("importlib.resources.files", side_effect=Exception("not found")),
        ):
            result = runner.invoke(app, ["ui"])
        assert result.exit_code == 1
        assert "Could not locate UI app" in result.output
