"""HTML exporter — renders a Report as a self-contained HTML page.

CSS and structure ported from velrichack ui/app.py.
"""

from __future__ import annotations

from trustandverify.core.models import Report, Verdict

_VERDICT_COLOUR = {
    Verdict.SUPPORTED: ("#00c853", "#f0fdf4"),
    Verdict.CONTESTED: ("#ff9100", "#fff8e1"),
    Verdict.REFUTED: ("#ff1744", "#fef2f2"),
    Verdict.NO_EVIDENCE: ("#9e9e9e", "#f5f5f5"),
}

_VERDICT_EMOJI = {
    Verdict.SUPPORTED: "✅",
    Verdict.CONTESTED: "⚠️",
    Verdict.REFUTED: "❌",
    Verdict.NO_EVIDENCE: "❓",
}

_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 900px; margin: 40px auto; padding: 0 20px; color: #1a1a1a; }
h1 { font-size: 2rem; font-weight: 700;
     background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
h2 { border-bottom: 2px solid #eee; padding-bottom: 6px; margin-top: 32px; }
.meta { color: #888; font-size: 0.9rem; margin-top: -8px; margin-bottom: 24px; }
.summary { background: #f0f4ff; border-left: 4px solid #667eea;
           padding: 16px 20px; border-radius: 8px; margin-bottom: 24px; }
.metrics { display: flex; gap: 16px; margin-bottom: 24px; }
.metric { flex: 1; text-align: center; background: #f8f9fa;
          border-radius: 12px; padding: 16px; }
.metric-value { font-size: 2rem; font-weight: 700; }
.metric-label { font-size: 0.8rem; color: #888; }
.claim { border-radius: 12px; padding: 20px; margin: 12px 0; border-left: 5px solid; }
.claim h3 { margin: 0 0 12px 0; font-size: 1rem; }
.opinion-bar { height: 20px; border-radius: 10px; display: flex;
               overflow: hidden; margin: 8px 0; }
.b { background: #00c853; }
.d { background: #ff1744; }
.u { background: #e0e0e0; }
.bar-legend { display: flex; justify-content: space-between;
              font-size: 0.75rem; color: #888; margin-bottom: 8px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 0.9rem; }
th, td { text-align: left; padding: 6px 12px; border-bottom: 1px solid #eee; }
th { background: #f5f5f5; font-weight: 600; }
.source { display: inline-block; background: #e8eaf6; padding: 3px 10px;
          border-radius: 6px; font-size: 0.8rem; margin: 2px; }
.source a { color: #3949ab; text-decoration: none; }
.conflict { background: #fff3e0; border-left: 4px solid #ff9100;
            padding: 10px 16px; border-radius: 6px; margin: 8px 0;
            font-size: 0.9rem; }
footer { margin-top: 40px; padding-top: 16px; border-top: 1px solid #eee;
         font-size: 0.8rem; color: #aaa; text-align: center; }
"""


class HtmlExporter:
    """Render a Report to a self-contained HTML page."""

    format_name = "html"
    file_extension = ".html"

    def render(self, report: Report) -> str:
        return _HTML_TEMPLATE.format(
            css=_CSS,
            query=_esc(report.query),
            generated=report.created_at.strftime("%Y-%m-%d %H:%M UTC"),
            num_claims=len(report.claims),
            summary_block=_summary_block(report),
            metrics_block=_metrics_block(report),
            claims_block=_claims_block(report),
            conflicts_block=_conflicts_block(report),
        )

    def render_to_file(self, report: Report, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.render(report))


# ── Block builders ─────────────────────────────────────────────────────────────


def _summary_block(report: Report) -> str:
    if not report.summary:
        return ""
    return f'<div class="summary"><strong>Executive Summary:</strong> {_esc(report.summary)}</div>'


def _metrics_block(report: Report) -> str:
    claims = report.claims
    supported = sum(1 for c in claims if c.verdict == Verdict.SUPPORTED)
    contested = sum(1 for c in claims if c.verdict == Verdict.CONTESTED)
    refuted = sum(1 for c in claims if c.verdict == Verdict.REFUTED)
    return f"""
    <div class="metrics">
      <div class="metric">
        <div class="metric-value">{len(claims)}</div>
        <div class="metric-label">Claims Verified</div>
      </div>
      <div class="metric">
        <div class="metric-value" style="color:#00c853">{supported}</div>
        <div class="metric-label">Supported</div>
      </div>
      <div class="metric">
        <div class="metric-value" style="color:#ff9100">{contested}</div>
        <div class="metric-label">Contested</div>
      </div>
      <div class="metric">
        <div class="metric-value" style="color:#ff1744">{refuted}</div>
        <div class="metric-label">Refuted</div>
      </div>
    </div>"""


def _claims_block(report: Report) -> str:
    parts = []
    for i, claim in enumerate(report.claims, 1):
        border_colour, bg_colour = _VERDICT_COLOUR.get(claim.verdict, ("#9e9e9e", "#f5f5f5"))
        emoji = _VERDICT_EMOJI.get(claim.verdict, "❓")
        op = claim.opinion

        if op:
            p = op.projected_probability()
            b_pct = op.belief * 100
            d_pct = op.disbelief * 100
            u_pct = op.uncertainty * 100
            bar = (
                f'<div class="opinion-bar">'
                f'<div class="b" style="width:{b_pct:.1f}%"></div>'
                f'<div class="d" style="width:{d_pct:.1f}%"></div>'
                f'<div class="u" style="width:{u_pct:.1f}%"></div>'
                f"</div>"
                f'<div class="bar-legend">'
                f"<span>🟢 Belief: {op.belief:.3f}</span>"
                f"<span>🔴 Disbelief: {op.disbelief:.3f}</span>"
                f"<span>⚪ Uncertainty: {op.uncertainty:.3f}</span>"
                f"</div>"
            )
            p_str = f"{p:.3f}"
        else:
            bar = ""
            p_str = "—"

        sources_html = ""
        if claim.evidence:
            chips = []
            for ev in claim.evidence:
                icon = "✅" if ev.supports_claim else "❌"
                chips.append(
                    f'<span class="source">{icon} '
                    f'<a href="{_esc(ev.source.url)}">{_esc(ev.source.title)}</a> '
                    f"(trust: {ev.source.trust_score:.2f})</span>"
                )
            sources_html = "<p><strong>Sources:</strong><br>" + " ".join(chips) + "</p>"

        assessment_html = ""
        if claim.assessment:
            assessment_html = f"<p><em>{_esc(claim.assessment)}</em></p>"

        parts.append(f"""
        <div class="claim" style="border-left-color:{border_colour}; background:{bg_colour}">
          <h3>{emoji} Claim {i}: {_esc(claim.text)}</h3>
          <table>
            <tr><th>Verdict</th><td><strong>{claim.verdict.value.upper()}</strong></td></tr>
            <tr><th>Projected probability</th><td>{p_str}</td></tr>
          </table>
          {bar}
          {assessment_html}
          {sources_html}
        </div>""")

    return "\n".join(parts)


def _conflicts_block(report: Report) -> str:
    if not report.conflicts:
        return ""
    parts = ["<h2>⚡ Evidence Conflicts</h2>"]
    for c in report.conflicts:
        parts.append(
            f'<div class="conflict">'
            f"<strong>{_esc(c.claim_text)}</strong> — "
            f"conflict degree: {c.conflict_degree:.3f} "
            f"({c.num_supporting} supporting vs {c.num_contradicting} contradicting)"
            f"</div>"
        )
    return "\n".join(parts)


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TrustGraph Report — {query}</title>
  <style>{css}</style>
</head>
<body>
  <h1>🔍 TrustGraph</h1>
  <p class="meta">
    <strong>Query:</strong> {query}<br>
    <strong>Generated:</strong> {generated} &nbsp;|&nbsp;
    <strong>Claims:</strong> {num_claims}
  </p>
  {summary_block}
  {metrics_block}
  <h2>🎯 Claims Analysis</h2>
  {claims_block}
  {conflicts_block}
  <footer>
    Generated by
    <a href="https://github.com/jemsbhai/trustandverify">trustandverify</a>
    using Subjective Logic confidence algebra (Jøsang 2016).
  </footer>
</body>
</html>"""
