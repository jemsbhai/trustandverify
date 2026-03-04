"""TrustGraph — Streamlit web UI for trustandverify.

Ported from velrichack ui/app.py. Replaces the subprocess + file-passing
approach with direct Python API calls via TrustAgent.
"""

from __future__ import annotations

import asyncio
import os
import sys


# ── Helpers (testable without Streamlit) ──────────────────────────────────────

def _opinion_bar(belief: float, disbelief: float, uncertainty: float) -> str:
    return (
        f'<div class="opinion-bar">'
        f'<div class="b-bar" style="width:{belief*100:.1f}%" title="Belief: {belief:.3f}"></div>'
        f'<div class="d-bar" style="width:{disbelief*100:.1f}%" title="Disbelief: {disbelief:.3f}"></div>'
        f'<div class="u-bar" style="width:{uncertainty*100:.1f}%" title="Uncertainty: {uncertainty:.3f}"></div>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#888">'
        f'<span>🟢 Belief: {belief:.3f}</span>'
        f'<span>🔴 Disbelief: {disbelief:.3f}</span>'
        f'<span>⚪ Uncertainty: {uncertainty:.3f}</span>'
        f'</div>'
    )


def _verdict_emoji(verdict: str) -> str:
    return {"supported": "✅", "contested": "⚠️", "refuted": "❌"}.get(verdict, "❓")


def _check_env() -> tuple[bool, list[str]]:
    missing = []
    if not os.environ.get("TAVILY_API_KEY"):
        missing.append("TAVILY_API_KEY")
    if not os.environ.get("GEMINI_API_KEY"):
        missing.append("GEMINI_API_KEY")
    return len(missing) == 0, missing


def _run_agent(query: str, num_claims: int) -> object:
    """Run TrustAgent synchronously (Streamlit is not async-native)."""
    from trustandverify.core.agent import TrustAgent
    from trustandverify.core.config import TrustConfig
    from trustandverify.llm.gemini import GeminiBackend
    from trustandverify.search.tavily import TavilySearch

    agent = TrustAgent(
        config=TrustConfig(num_claims=num_claims),
        search=TavilySearch(),
        llm=GeminiBackend(),
    )
    return asyncio.run(agent.verify(query, verbose=False))


# ── Streamlit runtime (only runs under `streamlit run`) ──────────────────────

def _streamlit_main() -> None:  # pragma: no cover
    import streamlit as st

    st.set_page_config(
        page_title="TrustGraph",
        page_icon="🔍",
        layout="wide",
    )

    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem; font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0;
        }
        .sub-header { font-size: 1.1rem; color: #888; margin-top: -10px; margin-bottom: 30px; }
        .opinion-bar { height: 24px; border-radius: 12px; display: flex;
                       overflow: hidden; margin: 8px 0; }
        .b-bar  { background: #00c853; }
        .d-bar  { background: #ff1744; }
        .u-bar  { background: #e0e0e0; }
        .metric-box { text-align: center; padding: 15px; border-radius: 12px; background: #f8f9fa; }
        .metric-value { font-size: 2rem; font-weight: 700; }
        .metric-label { font-size: 0.85rem; color: #888; }
        .source-chip { display: inline-block; background: #e8eaf6; padding: 4px 10px;
                       border-radius: 8px; font-size: 0.8rem; margin: 2px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">🔍 TrustGraph</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Agentic Knowledge Verification · '
        'Subjective Logic Confidence Algebra · trustandverify</div>',
        unsafe_allow_html=True,
    )

    # ── API key check ──
    env_ok, missing_keys = _check_env()
    if not env_ok:
        st.error(
            f"Missing environment variables: **{', '.join(missing_keys)}**\n\n"
            "Set them before launching the UI:\n"
            "```\nexport TAVILY_API_KEY=...\nexport GEMINI_API_KEY=...\n```"
        )
        st.stop()

    # ── Input ──
    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        query = st.text_input(
            "Research question or claim:",
            placeholder="e.g., Is intermittent fasting effective for weight loss?",
            key="query_input",
        )
    with col2:
        num_claims = st.slider(
            "Claims", min_value=0, max_value=8, value=0,
            help="0 = auto (3–5). Use 2–3 for speed, 6–8 for depth.",
        )
    with col3:
        st.write("")
        st.write("")
        run_clicked = st.button("🚀 Verify", type="primary", use_container_width=True)

    # Example queries
    st.markdown("**Try:**")
    ex_cols = st.columns(3)
    examples = [
        "Is remote work more productive than office work?",
        "Does intermittent fasting help with weight loss?",
        "Is nuclear energy safer than solar energy?",
    ]
    for i, ex in enumerate(examples):
        with ex_cols[i]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state["selected_example"] = ex
                st.rerun()

    if "selected_example" in st.session_state:
        query = st.session_state.pop("selected_example")
        run_clicked = True

    st.divider()

    # ── Run ──
    if run_clicked and query:
        with st.spinner("Agent running — searching the web and scoring evidence…"):
            try:
                report = _run_agent(query, num_claims)
            except Exception as e:  # noqa: BLE001
                st.error(f"Agent failed: {e}")
                st.stop()

        # ── Metrics ──
        st.markdown("### 📋 Verification Report")
        claims = report.claims
        supported = sum(1 for c in claims if c.verdict.value == "supported")
        contested = sum(1 for c in claims if c.verdict.value == "contested")
        refuted   = sum(1 for c in claims if c.verdict.value == "refuted")

        m1, m2, m3, m4 = st.columns(4)
        for col, val, label, colour in [
            (m1, len(claims), "Claims Verified", "#1a1a1a"),
            (m2, supported,  "Supported",        "#00c853"),
            (m3, contested,  "Contested",         "#ff9100"),
            (m4, refuted,    "Refuted",           "#ff1744"),
        ]:
            with col:
                st.markdown(
                    f'<div class="metric-box">'
                    f'<div class="metric-value" style="color:{colour}">{val}</div>'
                    f'<div class="metric-label">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("")

        if report.summary:
            st.info(f"**Executive Summary:** {report.summary}")

        st.markdown("")

        # ── Claims detail ──
        st.markdown("### 🎯 Claims Analysis")
        for i, claim in enumerate(report.claims, 1):
            op = claim.opinion
            p = op.projected_probability() if op else 0.5
            verdict = claim.verdict.value
            emoji = _verdict_emoji(verdict)

            with st.expander(
                f"{emoji} Claim {i}: {claim.text[:80]}{'…' if len(claim.text) > 80 else ''} "
                f"— **P={p:.3f}** ({verdict})",
                expanded=(i == 1),
            ):
                st.markdown(f"**Full claim:** {claim.text}")

                if op:
                    st.markdown(
                        _opinion_bar(op.belief, op.disbelief, op.uncertainty),
                        unsafe_allow_html=True,
                    )

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Projected probability", f"{p:.3f}")
                with c2:
                    st.metric("Verdict", verdict.upper())

                if claim.assessment:
                    st.markdown(f"**Assessment:** {claim.assessment}")

                if claim.evidence:
                    st.markdown("**Sources:**")
                    for ev in claim.evidence:
                        icon = "✅" if ev.supports_claim else "❌"
                        label = "supports" if ev.supports_claim else "contradicts"
                        st.markdown(
                            f'<span class="source-chip">{icon} '
                            f'<a href="{ev.source.url}">{ev.source.title}</a> '
                            f'(trust: {ev.source.trust_score:.2f}, {label})</span>',
                            unsafe_allow_html=True,
                        )

        # ── Conflicts ──
        if report.conflicts:
            st.markdown("### ⚡ Evidence Conflicts")
            for c in report.conflicts:
                st.warning(
                    f"**{c.claim_text}** — conflict degree: {c.conflict_degree:.3f} "
                    f"({c.num_supporting} supporting vs {c.num_contradicting} contradicting)"
                )

        # ── Export ──
        st.markdown("### 📦 Export")
        from trustandverify.export.jsonld import JsonLdExporter
        from trustandverify.export.markdown import MarkdownExporter
        from trustandverify.export.html import HtmlExporter

        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            st.download_button(
                "⬇️ JSON-LD",
                data=JsonLdExporter().render(report),
                file_name="report.jsonld",
                mime="application/ld+json",
                use_container_width=True,
            )
        with ex2:
            st.download_button(
                "⬇️ Markdown",
                data=MarkdownExporter().render(report),
                file_name="report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with ex3:
            st.download_button(
                "⬇️ HTML",
                data=HtmlExporter().render(report),
                file_name="report.html",
                mime="text/html",
                use_container_width=True,
            )

        # ── Raw JSON-LD ──
        with st.expander("View JSON-LD (SPARQL / RDF / PROV-O compatible)"):
            st.json(JsonLdExporter()._build_doc(report))

    elif not query and run_clicked:
        st.warning("Please enter a research question.")

    # ── Footer ──
    st.divider()
    st.markdown(
        '<div style="text-align:center;color:#888;font-size:0.85rem">'
        "<b>TrustGraph</b> · trustandverify · "
        "Subjective Logic (Jøsang 2016) · Every confidence score is mathematically grounded."
        "</div>",
        unsafe_allow_html=True,
    )


# Auto-run when executed by Streamlit
if "streamlit" in sys.modules:  # pragma: no cover
    _streamlit_main()
