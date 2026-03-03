"""trustandverify — Agentic knowledge verification using Subjective Logic.

Quick start::

    import asyncio
    from trustandverify import verify

    report = asyncio.run(verify("Is remote work more productive than office work?"))
    print(report.summary)
    for claim in report.claims:
        print(f"  [{claim.verdict}] {claim.text}")
        print(f"    P={claim.opinion.projected_probability():.3f}")
"""

from __future__ import annotations

from trustandverify._version import __version__
from trustandverify.core.agent import TrustAgent
from trustandverify.core.config import TrustConfig
from trustandverify.core.models import Claim, Conflict, Evidence, Report, Source, Verdict


async def verify(
    query: str,
    *,
    num_claims: int = 0,
    verbose: bool = False,
) -> Report:
    """One-liner verification using auto-configured backends from env vars.

    Requires ``GEMINI_API_KEY`` and ``TAVILY_API_KEY`` environment variables.

    Args:
        query:      The research question or claim to verify.
        num_claims: Number of claims to decompose into (0 = LLM decides).
        verbose:    Print step-by-step progress to stdout.

    Returns:
        A fully populated Report.

    Example::

        import asyncio
        report = asyncio.run(verify("Is coffee healthy?"))
    """
    from trustandverify.llm.gemini import GeminiBackend
    from trustandverify.search.tavily import TavilySearch

    agent = TrustAgent(
        config=TrustConfig(num_claims=num_claims),
        search=TavilySearch(),
        llm=GeminiBackend(),
    )
    return await agent.verify(query, verbose=verbose)


__all__ = [
    "__version__",
    "verify",
    "TrustAgent",
    "TrustConfig",
    "Report",
    "Claim",
    "Conflict",
    "Evidence",
    "Source",
    "Verdict",
]
