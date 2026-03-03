"""Prompt templates for all LLM calls in the pipeline.

These are ported directly from the byLLM function docstrings in
trustgraph.jac.  Keeping them in one place makes it easy to tune
wording without touching pipeline logic.
"""

from __future__ import annotations


def decompose_query(question: str, num_claims: int = 0) -> str:
    """Return a prompt that asks the LLM to decompose a question into claims.

    Args:
        question:   The research question.
        num_claims: Exact number of claims requested.  0 = let the LLM
                    decide (typically 3-5).
    """
    if num_claims > 0:
        return (
            f"Given the following research question, decompose it into exactly "
            f"{num_claims} specific, verifiable factual claims. Each claim should "
            f"be a concrete statement that can be checked against evidence.\n\n"
            f"Question: {question}\n\n"
            f"Return ONLY a JSON array of {num_claims} strings. "
            f"No markdown, no extra text.\n"
            f'Example: ["Claim one here", "Claim two here"]'
        )
    return (
        "Given a research question, decompose it into 3-5 specific, verifiable "
        "factual claims. Each claim should be a concrete statement that can be "
        "checked against evidence. Return only the claims as a JSON array of strings.\n\n"
        f"Question: {question}\n\n"
        "Return ONLY a JSON array of strings. No markdown, no extra text.\n"
        'Example: ["Claim one here", "Claim two here"]'
    )


def extract_evidence(claim: str, source_text: str) -> str:
    """Return a prompt that asks the LLM to extract evidence from source text."""
    return (
        "Given a claim and text from a source, extract the key evidence.\n"
        "Return ONLY a valid JSON object (no markdown, no extra text) with these fields:\n"
        '- "evidence": the specific text that supports or contradicts the claim\n'
        '- "supports": true if the evidence supports the claim, false if it contradicts\n'
        '- "relevance": a float 0-1 indicating how directly relevant this evidence is\n'
        '- "confidence": a float 0-1 indicating how confident you are in this assessment\n\n'
        f"Claim: {claim}\n\n"
        f"Source text: {source_text}\n\n"
        'Example output: {"evidence": "Studies show X", "supports": true, '
        '"relevance": 0.8, "confidence": 0.85}'
    )


def assess_claim(
    claim: str,
    supporting: list[str],
    contradicting: list[str],
    confidence_score: float,
) -> str:
    """Return a prompt that asks the LLM to write a claim assessment."""
    sup_block = "\n".join(f"  - {e}" for e in supporting) or "  (none)"
    con_block = "\n".join(f"  - {e}" for e in contradicting) or "  (none)"
    return (
        "Given a claim and all the evidence collected for and against it, "
        "write a concise 2-3 sentence assessment summarizing the finding. "
        "Include the confidence level and note any conflicts between sources.\n\n"
        f"Claim: {claim}\n\n"
        f"Supporting evidence:\n{sup_block}\n\n"
        f"Contradicting evidence:\n{con_block}\n\n"
        f"Confidence score: {confidence_score:.3f}\n\n"
        "Write only the assessment text, no headings or JSON."
    )


def write_summary(question: str, assessed_claims: list[str]) -> str:
    """Return a prompt that asks the LLM to write an executive summary."""
    claims_block = "\n\n".join(assessed_claims)
    return (
        "Given a research question and a list of assessed claims with confidence "
        "scores, write a concise executive summary (3-5 sentences) of the overall "
        "findings. Highlight areas of agreement and disagreement among sources.\n\n"
        f"Question: {question}\n\n"
        f"Assessed claims:\n{claims_block}\n\n"
        "Write only the summary text, no headings or JSON."
    )


def claim_to_search_query(claim: str) -> str:
    """Return a prompt that asks the LLM to generate an optimised search query."""
    return (
        "Given a claim, generate a good web search query to find evidence for or "
        "against it. Return just the search query string, optimized for finding "
        "research and data. No quotes, no explanation, just the query.\n\n"
        f"Claim: {claim}"
    )
