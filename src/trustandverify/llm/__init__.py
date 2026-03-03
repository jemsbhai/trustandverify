"""trustandverify.llm — public exports."""

from trustandverify.llm.anthropic import AnthropicBackend
from trustandverify.llm.gemini import GeminiBackend
from trustandverify.llm.ollama import OllamaBackend
from trustandverify.llm.openai import OpenAIBackend
from trustandverify.llm.protocol import LLMBackend
from trustandverify.llm.prompts import (
    assess_claim,
    claim_to_search_query,
    decompose_query,
    extract_evidence,
    write_summary,
)

__all__ = [
    "LLMBackend",
    "GeminiBackend",
    "OpenAIBackend",
    "AnthropicBackend",
    "OllamaBackend",
    "decompose_query",
    "extract_evidence",
    "assess_claim",
    "write_summary",
    "claim_to_search_query",
]
