"""LLMBackend protocol — any LLM provider."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    """Structural protocol for LLM providers.

    Used by the pipeline for claim decomposition, evidence extraction,
    claim assessment, and summary generation.
    """

    name: str
    model: str

    async def complete(self, prompt: str, system: str = "") -> str:
        """Send a prompt and return the completion as a string.

        Args:
            prompt: The user prompt.
            system: Optional system message.

        Returns:
            Completion text.
        """
        ...

    async def complete_json(
        self, prompt: str, system: str = "", defaults: dict | None = None
    ) -> dict:
        """Send a prompt and return parsed JSON.

        Implementations must handle LLM responses that include markdown
        fences (```json ... ```) and other formatting artefacts, stripping
        them before parsing.

        Args:
            prompt:   The user prompt.
            system:   Optional system message.
            defaults: Fallback dict returned when parsing fails.
                      Defaults to empty dict.

        Returns:
            Parsed dict.
        """
        ...

    def is_available(self) -> bool:
        """Return True if this backend is configured and ready to use."""
        ...
