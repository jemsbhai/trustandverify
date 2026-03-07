"""AnthropicBackend — async LLM backend using the anthropic package."""

from __future__ import annotations

import os

from trustandverify.llm.gemini import _parse_json_robust


class AnthropicBackend:
    """LLM backend using Anthropic Claude.

    Requires the ``ANTHROPIC_API_KEY`` environment variable.
    Install with: pip install trustandverify[anthropic]
    """

    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self._api_key)

    async def complete(self, prompt: str, system: str = "") -> str:
        try:
            import anthropic  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "AnthropicBackend requires anthropic. "
                "Install with: pip install trustandverify[anthropic]"
            ) from e

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        kwargs: dict = {
            "model": self.model,
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = await client.messages.create(**kwargs)
        return str(response.content[0].text)

    async def complete_json(
        self, prompt: str, system: str = "", defaults: dict | None = None
    ) -> dict:
        raw = await self.complete(prompt, system=system)
        return _parse_json_robust(raw, defaults=defaults or {})
