"""OpenAIBackend — async LLM backend using the openai package."""

from __future__ import annotations

import os

from trustandverify.llm.gemini import _parse_json_robust


class OpenAIBackend:
    """LLM backend using OpenAI (gpt-4o by default).

    Requires the ``OPENAI_API_KEY`` environment variable.
    Install with: pip install trustandverify[openai]
    """

    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self._api_key)

    async def complete(self, prompt: str, system: str = "") -> str:
        try:
            from openai import AsyncOpenAI  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "OpenAIBackend requires openai. Install with: pip install trustandverify[openai]"
            ) from e

        client = AsyncOpenAI(api_key=self._api_key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return str(response.choices[0].message.content)

    async def complete_json(
        self, prompt: str, system: str = "", defaults: dict | None = None
    ) -> dict:
        raw = await self.complete(prompt, system=system)
        return _parse_json_robust(raw, defaults=defaults or {})
