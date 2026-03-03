"""OllamaBackend — async LLM backend using local Ollama."""

from __future__ import annotations

from trustandverify.llm.gemini import _parse_json_robust


class OllamaBackend:
    """LLM backend using Ollama (local models).

    Requires Ollama running locally at http://localhost:11434.
    Install with: pip install trustandverify[ollama]
    """

    name = "ollama"

    def __init__(
        self,
        model: str = "llama3",
        host: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self._host = host

    def is_available(self) -> bool:
        """Returns True — availability depends on Ollama running locally."""
        return True

    async def complete(self, prompt: str, system: str = "") -> str:
        try:
            from ollama import AsyncClient  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "OllamaBackend requires ollama. "
                "Install with: pip install trustandverify[ollama]"
            ) from e

        client = AsyncClient(host=self._host)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await client.chat(model=self.model, messages=messages)
        return str(response["message"]["content"])

    async def complete_json(self, prompt: str, system: str = "", defaults: dict | None = None) -> dict:
        raw = await self.complete(prompt, system=system)
        return _parse_json_robust(raw, defaults=defaults or {})
