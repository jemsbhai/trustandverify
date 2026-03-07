"""GeminiBackend — async LLM backend via LiteLLM (gemini/gemini-2.5-flash).

Ports the LLM call pattern and robust JSON parsing from parse_evidence_json()
in trustgraph.jac.
"""

from __future__ import annotations

import json
import os


class GeminiBackend:
    """LLM backend using Google Gemini via LiteLLM.

    Requires the ``GEMINI_API_KEY`` environment variable.
    Default model matches the velrichack jac.toml: gemini/gemini-2.5-flash.
    """

    name = "gemini"

    def __init__(
        self,
        model: str = "gemini/gemini-2.5-flash",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self._api_key)

    async def complete(self, prompt: str, system: str = "") -> str:
        """Send a prompt and return the completion text."""
        try:
            from litellm import acompletion  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "GeminiBackend requires litellm. Install with: pip install trustandverify[gemini]"
            ) from e

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await acompletion(
            model=self.model,
            messages=messages,
            api_key=self._api_key,
        )
        return str(response.choices[0].message.content)

    async def complete_json(
        self, prompt: str, system: str = "", defaults: dict | None = None
    ) -> dict:
        """Send a prompt and return parsed JSON with robust fallback parsing.

        Handles markdown fences, unicode whitespace, and partial JSON —
        ported directly from parse_evidence_json() in trustgraph.jac.
        """
        raw = await self.complete(prompt, system=system)
        return _parse_json_robust(raw, defaults=defaults or {})


# ── Robust JSON parsing (ported from parse_evidence_json in trustgraph.jac) ──


def _parse_json_robust(raw: str, defaults: dict | None = None) -> dict:
    """Parse an LLM response as JSON, handling common formatting artefacts.

    Strategy (same as trustgraph.jac):
        1. Direct parse.
        2. Strip markdown code fences and retry.
        3. Find first { ... } block and parse that.
        4. Return defaults if all else fails.

    Args:
        raw:      Raw LLM response string.
        defaults: Fallback dict if parsing fails.  Defaults to empty dict.

    Returns:
        Parsed dict.
    """
    if defaults is None:
        defaults = {}

    # Sanitize unicode whitespace (non-breaking space, zero-width, etc.)
    cleaned = (
        raw.replace("\xa0", " ").replace("\u200b", "").replace("\u2009", " ").replace("\u202f", " ")
    )

    # 1. Direct parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
        if isinstance(result, list):
            return {"items": result}
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Strip markdown fences (```json ... ``` or ``` ... ```)
    fence_stripped = cleaned.strip()
    if fence_stripped.startswith("```"):
        first_nl = fence_stripped.find("\n")
        if first_nl > 0:
            fence_stripped = fence_stripped[first_nl + 1 :]
        if fence_stripped.rstrip().endswith("```"):
            fence_stripped = fence_stripped.rstrip()[:-3].rstrip()
    try:
        result = json.loads(fence_stripped)
        if isinstance(result, dict):
            return result
        if isinstance(result, list):
            return {"items": result}
    except (json.JSONDecodeError, ValueError):
        pass

    # 3. Extract first { ... } block
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            result = json.loads(cleaned[start : end + 1])
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    # 4. Extract first [ ... ] block (for array responses)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start >= 0 and end > start:
        try:
            result = json.loads(cleaned[start : end + 1])
            if isinstance(result, list):
                return {"items": result}
        except (json.JSONDecodeError, ValueError):
            pass

    print(f"[GeminiBackend] Could not parse JSON. Raw[0:120]: {raw[:120]}")
    return defaults
