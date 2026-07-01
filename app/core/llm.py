"""
Shared LLM client.

A thin wrapper around the OpenAI Chat Completions API with a single `complete`
method. If no API key is configured, `available` is False and callers use their
own deterministic fallback so the app runs offline.
"""
from __future__ import annotations

import logging

from app.core.config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()


class LLMClient:
    def __init__(self) -> None:
        self._client = None
        if _settings.openai_api_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=_settings.openai_api_key)
            except Exception as exc:  # pragma: no cover
                logger.warning("OpenAI client init failed (%s).", exc)
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        if self._client is None:
            raise RuntimeError("LLM not configured.")
        resp = self._client.chat.completions.create(
            model=_settings.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()


_llm: LLMClient | None = None


def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm
