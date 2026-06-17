"""
LLM providers: OpenAI GPT-4o-mini (primary) + Anthropic Claude (standby).

Both expose the same :class:`~app.providers.base.LLMProvider` shape. The
Anthropic SDK is imported lazily so the package works (and the test suite runs)
even when ``anthropic`` is not installed or ``ANTHROPIC_API_KEY`` is unset — in
that case the standby simply reports ``available() == False``.
"""
from __future__ import annotations

import logging

from .. import config
from .base import ProviderError, ProviderUnavailable
from .openai_client import get_openai_client, openai_available

log = logging.getLogger("funstay.providers.llm")


class OpenAILLMProvider:
    """Primary LLM — OpenAI chat completions (gpt-4o-mini)."""

    name = "openai"

    def available(self) -> bool:
        return openai_available()

    def complete(self, *, system: str, user: str, temperature: float, max_tokens: int) -> str:
        client = get_openai_client()
        try:
            completion = client.chat.completions.create(
                model=config.OPENAI_LLM_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:  # SDK already retried; treat as a provider failure
            raise ProviderError(f"OpenAI LLM call failed: {e}") from e
        content = completion.choices[0].message.content
        if not content:
            raise ProviderError("OpenAI LLM returned empty content")
        return content.strip()


class AnthropicLLMProvider:
    """Standby LLM — Anthropic Claude (Haiku tier, matching the cheap primary).

    Sampling parameters are intentionally not forwarded: the modern Claude
    models reject ``temperature`` and the concierge does not need it on the
    fallback path. ``max_tokens`` is the only generation control we pass.
    """

    name = "anthropic"

    def available(self) -> bool:
        if not config.ANTHROPIC_API_KEY:
            return False
        try:
            import anthropic  # noqa: F401  (presence check only)
        except ImportError:
            log.warning("ANTHROPIC_API_KEY set but the 'anthropic' package is not installed")
            return False
        return True

    def _client(self):
        if not config.ANTHROPIC_API_KEY:
            raise ProviderUnavailable("ANTHROPIC_API_KEY not set")
        try:
            import anthropic
        except ImportError as e:
            raise ProviderUnavailable("anthropic package not installed") from e
        return anthropic.Anthropic(
            api_key=config.ANTHROPIC_API_KEY,
            timeout=config.ANTHROPIC_TIMEOUT,
            max_retries=config.ANTHROPIC_MAX_RETRIES,
        )

    def complete(self, *, system: str, user: str, temperature: float, max_tokens: int) -> str:
        client = self._client()
        try:
            message = client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=max(max_tokens, 1),
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as e:
            raise ProviderError(f"Anthropic LLM call failed: {e}") from e
        text = next(
            (b.text for b in message.content if getattr(b, "type", None) == "text"),
            "",
        )
        if not text:
            raise ProviderError("Anthropic LLM returned no text content")
        return text.strip()
