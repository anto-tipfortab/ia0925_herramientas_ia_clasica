"""
Embedding providers.

Only OpenAI ``text-embedding-3-small`` is wired today (no second embedding
backend was specified), but it still goes through the circuit-breaker +
failover machinery so the call site, error type and observability are uniform
with the LLM/TTS paths. A second embedding provider can be appended to the
chain in :mod:`app.providers.registry` with no call-site change.
"""
from __future__ import annotations

from typing import List

from .. import config
from .base import ProviderError
from .openai_client import get_openai_client, openai_available


class OpenAIEmbeddingProvider:
    name = "openai"

    def available(self) -> bool:
        return openai_available()

    def embed(self, text: str) -> List[float]:
        client = get_openai_client()
        try:
            response = client.embeddings.create(model=config.OPENAI_EMBED_MODEL, input=text)
        except Exception as e:
            raise ProviderError(f"OpenAI embedding call failed: {e}") from e
        return list(response.data[0].embedding)
