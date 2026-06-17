"""
Provider interface for external model services.

Every external service (LLM, embeddings, TTS) is reached through one of these
thin Protocols so the rest of the app never imports an SDK directly and tests
can substitute a mock with no network. Each concrete provider owns its own
timeout + retry policy (via the SDK client it wraps); the circuit breaker and
failover are layered *on top* by :mod:`app.providers.failover`.
"""
from __future__ import annotations

from typing import List, Protocol, runtime_checkable


class ProviderError(RuntimeError):
    """A provider call failed after its own timeout + retries were exhausted."""


class ProviderUnavailable(ProviderError):
    """The provider is not usable in this process (e.g. an unset standby key).

    Raised lazily on use; ``available()`` lets the failover skip it cleanly so
    a missing standby key never crashes the service.
    """


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def available(self) -> bool:
        """True if this provider is configured well enough to attempt a call."""
        ...

    def complete(self, *, system: str, user: str, temperature: float, max_tokens: int) -> str:
        """Return the model's reply text. Raise :class:`ProviderError` on failure."""
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str

    def available(self) -> bool: ...

    def embed(self, text: str) -> List[float]:
        """Return an embedding vector. Raise :class:`ProviderError` on failure."""


@runtime_checkable
class TTSProvider(Protocol):
    name: str

    def available(self) -> bool: ...

    def synthesize(self, *, text: str, language_code: str) -> bytes:
        """Return synthesized audio (mp3 bytes). Raise :class:`ProviderError` on failure."""
