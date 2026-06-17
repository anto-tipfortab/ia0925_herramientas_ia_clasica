"""
Failover wrappers.

Each wrapper holds an ordered chain of ``(provider, circuit_breaker)`` pairs:
primary first, standby(s) after. For a call it walks the chain, skipping any
provider that is unavailable (e.g. unset standby key) or whose breaker is OPEN
(fast fail), and returns the first success. If every provider fails it raises
:class:`AllProvidersFailed` carrying the per-provider reasons.

Order of defenses for one logical call:
    provider's own timeout+retry  →  circuit breaker  →  next provider
"""
from __future__ import annotations

import logging
from typing import List, Tuple

from ..circuit import CircuitBreaker, CircuitOpenError
from .base import EmbeddingProvider, LLMProvider, ProviderError, TTSProvider

log = logging.getLogger("funstay.providers.failover")

LLMChain = List[Tuple[LLMProvider, CircuitBreaker]]
EmbeddingChain = List[Tuple[EmbeddingProvider, CircuitBreaker]]
TTSChain = List[Tuple[TTSProvider, CircuitBreaker]]


class AllProvidersFailed(ProviderError):
    """No provider in the chain could serve the call."""

    def __init__(self, kind: str, errors: List[Tuple[str, str]]) -> None:
        self.kind = kind
        self.errors = errors
        detail = "; ".join(f"{name}: {reason}" for name, reason in errors) or "no providers configured"
        super().__init__(f"all {kind} providers failed ({detail})")


def _attempt(kind: str, name: str, breaker: CircuitBreaker, errors: List[Tuple[str, str]], call):
    """Run one provider through its breaker; record + swallow failures, return (ok, value)."""
    try:
        return True, breaker.call(call)
    except CircuitOpenError:
        errors.append((name, "circuit open"))
        log.warning("%s provider %s skipped: circuit open", kind, name)
    except ProviderError as e:
        errors.append((name, str(e)))
        log.warning("%s provider %s failed: %s", kind, name, e)
    except Exception as e:  # defensive: never let a provider bug escape the chain
        errors.append((name, repr(e)))
        log.warning("%s provider %s errored: %r", kind, name, e)
    return False, None


class FailoverLLM:
    def __init__(self, chain: LLMChain) -> None:
        self._chain = chain

    def complete(self, *, system: str, user: str, temperature: float, max_tokens: int) -> str:
        errors: List[Tuple[str, str]] = []
        for provider, breaker in self._chain:
            if not provider.available():
                errors.append((provider.name, "unavailable"))
                continue
            ok, value = _attempt(
                "LLM",
                provider.name,
                breaker,
                errors,
                lambda p=provider: p.complete(
                    system=system, user=user, temperature=temperature, max_tokens=max_tokens
                ),
            )
            if ok:
                return value
        raise AllProvidersFailed("LLM", errors)


class FailoverEmbedding:
    def __init__(self, chain: EmbeddingChain) -> None:
        self._chain = chain

    def embed(self, text: str) -> List[float]:
        errors: List[Tuple[str, str]] = []
        for provider, breaker in self._chain:
            if not provider.available():
                errors.append((provider.name, "unavailable"))
                continue
            ok, value = _attempt(
                "embedding", provider.name, breaker, errors, lambda p=provider: p.embed(text)
            )
            if ok:
                return value
        raise AllProvidersFailed("embedding", errors)


class FailoverTTS:
    def __init__(self, chain: TTSChain) -> None:
        self._chain = chain

    def synthesize(self, *, text: str, language_code: str) -> bytes:
        errors: List[Tuple[str, str]] = []
        for provider, breaker in self._chain:
            if not provider.available():
                errors.append((provider.name, "unavailable"))
                continue
            ok, value = _attempt(
                "TTS",
                provider.name,
                breaker,
                errors,
                lambda p=provider: p.synthesize(text=text, language_code=language_code),
            )
            if ok:
                return value
        raise AllProvidersFailed("TTS", errors)
