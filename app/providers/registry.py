"""
Provider registry — the single place that wires concrete providers, their
per-provider circuit breakers, and the failover chains together.

Everything is keyed under :data:`app.config.TENANT_ID` so a future multi-tenant
build keys circuits/chains per tenant by parameterizing these helpers rather
than rewriting call sites. The failover objects and circuit breakers are
process singletons (built lazily) so circuit state persists across requests.
"""
from __future__ import annotations

from typing import Dict, Optional

from .. import config
from ..circuit import CircuitBreaker
from .embeddings import OpenAIEmbeddingProvider
from .failover import FailoverEmbedding, FailoverLLM, FailoverTTS
from .llm import AnthropicLLMProvider, OpenAILLMProvider
from .tts import PollyTTSProvider, make_tts_standby

_circuits: Dict[str, CircuitBreaker] = {}
_llm: Optional[FailoverLLM] = None
_embedding: Optional[FailoverEmbedding] = None
_tts: Optional[FailoverTTS] = None


def _circuit(name: str) -> CircuitBreaker:
    """Get-or-create the named breaker (namespaced by tenant)."""
    key = f"{config.TENANT_ID}:{name}"
    if key not in _circuits:
        _circuits[key] = CircuitBreaker(
            key,
            failure_threshold=config.CIRCUIT_FAILURE_THRESHOLD,
            cooldown=config.CIRCUIT_COOLDOWN_SECONDS,
        )
    return _circuits[key]


def get_circuit(name: str) -> CircuitBreaker:
    """Get-or-create a named circuit breaker in the shared registry (tenant-scoped).

    Public so non-provider subsystems (e.g. the NLU failover) register their
    breaker in the same dict that :func:`all_circuits` reports to /ready.
    """
    return _circuit(name)


def get_llm_failover() -> FailoverLLM:
    global _llm
    if _llm is None:
        _llm = FailoverLLM([
            (OpenAILLMProvider(), _circuit("llm:openai")),
            (AnthropicLLMProvider(), _circuit("llm:anthropic")),
        ])
    return _llm


def get_embedding_failover() -> FailoverEmbedding:
    global _embedding
    if _embedding is None:
        _embedding = FailoverEmbedding([
            (OpenAIEmbeddingProvider(), _circuit("embed:openai")),
        ])
    return _embedding


def get_tts_failover() -> FailoverTTS:
    global _tts
    if _tts is None:
        standby = make_tts_standby()
        _tts = FailoverTTS([
            (PollyTTSProvider(), _circuit("tts:polly")),
            (standby, _circuit(f"tts:{standby.name}")),
        ])
    return _tts


def all_circuits() -> Dict[str, CircuitBreaker]:
    """Every breaker created so far — used by the /ready probe (Lane 5)."""
    # Ensure the standard chains exist so a fresh process reports them.
    get_llm_failover()
    get_embedding_failover()
    get_tts_failover()
    return dict(_circuits)


def reset() -> None:
    """Drop cached failover chains, breakers and SDK clients (test helper).

    Also clears the cached OpenAI/Polly clients so a re-keyed test environment
    is honoured on the next call instead of reusing a stale client.
    """
    global _llm, _embedding, _tts
    from .openai_client import reset_openai_client
    from .tts import reset_polly_client

    _circuits.clear()
    _llm = _embedding = _tts = None
    reset_openai_client()
    reset_polly_client()
