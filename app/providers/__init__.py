"""
Provider interface package: config-driven LLM / embedding / TTS providers with
per-provider circuit breakers and primary→standby failover.

Public surface used by the rest of the app:

    from .providers import (
        get_llm_failover, get_embedding_failover, get_tts_failover,
    )
"""
from .base import (
    EmbeddingProvider,
    LLMProvider,
    ProviderError,
    ProviderUnavailable,
    TTSProvider,
)
from .failover import AllProvidersFailed, FailoverEmbedding, FailoverLLM, FailoverTTS
from .registry import (
    all_circuits,
    any_circuit_open,
    get_circuit,
    get_embedding_failover,
    get_llm_failover,
    get_tts_failover,
    reset,
)

__all__ = [
    "EmbeddingProvider",
    "LLMProvider",
    "TTSProvider",
    "ProviderError",
    "ProviderUnavailable",
    "AllProvidersFailed",
    "FailoverLLM",
    "FailoverEmbedding",
    "FailoverTTS",
    "get_llm_failover",
    "get_embedding_failover",
    "get_tts_failover",
    "get_circuit",
    "all_circuits",
    "any_circuit_open",
    "reset",
]
