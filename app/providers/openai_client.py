"""
Shared lazy OpenAI client.

The primary LLM, the embedding provider and the OpenAI-TTS standby all talk to
the same OpenAI account, so they share one client (built once, with the
configured timeout + retry policy). Construction is lazy and key-gated so that
importing the providers package never requires a key — only *using* a provider
does.
"""
from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI

from .. import config
from .base import ProviderUnavailable

_client: Optional[OpenAI] = None


def openai_available() -> bool:
    """True if an OpenAI API key is present in the environment."""
    return bool(os.environ.get("OPENAI_API_KEY"))


def get_openai_client() -> OpenAI:
    """Return the shared OpenAI client, building it on first use.

    The SDK applies a per-request timeout and exponential backoff between
    retries (``OPENAI_TIMEOUT`` / ``OPENAI_MAX_RETRIES``) — that is the
    "timeout + retry" step that runs *before* the circuit breaker.
    """
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ProviderUnavailable("OPENAI_API_KEY not set")
        _client = OpenAI(
            api_key=api_key,
            timeout=config.OPENAI_TIMEOUT,
            max_retries=config.OPENAI_MAX_RETRIES,
        )
    return _client


def reset_openai_client() -> None:
    """Drop the cached client (test helper so a re-keyed env is picked up)."""
    global _client
    _client = None
