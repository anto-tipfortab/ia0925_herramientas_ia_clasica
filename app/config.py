"""
Centralized, env-driven configuration for the FunStay Concierge service.

Every operational and security knob is read from the environment with a secure
default, so the same image runs locally and in production by changing env vars
only — never code. Values are read at import time (the process env is fixed for
the life of a worker); tests may monkeypatch module attributes directly.
"""
import os
from typing import Optional


def _get_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _get_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if not val or not val.strip():
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if not val or not val.strip():
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _get_list(name: str) -> list[str]:
    val = os.environ.get(name, "")
    return [item.strip() for item in val.split(",") if item.strip()]


# ---- Endpoint authentication (HMAC preferred over shared secret) ----
WEBHOOK_HMAC_SECRET: Optional[str] = os.environ.get("WEBHOOK_HMAC_SECRET") or None
WEBHOOK_SHARED_SECRET: Optional[str] = os.environ.get("WEBHOOK_SHARED_SECRET") or None
WEBHOOK_AUTH_HEADER: str = os.environ.get("WEBHOOK_AUTH_HEADER", "X-Funstay-Token")
WEBHOOK_SIGNATURE_HEADER: str = os.environ.get("WEBHOOK_SIGNATURE_HEADER", "X-Funstay-Signature")
# Fail closed: when true, protected endpoints refuse to serve unless a secret is
# configured — prevents accidentally shipping a publicly open webhook.
WEBHOOK_AUTH_REQUIRED: bool = _get_bool("WEBHOOK_AUTH_REQUIRED", False)

# ---- CORS allowlist (empty default = no cross-origin requests allowed) ----
CORS_ALLOW_ORIGINS: list[str] = _get_list("CORS_ALLOW_ORIGINS")

# ---- Request body size limits (bytes) ----
MAX_WEBHOOK_BYTES: int = _get_int("MAX_WEBHOOK_BYTES", 256 * 1024)        # 256 KB
MAX_VOICE_BYTES: int = _get_int("MAX_VOICE_BYTES", 10 * 1024 * 1024)      # 10 MB

# ---- OpenAI client resilience (SDK does exponential backoff between retries) ----
OPENAI_TIMEOUT: float = _get_float("OPENAI_TIMEOUT", 30.0)
OPENAI_MAX_RETRIES: int = _get_int("OPENAI_MAX_RETRIES", 3)

# ---- AWS Polly client resilience (botocore standard retry mode = backoff) ----
POLLY_CONNECT_TIMEOUT: float = _get_float("POLLY_CONNECT_TIMEOUT", 5.0)
POLLY_READ_TIMEOUT: float = _get_float("POLLY_READ_TIMEOUT", 15.0)
POLLY_MAX_ATTEMPTS: int = _get_int("POLLY_MAX_ATTEMPTS", 3)


def auth_is_configured() -> bool:
    """True if any endpoint-auth secret is present in the environment."""
    return bool(WEBHOOK_HMAC_SECRET or WEBHOOK_SHARED_SECRET)
