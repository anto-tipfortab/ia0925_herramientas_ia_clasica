"""
Endpoint authentication for FunStay Concierge.

Two interchangeable, env-driven mechanisms (HMAC preferred when both are set):
  * HMAC-SHA256 over the raw request body, sent in ``WEBHOOK_SIGNATURE_HEADER``
    (optionally prefixed ``sha256=``). Used for the Dialogflow ``/webhook``.
  * A static shared secret sent in ``WEBHOOK_AUTH_HEADER``. Used where the body
    is binary/multipart (e.g. ``/voice``) and HMAC signing is impractical.

All comparisons are constant-time. When no secret is configured the request is
allowed but a loud warning is logged — unless ``WEBHOOK_AUTH_REQUIRED`` is set,
in which case the request is rejected (fail closed).
"""
import hmac
import hashlib
import logging
from typing import Mapping

from . import config

log = logging.getLogger("funstay.security")


class AuthError(Exception):
    """Raised when endpoint authentication fails (maps to HTTP 401)."""


def _check_shared_secret(headers: Mapping[str, str], secret: str) -> None:
    provided = headers.get(config.WEBHOOK_AUTH_HEADER, "")
    if not provided or not hmac.compare_digest(provided, secret):
        raise AuthError("invalid or missing shared secret")


def _check_hmac(headers: Mapping[str, str], raw_body: bytes, secret: str) -> None:
    provided = headers.get(config.WEBHOOK_SIGNATURE_HEADER, "")
    if provided.startswith("sha256="):
        provided = provided.split("=", 1)[1]
    expected = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    if not provided or not hmac.compare_digest(provided, expected):
        raise AuthError("invalid or missing HMAC signature")


def verify_request(headers: Mapping[str, str], raw_body: bytes, *, allow_hmac: bool = True) -> None:
    """Validate an inbound protected request. Raises ``AuthError`` on failure.

    ``allow_hmac=False`` forces shared-secret auth (for endpoints whose body is
    binary, so HMAC-over-body signing isn't practical for the caller).
    """
    if allow_hmac and config.WEBHOOK_HMAC_SECRET:
        _check_hmac(headers, raw_body, config.WEBHOOK_HMAC_SECRET)
        return

    if config.WEBHOOK_SHARED_SECRET:
        _check_shared_secret(headers, config.WEBHOOK_SHARED_SECRET)
        return

    # Reached here = no mechanism applied. Fail closed if auth IS configured but
    # just not usable for this endpoint (e.g. only HMAC set, but allow_hmac=False
    # for /voice) — never silently serve a protected endpoint unauthenticated.
    if config.auth_is_configured():
        raise AuthError("no applicable auth mechanism for this endpoint")

    # No secret of any kind configured.
    if config.WEBHOOK_AUTH_REQUIRED:
        raise AuthError("endpoint auth required but no secret configured")
    log.warning(
        "Endpoint auth is DISABLED (no WEBHOOK_HMAC_SECRET / WEBHOOK_SHARED_SECRET set). "
        "Set one before exposing this service publicly."
    )
