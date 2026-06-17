"""
Fail-fast startup validation for secret configuration.

Secrets are loaded **only** from environment variables — never from committed
files or hardcoded literals. This guard runs once at application startup so a
misconfigured deployment crashes immediately and visibly, instead of appearing
healthy and then failing on the first user request.

Required:
  * OPENAI_API_KEY — no ambient fallback exists; always required.
  * AWS credentials — needed by Polly (/voice). Resolved via the standard
    boto3 credential chain (env vars, shared profile, or an IAM role), so
    role-based deploys work without static keys. Set FUNSTAY_REQUIRE_AWS=false
    for a webhook-only deployment that never calls Polly.
"""
import os
import logging

log = logging.getLogger("funstay.startup")

REQUIRED_ENV_SECRETS = ("OPENAI_API_KEY",)
_AWS_ENV_HINTS = ("AWS_ACCESS_KEY_ID", "AWS_PROFILE")


class MissingSecretError(RuntimeError):
    """Raised at startup when a required secret is absent from the environment."""


def _is_blank(name: str) -> bool:
    return not (os.environ.get(name) or "").strip()


def _truthy(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _aws_credentials_present() -> bool:
    """True if AWS credentials are resolvable.

    Fast path checks the documented env vars; only when those are absent do we
    fall back to the full boto3 chain (which also covers EC2/ECS IAM roles).
    """
    if any(not _is_blank(v) for v in _AWS_ENV_HINTS):
        return True
    try:
        import boto3
        return boto3.Session().get_credentials() is not None
    except Exception:  # boto3 missing or chain errored → treat as unresolved
        return False


def validate_required_secrets() -> None:
    """Raise MissingSecretError if any required secret is not configured."""
    missing = [name for name in REQUIRED_ENV_SECRETS if _is_blank(name)]

    if _truthy("FUNSTAY_REQUIRE_AWS", True) and not _aws_credentials_present():
        missing.append("AWS credentials (AWS_ACCESS_KEY_ID, AWS_PROFILE, or an IAM role)")

    if missing:
        raise MissingSecretError(
            "Missing required secret(s): "
            + ", ".join(missing)
            + ". Provide them via environment variables (see .env.example); "
            "secrets are never read from committed files."
        )
    log.info("Startup secret check passed (all required secrets present in env).")
