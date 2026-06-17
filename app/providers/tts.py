"""
TTS providers: AWS Polly (primary) + OpenAI-TTS / ElevenLabs (standby).

Polly keeps the existing neural→standard in-engine fallback; that is a
*quality* fallback within the primary. The *provider* failover (Polly → standby)
is handled one layer up by the circuit breaker + :mod:`app.providers.failover`.
"""
from __future__ import annotations

import logging
import os

from .. import config
from .base import ProviderError, ProviderUnavailable, TTSProvider
from .openai_client import get_openai_client, openai_available

log = logging.getLogger("funstay.providers.tts")

VOICE_ES = "Lucia"   # Polly es-ES neural
VOICE_EN = "Joanna"  # Polly en-US neural

_polly = None


def _polly_client():
    global _polly
    if _polly is None:
        import boto3
        from botocore.config import Config as BotoConfig

        region = os.environ.get("AWS_DEFAULT_REGION", "eu-west-1")
        _polly = boto3.client(
            "polly",
            region_name=region,
            config=BotoConfig(
                connect_timeout=config.POLLY_CONNECT_TIMEOUT,
                read_timeout=config.POLLY_READ_TIMEOUT,
                retries={"max_attempts": config.POLLY_MAX_ATTEMPTS, "mode": "standard"},
            ),
        )
    return _polly


def reset_polly_client() -> None:
    """Drop the cached Polly client (test helper)."""
    global _polly
    _polly = None


class PollyTTSProvider:
    """Primary TTS — AWS Polly neural (standard engine as in-engine fallback)."""

    name = "polly"

    def available(self) -> bool:
        # Polly is the configured primary; credential problems surface as a
        # ProviderError on call (then the breaker/standby take over).
        return True

    def synthesize(self, *, text: str, language_code: str) -> bytes:
        es = language_code.startswith("es")
        voice = VOICE_ES if es else VOICE_EN
        lang = "es-ES" if es else "en-US"
        polly = _polly_client()
        try:
            try:
                r = polly.synthesize_speech(
                    Text=text, OutputFormat="mp3", VoiceId=voice, Engine="neural", LanguageCode=lang
                )
            except Exception as e:  # neural unsupported / throttled → standard
                log.warning("Polly neural failed (%s); falling back to standard engine.", e)
                r = polly.synthesize_speech(
                    Text=text, OutputFormat="mp3", VoiceId=voice, Engine="standard"
                )
        except Exception as e:
            raise ProviderError(f"Polly synthesis failed: {e}") from e
        return bytes(r["AudioStream"].read())


class OpenAITTSProvider:
    """Standby TTS — OpenAI text-to-speech (mp3)."""

    name = "openai"

    def available(self) -> bool:
        return openai_available()

    def synthesize(self, *, text: str, language_code: str) -> bytes:
        client = get_openai_client()
        try:
            response = client.audio.speech.create(
                model=config.OPENAI_TTS_MODEL,
                voice=config.OPENAI_TTS_VOICE,
                input=text,
                response_format="mp3",
            )
            audio = response.read()
        except Exception as e:
            raise ProviderError(f"OpenAI TTS failed: {e}") from e
        return bytes(audio)


class ElevenLabsTTSProvider:
    """Standby TTS — ElevenLabs (HTTP). Used when ``TTS_STANDBY=elevenlabs``."""

    name = "elevenlabs"

    def available(self) -> bool:
        return bool(config.ELEVENLABS_API_KEY)

    def synthesize(self, *, text: str, language_code: str) -> bytes:
        if not config.ELEVENLABS_API_KEY:
            raise ProviderUnavailable("ELEVENLABS_API_KEY not set")
        import httpx

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{config.ELEVENLABS_VOICE_ID}"
        try:
            resp = httpx.post(
                url,
                headers={
                    "xi-api-key": config.ELEVENLABS_API_KEY,
                    "accept": "audio/mpeg",
                },
                json={"text": text, "model_id": config.ELEVENLABS_MODEL},
                timeout=config.ELEVENLABS_TIMEOUT,
            )
            resp.raise_for_status()
        except Exception as e:
            raise ProviderError(f"ElevenLabs TTS failed: {e}") from e
        return bytes(resp.content)


def make_tts_standby() -> TTSProvider:
    """Pick the configured TTS standby provider (resolved by the registry)."""
    if config.TTS_STANDBY == "elevenlabs":
        return ElevenLabsTTSProvider()
    return OpenAITTSProvider()
