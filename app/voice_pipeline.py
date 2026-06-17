"""
Voice pipeline for the browser demo (app/web UI).

Reuses the same external services as the terminal client, but instead of
recording/playing locally it accepts an uploaded audio blob and returns the
synthesized reply audio:

    browser mic (webm/opus)
        -> OpenAI Whisper      (STT, auto language)
        -> Dialogflow ES       (NLU / intents / contexts — TEXT only)
        -> AWS Polly (neural)  (TTS)
        -> reply audio (mp3) back to the browser

Dialogflow only ever receives and returns TEXT, complying with the rule that
the built-in Dialogflow STT/TTS must not be used.
"""
import os
import re
import base64
import tempfile
import logging

from google.cloud import dialogflow_v2 as dialogflow
from google.api_core.client_options import ClientOptions

from .cache import get_audio_cache
from .providers import get_tts_failover
from .providers.openai_client import get_openai_client
from .providers.tts import VOICE_EN, VOICE_ES

log = logging.getLogger("funstay.voice")

PROJECT = os.environ.get("GOOGLE_PROJECT_ID")

WHISPER_MODEL = "whisper-1"

_df = None

# Per-session last detected language. Whisper mis-detects very short clips
# ("Sí", "No", "Yes") — for those we keep the session's established language.
_last_lang: dict[str, str] = {}
_SHORT_CONFIRM = {"si", "sí", "no", "yes", "sì", "vale", "ok", "okay", "claro", "nope", "yeah", "yep"}


def _oa():
    # Whisper STT reuses the shared OpenAI client: key-gated (raises
    # ProviderUnavailable when OPENAI_API_KEY is unset) and on the same
    # timeout/retry policy as the LLM/embedding providers.
    return get_openai_client()


def _df_client():
    global _df
    if _df is None:
        # quota_project_id required when authenticating with user ADC
        _df = dialogflow.SessionsClient(
            client_options=ClientOptions(quota_project_id=PROJECT)
        )
    return _df


def transcribe(audio_bytes: bytes, suffix: str = ".webm") -> tuple[str, str]:
    """Whisper STT. Returns (text, 2-letter-language)."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    with open(path, "rb") as fh:
        resp = _oa().audio.transcriptions.create(
            model=WHISPER_MODEL, file=fh, response_format="verbose_json"
        )
    text = (resp.text or "").strip()
    lang = getattr(resp, "language", "es")
    if lang in ("spanish", "español"):
        lang = "es"
    elif lang in ("english", "inglés"):
        lang = "en"
    lang = (lang or "es")[:2]
    log.info(f"[web] Whisper: '{text}' (lang={lang})")
    return text, lang


def normalize_booking_code(text: str) -> str:
    """Whisper transcribes booking codes lowercase / spaced (e.g. 'fs 123456'),
    but the Dialogflow regexp entity is case-sensitive 'FS\\d{6}'. Normalise any
    spoken booking code to the canonical 'FS123456' form so the entity matches."""
    def repl(m):
        digits = re.sub(r"\D", "", m.group(2))[:6]
        return ("FS" + digits) if len(digits) == 6 else m.group(0)
    # 'fs' / 'f s' followed by six digits possibly separated by spaces/commas
    return re.sub(r"(?i)\b(f\s?s)[\s:.\-]*((?:\d[\s,]*){6})", repl, text)


def detect_intent(text: str, language_code: str, session_id: str) -> tuple[str, str]:
    """Dialogflow detect_intent (TEXT in/out). Returns (intent_name, reply_text)."""
    text = normalize_booking_code(text)
    client = _df_client()
    session_path = client.session_path(PROJECT, session_id)
    query_input = dialogflow.QueryInput(
        text=dialogflow.TextInput(text=text, language_code=language_code)
    )
    resp = client.detect_intent(
        request={"session": session_path, "query_input": query_input}, timeout=30
    )
    qr = resp.query_result
    return qr.intent.display_name or "", (qr.fulfillment_text or "")


def synthesize(text: str, language_code: str) -> bytes:
    """TTS -> mp3 bytes, audio-cache first then the provider failover.

    Read-through the audio cache (sha256(text|voice|lang)); on a miss, synthesize
    via the provider failover (Polly primary → standby, with its own breaker and
    neural→standard fallback) and write the result back so identical utterances —
    the common FAQ answers — are rendered once.

    Known limitation: the key uses the canonical primary voice, so audio produced
    by a standby/standard engine during an outage is cached under that key. The
    cached clip stays intelligible; voice consistency is best-effort across an
    outage window. (Lane 3 pre-synthesizes approved answers on the healthy path.)
    """
    es = language_code.startswith("es")
    voice = VOICE_ES if es else VOICE_EN
    lang = "es" if es else "en"

    cache = get_audio_cache()
    cached = cache.get(text, voice, lang)
    if cached is not None:
        return cached

    audio = get_tts_failover().synthesize(text=text, language_code=language_code)
    try:
        cache.put(text, voice, lang, audio)
    except OSError as e:  # a cache write must never break the live turn
        log.warning("[web] audio cache write failed: %s", e)
    return audio


def handle_turn(audio_bytes: bytes, suffix: str, session_id: str) -> dict:
    """Full pipeline for one spoken turn. Returns a JSON-serialisable dict."""
    text, lang = transcribe(audio_bytes, suffix=suffix)
    # Stabilise language for short confirmations that Whisper often mis-detects.
    norm = text.strip().lower().strip(".!?¡¿ ")
    if (len(norm) <= 5 or norm in _SHORT_CONFIRM) and session_id in _last_lang:
        lang = _last_lang[session_id]
    elif text:
        _last_lang[session_id] = lang
    if not text:
        msg = "No te he escuchado bien, ¿puedes repetirlo?"
        return {
            "user_text": "", "language": lang, "intent": "", "reply_text": msg,
            "audio_b64": base64.b64encode(synthesize(msg, lang)).decode(),
        }
    intent, reply = detect_intent(text, lang, session_id)
    if not reply:
        reply = "Lo siento, no he podido procesar eso." if lang.startswith("es") \
            else "Sorry, I couldn't process that."
    audio = synthesize(reply, lang)
    return {
        "user_text": text,
        "language": lang,
        "intent": intent,
        "reply_text": reply,
        "audio_b64": base64.b64encode(audio).decode(),
    }
