"""
FunStay Concierge — Voice Client (AWS + OpenAI Whisper variant)

Replaces the Azure-based voice stack with:
  - OpenAI Whisper API for STT (auto-language detection, ES + EN)
  - AWS Polly for TTS (neural voices: Lucia ES / Joanna EN)
  - sounddevice for mic capture and PCM playback (cross-platform, no afplay)

Flow per turn:
  1. Microphone → record until Enter pressed → WAV temp file
  2. WAV → OpenAI Whisper → transcribed text + detected language
  3. Text → Dialogflow ES detect_intent → response text
  4. Response text → AWS Polly (PCM output) → sounddevice playback

Usage:
  source .env
  python -m scripts.voice_client_aws

Env vars required:
  GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
  GOOGLE_PROJECT_ID=<your-gcp-project-id>
  OPENAI_API_KEY=sk-...
  AWS_ACCESS_KEY_ID=AKIA...
  AWS_SECRET_ACCESS_KEY=...
  AWS_DEFAULT_REGION=eu-west-1   (or us-east-1, any region with Polly neural voices)
  DIALOGFLOW_SESSION_ID=funstay-demo-session  (optional)
"""
import os
import sys
import uuid
import time
import logging
import tempfile
import threading

import numpy as np
import sounddevice as sd
import wave

import boto3
from openai import OpenAI
from google.cloud import dialogflow_v2 as dialogflow
from google.api_core.client_options import ClientOptions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("voice")

# ---------- Config ----------
GOOGLE_PROJECT_ID = os.environ.get("GOOGLE_PROJECT_ID")
SESSION_ID = os.environ.get("DIALOGFLOW_SESSION_ID", f"funstay-{uuid.uuid4().hex[:8]}")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "eu-west-1")

# Whisper transcription model
WHISPER_MODEL = "whisper-1"

# Polly neural voices per language
POLLY_VOICE_ES = "Lucia"     # es-ES neural female (clear castilian)
POLLY_VOICE_EN = "Joanna"    # en-US neural female

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1


def validate_config():
    missing = []
    for v in ["GOOGLE_PROJECT_ID", "OPENAI_API_KEY"]:
        if not os.environ.get(v):
            missing.append(v)
    # AWS (Polly) credentials may come from static keys OR a named profile
    # (AWS_PROFILE) resolved from ~/.aws. Accept either.
    has_static_aws = os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not has_static_aws and not os.environ.get("AWS_PROFILE"):
        missing.append("AWS credentials (AWS_ACCESS_KEY_ID/SECRET or AWS_PROFILE)")
    # Auth to Dialogflow uses Application Default Credentials (gcloud auth
    # application-default login), so GOOGLE_APPLICATION_CREDENTIALS is optional.
    if missing:
        log.error("Missing env vars: " + ", ".join(missing))
        sys.exit(1)


# ---------- Audio recording ----------

def record_until_enter() -> str:
    """Record mic input until user presses Enter. Returns path to WAV temp file."""
    print("\n🎙️  Recording... press ENTER to stop.")
    frames = []
    stop_flag = threading.Event()

    def callback(indata, frame_count, time_info, status):
        if status:
            log.debug(f"audio status: {status}")
        frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        callback=callback,
    )
    with stream:
        input()  # blocks until Enter
        stop_flag.set()

    if not frames:
        return ""

    audio = np.concatenate(frames)
    duration = len(audio) / SAMPLE_RATE
    log.info(f"Recorded {duration:.1f}s ({len(audio)} samples)")

    if duration < 0.4:
        log.warning("Recording too short, discarding.")
        return ""

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    tmp.close()
    return tmp.name


# ---------- OpenAI Whisper STT ----------

_openai = None

def _get_openai():
    global _openai
    if _openai is None:
        _openai = OpenAI()
    return _openai


def transcribe(wav_path: str) -> tuple[str, str]:
    """Send WAV to Whisper. Returns (text, language_code_2letter)."""
    client = _get_openai()
    with open(wav_path, "rb") as f:
        # verbose_json gives us the detected language
        response = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=f,
            response_format="verbose_json",
        )
    text = response.text.strip()
    # Whisper returns ISO 639-1 lowercased ('es', 'en', etc.)
    lang = getattr(response, "language", "es")
    # Normalize a few aliases
    if lang in ("spanish", "español"):
        lang = "es"
    elif lang in ("english", "inglés"):
        lang = "en"
    lang = lang[:2]
    log.info(f"Whisper: '{text}' (lang={lang})")
    return text, lang


# ---------- Dialogflow ----------

def _normalize_booking_code(text: str) -> str:
    """Normalise spoken booking codes ('fs 123456') to canonical 'FS123456'
    so the case-sensitive Dialogflow regexp entity FS\\d{6} matches."""
    import re
    def repl(m):
        digits = re.sub(r"\D", "", m.group(2))[:6]
        return ("FS" + digits) if len(digits) == 6 else m.group(0)
    return re.sub(r"(?i)\b(f\s?s)[\s:.\-]*((?:\d[\s,]*){6})", repl, text)


def detect_intent_text(text: str, language_code: str) -> str:
    """Send text to Dialogflow detect_intent; return the fulfillment text."""
    text = _normalize_booking_code(text)
    # quota_project_id is required when authenticating with user ADC
    session_client = dialogflow.SessionsClient(
        client_options=ClientOptions(quota_project_id=GOOGLE_PROJECT_ID)
    )
    session_path = session_client.session_path(GOOGLE_PROJECT_ID, SESSION_ID)

    text_input = dialogflow.TextInput(text=text, language_code=language_code)
    query_input = dialogflow.QueryInput(text=text_input)

    response = session_client.detect_intent(
        request={"session": session_path, "query_input": query_input}
    )
    qr = response.query_result
    log.info(
        f"DFL intent='{qr.intent.display_name}' confidence={qr.intent_detection_confidence:.2f}"
    )
    return qr.fulfillment_text or "(no response from Dialogflow)"


# ---------- AWS Polly TTS ----------

_polly = None

def _get_polly():
    global _polly
    if _polly is None:
        _polly = boto3.client("polly", region_name=AWS_REGION)
    return _polly


def speak(text: str, language_code: str):
    """Synthesize with Polly (PCM 16kHz mono) and play through default speaker."""
    voice = POLLY_VOICE_ES if language_code.startswith("es") else POLLY_VOICE_EN
    polly = _get_polly()

    try:
        response = polly.synthesize_speech(
            Text=text,
            OutputFormat="pcm",
            VoiceId=voice,
            Engine="neural",
            SampleRate="16000",
            LanguageCode="es-ES" if language_code.startswith("es") else "en-US",
        )
    except Exception as e:
        log.error(f"Polly error with neural engine: {e}. Trying standard engine.")
        response = polly.synthesize_speech(
            Text=text,
            OutputFormat="pcm",
            VoiceId=voice,
            Engine="standard",
            SampleRate="16000",
        )

    audio_bytes = response["AudioStream"].read()
    audio = np.frombuffer(audio_bytes, dtype=np.int16)

    print(f"🔊 Speaking ({voice})...")
    sd.play(audio, samplerate=16000)
    sd.wait()


# ---------- Main loop ----------

def main():
    validate_config()
    log.info(f"Session: {SESSION_ID}")
    log.info(f"AWS region: {AWS_REGION}")
    print("\n=== FunStay Concierge — Voice Client (AWS + Whisper) ===")
    print("Speak in Spanish or English.")
    print("Press ENTER when ready to record, then ENTER again to stop.")
    print("Say 'adiós' or 'goodbye' to exit.\n")

    while True:
        input("Press ENTER to start recording (or Ctrl+C to quit)...")
        wav_path = record_until_enter()
        if not wav_path:
            print("(no audio captured — try again)")
            continue

        try:
            user_text, detected_lang = transcribe(wav_path)
        except Exception as e:
            log.exception(f"Whisper error: {e}")
            print("(transcription failed)")
            continue
        finally:
            try:
                os.unlink(wav_path)
            except Exception:
                pass

        if not user_text:
            print("(silence — try again)")
            continue

        print(f"👤 You ({detected_lang}): {user_text}")

        if user_text.lower().strip().rstrip(".!?") in {
            "goodbye", "adiós", "adios", "exit", "salir", "bye",
            "good bye", "good-bye",
        }:
            speak(
                "¡Hasta luego!" if detected_lang.startswith("es") else "Goodbye!",
                detected_lang,
            )
            break

        try:
            reply = detect_intent_text(user_text, detected_lang)
        except Exception as e:
            log.exception(f"Dialogflow error: {e}")
            reply = (
                "Lo siento, ha habido un problema técnico."
                if detected_lang.startswith("es")
                else "Sorry, there was a technical problem."
            )

        print(f"🤖 Agent: {reply}")
        try:
            speak(reply, detected_lang)
        except Exception as e:
            log.exception(f"Polly error: {e}")
            print("(TTS failed — see logs)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBye!")
