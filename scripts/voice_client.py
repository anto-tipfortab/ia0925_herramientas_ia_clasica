"""
FunStay Concierge — Voice Client (end-to-end voice loop)

Flow per turn:
  1. Microphone → Azure Speech STT → text
  2. Text → Dialogflow ES detect_intent (via google-cloud-dialogflow) → response text
  3. Response text → Azure Speech TTS → audio playback

Language is auto-detected by Azure (es-ES + en-US) and kept consistent
with Dialogflow's language code.

Usage:
  source .env (or export the variables manually)
  python -m scripts.voice_client

Env vars required:
  GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
  GOOGLE_PROJECT_ID=<your-gcp-project-id>
  AZURE_SPEECH_KEY=<your-azure-speech-key>
  AZURE_SPEECH_REGION=<your-azure-region, e.g. westeurope>
  DIALOGFLOW_SESSION_ID=funstay-demo-session  (any string, identifies the session)
"""
import os
import sys
import uuid
import logging

import azure.cognitiveservices.speech as speechsdk
from google.cloud import dialogflow_v2 as dialogflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("voice")

# ---------- Config ----------
GOOGLE_PROJECT_ID = os.environ.get("GOOGLE_PROJECT_ID")
AZURE_SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "westeurope")
SESSION_ID = os.environ.get("DIALOGFLOW_SESSION_ID", str(uuid.uuid4()))

# Spanish voice (Castilian) — natural, female; alternatives: es-ES-AlvaroNeural (male)
TTS_VOICE_ES = "es-ES-ElviraNeural"
# English voice — US female; alternatives: en-US-GuyNeural (male)
TTS_VOICE_EN = "en-US-JennyNeural"

# Auto-detection candidate languages
STT_LANGUAGES = ["es-ES", "en-US"]


def validate_config():
    missing = [v for v in ["GOOGLE_PROJECT_ID", "AZURE_SPEECH_KEY"] if not os.environ.get(v)]
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        missing.append("GOOGLE_APPLICATION_CREDENTIALS")
    if missing:
        log.error("Missing env vars: " + ", ".join(missing))
        sys.exit(1)


# ---------- Azure STT ----------

def listen_from_mic() -> tuple[str, str]:
    """Listen to mic, return (recognized_text, detected_language_code).
    Returns ("", "") if nothing is heard."""
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    # Auto-detect language
    auto_detect = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=STT_LANGUAGES)

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        auto_detect_source_language_config=auto_detect,
        audio_config=audio_config,
    )

    print("\n🎙️  Listening... (speak now)")
    result = recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        # Detected language is in property
        detected = result.properties.get(
            speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
        )
        log.info(f"STT: '{result.text}' (lang={detected})")
        return result.text, (detected or "es-ES")
    elif result.reason == speechsdk.ResultReason.NoMatch:
        log.warning("STT: no speech detected.")
        return "", ""
    elif result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        log.error(f"STT canceled: {details.reason} | {details.error_details}")
        return "", ""
    return "", ""


# ---------- Dialogflow ----------

def detect_intent_text(text: str, language_code: str) -> str:
    """Send text to Dialogflow detect_intent; return the response text."""
    # Dialogflow ES uses two-letter codes ('es', 'en') as primary
    dfl_lang = language_code.split("-")[0]  # es-ES → es

    session_client = dialogflow.SessionsClient()
    session_path = session_client.session_path(GOOGLE_PROJECT_ID, SESSION_ID)

    text_input = dialogflow.TextInput(text=text, language_code=dfl_lang)
    query_input = dialogflow.QueryInput(text=text_input)

    response = session_client.detect_intent(
        request={"session": session_path, "query_input": query_input}
    )
    qr = response.query_result
    log.info(
        f"DFL intent='{qr.intent.display_name}' confidence={qr.intent_detection_confidence:.2f}"
    )
    return qr.fulfillment_text or "(no response from Dialogflow)"


# ---------- Azure TTS ----------

def speak(text: str, language_code: str):
    """Synthesize text and play through default speaker."""
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    voice = TTS_VOICE_ES if language_code.startswith("es") else TTS_VOICE_EN
    speech_config.speech_synthesis_voice_name = voice

    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    print(f"🔊 Speaking ({voice})... {text}")
    result = synthesizer.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        log.error(f"TTS canceled: {details.reason} | {details.error_details}")


# ---------- Main loop ----------

def main():
    validate_config()
    log.info(f"Session: {SESSION_ID}")
    print("\n=== FunStay Concierge — Voice Client ===")
    print("Speak in Spanish or English. Say 'goodbye' or 'adiós' to exit.\n")

    while True:
        user_text, detected_lang = listen_from_mic()
        if not user_text:
            print("(silence — try again)")
            continue

        print(f"👤 You ({detected_lang}): {user_text}")

        # Exit phrases
        if user_text.lower().strip().rstrip(".!?") in {
            "goodbye", "adiós", "adios", "exit", "salir", "bye",
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
        speak(reply, detected_lang)


if __name__ == "__main__":
    main()
