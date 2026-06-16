"""
FunStay Concierge — Webhook FastAPI
Handles all Dialogflow ES fulfillment requests with intent-based routing.

Endpoints:
  POST /webhook   → main Dialogflow webhook entrypoint
  GET  /health    → health check
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse

from .startup_checks import validate_required_secrets
from .handlers import (
    handle_validar_reserva,
    handle_codigo_puerta,
    handle_pool_jacuzzi,
    handle_recomendaciones,
    handle_transporte,
    handle_incidencia,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("funstay")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail fast on a misconfigured deploy: required secrets must be in the env.
    validate_required_secrets()
    yield


app = FastAPI(title="FunStay Concierge Webhook", version="1.0.0", lifespan=lifespan)


# Map Dialogflow intent name → handler function
INTENT_HANDLERS = {
    "proporcionar_codigo_reserva": handle_validar_reserva,
    "solicitar_codigo_puerta": handle_codigo_puerta,
    "pool_y_jacuzzi": handle_pool_jacuzzi,
    "recomendaciones_disney_orlando": handle_recomendaciones,
    "transporte_a_parques": handle_transporte,
    "reportar_incidencia": handle_incidencia,
}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "FunStay Concierge Webhook"}


@app.get("/demo")
async def demo_page():
    """Browser voice-demo UI (mic → Whisper → Dialogflow → Polly)."""
    return FileResponse(STATIC_DIR / "demo.html")


@app.post("/voice")
async def voice(audio: UploadFile = File(...), session_id: str = Form("web-demo")):
    """Full spoken-turn pipeline for the browser demo.

    Receives a recorded audio blob, runs STT → NLU → TTS, and returns the
    transcript, matched intent, reply text and synthesized reply audio (base64).
    """
    from .voice_pipeline import handle_turn
    from fastapi.concurrency import run_in_threadpool

    raw = await audio.read()
    suffix = "." + (audio.filename.rsplit(".", 1)[-1] if audio.filename and "." in audio.filename else "webm")
    try:
        # The pipeline makes BLOCKING calls (Whisper, Dialogflow, Polly). Crucially,
        # Dialogflow calls back into THIS server's /webhook, so the blocking work must
        # run in a threadpool to keep the event loop free — otherwise the webhook
        # callback starves and Dialogflow times out (returning static placeholders).
        result = await run_in_threadpool(handle_turn, raw, suffix, session_id)
        return JSONResponse(result)
    except Exception as e:
        log.exception(f"/voice pipeline error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/webhook")
async def webhook(request: Request):
    """Dialogflow ES fulfillment endpoint.

    Parses the request, detects which intent fired, dispatches to the
    appropriate handler, and returns a Dialogflow-compatible response.
    """
    body = await request.json()

    # Extract intent name + parameters + language + session
    query_result = body.get("queryResult", {})
    intent_name = query_result.get("intent", {}).get("displayName", "")
    parameters = query_result.get("parameters", {})
    contexts = query_result.get("outputContexts", [])
    language = query_result.get("languageCode", "es")
    query_text = query_result.get("queryText", "")
    session = body.get("session", "")

    log.info(f"Intent: '{intent_name}' | lang: {language} | params: {parameters} | query: {query_text!r}")

    handler = INTENT_HANDLERS.get(intent_name)
    if not handler:
        # Unknown intent: return a graceful fallback
        msg = (
            "Lo siento, no puedo procesar esa petición ahora mismo."
            if language.startswith("es")
            else "Sorry, I can't process that request right now."
        )
        return JSONResponse(_text_response(msg))

    try:
        response = await handler(
            parameters=parameters,
            contexts=contexts,
            language=language,
            session=session,
            query_text=query_text,
        )
        return JSONResponse(response)
    except Exception as e:
        log.exception(f"Handler error for '{intent_name}': {e}")
        msg = (
            "Tuve un problema procesando tu petición. ¿Puedes intentarlo de nuevo?"
            if language.startswith("es")
            else "I had trouble processing your request. Can you try again?"
        )
        return JSONResponse(_text_response(msg))


def _text_response(text: str) -> dict:
    """Build a minimal Dialogflow text response payload."""
    return {"fulfillmentText": text, "fulfillmentMessages": [{"text": {"text": [text]}}]}
