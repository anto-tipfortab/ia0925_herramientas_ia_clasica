"""
FunStay Concierge — Webhook FastAPI
Handles all Dialogflow ES fulfillment requests with intent-based routing.

Endpoints:
  POST /webhook   → main Dialogflow webhook entrypoint
  GET  /health    → health check
"""
import logging
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from . import config
from .security import verify_request, AuthError
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

app = FastAPI(title="FunStay Concierge Webhook", version="1.0.0")

def _max_bytes_for_path(path: str) -> int:
    """Per-path request body ceiling (bytes). Read live from config so the
    middleware pre-check and the in-handler check always agree. Anything other
    than /voice is held to the small webhook limit."""
    if path == "/voice":
        return config.MAX_VOICE_BYTES
    return config.MAX_WEBHOOK_BYTES


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """Reject oversized requests early via the Content-Length header (cheap
    pre-check before the body is read). Handlers re-check the actual byte count
    to defend against a missing/spoofed Content-Length."""

    async def dispatch(self, request: Request, call_next):
        limit = _max_bytes_for_path(request.url.path)
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > limit:
                    return JSONResponse({"error": "request too large"}, status_code=413)
            except ValueError:
                return JSONResponse({"error": "invalid content-length"}, status_code=400)
        return await call_next(request)


app.add_middleware(MaxBodySizeMiddleware)

# CORS allowlist — empty by default (no cross-origin requests permitted).
if config.CORS_ALLOW_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ALLOW_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


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
async def voice(request: Request, audio: UploadFile = File(...), session_id: str = Form("web-demo")):
    """Full spoken-turn pipeline for the browser demo.

    Receives a recorded audio blob, runs STT → NLU → TTS, and returns the
    transcript, matched intent, reply text and synthesized reply audio (base64).
    """
    from .voice_pipeline import handle_turn
    from fastapi.concurrency import run_in_threadpool

    # Binary multipart body → shared-secret auth (HMAC-over-body isn't practical
    # for the caller here).
    try:
        verify_request(request.headers, b"", allow_hmac=False)
    except AuthError as e:
        log.warning(f"/voice auth rejected: {e}")
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    raw = await audio.read()
    if len(raw) > config.MAX_VOICE_BYTES:
        return JSONResponse({"error": "request too large"}, status_code=413)
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
    raw = await request.body()
    if len(raw) > config.MAX_WEBHOOK_BYTES:
        return JSONResponse({"error": "request too large"}, status_code=413)

    # Authenticate before doing any work (HMAC over the exact bytes received).
    try:
        verify_request(request.headers, raw, allow_hmac=True)
    except AuthError as e:
        log.warning(f"/webhook auth rejected: {e}")
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    try:
        body = await request.json()
    except ValueError:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

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
