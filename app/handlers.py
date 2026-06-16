"""
Per-intent handlers. Each receives (parameters, contexts, language, session)
and returns a Dialogflow-ES-compatible response dict.
"""
import logging
from .data import lookup_reservation, generate_door_code, open_maintenance_ticket
from .rag import answer_with_rag

log = logging.getLogger("funstay.handlers")


# ---------- response builders ----------

def text_response(text: str, output_contexts: list | None = None) -> dict:
    """Standard Dialogflow ES response with optional output contexts."""
    resp = {
        "fulfillmentText": text,
        "fulfillmentMessages": [{"text": {"text": [text]}}],
    }
    if output_contexts:
        resp["outputContexts"] = output_contexts
    return resp


def _i18n(language: str, es: str, en: str) -> str:
    return es if language.startswith("es") else en


# ---------- handlers ----------

async def handle_validar_reserva(parameters, contexts, language, session, query_text=""):
    """Intent: proporcionar_codigo_reserva
    Validate booking code against mock DB, return personalized greeting,
    and reinforce the reserva-validada context."""
    booking_code = parameters.get("codigo_reserva", "").upper()
    reservation = lookup_reservation(booking_code)

    if not reservation:
        msg = _i18n(
            language,
            f"No encuentro la reserva {booking_code} en el sistema. ¿Puedes verificarla? Debe empezar por FS seguido de seis dígitos.",
            f"I can't find booking {booking_code} in the system. Can you double-check it? It should start with FS followed by six digits.",
        )
        # Reset reserva-validada (set lifespan 0)
        reset_context = [{
            "name": f"{session}/contexts/reserva-validada",
            "lifespanCount": 0,
        }]
        return text_response(msg, output_contexts=reset_context)

    name = reservation["guest_name_localized"].get(language[:2], reservation["guest_name"])
    msg = _i18n(
        language,
        f"¡Bienvenido a FunStay, {name}! Tu reserva en {reservation['property']} está confirmada del {reservation['check_in']} al {reservation['check_out']} para {reservation['guests']} huéspedes. ¿En qué puedo ayudarte?",
        f"Welcome to FunStay, {name}! Your reservation at {reservation['property']} is confirmed from {reservation['check_in']} to {reservation['check_out']} for {reservation['guests']} guests. How can I help you?",
    )
    # Persist reserva-validada with stored guest info
    keep_context = [{
        "name": f"{session}/contexts/reserva-validada",
        "lifespanCount": 50,
        "parameters": {
            "codigo_reserva": booking_code,
            "property": reservation["property"],
            "guest_name": name,
        },
    }]
    return text_response(msg, output_contexts=keep_context)


async def handle_codigo_puerta(parameters, contexts, language, session, query_text=""):
    """Intent: solicitar_codigo_puerta
    Generate one-time door code (simulates Seam.co / Nuki API)."""
    # Try to pull property + guest from validated context
    booking = _get_context_param(contexts, "reserva-validada", "codigo_reserva")
    prop = _get_context_param(contexts, "reserva-validada", "property") or "tu propiedad"

    if not booking:
        msg = _i18n(
            language,
            "Necesito validar tu reserva primero. ¿Me das tu código de reserva?",
            "I need to validate your booking first. Can you share your booking code?",
        )
        return text_response(msg)

    code = generate_door_code()
    msg = _i18n(
        language,
        f"He generado tu código de acceso para {prop}: {code}. Es de un solo uso y se desactiva al hacer check-out. Marca el código en el teclado de la puerta y pulsa la tecla de candado.",
        f"I've generated your access code for {prop}: {code}. It's single-use and deactivates at check-out. Enter the code on the door keypad and press the lock key.",
    )
    return text_response(msg)


async def handle_pool_jacuzzi(parameters, contexts, language, session, query_text=""):
    """Intent: pool_y_jacuzzi → RAG over property manual.
    Uses the guest's actual question so the anti-hallucination guardrail can
    decline genuinely out-of-scope requests (e.g. a feature not in the manual)."""
    query = query_text or _i18n(
        language,
        "Instrucciones de la piscina y el jacuzzi: cómo calentar, encender, horarios y coste.",
        "Pool and hot tub instructions: how to heat, turn on, schedules and cost.",
    )
    answer = answer_with_rag(query, language=language, topic_filter="pool")
    return text_response(answer)


async def handle_recomendaciones(parameters, contexts, language, session, query_text=""):
    """Intent: recomendaciones_disney_orlando → RAG over Orlando area guide."""
    query = query_text or _i18n(
        language,
        "Recomendaciones de restaurantes, atracciones y supermercados cerca del área de Disney.",
        "Recommendations for restaurants, attractions, and grocery stores near the Disney area.",
    )
    answer = answer_with_rag(query, language=language, topic_filter="orlando_guide")
    return text_response(answer)


async def handle_transporte(parameters, contexts, language, session, query_text=""):
    """Intent: transporte_a_parques → RAG over transportation info."""
    query = query_text or _i18n(
        language,
        "Cómo llegar a los parques de Disney, Universal y al aeropuerto MCO. Distancias, tiempos y opciones de transporte.",
        "How to get to Disney parks, Universal, and MCO airport. Distances, times, and transportation options.",
    )
    answer = answer_with_rag(query, language=language, topic_filter="transport")
    return text_response(answer)


async def handle_incidencia(parameters, contexts, language, session, query_text=""):
    """Intent: reportar_incidencia
    Open a maintenance ticket and notify Mike (simulated)."""
    booking = _get_context_param(contexts, "reserva-validada", "codigo_reserva") or "unknown"
    prop = _get_context_param(contexts, "reserva-validada", "property") or "the property"

    # Use the user's full utterance as the issue summary
    # (could be enriched by an LLM in production)
    issue = "Reported via voice concierge"

    ticket_id = open_maintenance_ticket(booking, issue)

    msg = _i18n(
        language,
        f"He registrado tu incidencia en {prop}. Tu ticket es {ticket_id}. ¿Confirmas que avise a Mike y al equipo de mantenimiento por WhatsApp?",
        f"I've registered your issue at {prop}. Your ticket is {ticket_id}. Do you confirm I should notify Mike and the maintenance team via WhatsApp?",
    )
    keep_context = [{
        "name": f"{session}/contexts/confirmar-incidencia",
        "lifespanCount": 2,
        "parameters": {"ticket_id": ticket_id},
    }]
    return text_response(msg, output_contexts=keep_context)


# ---------- helpers ----------

def _get_context_param(contexts: list, context_name: str, param: str):
    """Extract a parameter value from a named output context."""
    for ctx in contexts:
        if ctx.get("name", "").endswith(f"/contexts/{context_name}"):
            return ctx.get("parameters", {}).get(param)
    return None
