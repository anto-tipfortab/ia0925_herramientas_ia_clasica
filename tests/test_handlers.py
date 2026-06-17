"""Tests for the per-intent handlers (app.handlers).

Handlers are async and return Dialogflow-ES response dicts. Data-layer helpers
and the RAG call are mocked where needed; nothing touches the network.
"""
import asyncio

import app.handlers as handlers


def run(coro):
    return asyncio.run(coro)


def _ctx(name, params):
    """Build a Dialogflow output-context entry."""
    return {"name": f"projects/p/agent/sessions/s/contexts/{name}", "parameters": params}


# ---------- handle_validar_reserva ----------

def test_valid_booking_greets_and_persists_context():
    resp = run(handlers.handle_validar_reserva(
        parameters={"codigo_reserva": "fs123456"},  # lowercase → handler upper-cases
        contexts=[], language="es", session="projects/p/agent/sessions/s",
    ))
    text = resp["fulfillmentText"]
    assert "Anto" in text and "ChampionsGate Villa" in text
    ctxs = resp["outputContexts"]
    validada = next(c for c in ctxs if c["name"].endswith("/contexts/reserva-validada"))
    assert validada["lifespanCount"] == 50
    assert validada["parameters"]["codigo_reserva"] == "FS123456"


def test_unknown_booking_is_rejected_and_context_reset():
    resp = run(handlers.handle_validar_reserva(
        parameters={"codigo_reserva": "FS999999"},
        contexts=[], language="es", session="projects/p/agent/sessions/s",
    ))
    assert "No encuentro la reserva" in resp["fulfillmentText"]
    validada = next(c for c in resp["outputContexts"] if c["name"].endswith("/contexts/reserva-validada"))
    assert validada["lifespanCount"] == 0  # reset


def test_valid_booking_english_localization():
    resp = run(handlers.handle_validar_reserva(
        parameters={"codigo_reserva": "FS123456"},
        contexts=[], language="en", session="s",
    ))
    assert "Welcome to FunStay" in resp["fulfillmentText"]


# ---------- handle_codigo_puerta ----------

def test_door_code_requires_validated_reservation():
    resp = run(handlers.handle_codigo_puerta(
        parameters={}, contexts=[], language="es", session="s",
    ))
    assert "validar tu reserva primero" in resp["fulfillmentText"]


def test_door_code_generated_with_context(monkeypatch):
    monkeypatch.setattr(handlers, "generate_door_code", lambda: "654321")
    contexts = [_ctx("reserva-validada", {"codigo_reserva": "FS123456", "property": "ChampionsGate Villa"})]
    resp = run(handlers.handle_codigo_puerta(
        parameters={}, contexts=contexts, language="es", session="s",
    ))
    text = resp["fulfillmentText"]
    assert "654321" in text and "ChampionsGate Villa" in text


# ---------- handle_incidencia ----------

def test_incidencia_opens_ticket_and_sets_confirm_context(monkeypatch):
    monkeypatch.setattr(handlers, "open_maintenance_ticket", lambda booking, issue: "TKT-12345")
    contexts = [_ctx("reserva-validada", {"codigo_reserva": "FS123456", "property": "Reunion Resort"})]
    resp = run(handlers.handle_incidencia(
        parameters={}, contexts=contexts, language="es", session="s",
    ))
    assert "TKT-12345" in resp["fulfillmentText"]
    confirm = next(c for c in resp["outputContexts"] if c["name"].endswith("/contexts/confirmar-incidencia"))
    assert confirm["parameters"]["ticket_id"] == "TKT-12345"


# ---------- RAG-backed handlers ----------

def test_pool_handler_passes_query_and_topic_to_rag(monkeypatch):
    captured = {}

    def fake_rag(query, language="es", topic_filter=None):
        captured["query"] = query
        captured["topic"] = topic_filter
        return "POOL ANSWER"

    monkeypatch.setattr(handlers, "answer_with_rag", fake_rag)
    resp = run(handlers.handle_pool_jacuzzi(
        parameters={}, contexts=[], language="en", session="s",
        query_text="how do I turn on the hot tub?",
    ))
    assert resp["fulfillmentText"] == "POOL ANSWER"
    assert captured["query"] == "how do I turn on the hot tub?"
    assert captured["topic"] == "pool"


def test_recomendaciones_handler_uses_orlando_topic(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        handlers, "answer_with_rag",
        lambda query, language="es", topic_filter=None: captured.update(topic=topic_filter) or "REC",
    )
    resp = run(handlers.handle_recomendaciones(
        parameters={}, contexts=[], language="es", session="s", query_text="",
    ))
    assert resp["fulfillmentText"] == "REC"
    assert captured["topic"] == "orlando_guide"
