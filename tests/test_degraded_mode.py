"""Tests for /ready readiness + degraded-mode messaging (Lane 5).

No network: Chroma, circuits and the readiness probe are mocked. The TestClient
runs the app lifespan, so AWS is made optional for the suite.
"""
import os

os.environ.setdefault("FUNSTAY_REQUIRE_AWS", "false")  # lifespan secret check

from fastapi.testclient import TestClient  # noqa: E402

import app.health as health  # noqa: E402
import app.main as main  # noqa: E402
import app.voice_pipeline as vp  # noqa: E402
from app.cache import pre_rendered_decline  # noqa: E402


class FakeBreaker:
    def __init__(self, name, state):
        self._name = name
        self._state = state

    def snapshot(self):
        return {"name": self._name, "state": self._state,
                "consecutive_failures": 0, "failure_threshold": 3}


# ---------- readiness() unit ----------

def test_readiness_ready(monkeypatch):
    monkeypatch.setattr(health, "check_chroma", lambda: {"reachable": True, "documents": 42})
    monkeypatch.setattr(health, "all_circuits", lambda: {"funstay:llm:openai": FakeBreaker("x", "closed")})
    ok, body = health.readiness()
    assert ok is True
    assert body["status"] == "ready" and body["open_circuits"] == []
    assert body["dependencies"]["chroma"]["documents"] == 42


def test_readiness_degraded_when_chroma_down(monkeypatch):
    monkeypatch.setattr(health, "check_chroma", lambda: {"reachable": False, "error": "boom"})
    monkeypatch.setattr(health, "all_circuits", lambda: {})
    ok, body = health.readiness()
    assert ok is False and body["status"] == "degraded"


def test_readiness_degraded_when_circuit_open(monkeypatch):
    monkeypatch.setattr(health, "check_chroma", lambda: {"reachable": True, "documents": 1})
    monkeypatch.setattr(health, "all_circuits", lambda: {
        "funstay:llm:openai": FakeBreaker("a", "open"),
        "funstay:tts:polly": FakeBreaker("b", "closed"),
    })
    ok, body = health.readiness()
    assert ok is False and body["status"] == "degraded"
    assert body["open_circuits"] == ["funstay:llm:openai"]


# ---------- endpoints ----------

def test_health_is_liveness():
    with TestClient(main.app) as client:
        r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_ready_endpoint_200_when_ready(monkeypatch):
    monkeypatch.setattr(main, "readiness", lambda: (True, {"status": "ready"}))
    with TestClient(main.app) as client:
        r = client.get("/ready")
    assert r.status_code == 200 and r.json()["status"] == "ready"


def test_ready_endpoint_503_when_degraded(monkeypatch):
    monkeypatch.setattr(main, "readiness", lambda: (False, {"status": "degraded", "open_circuits": ["x"]}))
    with TestClient(main.app) as client:
        r = client.get("/ready")
    assert r.status_code == 503 and r.json()["status"] == "degraded"


# ---------- degraded webhook (unknown intent) ----------

def _webhook_body(intent="unknown_intent", lang="es"):
    return {"queryResult": {"intent": {"displayName": intent}, "languageCode": lang}}


def test_webhook_unknown_intent_degraded_decline(monkeypatch):
    monkeypatch.setattr(main, "any_circuit_open", lambda: True)
    with TestClient(main.app) as client:
        r = client.post("/webhook", json=_webhook_body(lang="es"))
    assert r.status_code == 200
    assert r.json()["fulfillmentText"] == pre_rendered_decline("es")


def test_webhook_unknown_intent_normal_fallback(monkeypatch):
    monkeypatch.setattr(main, "any_circuit_open", lambda: False)
    with TestClient(main.app) as client:
        r = client.post("/webhook", json=_webhook_body(lang="en"))
    assert r.status_code == 200
    assert "can't process that request" in r.json()["fulfillmentText"]


# ---------- degraded voice reply ----------

def test_handle_turn_degraded_reply(monkeypatch):
    monkeypatch.setattr(vp, "transcribe", lambda *a, **k: ("hola", "es"))
    monkeypatch.setattr(vp, "detect_intent", lambda *a, **k: ("", ""))  # no reply
    monkeypatch.setattr(vp, "synthesize", lambda text, lang: b"AUDIO")
    monkeypatch.setattr("app.providers.any_circuit_open", lambda: True)

    result = vp.handle_turn(b"x", ".webm", "s1")
    assert result["reply_text"] == pre_rendered_decline("es")


def test_handle_turn_normal_fallback(monkeypatch):
    # distinct session + a >5-char utterance so the short-clip language
    # stabilization can't inherit another test's session language.
    monkeypatch.setattr(vp, "transcribe", lambda *a, **k: ("hello there", "en"))
    monkeypatch.setattr(vp, "detect_intent", lambda *a, **k: ("", ""))
    monkeypatch.setattr(vp, "synthesize", lambda text, lang: b"AUDIO")
    monkeypatch.setattr("app.providers.any_circuit_open", lambda: False)

    result = vp.handle_turn(b"x", ".webm", "s-normal")
    assert "couldn't process that" in result["reply_text"]
