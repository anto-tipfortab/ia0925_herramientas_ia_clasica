"""Tests for the RAG anti-hallucination guardrail (app.rag.answer_with_rag).

The guardrail declines (no LLM call) when retrieval is empty or all distances
exceed MIN_CONFIDENCE_DISTANCE; otherwise it calls the LLM with the retrieved
context. Retrieval and the LLM-provider seam (rag._complete) are mocked — no
network.
"""
from unittest.mock import MagicMock

import app.rag as rag


def test_declines_when_no_documents(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda *a, **k: ([], [], []))
    spy = MagicMock(return_value="SHOULD NOT BE CALLED")
    monkeypatch.setattr(rag, "_complete", spy)

    out = rag.answer_with_rag("¿hay piscina climatizada?", language="es")

    assert "No tengo esa información" in out
    spy.assert_not_called()  # guardrail blocked the LLM


def test_declines_when_all_distances_too_weak(monkeypatch):
    # docs exist but every distance is beyond the confidence threshold.
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **k: (["irrelevant"], [rag.MIN_CONFIDENCE_DISTANCE + 0.5], [{"source": "x"}]),
    )
    spy = MagicMock(return_value="SHOULD NOT BE CALLED")
    monkeypatch.setattr(rag, "_complete", spy)

    out = rag.answer_with_rag("question", language="en")

    assert "I don't have that information" in out
    spy.assert_not_called()


def test_answers_when_context_is_strong(monkeypatch):
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **k: (["The pool heater is on panel B."], [0.21], [{"source": "manual.pdf"}]),
    )
    spy = MagicMock(return_value="Heat the pool from panel B.")
    monkeypatch.setattr(rag, "_complete", spy)

    out = rag.answer_with_rag("how do I heat the pool?", language="en")

    assert out == "Heat the pool from panel B."
    spy.assert_called_once()    # guardrail allowed the LLM


def test_threshold_boundary_is_inclusive_of_answering(monkeypatch):
    # Distance exactly at the threshold must NOT be declined (guardrail uses '>').
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **k: (["doc"], [rag.MIN_CONFIDENCE_DISTANCE], [{"source": "x"}]),
    )
    spy = MagicMock(return_value="Answer at boundary.")
    monkeypatch.setattr(rag, "_complete", spy)

    out = rag.answer_with_rag("q", language="en")

    assert out == "Answer at boundary."
    spy.assert_called_once()


def test_complete_routes_through_llm_failover(monkeypatch):
    """rag._complete delegates to the LLM-provider failover."""
    fake = MagicMock()
    fake.complete.return_value = "  routed answer  "
    monkeypatch.setattr(rag, "get_llm_failover", lambda: fake)

    out = rag._complete("sys", "user")

    assert out == "  routed answer  "
    _, kwargs = fake.complete.call_args
    assert kwargs["system"] == "sys" and kwargs["user"] == "user"
