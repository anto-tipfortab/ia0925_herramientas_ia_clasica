"""Tests for the RAG anti-hallucination guardrail (app.rag._live_rag).

The guardrail declines (no LLM call) when retrieval is empty or all distances
exceed MIN_CONFIDENCE_DISTANCE, returning grounded=False; otherwise it calls
the LLM with the retrieved context and returns grounded=True. Retrieval and the
LLM-provider seam (rag._complete) are mocked — no network.
"""
from unittest.mock import MagicMock

import app.rag as rag


def test_declines_when_no_documents(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda *a, **k: ([], [], []))
    spy = MagicMock(return_value="SHOULD NOT BE CALLED")
    monkeypatch.setattr(rag, "_complete", spy)

    answer, grounded, refs = rag._live_rag("¿hay piscina climatizada?", "es", None)

    assert "No tengo esa información" in answer
    assert grounded is False and refs == []
    spy.assert_not_called()  # guardrail blocked the LLM


def test_declines_when_all_distances_too_weak(monkeypatch):
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **k: (["irrelevant"], [rag.MIN_CONFIDENCE_DISTANCE + 0.5], [{"source": "x"}]),
    )
    spy = MagicMock(return_value="SHOULD NOT BE CALLED")
    monkeypatch.setattr(rag, "_complete", spy)

    answer, grounded, _ = rag._live_rag("question", "en", None)

    assert "I don't have that information" in answer
    assert grounded is False
    spy.assert_not_called()


def test_answers_when_context_is_strong(monkeypatch):
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **k: (["The pool heater is on panel B."], [0.21], [{"source": "manual.pdf"}]),
    )
    spy = MagicMock(return_value="Heat the pool from panel B.")
    monkeypatch.setattr(rag, "_complete", spy)

    answer, grounded, refs = rag._live_rag("how do I heat the pool?", "en", "pool")

    assert answer == "Heat the pool from panel B."
    assert grounded is True and refs == ["manual.pdf"]
    spy.assert_called_once()


def test_threshold_boundary_is_inclusive_of_answering(monkeypatch):
    # Distance exactly at the threshold must NOT be declined (guardrail uses '>').
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **k: (["doc"], [rag.MIN_CONFIDENCE_DISTANCE], [{"source": "x"}]),
    )
    monkeypatch.setattr(rag, "_complete", MagicMock(return_value="Answer at boundary."))

    answer, grounded, _ = rag._live_rag("q", "en", None)

    assert answer == "Answer at boundary." and grounded is True


def test_complete_routes_through_llm_failover(monkeypatch):
    """rag._complete delegates to the LLM-provider failover."""
    fake = MagicMock()
    fake.complete.return_value = "  routed answer  "
    monkeypatch.setattr(rag, "get_llm_failover", lambda: fake)

    out = rag._complete("sys", "user")

    assert out == "  routed answer  "
    _, kwargs = fake.complete.call_args
    assert kwargs["system"] == "sys" and kwargs["user"] == "user"
