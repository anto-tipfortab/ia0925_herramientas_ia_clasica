"""Tests for the RAG anti-hallucination guardrail (app.rag.answer_with_rag).

The guardrail declines (no LLM call) when retrieval is empty or all distances
exceed MIN_CONFIDENCE_DISTANCE; otherwise it calls the LLM with the retrieved
context. Both retrieval and the OpenAI client are mocked — no network.
"""
from unittest.mock import MagicMock

import app.rag as rag


def _fake_openai_returning(text):
    """Build a MagicMock OpenAI client whose chat completion yields `text`."""
    client = MagicMock()
    client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content=text))
    ]
    return client


def test_declines_when_no_documents(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda *a, **k: ([], [], []))
    spy = _fake_openai_returning("SHOULD NOT BE CALLED")
    monkeypatch.setattr(rag, "_get_openai", lambda: spy)

    out = rag.answer_with_rag("¿hay piscina climatizada?", language="es")

    assert "No tengo esa información" in out
    spy.chat.completions.create.assert_not_called()  # guardrail blocked the LLM


def test_declines_when_all_distances_too_weak(monkeypatch):
    # docs exist but every distance is beyond the confidence threshold.
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **k: (["irrelevant"], [rag.MIN_CONFIDENCE_DISTANCE + 0.5], [{"source": "x"}]),
    )
    spy = _fake_openai_returning("SHOULD NOT BE CALLED")
    monkeypatch.setattr(rag, "_get_openai", lambda: spy)

    out = rag.answer_with_rag("question", language="en")

    assert "I don't have that information" in out
    spy.chat.completions.create.assert_not_called()


def test_answers_when_context_is_strong(monkeypatch):
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **k: (["The pool heater is on panel B."], [0.21], [{"source": "manual.pdf"}]),
    )
    spy = _fake_openai_returning("  Heat the pool from panel B.  ")
    monkeypatch.setattr(rag, "_get_openai", lambda: spy)

    out = rag.answer_with_rag("how do I heat the pool?", language="en")

    assert out == "Heat the pool from panel B."         # stripped LLM answer
    spy.chat.completions.create.assert_called_once()    # guardrail allowed the LLM


def test_threshold_boundary_is_inclusive_of_answering(monkeypatch):
    # Distance exactly at the threshold must NOT be declined (guardrail uses '>').
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **k: (["doc"], [rag.MIN_CONFIDENCE_DISTANCE], [{"source": "x"}]),
    )
    spy = _fake_openai_returning("Answer at boundary.")
    monkeypatch.setattr(rag, "_get_openai", lambda: spy)

    out = rag.answer_with_rag("q", language="en")

    assert out == "Answer at boundary."
    spy.chat.completions.create.assert_called_once()
