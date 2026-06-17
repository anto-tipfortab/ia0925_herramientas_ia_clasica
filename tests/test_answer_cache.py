"""Tests for app.cache (AnswerCache + AudioCache) and the 4-step ladder in
app.rag.answer_with_rag. Embeddings, retrieval and the LLM are mocked — no
network, no keys.
"""
from unittest.mock import MagicMock

import pytest

import app.rag as rag
from app.cache import AnswerCache, AudioCache, cosine_distance, pre_rendered_decline
from app.providers import ProviderError

TENANT = "funstay"


@pytest.fixture
def cache(tmp_path):
    return AnswerCache(str(tmp_path / "answers.sqlite3"))


# ---------- AnswerCache unit ----------

def test_curated_lookup_by_topic(cache):
    cache.upsert_curated(
        tenant_id=TENANT, topic="pool", canonical_q="pool hours",
        query_embedding=None, answer_es="ES pool", answer_en="EN pool", status="curated",
    )
    row = cache.get_curated(TENANT, "pool")
    assert row is not None
    assert row["answer_es"] == "ES pool" and row["answer_en"] == "EN pool"
    assert cache.get_curated(TENANT, "transport") is None


def test_semantic_match_within_threshold(cache):
    cache.write_through(
        tenant_id=TENANT, topic="pool", canonical_q="how warm is the pool",
        query_embedding=[1.0, 0.0, 0.0], language="en", answer="WARM", status="auto",
    )
    # near-identical direction → small distance → match
    hit = cache.semantic_match(TENANT, [0.99, 0.01, 0.0], "pool", threshold=0.25)
    assert hit is not None
    row, dist = hit
    assert row["answer_en"] == "WARM" and dist <= 0.25
    # orthogonal vector → distance 1.0 → no match
    assert cache.semantic_match(TENANT, [0.0, 1.0, 0.0], "pool", threshold=0.25) is None


def test_semantic_match_respects_topic_scope(cache):
    cache.write_through(
        tenant_id=TENANT, topic="pool", canonical_q="q", query_embedding=[1.0, 0.0],
        language="en", answer="A", status="auto",
    )
    assert cache.semantic_match(TENANT, [1.0, 0.0], "transport", threshold=0.25) is None


def test_write_through_merges_languages(cache):
    cache.write_through(tenant_id=TENANT, topic="pool", canonical_q="q",
                        query_embedding=[1.0], language="es", answer="ES")
    cache.write_through(tenant_id=TENANT, topic="pool", canonical_q="q",
                        query_embedding=[1.0], language="en", answer="EN")
    row = cache.get_curated(TENANT, "pool")  # status is 'auto', so not curated
    assert row is None
    hit = cache.semantic_match(TENANT, [1.0], "pool", threshold=0.25)
    assert hit is not None
    assert hit[0]["answer_es"] == "ES" and hit[0]["answer_en"] == "EN"


def test_draft_status_not_served(cache):
    cache.upsert_curated(tenant_id=TENANT, topic="pool", canonical_q="q",
                         query_embedding=[1.0], answer_en="DRAFT", status="draft")
    assert cache.semantic_match(TENANT, [1.0], "pool", threshold=0.25) is None


def test_cosine_distance_edge_cases():
    assert cosine_distance([1, 0], [1, 0]) == pytest.approx(0.0)
    assert cosine_distance([1, 0], [0, 1]) == pytest.approx(1.0)
    assert cosine_distance([], [1]) == 1.0          # mismatched/empty → 1.0
    assert cosine_distance([0, 0], [1, 1]) == 1.0   # zero vector → 1.0


# ---------- AudioCache unit ----------

def test_audio_cache_roundtrip(tmp_path):
    ac = AudioCache(str(tmp_path))
    assert ac.get("hola", "Lucia", "es") is None
    path = ac.put("hola", "Lucia", "es", b"MP3BYTES")
    assert path.exists()
    assert ac.get("hola", "Lucia", "es") == b"MP3BYTES"
    # key is sensitive to text/voice/lang
    assert ac.get("hola", "Joanna", "es") is None
    assert ac.get("adios", "Lucia", "es") is None


# ---------- 4-step ladder ----------

def _wire(monkeypatch, cache, *, embed=None, live=None):
    monkeypatch.setattr(rag, "get_answer_cache", lambda: cache)
    if embed is not None:
        monkeypatch.setattr(rag, "embed_text", embed)
    if live is not None:
        monkeypatch.setattr(rag, "_live_rag", live)


def test_step1_curated_hit_skips_embedding(monkeypatch, cache):
    cache.upsert_curated(tenant_id=TENANT, topic="pool", canonical_q="canon",
                         query_embedding=None, answer_en="CURATED EN", status="curated")

    def boom(_):
        raise AssertionError("embed_text must not be called on a curated hit")

    _wire(monkeypatch, cache, embed=boom)
    out = rag.answer_with_rag("anything", language="en", topic_filter="pool")
    assert out == "CURATED EN"


def test_step2_semantic_hit(monkeypatch, cache):
    cache.write_through(tenant_id=TENANT, topic="pool", canonical_q="how warm",
                        query_embedding=[1.0, 0.0], language="en", answer="SEMANTIC", status="auto")
    live = MagicMock()
    _wire(monkeypatch, cache, embed=lambda q: [1.0, 0.0], live=live)
    out = rag.answer_with_rag("is the pool warm?", language="en", topic_filter="pool")
    assert out == "SEMANTIC"
    live.assert_not_called()  # served from cache, no live RAG


def test_step3_miss_runs_live_and_writes_through(monkeypatch, cache):
    live = MagicMock(return_value=("LIVE ANSWER", True, ["manual.pdf"]))
    _wire(monkeypatch, cache, embed=lambda q: [0.0, 1.0], live=live)

    out = rag.answer_with_rag("novel question", language="en", topic_filter="pool")
    assert out == "LIVE ANSWER"
    live.assert_called_once()
    # written through as auto → now a semantic hit on the same embedding
    hit = cache.semantic_match(TENANT, [0.0, 1.0], "pool", threshold=0.25)
    assert hit is not None and hit[0]["answer_en"] == "LIVE ANSWER"
    assert hit[0]["status"] == "auto"


def test_step2_match_missing_language_merges_onto_same_row(monkeypatch, cache):
    # Stored auto row has only English; a Spanish query that matches semantically
    # must fill answer_es onto the SAME row, not create a duplicate.
    cache.write_through(tenant_id=TENANT, topic="pool", canonical_q="how warm",
                        query_embedding=[1.0, 0.0], language="en", answer="EN", status="auto")
    live = MagicMock(return_value=("ES NUEVA", True, []))
    _wire(monkeypatch, cache, embed=lambda q: [1.0, 0.0], live=live)

    out = rag.answer_with_rag("está caliente la piscina", language="es", topic_filter="pool")
    assert out == "ES NUEVA"
    # exactly one row, now bilingual
    rows = cache.candidates(TENANT, "pool")
    assert len(rows) == 1
    assert rows[0]["answer_en"] == "EN" and rows[0]["answer_es"] == "ES NUEVA"


def test_step3_guardrail_decline_not_cached(monkeypatch, cache):
    live = MagicMock(return_value=("no info", False, []))
    _wire(monkeypatch, cache, embed=lambda q: [0.5, 0.5], live=live)

    out = rag.answer_with_rag("out of scope", language="en", topic_filter="pool")
    assert out == "no info"
    # not grounded → must not be written through
    assert cache.semantic_match(TENANT, [0.5, 0.5], "pool", threshold=0.25) is None


def test_step4_decline_when_embeddings_down(monkeypatch, cache):
    def down(_):
        raise ProviderError("all embedding providers failed")

    _wire(monkeypatch, cache, embed=down, live=MagicMock())
    out = rag.answer_with_rag("question", language="es", topic_filter="pool")
    assert out == pre_rendered_decline("es")


def test_step4_decline_when_live_rag_down(monkeypatch, cache):
    def live_down(*a, **k):
        raise ProviderError("all LLM providers failed")

    _wire(monkeypatch, cache, embed=lambda q: [0.1, 0.2], live=live_down)
    out = rag.answer_with_rag("question", language="en", topic_filter="pool")
    assert out == pre_rendered_decline("en")


def test_curated_hit_resilient_to_embedding_outage(monkeypatch, cache):
    """A curated topic answer is served even when embeddings are completely down."""
    cache.upsert_curated(tenant_id=TENANT, topic="pool", canonical_q="c",
                         query_embedding=None, answer_es="CURADO", status="curated")

    def down(_):
        raise ProviderError("embeddings down")

    _wire(monkeypatch, cache, embed=down)
    assert rag.answer_with_rag("x", language="es", topic_filter="pool") == "CURADO"
