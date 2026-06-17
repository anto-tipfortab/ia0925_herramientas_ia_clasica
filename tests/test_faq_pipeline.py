"""Tests for the FAQ pipeline (scripts/generate_faq, scripts/review_faq) and
corpus invalidation. Embeddings, RAG and TTS are mocked — no network, no keys.
"""
from unittest.mock import MagicMock

import pytest

import scripts.generate_faq as gen
import scripts.ingest as ingest
import scripts.review_faq as review
from app.cache import AnswerCache, AudioCache

TENANT = "funstay"


@pytest.fixture
def cache(tmp_path):
    return AnswerCache(str(tmp_path / "answers.sqlite3"))


@pytest.fixture
def audio_cache(tmp_path):
    return AudioCache(str(tmp_path / "audio"))


def _grounded(answer):
    return lambda q, lang, topic: (f"{answer}-{lang}", True, ["manual.pdf"])


# ---------- generate_faq ----------

def test_generate_drafts_creates_bilingual_drafts(cache):
    seed = {"pool": ["q1", "q2"], "transport": ["q3"]}
    n = gen.generate_drafts(cache, seed, embed_fn=lambda q: [0.1, 0.2],
                            rag_fn=_grounded("A"), tenant_id=TENANT)
    assert n == 3
    drafts = cache.list_by_status(TENANT, "draft")
    assert len(drafts) == 3
    row = drafts[0]
    assert row["status"] == "draft"
    assert row["answer_es"] and row["answer_es"].endswith("-es")
    assert row["answer_en"] and row["answer_en"].endswith("-en")


def test_generate_drafts_skips_ungrounded(cache):
    seed = {"pool": ["q1"]}
    rag_fn = lambda q, lang, topic: ("", False, [])  # noqa: E731
    n = gen.generate_drafts(cache, seed, embed_fn=lambda q: [0.1], rag_fn=rag_fn, tenant_id=TENANT)
    assert n == 0
    assert cache.list_by_status(TENANT, "draft") == []


# ---------- review_faq.approve (pre-synthesis) ----------

def _seed_draft(cache, *, es="ES", en="EN", topic="pool", q="q"):
    cache.upsert_curated(tenant_id=TENANT, topic=topic, canonical_q=q,
                         query_embedding=[0.1, 0.2], answer_es=es, answer_en=en, status="draft")
    return cache.list_by_status(TENANT, "draft")[0]["id"]


def test_approve_presynthesizes_both_languages(cache, audio_cache):
    row_id = _seed_draft(cache)
    synth = MagicMock(return_value=b"AUDIO")
    n = review.approve(cache, row_id, audio_cache=audio_cache, synth_fn=synth, tenant_id=TENANT)
    assert n == 2  # es + en synthesized
    row = cache.get_by_id(TENANT, row_id)
    assert row["status"] == "approved"
    assert row["audio_es_path"] and row["audio_en_path"]
    # the audio files exist
    assert audio_cache.get("ES", review.VOICE_ES, "es", TENANT) == b"AUDIO"


def test_reapprove_unchanged_does_not_resynthesize(cache, audio_cache):
    row_id = _seed_draft(cache)
    synth = MagicMock(return_value=b"AUDIO")
    review.approve(cache, row_id, audio_cache=audio_cache, synth_fn=synth, tenant_id=TENANT)
    synth.reset_mock()
    n = review.approve(cache, row_id, audio_cache=audio_cache, synth_fn=synth, tenant_id=TENANT)
    assert n == 0
    synth.assert_not_called()  # text unchanged -> audio cache hit -> no re-synth


def test_edit_then_approve_resynthesizes_only_changed_language(cache, audio_cache):
    row_id = _seed_draft(cache)
    synth = MagicMock(return_value=b"AUDIO")
    review.approve(cache, row_id, audio_cache=audio_cache, synth_fn=synth, tenant_id=TENANT)
    # edit ES only -> its audio path is cleared; text hash changes
    review.edit(cache, row_id, es="ES NUEVO", tenant_id=TENANT)
    row = cache.get_by_id(TENANT, row_id)
    assert row["audio_es_path"] is None and row["answer_es"] == "ES NUEVO"

    synth.reset_mock()
    n = review.approve(cache, row_id, audio_cache=audio_cache, synth_fn=synth, tenant_id=TENANT)
    assert n == 1  # only ES re-synthesized (EN text unchanged -> cache hit)
    synth.assert_called_once()


def test_edit_requires_a_language(cache):
    row_id = _seed_draft(cache)
    with pytest.raises(ValueError):
        review.edit(cache, row_id, tenant_id=TENANT)


# ---------- corpus invalidation + regenerate ----------

def test_invalidate_topics_marks_served_rows_stale(cache):
    # an approved (served) row and a draft (not served)
    cache.upsert_curated(tenant_id=TENANT, topic="pool", canonical_q="served",
                         query_embedding=[1.0], answer_en="A", status="approved")
    cache.upsert_curated(tenant_id=TENANT, topic="pool", canonical_q="draft",
                         query_embedding=[1.0], answer_en="B", status="draft")
    n = cache.invalidate_topics(TENANT, ["pool"])
    assert n == 1  # only the served row
    stale = cache.stale_rows(TENANT)
    assert len(stale) == 1 and stale[0]["canonical_q"] == "served"
    # draft untouched
    assert len(cache.list_by_status(TENANT, "draft")) == 1


def test_regenerate_stale_returns_to_draft(cache):
    cache.upsert_curated(tenant_id=TENANT, topic="pool", canonical_q="q",
                         query_embedding=[1.0], answer_en="OLD", status="approved")
    cache.invalidate_topics(TENANT, ["pool"])
    assert len(cache.stale_rows(TENANT)) == 1

    n = gen.regenerate_stale(cache, embed_fn=lambda q: [1.0],
                             rag_fn=_grounded("NEW"), tenant_id=TENANT)
    assert n == 1
    assert cache.stale_rows(TENANT) == []
    drafts = cache.list_by_status(TENANT, "draft")
    assert len(drafts) == 1 and drafts[0]["answer_en"] == "NEW-en"


def test_regenerate_partial_keeps_stale(cache):
    # bilingual served row goes stale; regen only grounds ES -> must stay stale.
    cache.upsert_curated(tenant_id=TENANT, topic="pool", canonical_q="q",
                         query_embedding=[1.0], answer_es="OLD ES", answer_en="OLD EN",
                         status="approved")
    cache.invalidate_topics(TENANT, ["pool"])

    def es_only(q, lang, topic):
        return (f"NEW-{lang}", lang == "es", ["m.pdf"])  # only ES grounded

    n = gen.regenerate_stale(cache, embed_fn=lambda q: [1.0], rag_fn=es_only, tenant_id=TENANT)
    assert n == 0  # not fully regenerated
    assert len(cache.stale_rows(TENANT)) == 1  # still stale, not approvable
    assert cache.list_by_status(TENANT, "draft") == []


def test_ingest_invalidate_cache_for_topics(monkeypatch, cache):
    cache.upsert_curated(tenant_id=TENANT, topic="transport", canonical_q="q",
                         query_embedding=[1.0], answer_en="A", status="auto")
    monkeypatch.setattr(ingest, "get_answer_cache", lambda: cache)
    n = ingest.invalidate_cache_for_topics({"transport", "pool"}, tenant_id=TENANT)
    assert n == 1
    assert cache.stale_rows(TENANT)[0]["topic"] == "transport"
