"""
Answer + audio cache for the concierge.

Two cooperating caches let the common path be served *without* a live model
call, and let the service keep answering when a provider is down:

* :class:`AnswerCache` — a SQLite table of FAQ answers (curated + auto-written).
  Lookup is a 4-step ladder (curated exact → semantic → live → decline) driven
  from :func:`app.rag.answer_with_rag`.
* :class:`AudioCache` — pre-rendered TTS audio as files keyed by
  ``sha256(text|voice|lang)`` so identical utterances are synthesized once.

Everything is namespaced by :data:`app.config.TENANT_ID` (a column on the answer
table, a subdirectory for audio) so multi-tenant is a config change, not a
migration. No network here: embeddings/LLM/TTS are passed in or reached through
the Lane-1 providers by the caller.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sqlite3
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from . import config

log = logging.getLogger("funstay.cache")

# Statuses that may be *served* to a guest (draft answers are review-only).
SERVED_STATUSES: Tuple[str, ...] = ("curated", "auto", "approved")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id     TEXT NOT NULL,
    topic         TEXT,
    canonical_q   TEXT NOT NULL,
    q_embedding   TEXT,
    answer_es     TEXT,
    answer_en     TEXT,
    audio_es_path TEXT,
    audio_en_path TEXT,
    source_refs   TEXT,
    corpus_version  TEXT,
    prompt_version  TEXT,
    status        TEXT NOT NULL DEFAULT 'auto',
    updated_at    TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_answers_key
    ON answers (tenant_id, topic, canonical_q);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine distance in [0, 2]; 0 = identical direction. Empty/zero → 1.0."""
    if not a or not b or len(a) != len(b):
        return 1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    return 1.0 - (dot / (na * nb))


def answer_column(language: str) -> str:
    return "answer_es" if language.startswith("es") else "answer_en"


def _topic_key(topic: Optional[str]) -> str:
    """Normalise a topic for the unique key. NULL != NULL in a SQLite UNIQUE
    index, which would break ON CONFLICT for topic-less rows — so store "" not
    NULL and key on it consistently."""
    return topic or ""


class AnswerCache:
    """SQLite-backed FAQ answer store."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with closing(self._conn()) as conn, conn:
            conn.executescript(_SCHEMA)

    # ---- step 1: curated exact match by topic (no embedding) ----
    def get_curated(self, tenant_id: str, topic: str) -> Optional[sqlite3.Row]:
        """Most-recent curated answer for a topic, or None. No embedding needed —
        this is the resilient fast path that works even when embeddings are down."""
        with closing(self._conn()) as conn:
            cur = conn.execute(
                "SELECT * FROM answers WHERE tenant_id=? AND topic=? AND status='curated' "
                "ORDER BY updated_at DESC LIMIT 1",
                (tenant_id, _topic_key(topic)),
            )
            return cur.fetchone()

    # ---- step 2: semantic match over stored question embeddings ----
    def candidates(self, tenant_id: str, topic: Optional[str] = None) -> List[sqlite3.Row]:
        """Servable rows that carry an embedding, optionally scoped to a topic."""
        placeholders = ",".join("?" for _ in SERVED_STATUSES)
        sql = (
            f"SELECT * FROM answers WHERE tenant_id=? AND q_embedding IS NOT NULL "
            f"AND status IN ({placeholders})"
        )
        params: List[object] = [tenant_id, *SERVED_STATUSES]
        if topic:
            sql += " AND topic=?"
            params.append(_topic_key(topic))
        with closing(self._conn()) as conn:
            return list(conn.execute(sql, params).fetchall())

    def semantic_match(
        self,
        tenant_id: str,
        query_embedding: Sequence[float],
        topic: Optional[str],
        threshold: float,
    ) -> Optional[Tuple[sqlite3.Row, float]]:
        """Nearest stored question within ``threshold`` cosine distance, or None."""
        best: Optional[Tuple[sqlite3.Row, float]] = None
        for row in self.candidates(tenant_id, topic):
            try:
                emb = json.loads(row["q_embedding"])
            except (TypeError, ValueError):
                continue
            dist = cosine_distance(query_embedding, emb)
            if best is None or dist < best[1]:
                best = (row, dist)
        if best is not None and best[1] <= threshold:
            return best
        return None

    # ---- step 3: write-through of a live answer ----
    def write_through(
        self,
        *,
        tenant_id: str,
        topic: Optional[str],
        canonical_q: str,
        query_embedding: Optional[Sequence[float]],
        language: str,
        answer: str,
        source_refs: Optional[List[str]] = None,
        status: str = "auto",
    ) -> None:
        """Insert/merge a cached answer for one language (keeps the other language).

        Atomic upsert via ON CONFLICT so concurrent write-throughs for the same
        (tenant, topic, question) merge instead of racing the unique index. The
        ``status`` is only set on INSERT — a later auto write-through never
        downgrades a curated/approved row, it just fills in the other language.
        ``col`` is a fixed literal from :func:`answer_column` (not user input).
        """
        col = answer_column(language)
        emb = json.dumps(list(query_embedding)) if query_embedding is not None else None
        refs = json.dumps(source_refs) if source_refs is not None else None
        with closing(self._conn()) as conn, conn:
            conn.execute(
                f"INSERT INTO answers (tenant_id, topic, canonical_q, q_embedding, {col}, "
                "source_refs, corpus_version, prompt_version, status, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(tenant_id, topic, canonical_q) DO UPDATE SET "
                f"{col}=excluded.{col}, "
                "q_embedding=COALESCE(excluded.q_embedding, q_embedding), "
                "source_refs=COALESCE(excluded.source_refs, source_refs), "
                "corpus_version=excluded.corpus_version, "
                "prompt_version=excluded.prompt_version, updated_at=excluded.updated_at",
                (tenant_id, _topic_key(topic), canonical_q, emb, answer, refs,
                 config.CORPUS_VERSION, config.PROMPT_VERSION, status, _now()),
            )

    def upsert_curated(
        self,
        *,
        tenant_id: str,
        topic: Optional[str],
        canonical_q: str,
        query_embedding: Optional[Sequence[float]],
        answer_es: Optional[str] = None,
        answer_en: Optional[str] = None,
        source_refs: Optional[List[str]] = None,
        status: str = "curated",
    ) -> None:
        """Insert/replace a curated or draft FAQ entry (used by Lane 3 tooling)."""
        emb = json.dumps(list(query_embedding)) if query_embedding is not None else None
        refs = json.dumps(source_refs) if source_refs is not None else None
        with closing(self._conn()) as conn, conn:
            conn.execute(
                "INSERT INTO answers (tenant_id, topic, canonical_q, q_embedding, answer_es, "
                "answer_en, source_refs, corpus_version, prompt_version, status, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(tenant_id, topic, canonical_q) DO UPDATE SET "
                "q_embedding=COALESCE(excluded.q_embedding, q_embedding), "
                "answer_es=COALESCE(excluded.answer_es, answer_es), "
                "answer_en=COALESCE(excluded.answer_en, answer_en), "
                "source_refs=COALESCE(excluded.source_refs, source_refs), "
                "status=excluded.status, updated_at=excluded.updated_at",
                (tenant_id, _topic_key(topic), canonical_q, emb, answer_es, answer_en, refs,
                 config.CORPUS_VERSION, config.PROMPT_VERSION, status, _now()),
            )

    # ---- review + lifecycle helpers (Lane 3) ----
    def list_by_status(self, tenant_id: str, status: str) -> List[sqlite3.Row]:
        with closing(self._conn()) as conn:
            return list(conn.execute(
                "SELECT * FROM answers WHERE tenant_id=? AND status=? ORDER BY topic, canonical_q",
                (tenant_id, status),
            ).fetchall())

    def get_by_id(self, tenant_id: str, row_id: int) -> Optional[sqlite3.Row]:
        with closing(self._conn()) as conn:
            return conn.execute(
                "SELECT * FROM answers WHERE tenant_id=? AND id=?", (tenant_id, row_id),
            ).fetchone()

    def set_status(self, row_id: int, status: str, tenant_id: str = config.TENANT_ID) -> None:
        with closing(self._conn()) as conn, conn:
            conn.execute("UPDATE answers SET status=?, updated_at=? WHERE id=? AND tenant_id=?",
                         (status, _now(), row_id, tenant_id))

    def update_answer(self, row_id: int, language: str, text: str,
                      tenant_id: str = config.TENANT_ID) -> None:
        """Edit one language's answer and clear its stale audio path — the audio
        will be re-synthesized on the next approval because the text (and thus the
        sha256(text|voice|lang) key) changed."""
        col = answer_column(language)
        audio_col = "audio_es_path" if language.startswith("es") else "audio_en_path"
        with closing(self._conn()) as conn, conn:
            conn.execute(
                f"UPDATE answers SET {col}=?, {audio_col}=NULL, updated_at=? WHERE id=? AND tenant_id=?",
                (text, _now(), row_id, tenant_id),
            )

    def set_audio_path(self, row_id: int, language: str, path: str,
                       tenant_id: str = config.TENANT_ID) -> None:
        audio_col = "audio_es_path" if language.startswith("es") else "audio_en_path"
        with closing(self._conn()) as conn, conn:
            conn.execute(f"UPDATE answers SET {audio_col}=?, updated_at=? WHERE id=? AND tenant_id=?",
                         (path, _now(), row_id, tenant_id))

    def invalidate_topics(self, tenant_id: str, topics: Sequence[str]) -> int:
        """Mark served answers of the given topics 'stale' (corpus changed →
        regenerate). Returns the number of rows affected. Draft rows are left
        alone (not yet served)."""
        if not topics:
            return 0
        served = ",".join("?" for _ in SERVED_STATUSES)
        topic_ph = ",".join("?" for _ in topics)
        with closing(self._conn()) as conn, conn:
            cur = conn.execute(
                f"UPDATE answers SET status='stale', updated_at=? "
                f"WHERE tenant_id=? AND status IN ({served}) AND topic IN ({topic_ph})",
                (_now(), tenant_id, *SERVED_STATUSES, *[_topic_key(t) for t in topics]),
            )
            return cur.rowcount

    def stale_rows(self, tenant_id: str) -> List[sqlite3.Row]:
        return self.list_by_status(tenant_id, "stale")


class AudioCache:
    """Pre-rendered TTS audio stored as files keyed by sha256(text|voice|lang)."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)

    @staticmethod
    def _key(text: str, voice: str, lang: str) -> str:
        return hashlib.sha256(f"{text}|{voice}|{lang}".encode()).hexdigest()

    def path_for(self, text: str, voice: str, lang: str, tenant_id: Optional[str] = None) -> Path:
        tenant = tenant_id or config.TENANT_ID
        return self.base_dir / tenant / f"{self._key(text, voice, lang)}.mp3"

    def get(self, text: str, voice: str, lang: str, tenant_id: Optional[str] = None) -> Optional[bytes]:
        path = self.path_for(text, voice, lang, tenant_id)
        if path.exists():
            return path.read_bytes()
        return None

    def put(self, text: str, voice: str, lang: str, audio: bytes,
            tenant_id: Optional[str] = None) -> Path:
        path = self.path_for(text, voice, lang, tenant_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a unique temp file then atomically rename, so concurrent
        # writers of the same utterance never corrupt each other's bytes and a
        # reader never sees a half-written file.
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(audio)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return path


def pre_rendered_decline(language: str) -> str:
    """Graceful decline when live lookups are unavailable (degraded mode / step 4)."""
    if language.startswith("es"):
        return (
            "Puedo responder preguntas frecuentes; las consultas en vivo no están "
            "disponibles en este momento. Inténtalo de nuevo en unos minutos."
        )
    return (
        "I can answer common questions; live lookups are briefly unavailable. "
        "Please try again in a few minutes."
    )


# ---- process singletons ----
_answer_cache: Optional[AnswerCache] = None
_audio_cache: Optional[AudioCache] = None


def get_answer_cache() -> AnswerCache:
    global _answer_cache
    if _answer_cache is None:
        _answer_cache = AnswerCache(config.CACHE_DB_PATH)
    return _answer_cache


def get_audio_cache() -> AudioCache:
    global _audio_cache
    if _audio_cache is None:
        _audio_cache = AudioCache(config.AUDIO_CACHE_DIR)
    return _audio_cache


def reset() -> None:
    """Drop cached singletons (test helper)."""
    global _answer_cache, _audio_cache
    _answer_cache = None
    _audio_cache = None
