"""
RAG pipeline for FunStay Concierge.

- ChromaDB persistent vector store
- OpenAI text-embedding-3-small for embeddings (cheap + fast)
- GPT-4o-mini for grounded answer generation
- Per-language responses (ES + EN)
- Anti-hallucination guardrail: refuses when retrieved context is insufficient

The vector store is built offline by scripts/ingest.py. At runtime we just
load the persistent client and query.
"""
import logging
from typing import List, Optional, Tuple
from pathlib import Path

import chromadb

from . import config
from .cache import answer_column, get_answer_cache, pre_rendered_decline
from .providers import ProviderError, get_embedding_failover, get_llm_failover

log = logging.getLogger("funstay.rag")

CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"
COLLECTION_NAME = "funstay_knowledge"

# Embedding + LLM model selection now lives in app.config (OPENAI_EMBED_MODEL /
# OPENAI_LLM_MODEL) and is read by the providers — not duplicated here.

# Retrieval config
TOP_K = 4
MIN_CONFIDENCE_DISTANCE = 1.2  # cosine distance threshold; > means context too weak

# Lazy clients
_chroma = None
_collection = None


def _get_collection():
    global _chroma, _collection
    if _collection is None:
        _chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _chroma.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
    return _collection


def embed_text(text: str) -> List[float]:
    """Generate an embedding vector for a single piece of text.

    Routed through the embedding-provider failover (timeout+retry → circuit
    breaker → standby) so a backend outage degrades gracefully instead of
    raising raw SDK errors.
    """
    return get_embedding_failover().embed(text)


def _complete(system: str, user: str) -> str:
    """Grounded answer generation via the LLM-provider failover (OpenAI → Claude)."""
    return get_llm_failover().complete(
        system=system, user=user, temperature=0.3, max_tokens=250
    )


def retrieve(query: str, topic_filter: str | None = None, k: int = TOP_K,
             query_embedding: Optional[List[float]] = None):
    """Retrieve top-k chunks for a query. Optionally filter by topic metadata.

    Accepts a precomputed ``query_embedding`` so the 4-step cache ladder can
    embed the query exactly once.
    """
    collection = _get_collection()
    if query_embedding is None:
        query_embedding = embed_text(query)

    where = {"topic": topic_filter} if topic_filter else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where,
    )

    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    log.info(f"RAG retrieved {len(docs)} chunks for query '{query[:60]}...' (topic={topic_filter})")
    if distances:
        log.info(f"  distances: {[round(d, 3) for d in distances]}")

    return docs, distances, metadatas


def _guardrail_decline(language: str) -> str:
    """Anti-hallucination decline: context too weak to answer (a grounded negative)."""
    if language.startswith("es"):
        return (
            "No tengo esa información en el manual de tu propiedad. "
            "Si quieres, abro un ticket para que Mike te contacte directamente."
        )
    return (
        "I don't have that information in your property manual. "
        "If you'd like, I can open a ticket so Mike can contact you directly."
    )


def _live_rag(
    query: str,
    language: str,
    topic_filter: str | None,
    query_embedding: Optional[List[float]] = None,
) -> Tuple[str, bool, List[str]]:
    """Live retrieval + grounded generation.

    Returns ``(answer, grounded, source_refs)``. ``grounded`` is False when the
    anti-hallucination guardrail declined (too-weak context) — that answer is
    legitimate but must NOT be cached as a FAQ. Raises ProviderError if the
    embedding/LLM providers are all unavailable (the caller maps that to the
    step-4 decline).
    """
    docs, distances, metadatas = retrieve(
        query, topic_filter=topic_filter, query_embedding=query_embedding
    )

    if not docs or (distances and min(distances) > MIN_CONFIDENCE_DISTANCE):
        log.info("RAG: insufficient context, declining to answer.")
        return _guardrail_decline(language), False, []

    # Chroma can return None metadata for a chunk ingested without it — guard so
    # a missing source never raises (which would bypass the decline path).
    source_refs = sorted({(meta or {}).get("source", "unknown") for meta in metadatas})
    context_block = "\n\n".join(
        f"[Fuente: {(meta or {}).get('source', 'unknown')}]\n{doc}"
        for doc, meta in zip(docs, metadatas)
    )

    system_prompt_es = (
        "Eres el concierge virtual de FunStay, gestor de propiedades vacacionales en el área de Disney (Orlando). "
        "Respondes basándote EXCLUSIVAMENTE en el contexto proporcionado. "
        "Si la información no está en el contexto, di claramente que no la tienes. NUNCA inventes datos. "
        "Sé breve (2-4 frases máximo), cordial y directo. Estás hablando por voz, así que evita listas largas y formato markdown. "
        "Usa el español de España con tono profesional pero cercano."
    )
    system_prompt_en = (
        "You are FunStay's virtual concierge, managing vacation rentals in the Disney area (Orlando). "
        "Answer based EXCLUSIVELY on the provided context. "
        "If the information is not in the context, clearly say you don't have it. NEVER make up data. "
        "Be brief (2-4 sentences max), warm and direct. You are speaking via voice, so avoid long lists and markdown formatting. "
        "Use natural conversational English."
    )

    system = system_prompt_es if language.startswith("es") else system_prompt_en
    user = f"Contexto:\n{context_block}\n\nPregunta del huésped: {query}"

    answer = _complete(system, user)
    log.info(f"RAG answer: {answer[:100]}...")
    return answer, True, source_refs


def answer_with_rag(query: str, language: str = "es", topic_filter: str | None = None) -> str:
    """4-step answer ladder (cache-first, fail-soft):

    1. Curated exact match by topic — no embedding, so it survives an embedding outage.
    2. Semantic match against stored FAQ question embeddings (distance threshold).
    3. Miss → live RAG via the Lane-1 LLM provider, written through as status='auto'.
    4. Live RAG unavailable (providers down) → pre-rendered graceful decline.
    """
    cache = get_answer_cache()
    tenant = config.TENANT_ID

    # Step 1 — curated exact match by intent/topic (no embedding needed).
    if topic_filter:
        row = cache.get_curated(tenant, topic_filter)
        if row is not None:
            cached = row[answer_column(language)]
            if cached:
                log.info("cache: curated hit (topic=%s)", topic_filter)
                return cached

    # Steps 2–4 need the embedding/LLM providers; degrade to a decline if they are down.
    try:
        q_emb = embed_text(query)
    except ProviderError as e:
        log.warning("cache: embeddings unavailable (%s); serving decline", e)
        return pre_rendered_decline(language)

    # Step 2 — semantic match against stored FAQ questions. On a match that is
    # missing the requested language, reuse that row's key so step 3 fills the
    # other language onto the SAME row instead of creating a duplicate.
    write_topic, write_q = topic_filter, query
    match = cache.semantic_match(tenant, q_emb, topic_filter, config.CACHE_SEMANTIC_DISTANCE)
    if match is not None:
        row, dist = match
        cached = row[answer_column(language)]
        if cached:
            log.info("cache: semantic hit (topic=%s, distance=%.3f)", topic_filter, dist)
            return cached
        write_topic, write_q = row["topic"], row["canonical_q"]

    # Step 3 — live RAG, write-through on a grounded answer.
    try:
        answer, grounded, source_refs = _live_rag(query, language, topic_filter, query_embedding=q_emb)
    except ProviderError as e:
        log.warning("cache: live RAG unavailable (%s); serving decline", e)
        return pre_rendered_decline(language)

    if grounded:
        try:
            cache.write_through(
                tenant_id=tenant,
                topic=write_topic,
                canonical_q=write_q,
                query_embedding=q_emb,
                language=language,
                answer=answer,
                source_refs=source_refs,
                status="auto",
            )
        except Exception as e:  # a cache write must never break the live answer
            log.warning("cache: write-through failed (%s)", e)

    return answer
