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
from typing import List
from pathlib import Path

import chromadb

from .providers import get_embedding_failover, get_llm_failover

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


def retrieve(query: str, topic_filter: str | None = None, k: int = TOP_K):
    """Retrieve top-k chunks for a query. Optionally filter by topic metadata."""
    collection = _get_collection()
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


def answer_with_rag(query: str, language: str = "es", topic_filter: str | None = None) -> str:
    """
    Full RAG pipeline:
    1. Retrieve top-k relevant chunks
    2. If retrieval is too weak → return "I don't have that info" (no hallucination)
    3. Else call LLM with retrieved context + strict instructions
    """
    docs, distances, metadatas = retrieve(query, topic_filter=topic_filter)

    # Anti-hallucination guardrail: if all results are weak, decline
    if not docs or (distances and min(distances) > MIN_CONFIDENCE_DISTANCE):
        log.info("RAG: insufficient context, declining to answer.")
        if language.startswith("es"):
            return (
                "No tengo esa información en el manual de tu propiedad. "
                "Si quieres, abro un ticket para que Mike te contacte directamente."
            )
        return (
            "I don't have that information in your property manual. "
            "If you'd like, I can open a ticket so Mike can contact you directly."
        )

    # Build context block with source attribution
    context_block = "\n\n".join(
        f"[Fuente: {meta.get('source', 'unknown')}]\n{doc}"
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
    return answer
