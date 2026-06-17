"""
Curated-FAQ generation.

For a seed list of questions per topic, run the live RAG path to produce
grounded ES + EN answers and store them as ``status='draft'`` for human review
(``scripts/review_faq.py``). Also regenerates answers that were marked ``stale``
by a corpus bump.

Usage:
  cd funstay_webhook
  python -m scripts.generate_faq            # generate drafts from the seed list
  python -m scripts.generate_faq --stale    # regenerate stale entries -> draft

Requires OPENAI_API_KEY + an ingested ChromaDB at runtime; everything is reached
through the Lane-1 providers so a key/provider outage degrades (a question with
no grounded answer is skipped, not invented).
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config
from app.cache import AnswerCache, answer_column, get_answer_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("generate_faq")

# Seed questions per RAG topic. ES + EN phrasings so both language columns get a
# grounded answer. In production this list is owned by the concierge team.
SEED_QUESTIONS = {
    "pool": [
        "¿Cómo enciendo la calefacción de la piscina y cuánto cuesta?",
        "What are the pool and hot tub hours?",
        "¿El jacuzzi está incluido o tiene coste adicional?",
    ],
    "orlando_guide": [
        "¿Qué supermercados y restaurantes hay cerca de la propiedad?",
        "What are the best attractions near the Disney area?",
    ],
    "transport": [
        "¿Cómo llego a los parques de Disney y a Universal desde la propiedad?",
        "How far is Orlando International Airport (MCO) and how do I get there?",
    ],
}

LANGUAGES = ("es", "en")


def generate_drafts(
    cache: AnswerCache,
    questions_by_topic,
    *,
    embed_fn,
    rag_fn,
    languages=LANGUAGES,
    tenant_id: str = config.TENANT_ID,
) -> int:
    """Run live RAG for each seed question and store grounded answers as drafts.

    ``rag_fn(query, language, topic) -> (answer, grounded, source_refs)`` (i.e.
    ``app.rag._live_rag``). ``embed_fn(text) -> list[float]``. Returns the number
    of draft rows written.
    """
    created = 0
    for topic, questions in questions_by_topic.items():
        for question in questions:
            try:
                emb = embed_fn(question)
                answers: dict[str, str] = {}
                refs: set[str] = set()
                for lang in languages:
                    answer, grounded, source_refs = rag_fn(question, lang, topic)
                    if grounded:
                        answers[lang] = answer
                        refs.update(source_refs or [])
            except Exception as e:  # one transient error must not abort the batch
                log.warning("error generating '%s' (%s); skipping", question, e)
                continue
            if not answers:
                log.warning("skip (no grounded answer): %s", question)
                continue
            cache.upsert_curated(
                tenant_id=tenant_id,
                topic=topic,
                canonical_q=question,
                query_embedding=emb,
                answer_es=answers.get("es"),
                answer_en=answers.get("en"),
                source_refs=sorted(refs),
                status="draft",
            )
            created += 1
            log.info("draft: [%s] %s", topic, question)
    return created


def regenerate_stale(
    cache: AnswerCache,
    *,
    embed_fn,
    rag_fn,
    languages=LANGUAGES,
    tenant_id: str = config.TENANT_ID,
) -> int:
    """Regenerate rows marked 'stale' (by a corpus bump) back into 'draft' so a
    human re-approves them. Audio re-synthesis happens at approval and only when
    the answer text actually changed (the audio cache is keyed by text)."""
    regenerated = 0
    for row in cache.stale_rows(tenant_id):
        question = row["canonical_q"]
        topic = row["topic"] or None
        # languages this row previously served — all must refresh to leave 'stale'.
        had_langs = [lang for lang in languages if row[answer_column(lang)]]
        try:
            emb = embed_fn(question)
            answers: dict[str, str] = {}
            refs: set[str] = set()
            for lang in had_langs:
                answer, grounded, source_refs = rag_fn(question, lang, topic)
                if grounded:
                    answers[lang] = answer
                    refs.update(source_refs or [])
        except Exception as e:
            log.warning("error regenerating '%s' (%s); leaving stale", question, e)
            continue
        if not answers:
            log.warning("stale row still ungrounded, leaving stale: %s", question)
            continue
        # Only promote to 'draft' when EVERY previously-served language refreshed;
        # otherwise keep 'stale' so a half-regenerated answer is never approved.
        complete = set(answers) >= set(had_langs)
        cache.upsert_curated(
            tenant_id=tenant_id,
            topic=topic,
            canonical_q=question,
            query_embedding=emb,
            answer_es=answers.get("es"),
            answer_en=answers.get("en"),
            source_refs=sorted(refs),
            status="draft" if complete else "stale",
        )
        if complete:
            regenerated += 1
            log.info("regenerated -> draft: [%s] %s", topic, question)
        else:
            log.warning("partially regenerated, kept stale: [%s] %s", topic, question)
    return regenerated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate/regenerate curated FAQ drafts.")
    parser.add_argument("--stale", action="store_true",
                        help="regenerate stale entries instead of seeding new drafts")
    args = parser.parse_args()

    # Imported here so the module is importable for tests without the heavy deps.
    from app.rag import _live_rag, embed_text

    cache = get_answer_cache()
    if args.stale:
        n = regenerate_stale(cache, embed_fn=embed_text, rag_fn=_live_rag)
        log.info("regenerated %d stale entries into drafts", n)
    else:
        n = generate_drafts(cache, SEED_QUESTIONS, embed_fn=embed_text, rag_fn=_live_rag)
        log.info("wrote %d FAQ drafts", n)


if __name__ == "__main__":
    main()
