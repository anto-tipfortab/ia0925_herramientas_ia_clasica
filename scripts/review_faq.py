"""
FAQ review CLI: list / approve / edit draft answers.

Approving an entry pre-synthesizes its audio via the TTS provider and records
the file paths. Editing an answer clears that language's audio path so it is
re-synthesized on the next approval — audio is only re-rendered when the answer
text (and thus the sha256(text|voice|lang) key) actually changes.

Usage:
  cd funstay_webhook
  python -m scripts.review_faq list [--status draft]
  python -m scripts.review_faq approve <id>
  python -m scripts.review_faq edit <id> [--es "..."] [--en "..."]
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config
from app.cache import AnswerCache, AudioCache, get_answer_cache, get_audio_cache
from app.providers.tts import VOICE_EN, VOICE_ES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("review_faq")

# (language code, audio-cache lang, Polly voice label) per supported language.
_LANGS = (("es", "es", VOICE_ES), ("en", "en", VOICE_EN))


def list_drafts(cache: AnswerCache, *, tenant_id: str = config.TENANT_ID, status: str = "draft"):
    rows = cache.list_by_status(tenant_id, status)
    for row in rows:
        print(f"#{row['id']} [{row['topic']}] ({row['status']}) {row['canonical_q']}")
        if row["answer_es"]:
            print(f"    ES: {row['answer_es']}")
        if row["answer_en"]:
            print(f"    EN: {row['answer_en']}")
    return rows


def approve(
    cache: AnswerCache,
    row_id: int,
    *,
    audio_cache: AudioCache,
    synth_fn,
    tenant_id: str = config.TENANT_ID,
) -> int:
    """Mark a row approved and pre-synthesize audio for each populated language.

    ``synth_fn(text, lang_code) -> bytes`` is only called on an audio-cache miss,
    so re-approving an unchanged answer does NOT re-synthesize. Returns the count
    of languages synthesized this call.
    """
    # Validate the row (scoped to tenant) BEFORE mutating any state.
    row = cache.get_by_id(tenant_id, row_id)
    if row is None:
        raise ValueError(f"no FAQ row #{row_id} for tenant {tenant_id}")
    cache.set_status(row_id, "approved", tenant_id=tenant_id)
    synthesized = 0
    for lang_code, audio_lang, voice in _LANGS:
        text = row["answer_es" if lang_code == "es" else "answer_en"]
        if not text:
            continue
        if audio_cache.get(text, voice, audio_lang, tenant_id) is None:
            audio = synth_fn(text, lang_code)
            audio_cache.put(text, voice, audio_lang, audio, tenant_id)
            synthesized += 1
        path = audio_cache.path_for(text, voice, audio_lang, tenant_id)
        cache.set_audio_path(row_id, lang_code, str(path), tenant_id=tenant_id)
    log.info("approved #%d (%d audio synthesized)", row_id, synthesized)
    return synthesized


def edit(
    cache: AnswerCache,
    row_id: int,
    *,
    es: str | None = None,
    en: str | None = None,
    tenant_id: str = config.TENANT_ID,
) -> None:
    if es is None and en is None:
        raise ValueError("nothing to edit: pass --es and/or --en")
    if es is not None:
        cache.update_answer(row_id, "es", es, tenant_id=tenant_id)
    if en is not None:
        cache.update_answer(row_id, "en", en, tenant_id=tenant_id)
    log.info("edited #%d (audio path cleared for changed languages)", row_id)


def _default_synth():
    """Real TTS synth via the provider failover (only built when actually used)."""
    from app.providers import get_tts_failover

    def synth(text: str, lang_code: str) -> bytes:
        return get_tts_failover().synthesize(text=text, language_code=lang_code)

    return synth


def main() -> None:
    parser = argparse.ArgumentParser(description="Review curated FAQ drafts.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="list entries by status")
    p_list.add_argument("--status", default="draft")

    p_approve = sub.add_parser("approve", help="approve an entry + pre-synthesize audio")
    p_approve.add_argument("id", type=int)

    p_edit = sub.add_parser("edit", help="edit an answer (clears audio for that language)")
    p_edit.add_argument("id", type=int)
    p_edit.add_argument("--es")
    p_edit.add_argument("--en")

    args = parser.parse_args()
    cache = get_answer_cache()

    if args.cmd == "list":
        list_drafts(cache, status=args.status)
    elif args.cmd == "approve":
        approve(cache, args.id, audio_cache=get_audio_cache(), synth_fn=_default_synth())
    elif args.cmd == "edit":
        edit(cache, args.id, es=args.es, en=args.en)


if __name__ == "__main__":
    main()
