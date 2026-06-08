"""
Offline ingestion pipeline for FunStay's RAG knowledge base.

Pipeline:
  1. Rasterize each PDF page to an image (pdf2image / pypdfium2)
  2. Run OCR with pytesseract (open-source) — could be swapped for Azure
     Document Intelligence or Mistral OCR
  3. Chunk text by paragraph (with size limits)
  4. Embed with OpenAI text-embedding-3-small
  5. Persist into ChromaDB with metadata (source, topic, page)

Usage:
  cd funstay_webhook
  python -m scripts.ingest

Requires:
  - tesseract-ocr installed (sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-spa)
  - poppler-utils installed (sudo apt-get install poppler-utils)
  - OPENAI_API_KEY in environment
"""
import os
import sys
import logging
import re
from pathlib import Path
from typing import List

# Add parent dir to path so we can import app.rag
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdf2image import convert_from_path
import pytesseract

from app.rag import _get_collection, embed_text, COLLECTION_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ingest")

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

# Topic mapping: which PDF maps to which RAG topic filter
# (the webhook uses these to scope retrieval per intent)
DOC_TOPIC_MAP = {
    "01_ChampionsGate_Property_Manual.pdf": "pool",
    "02_Reunion_Welcome_Book.pdf": "pool",  # also has property info
    "03_Orlando_Area_Guide.pdf": "orlando_guide",
}

# Some chunks are clearly about transport — relabel them after extraction
TRANSPORT_KEYWORDS = ["distance", "drive time", "shuttle", "MCO", "airport",
                       "Uber", "Lyft", "Mears", "kilometer", "mile"]


def ocr_pdf_to_text_per_page(pdf_path: Path, languages: str = "eng+spa") -> List[str]:
    """Convert PDF pages to images and OCR each one. Returns text per page."""
    log.info(f"Rasterizing {pdf_path.name}...")
    images = convert_from_path(str(pdf_path), dpi=200)
    pages_text = []
    for i, img in enumerate(images, 1):
        log.info(f"  OCR page {i}/{len(images)}")
        text = pytesseract.image_to_string(img, lang=languages)
        pages_text.append(text)
    return pages_text


def chunk_text(text: str, max_chars: int = 800) -> List[str]:
    """Chunk by paragraph (double newline), respecting max_chars per chunk."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    buf = ""
    for p in paragraphs:
        # Collapse internal newlines
        p_clean = re.sub(r"\s+", " ", p)
        if len(p_clean) < 30:  # skip tiny fragments
            continue
        if len(buf) + len(p_clean) + 2 <= max_chars:
            buf = (buf + "\n\n" + p_clean) if buf else p_clean
        else:
            if buf:
                chunks.append(buf)
            buf = p_clean
    if buf:
        chunks.append(buf)
    return chunks


def detect_topic(text: str, default_topic: str) -> str:
    """Lightweight topic re-labeling — if a chunk mentions transport, override."""
    lower = text.lower()
    if any(k.lower() in lower for k in TRANSPORT_KEYWORDS):
        return "transport"
    return default_topic


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY not set. Aborting.")
        sys.exit(1)

    collection = _get_collection()

    # Reset collection to avoid duplicates on re-runs
    try:
        existing = collection.count()
        if existing > 0:
            log.info(f"Clearing {existing} existing chunks from collection...")
            # Easiest way: delete all docs by getting their IDs
            all_data = collection.get()
            if all_data["ids"]:
                collection.delete(ids=all_data["ids"])
    except Exception as e:
        log.warning(f"Could not clear collection: {e}")

    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        log.error(f"No PDFs found in {DOCS_DIR}")
        sys.exit(1)

    total_chunks = 0
    for pdf_path in pdf_files:
        default_topic = DOC_TOPIC_MAP.get(pdf_path.name, "general")
        pages = ocr_pdf_to_text_per_page(pdf_path)

        for page_num, page_text in enumerate(pages, 1):
            chunks = chunk_text(page_text)
            for chunk_idx, chunk in enumerate(chunks):
                topic = detect_topic(chunk, default_topic)
                chunk_id = f"{pdf_path.stem}_p{page_num}_c{chunk_idx}"

                try:
                    emb = embed_text(chunk)
                except Exception as e:
                    log.warning(f"Embedding failed for {chunk_id}: {e}")
                    continue

                collection.add(
                    ids=[chunk_id],
                    embeddings=[emb],
                    documents=[chunk],
                    metadatas=[{
                        "source": pdf_path.name,
                        "page": page_num,
                        "topic": topic,
                    }],
                )
                total_chunks += 1
                log.info(f"  + {chunk_id} [topic={topic}, len={len(chunk)}]")

    log.info(f"\n✓ Ingested {total_chunks} chunks into ChromaDB collection '{COLLECTION_NAME}'.")
    log.info(f"  Persisted to: {collection._client.get_settings().persist_directory if hasattr(collection, '_client') else 'data/chroma'}")


if __name__ == "__main__":
    main()
