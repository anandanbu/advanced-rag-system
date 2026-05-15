"""
ingestion/chunker.py
────────────────────
Splits loaded documents into overlapping chunks suitable for embedding.

═══════════════════════════════════════════════════════════════
ROOT CAUSE FIX in this file:
═══════════════════════════════════════════════════════════════

PROBLEM — DuplicateIDError on re-upload (and first upload of
          multiple documents with similar names):

  Old code: chunk_id = md5(f"{source}_{chunk_index}")

  Two failure modes:
    1. Re-upload the same file → same source+index → same ID
       → ChromaDB collection.add() raises DuplicateIDError
       → Server crashes or hangs depending on ChromaDB version

    2. Two files with the same name uploaded to different sessions
       → same source+index → same ID collision in shared collection

  FIX: chunk_id = md5(source + chunk_index + chunk_text_prefix)

  Including a hash of the first 64 chars of the chunk text makes IDs:
    - Stable within a session (same text → same ID = safe deduplication)
    - Different across different documents (different text → different ID)
    - Different when content changes on re-upload (updated file → new ID)

  Note: We also switched the store to use upsert() instead of add(),
  so even if two chunks happen to share an ID, the newer one wins
  cleanly instead of raising an exception.
═══════════════════════════════════════════════════════════════
"""

import hashlib
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def chunk_documents(
    documents: list[dict],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[dict]:
    """
    Split a list of loaded documents into smaller overlapping chunks.

    Args:
        documents    : Output from ingestion/loader.py — list of
                       {"text": str, "metadata": dict} dicts
        chunk_size   : Max characters per chunk (default from settings)
        chunk_overlap: Overlap characters between chunks (default from settings)

    Returns:
        List of chunk dicts, each with:
          - chunk_id : Stable unique ID (safe for ChromaDB upsert)
          - text     : Chunk string
          - metadata : Source metadata + chunk_index, chunk_total, char_count
    """
    chunk_size    = chunk_size    or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    # RecursiveCharacterTextSplitter tries separators in order:
    # paragraph → sentence → word → character.
    # This preserves as much semantic coherence as possible per chunk.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )

    all_chunks: list[dict] = []

    for doc in documents:
        raw_text = doc.get("text", "").strip()
        if not raw_text:
            continue

        text_chunks = splitter.split_text(raw_text)

        for chunk_idx, chunk_text in enumerate(text_chunks):
            chunk_text = chunk_text.strip()
            if len(chunk_text) < 30:
                # Skip near-empty chunks: page numbers, lone headers, etc.
                continue

            chunk_metadata = {
                **doc.get("metadata", {}),
                "chunk_index": chunk_idx,
                "chunk_total": len(text_chunks),
                "char_count":  len(chunk_text),
            }

            # FIX: Include chunk text prefix in the ID hash.
            # This makes IDs content-addressable:
            #   - Same file re-uploaded with identical content → same ID → upsert no-ops
            #   - Same file re-uploaded with changed content  → new ID  → upsert updates
            #   - Two different files with same name          → different IDs (different text)
            source   = chunk_metadata.get("source", "unknown")
            id_input = f"{source}::{chunk_idx}::{chunk_text[:64]}"
            chunk_id = hashlib.sha256(id_input.encode("utf-8")).hexdigest()[:32]

            all_chunks.append({
                "chunk_id": chunk_id,
                "text":     chunk_text,
                "metadata": chunk_metadata,
            })

    logger.info(
        f"Chunked {len(documents)} doc(s) → {len(all_chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return all_chunks
