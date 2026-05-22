"""
vectorstore/chroma_store.py
───────────────────────────
Fixed: KeyError '_type' crash caused by ChromaDB version mismatch.

ROOT CAUSE:
  ChromaDB 0.4.x stored collection config WITHOUT a '_type' field.
  ChromaDB 0.5.x expects '_type' to exist when reading old databases.
  Result: KeyError '_type' on every request → 500 error.

  This happens on Render because the project src/ directory persists
  between deploys, so old 0.4.x database files survive the upgrade
  to 0.5.x and corrupt every startup.

FIXES APPLIED:
  1. _safe_init_collection() — wraps collection creation in try/except.
     If the existing database is corrupt/incompatible, it deletes the
     database directory and creates a fresh one automatically.
     No manual intervention needed on Render.

  2. CHROMA_PERSIST_DIR should be /tmp/chroma_db (set this in Render's
     environment variables). /tmp is wiped on every Render restart,
     guaranteeing a fresh database — no version conflicts ever.

  3. upsert() instead of add() — handles re-uploads without
     DuplicateIDError regardless of database state.

  4. Sub-batching at 500 items — avoids ChromaDB's SQLite parameter
     limit which causes silent hangs on large documents.
"""

import os
import shutil
from functools import lru_cache
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings
from embeddings.embedder import get_embedder
from utils.logger import get_logger
from utils.helpers import ensure_dir

logger = get_logger(__name__)

# Never submit more than this many items to ChromaDB at once.
# ChromaDB's SQLite backend has a ~5461 row parameter limit per call.
CHROMA_BATCH_SIZE = 500


class ChromaVectorStore:
    """
    ChromaDB wrapper with auto-recovery from corrupt/incompatible databases.
    Safe for Render free tier — handles version mismatches automatically.
    """

    def __init__(self, persist_dir: str, collection_name: str):
        ensure_dir(persist_dir)
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._embedder = get_embedder()

        # Try to init — auto-recover if DB is corrupt
        self._client, self._collection = self._safe_init_collection(
            persist_dir, collection_name
        )

        count = self._collection.count()
        logger.info(
            f"ChromaDB ready — collection='{collection_name}', "
            f"chunks={count}, dir='{persist_dir}'"
        )

    # ── Safe initialisation ───────────────────────────────────────────────────

    def _safe_init_collection(self, persist_dir: str, collection_name: str):
        """
        Create ChromaDB client and collection.

        If the existing database is corrupt (e.g. created by an older
        ChromaDB version — the '_type' KeyError), this method:
          1. Logs a warning
          2. Deletes the corrupt database directory
          3. Creates a fresh database

        This means documents need to be re-uploaded after auto-recovery,
        but the server starts correctly instead of crashing.
        """
        for attempt in range(2):
            try:
                client = chromadb.PersistentClient(
                    path=persist_dir,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                collection = client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                return client, collection

            except (KeyError, Exception) as e:
                error_str = str(e)

                # Detect the specific _type version-mismatch error
                is_type_error = (
                    "_type" in error_str
                    or "from_json" in error_str
                    or "configuration" in error_str.lower()
                )

                if attempt == 0 and is_type_error:
                    logger.warning(
                        f"ChromaDB database is incompatible with current version "
                        f"('{error_str}'). "
                        f"This is caused by a ChromaDB version upgrade. "
                        f"Auto-recovering: deleting old database and creating fresh one."
                    )
                    logger.warning(
                        "All previously uploaded documents have been cleared. "
                        "Please re-upload your documents."
                    )
                    # Delete the corrupt database directory
                    if os.path.exists(persist_dir):
                        shutil.rmtree(persist_dir)
                        logger.info(f"Deleted corrupt database at: {persist_dir}")
                    ensure_dir(persist_dir)
                    # Loop continues — attempt 1 will succeed with fresh DB
                    continue

                # Any other error on attempt 0: try once more
                if attempt == 0:
                    logger.warning(f"ChromaDB init failed ({e}), retrying…")
                    continue

                # Both attempts failed — re-raise
                logger.error(f"ChromaDB init failed after recovery attempt: {e}")
                raise

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[dict]) -> int:
        """
        Embed and upsert chunks into ChromaDB.

        Uses upsert() — safe for re-uploads, no DuplicateIDError.
        Splits into batches of CHROMA_BATCH_SIZE to avoid SQLite limits.
        """
        if not chunks:
            return 0

        texts     = [c["text"]     for c in chunks]
        ids       = [c["chunk_id"] for c in chunks]
        metadatas = [_sanitize_metadata(c["metadata"]) for c in chunks]

        logger.info(f"Embedding {len(chunks)} chunk(s)…")
        embeddings = self._embedder.embed_batch(texts)
        logger.info(f"Embedding done. Writing to ChromaDB in batches…")

        total_added = 0
        total_batches = (len(chunks) + CHROMA_BATCH_SIZE - 1) // CHROMA_BATCH_SIZE

        for i in range(0, len(chunks), CHROMA_BATCH_SIZE):
            batch_num = i // CHROMA_BATCH_SIZE + 1
            b_ids   = ids[i : i + CHROMA_BATCH_SIZE]
            b_texts = texts[i : i + CHROMA_BATCH_SIZE]
            b_embs  = embeddings[i : i + CHROMA_BATCH_SIZE]
            b_metas = metadatas[i : i + CHROMA_BATCH_SIZE]

            logger.info(f"ChromaDB batch {batch_num}/{total_batches} ({len(b_ids)} chunks)…")

            self._collection.upsert(
                ids=b_ids,
                documents=b_texts,
                embeddings=b_embs,
                metadatas=b_metas,
            )
            total_added += len(b_ids)

        final_count = self._collection.count()
        logger.info(
            f"ChromaDB upsert complete — submitted={total_added}, "
            f"collection_total={final_count}"
        )
        return total_added

    # ── Read ──────────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[dict] = None,
    ) -> list[dict]:
        """
        Embed the query and return the top-k most similar chunks.
        Returns empty list if store is empty — never crashes.
        """
        top_k = top_k or settings.retriever_top_k
        count  = self._collection.count()

        if count == 0:
            logger.warning("Vector store is empty — no results.")
            return []

        query_embedding = self._embedder.embed_text(query)
        n_results = min(top_k, count)

        query_kwargs: dict = dict(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        if filter_metadata:
            query_kwargs["where"] = filter_metadata

        results = self._collection.query(**query_kwargs)

        chunks = []
        for i in range(len(results["ids"][0])):
            distance   = results["distances"][0][i]
            similarity = 1.0 - (distance / 2.0)
            chunks.append({
                "chunk_id": results["ids"][0][i],
                "text":     results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score":    round(similarity, 4),
            })

        if chunks:
            logger.debug(
                f"Retrieved {len(chunks)} chunk(s) | "
                f"top_score={chunks[0]['score']:.3f} | "
                f"query='{query[:60]}'"
            )
        return chunks

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_by_source(self, source_name: str) -> int:
        """Remove all chunks for a given source document."""
        results = self._collection.get(
            where={"source": source_name},
            include=["documents"],
        )
        ids_to_delete = results.get("ids", [])
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} chunks for '{source_name}'")
        return len(ids_to_delete)

    def delete_collection(self) -> None:
        """Wipe entire collection. Irreversible."""
        self._client.delete_collection(self._collection.name)
        logger.warning(f"Deleted collection '{self._collection.name}'")

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return collection statistics."""
        count = self._collection.count()
        sources: list[str] = []
        if count > 0:
            all_meta = self._collection.get(include=["metadatas"])["metadatas"]
            sources  = sorted({m.get("source", "unknown") for m in all_meta})
        return {
            "total_chunks":    count,
            "collection_name": self._collection.name,
            "sources":         sources,
            "embedding_model": self._embedder.model_name,
            "embedding_dim":   self._embedder.dimension,
        }


# ── Metadata helper ───────────────────────────────────────────────────────────

def _sanitize_metadata(meta: dict) -> dict:
    """
    ChromaDB only accepts str / int / float / bool metadata values.
    Converts everything else (lists, None, dicts) to strings.
    """
    clean: dict = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif v is None:
            clean[k] = ""
        else:
            clean[k] = str(v)
    return clean


# ── Singleton ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_vector_store() -> ChromaVectorStore:
    """
    Singleton factory — one ChromaDB client per process.

    IMPORTANT: CHROMA_PERSIST_DIR must be /tmp/chroma_db on Render.
    Set this in Render → Environment Variables:
      Key:   CHROMA_PERSIST_DIR
      Value: /tmp/chroma_db

    Using /tmp guarantees a fresh database on every restart,
    eliminating the version-mismatch error permanently.
    """
    return ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
    )
