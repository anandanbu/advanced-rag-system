"""
vectorstore/chroma_store.py
───────────────────────────
ChromaDB wrapper for storing and retrieving document embeddings.

═══════════════════════════════════════════════════════════════
ROOT CAUSE FIXES in this file:
═══════════════════════════════════════════════════════════════

PROBLEM 1 — ChromaDB batch size limit:
  chromadb.Collection.add() has an undocumented internal limit of
  ~5461 items per call (derived from SQLite parameter limits: 32766
  max params ÷ 6 columns = 5461). Submitting a large PDF that
  produces >5461 chunks causes a silent hang or cryptic SQLite error.

  FIX: Split add() calls into sub-batches of CHROMA_BATCH_SIZE=500.
       Each sub-batch is committed independently with a log line,
       so you can see progress in the terminal instead of silence.

PROBLEM 2 — DuplicateIDError on re-upload:
  When re-uploading the same document, the old deterministic chunk IDs
  (md5 of source+index) already exist in the collection. ChromaDB raises
  DuplicateIDError instead of upserting silently.

  FIX: Use collection.upsert() instead of collection.add().
       upsert() inserts new IDs and updates existing ones — idempotent.
       This also means re-uploading an updated document works correctly.

PROBLEM 3 — SQLite thread-safety:
  ChromaDB's PersistentClient uses SQLite under the hood. SQLite in WAL
  mode is safe for concurrent reads but requires serialized writes.
  Calling add() from multiple threads simultaneously causes lock errors.

  FIX: The ThreadPoolExecutor in upload.py uses max_workers=1,
       so ChromaDB writes are already serialized. No additional
       locking needed here, but documented clearly.
═══════════════════════════════════════════════════════════════
"""

from functools import lru_cache
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings
from embeddings.embedder import get_embedder
from utils.logger import get_logger
from utils.helpers import ensure_dir

logger = get_logger(__name__)

# FIX: Never submit more than this many items to ChromaDB at once.
# ChromaDB's SQLite backend has a parameter limit that causes hangs
# above ~5461 rows. 500 is a safe, fast sub-batch size.
CHROMA_BATCH_SIZE = 500


class ChromaVectorStore:
    """
    Manages a ChromaDB collection: add chunks, query by similarity,
    delete by source, and get collection stats.

    Write safety: designed to be called from a single thread
    (enforced by the ThreadPoolExecutor in upload.py).
    """

    def __init__(self, persist_dir: str, collection_name: str):
        ensure_dir(persist_dir)

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = get_embedder()

        count = self._collection.count()
        logger.info(
            f"ChromaDB ready — collection='{collection_name}', "
            f"chunks={count}, persist='{persist_dir}'"
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[dict]) -> int:
        """
        Embed and upsert a list of chunk dicts from chunker.py.

        Uses upsert() instead of add() to handle re-uploads gracefully.
        Splits into sub-batches of CHROMA_BATCH_SIZE to avoid SQLite
        parameter limits that cause silent hangs on large documents.

        Returns: count of chunks submitted (includes updates).
        """
        if not chunks:
            return 0

        texts     = [c["text"]      for c in chunks]
        ids       = [c["chunk_id"]  for c in chunks]
        metadatas = [_sanitize_metadata(c["metadata"]) for c in chunks]

        # Embed ALL chunks in one call (efficient batched forward pass)
        logger.info(f"Embedding {len(chunks)} chunk(s)…")
        embeddings = self._embedder.embed_batch(texts)
        logger.info(f"Embedding done. Storing in ChromaDB…")

        # FIX: Submit to ChromaDB in sub-batches to avoid SQLite limits.
        # Without this, a 300-page PDF (>5461 chunks) hangs forever.
        total_added = 0
        for batch_start in range(0, len(chunks), CHROMA_BATCH_SIZE):
            batch_end = batch_start + CHROMA_BATCH_SIZE
            batch_num = (batch_start // CHROMA_BATCH_SIZE) + 1
            total_batches = (len(chunks) + CHROMA_BATCH_SIZE - 1) // CHROMA_BATCH_SIZE

            b_ids   = ids[batch_start:batch_end]
            b_texts = texts[batch_start:batch_end]
            b_embs  = embeddings[batch_start:batch_end]
            b_metas = metadatas[batch_start:batch_end]

            logger.info(
                f"ChromaDB batch {batch_num}/{total_batches} "
                f"({len(b_ids)} chunks)…"
            )

            # FIX: upsert() instead of add() — handles re-uploads without
            # DuplicateIDError. New IDs are inserted, existing IDs updated.
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

    # ── Read / Search ─────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[dict] = None,
    ) -> list[dict]:
        """
        Embed the query and return the top-k most similar chunks.
        """
        top_k = top_k or settings.retriever_top_k

        count = self._collection.count()
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
            similarity = 1.0 - (distance / 2.0)  # cosine distance → similarity
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
        """Remove all chunks for a given source filename."""
        results = self._collection.get(
            where={"source": source_name},
            include=["documents"],
        )
        ids_to_delete = results.get("ids", [])
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
            logger.info(
                f"Deleted {len(ids_to_delete)} chunk(s) for source='{source_name}'"
            )
        return len(ids_to_delete)

    def delete_collection(self) -> None:
        """Wipe the entire collection. Irreversible."""
        self._client.delete_collection(self._collection.name)
        logger.warning(f"Deleted entire collection '{self._collection.name}'")

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        count = self._collection.count()
        sources: list[str] = []
        if count > 0:
            all_meta = self._collection.get(include=["metadatas"])["metadatas"]
            sources = sorted({m.get("source", "unknown") for m in all_meta})
        return {
            "total_chunks":    count,
            "collection_name": self._collection.name,
            "sources":         sources,
            "embedding_model": self._embedder.model_name,
            "embedding_dim":   self._embedder.dimension,
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _get_existing_ids(self, ids: list[str]) -> set[str]:
        """Check which IDs already exist in the collection."""
        if not ids:
            return set()
        try:
            result = self._collection.get(ids=ids, include=[])
            return set(result.get("ids", []))
        except Exception:
            return set()


def _sanitize_metadata(meta: dict) -> dict:
    """
    ChromaDB only accepts str, int, float, bool metadata values.
    Converts lists, None, and other types to strings.
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


@lru_cache(maxsize=1)
def get_vector_store() -> ChromaVectorStore:
    """Singleton factory — one ChromaDB client per process."""
    return ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
    )
