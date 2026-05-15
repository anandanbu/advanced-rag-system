"""
api/routes/upload.py
─────────────────────
POST /upload        — upload and ingest a document
GET  /upload/stats  — collection statistics
DELETE /upload/{source} — remove a document from the store

═══════════════════════════════════════════════════════════════
ROOT CAUSE FIX — why the old code hung forever:
═══════════════════════════════════════════════════════════════

The old code called ingest_file() directly inside an `async def` endpoint:

    async def upload_document(...):
        stats = ingest_file(file_bytes, filename)   ← BLOCKING

This is the #1 FastAPI mistake. ingest_file() does:
  - file I/O (temp file write)
  - sentence-transformers model.encode()  ← uses torch DataLoader internally
  - ChromaDB collection.add()             ← SQLite write

ALL of these are synchronous and CPU/IO-bound. Calling them directly in
an async def blocks the SINGLE uvicorn event loop thread. While blocked:
  - Swagger UI cannot get a response (spins forever)
  - /health cannot respond
  - No other requests are processed at all

FIX: Use asyncio.get_event_loop().run_in_executor(None, fn)
This hands the blocking work to Python's ThreadPoolExecutor,
releasing the event loop immediately. The async endpoint then
`await`s the thread's completion without blocking anything.

Additionally: torch's DataLoader internally uses fork() to spawn workers.
fork() inside an asyncio context (Python 3.12+) causes deadlocks because
the child inherits the parent's asyncio event loop state and mutexes.
This is fixed in embedder.py (num_workers=0, no forking at all).
═══════════════════════════════════════════════════════════════
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from fastapi import APIRouter, UploadFile, File, HTTPException

from rag.pipeline import ingest_file
from vectorstore.chroma_store import get_vector_store
from utils.helpers import is_supported_file, sanitize_filename
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/upload", tags=["Documents"])

MAX_FILE_SIZE = 10 * 1024 * 1024  # 50 MB

# Dedicated single-threaded executor for ingestion.
# Single thread is intentional: ChromaDB SQLite cannot safely handle
# concurrent writes from multiple threads without extra locking.
_ingest_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")


@router.post("", summary="Upload and ingest a document into the vector store")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF/TXT/DOCX/CSV).

    Pipeline (all heavy work runs in a background thread):
      1. Validate file type and size
      2. Extract text (format-aware loader)
      3. Split into overlapping chunks
      4. Embed with sentence-transformers  ← CPU-bound, runs in thread
      5. Store in ChromaDB                 ← IO-bound, runs in thread

    The async endpoint releases the event loop during steps 4-5
    via run_in_executor, so Swagger UI and /health stay responsive.
    """

    # ── Validation (fast, no blocking) ────────────────────────────────────────
    if not is_supported_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: '{file.filename}'. "
                "Supported: PDF, TXT, DOCX, CSV"
            ),
        )

    # await file.read() is truly async — fine here
    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large ({len(file_bytes) // 1024}KB). "
                f"Max: {MAX_FILE_SIZE // 1024}KB"
            ),
        )

    safe_filename = sanitize_filename(file.filename)
    logger.info(f"Upload received: '{safe_filename}' ({len(file_bytes):,} bytes)")

    # ── Offload ALL sync work to thread pool ──────────────────────────────────
    # This is the critical fix. run_in_executor:
    #   1. Submits ingest_file to the ThreadPoolExecutor
    #   2. Returns control to the event loop immediately
    #   3. Resumes this coroutine when the thread finishes
    # The event loop is FREE to handle /health, /docs, etc. while ingesting.
    loop = asyncio.get_event_loop()
    try:
        stats = await loop.run_in_executor(
            _ingest_executor,
            partial(ingest_file, file_bytes, safe_filename),
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Ingestion error for '{safe_filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    logger.info(f"Upload complete: '{safe_filename}' → {stats}")
    return {
        "message": f"'{safe_filename}' ingested successfully.",
        "stats": stats,
    }


@router.get("/stats", summary="Get vector store statistics")
async def vector_store_stats():
    """Returns collection stats: total chunks, sources, embedding model info."""
    loop = asyncio.get_event_loop()
    try:
        # get_stats() hits SQLite — also run in executor for safety
        stats = await loop.run_in_executor(
            _ingest_executor,
            lambda: get_vector_store().get_stats(),
        )
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{source_name}", summary="Remove a document from the vector store")
async def delete_document(source_name: str):
    """Delete all chunks belonging to a source document."""
    loop = asyncio.get_event_loop()
    try:
        deleted = await loop.run_in_executor(
            _ingest_executor,
            lambda: get_vector_store().delete_by_source(source_name),
        )
        if deleted == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No chunks found for source: '{source_name}'",
            )
        return {
            "message": f"Deleted {deleted} chunks for '{source_name}'",
            "source": source_name,
            "chunks_deleted": deleted,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
