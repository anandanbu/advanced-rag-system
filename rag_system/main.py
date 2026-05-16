"""
main.py
───────
FastAPI application entry point.

═══════════════════════════════════════════════════════════════
CRITICAL FIX at the top of this file:
═══════════════════════════════════════════════════════════════

multiprocessing.set_start_method("spawn") is called BEFORE any
torch or sentence-transformers import. This is mandatory on
Python 3.12+ and Python 3.14.

WHY:
  Python's default start method on Linux/Mac is "fork".
  fork() copies the entire process memory including:
    - asyncio event loop (in a half-initialized state)
    - torch's internal thread pool locks
    - OpenMP thread mutexes

  When the child process tries to use any of these, it deadlocks
  waiting for locks that will never be released (the parent thread
  that held them was not copied into the child).

  "spawn" creates a clean child process with no inherited state.
  It is slower to start but completely safe.

  Note: sentence-transformers encode() with our fix (show_progress_bar=False)
  does NOT fork at all during inference. This setting is a belt-and-suspenders
  protection for any other library that might fork unexpectedly.
═══════════════════════════════════════════════════════════════

Run:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
  python main.py
"""

# ── MUST BE FIRST — before any torch/transformers/chromadb import ─────────────
import multiprocessing
import os

# Set spawn BEFORE anything that touches torch or multiprocessing workers.
# This prevents fork-related deadlocks with torch DataLoader on Python 3.12+.
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set — fine, just continue

# Force single-threaded torch to eliminate OpenMP fork issues
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── Standard imports (after env vars are set) ─────────────────────────────────
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from config.settings import settings
from api.middleware import setup_middleware
from api.routes.chat import router as chat_router
from api.routes.upload import router as upload_router
from api.routes.history import router as history_router
from api.routes.agent import router as agent_router
from api.routes.summary import router as summary_router
from utils.logger import get_logger
from utils.helpers import ensure_dir

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("  Advanced RAG System v2.0 — Starting Up")
    logger.info("=" * 60)

    ensure_dir("./data/uploads")
    ensure_dir(settings.chroma_persist_dir)
    ensure_dir(settings.memory_dir)

    if not settings.groq_api_key or settings.groq_api_key == "your_groq_api_key_here":
        logger.error("GROQ_API_KEY is not set!")
        logger.error("Copy .env.example → .env and add your key from console.groq.com")
        sys.exit(1)

    #logger.info("Pre-loading embedding model (this takes ~5s on first run)…")
    #try:
        #from embeddings.embedder import get_embedder
        #embedder = get_embedder()
        #logger.info(
            #f"Embedding model ready: '{embedder.model_name}' "
            #f"(dim={embedder.dimension})"
       # )
    #except Exception as e:
        #logger.error(f"Failed to load embedding model: {e}")
        #sys.exit(1)

    #logger.info("Initializing ChromaDB vector store…")
    #try:
       # from vectorstore.chroma_store import get_vector_store
       # store = get_vector_store()
       # stats = store.get_stats()
        #logger.info(
            #f"Vector store ready: {stats['total_chunks']} chunks | "
           # f"{len(stats['sources'])} source(s)"
        #)
    #except Exception as e:
        #logger.error(f"Vector store initialization failed: {e}")
        #sys.exit(1)

    logger.info(f"✅ Server ready → http://localhost:{settings.api_port}")
    logger.info(f"📖 API docs    → http://localhost:{settings.api_port}/docs")
    logger.info("=" * 60)

    yield  # ← server runs here

    logger.info("Shutting down…")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Advanced RAG System",
        description=(
            "Production-ready Retrieval-Augmented Generation API with "
            "memory, critic evaluation, self-improvement, and ReAct agent."
        ),
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    setup_middleware(app)

    app.include_router(chat_router)
    app.include_router(upload_router)
    app.include_router(history_router)
    app.include_router(agent_router)
    app.include_router(summary_router)

    @app.get("/health", tags=["System"], summary="Health check")
    async def health():
        from vectorstore.chroma_store import get_vector_store
        store = get_vector_store()
        stats = store.get_stats()
        return {
            "status": "ok",
            "version": "2.0.0",
            "model": settings.groq_model,
            "embedding_model": settings.embedding_model,
            "python_version": sys.version,
            "vector_store": {
                "total_chunks": stats["total_chunks"],
                "sources": stats["sources"],
            },
        }

    @app.get("/", tags=["System"])
    async def root():
        return {
            "message": "Advanced RAG System v2.0",
            "docs": "/docs",
            "health": "/health",
            "endpoints": {
                "chat":    "POST /chat",
                "upload":  "POST /upload",
                "agent":   "POST /agent",
                "history": "GET  /history/{session_id}",
                "summary": "GET  /summary/{source_name}",
            },
        }

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
