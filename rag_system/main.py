"""
main.py
───────
FastAPI application entry point — memory-optimized for Render free tier.

STARTUP SEQUENCE (ORDER MATTERS):
  1. Set all thread-limit env vars (before any import touches torch)
  2. Set multiprocessing start method to 'spawn' (before asyncio starts)
  3. Import FastAPI and other lightweight packages
  4. Define lifespan (pre-load model + warm up ChromaDB)
  5. Register all routes
  6. Start uvicorn with --workers 1 (critical — see note below)

WHY --workers 1:
  Each uvicorn worker is a separate process that loads its own copy
  of the embedding model. On Render free tier, 2 workers = 2 × 385 MB
  = 770 MB → instant OOM. Always use exactly 1 worker.
"""

# ── STEP 1: Thread limits — must be FIRST, before any numpy/torch import ──────
import os
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"
os.environ["NUMEXPR_NUM_THREADS"]    = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── STEP 2: Multiprocessing spawn mode — before asyncio starts ────────────────
import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

# ── STEP 3: Standard imports ──────────────────────────────────────────────────
import sys
import gc
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


# ── STEP 4: Lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup, once at shutdown.
    Pre-loads the embedding model so the first request is never slow.
    """
    logger.info("=" * 60)
    logger.info("  Advanced RAG System — Starting (Render free tier mode)")
    logger.info("=" * 60)

    # Create data directories in /tmp (the only writable path on Render)
    ensure_dir("/tmp/chroma_db")
    ensure_dir("/tmp/memory")
    ensure_dir("/tmp/uploads")
    ensure_dir("./data/uploads")

    # Validate critical config
    if not settings.groq_api_key or settings.groq_api_key == "your_groq_api_key_here":
        logger.error("GROQ_API_KEY is not set — add it in Render's Environment tab")
        sys.exit(1)

    # Pre-load embedding model (downloads ONNX model on first deploy, then cached)
    logger.info("Loading embedding model with ONNX backend…")
    try:
        from embeddings.embedder import get_embedder
        embedder = get_embedder()
        logger.info(
            f"Embedder ready — model='{embedder.model_name}', "
            f"dim={embedder.dimension}"
        )
    except Exception as e:
        logger.error(f"Embedder failed to load: {e}", exc_info=True)
        sys.exit(1)

    # Force GC after model load to reclaim any temporary allocations
    gc.collect()

    # Initialize ChromaDB
    logger.info("Initializing ChromaDB…")
    try:
        from vectorstore.chroma_store import get_vector_store
        store = get_vector_store()
        stats = store.get_stats()
        logger.info(
            f"ChromaDB ready — {stats['total_chunks']} chunks, "
            f"{len(stats['sources'])} source(s)"
        )
    except Exception as e:
        logger.error(f"ChromaDB failed: {e}", exc_info=True)
        sys.exit(1)

    # Log memory usage at startup for monitoring
    try:
        import resource
        mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        logger.info(f"Memory at startup: {mem_mb:.0f} MB / 512 MB (Render limit)")
    except Exception:
        pass

    logger.info(f"✅ Server ready → http://localhost:{settings.api_port}")
    logger.info(f"📖 API docs    → http://localhost:{settings.api_port}/docs")
    logger.info("=" * 60)

    yield  # ← server runs here

    logger.info("Shutting down…")
    gc.collect()


# ── STEP 5: App factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Advanced RAG System",
        description=(
            "Production-ready RAG API — optimized for Render free tier. "
            "Memory usage ~385 MB idle, ~430 MB peak."
        ),
        version="2.1.0",
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

    @app.get("/health", tags=["System"])
    async def health():
        """Health check — also reports current memory usage."""
        from vectorstore.chroma_store import get_vector_store
        stats = get_vector_store().get_stats()

        # Report memory so you can monitor it in Render's logs
        mem_info = {}
        try:
            import resource
            mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            mem_info = {
                "used_mb": round(mem_mb, 1),
                "limit_mb": 512,
                "headroom_mb": round(512 - mem_mb, 1),
            }
        except Exception:
            pass

        return {
            "status": "ok",
            "version": "2.1.0",
            "model": settings.groq_model,
            "embedding_model": settings.embedding_model,
            "embedding_backend": "onnx",
            "memory": mem_info,
            "vector_store": {
                "total_chunks": stats["total_chunks"],
                "sources": stats["sources"],
            },
        }

    @app.get("/", tags=["System"])
    async def root():
        return {
            "message": "Advanced RAG System v2.1 — running on Render free tier",
            "docs": "/docs",
            "health": "/health",
        }

    return app


# ── STEP 6: Entry point ───────────────────────────────────────────────────────

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,         # Never use reload=True on Render — loads model twice
        workers=1,            # Critical: 1 worker only — see module docstring
        log_level=settings.log_level.lower(),
    )
