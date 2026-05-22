"""
main.py
───────
FastAPI application entry point — optimized for Render free tier.

IMPORTANT CHANGE:
  - Removed heavy embedding/chromadb preload from startup.
  - FastAPI now opens port immediately.
  - Models load lazily on first upload/chat request.
"""

# ── STEP 1: Thread limits ─────────────────────────────────────────────────────
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── STEP 2: Multiprocessing spawn mode ───────────────────────────────────────
import multiprocessing

if multiprocessing.get_start_method(allow_none=True) != "spawn":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

# ── STEP 3: Imports ───────────────────────────────────────────────────────────
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
    Lightweight startup for Render free tier.

    DO NOT preload:
      - embedding model
      - chromadb
      - sentence-transformers
      - ONNX

    They load lazily during the first request.
    """

    logger.info("=" * 60)
    logger.info("  Advanced RAG System — Starting")
    logger.info("=" * 60)

    # Create writable directories
    ensure_dir("/tmp/chroma_db")
    ensure_dir("/tmp/memory")
    ensure_dir("/tmp/uploads")
    ensure_dir("./data/uploads")

    # Validate config
    if (
        not settings.groq_api_key
        or settings.groq_api_key == "your_groq_api_key_here"
    ):
        logger.error("GROQ_API_KEY is not set")
        sys.exit(1)

    logger.info("FastAPI startup complete")
    logger.info(f"API running on port {settings.api_port}")

    yield

    logger.info("Shutting down...")
    gc.collect()


# ── STEP 5: App Factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Advanced RAG System",
        description="Production-ready RAG API optimized for Render free tier.",
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
        """
        Lightweight health endpoint.
        Avoid loading vector store here.
        """

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
        }

    @app.get("/", tags=["System"])
    async def root():
        return {
            "message": "Advanced RAG System v2.1 running",
            "docs": "/docs",
            "health": "/health",
        }

    return app


# ── STEP 6: Entry Point ───────────────────────────────────────────────────────

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        workers=1,
        log_level=settings.log_level.lower(),
    )s
