"""
api/middleware.py
─────────────────
FastAPI middleware stack:
  1. CORS — allows frontend (Streamlit / React) to call the API
  2. Request logging — logs every request with method, path, latency, status
  3. Global exception handler — catches unhandled errors and returns clean JSON

Register all middleware in main.py via setup_middleware(app).
"""

import time
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Request Logging Middleware ────────────────────────────────────────────────

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every HTTP request: method, path, status code, latency.
    Skips health check endpoints to avoid log spam.
    """

    SKIP_PATHS = {"/health", "/favicon.ico", "/docs", "/openapi.json"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        start = time.time()
        response = None

        try:
            response = await call_next(request)
            latency_ms = (time.time() - start) * 1000
            logger.info(
                f"{request.method} {request.url.path} "
                f"→ {response.status_code} ({latency_ms:.0f}ms)"
            )
            return response

        except Exception as exc:
            latency_ms = (time.time() - start) * 1000
            logger.error(
                f"{request.method} {request.url.path} "
                f"→ 500 ({latency_ms:.0f}ms) | {exc}"
            )
            raise


# ── Global Exception Handler ──────────────────────────────────────────────────

async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all for unhandled exceptions.
    Returns a clean JSON error instead of a raw Python traceback.
    """
    tb = traceback.format_exc()
    logger.error(f"Unhandled exception on {request.url.path}:\n{tb}")

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url.path),
        },
    )


# ── Setup Function ────────────────────────────────────────────────────────────

def setup_middleware(app: FastAPI) -> None:
    """
    Register all middleware and exception handlers on the FastAPI app.
    Call this once in main.py before the app starts.
    """
    # CORS — must be added before other middleware
    origins = (
        ["*"]
        if settings.allowed_origins == "*"
        else [o.strip() for o in settings.allowed_origins.split(",")]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging
    app.add_middleware(RequestLoggingMiddleware)

    # Global exception handler
    app.add_exception_handler(Exception, global_exception_handler)

    logger.info(f"Middleware configured | CORS origins: {origins}")
