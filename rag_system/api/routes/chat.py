"""
api/routes/chat.py
──────────────────
POST /chat  — the main RAG chat endpoint
POST /chat/clear — clear a session's conversation history

Request body:
  {
    "message": "What is photosynthesis?",
    "session_id": "optional-uuid-string",
    "use_critic": true,
    "filter_source": null
  }

Response:
  {
    "answer": "...",
    "session_id": "...",
    "sources": [...],
    "critic_score": 0.87,
    "critic_passed": true,
    "hallucination_detected": false,
    "improvement_iterations": 0,
    "retrieval_count": 5,
    "latency_ms": 1234.5,
    "mode": "rag"
  }
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from rag.pipeline import RAGPipeline
from memory.conversation import clear_session, get_session_summary
from utils.helpers import generate_session_id
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])

# Singleton pipeline — loaded once when the module is imported
_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


# ── Request / Response Schemas ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User's question")
    session_id: Optional[str] = Field(None, description="Session ID (auto-generated if omitted)")
    use_critic: bool = Field(True, description="Run critic evaluation (set False for faster responses)")
    filter_source: Optional[str] = Field(None, description="Restrict retrieval to a specific document")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Override number of retrieved chunks")


class SourceChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: dict
    score: float


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[SourceChunk]
    critic_score: float
    critic_passed: bool
    hallucination_detected: bool
    improvement_iterations: int
    retrieval_count: int
    latency_ms: float
    mode: str


class ClearRequest(BaseModel):
    session_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("", response_model=ChatResponse, summary="Send a message to the RAG assistant")
async def chat(request: ChatRequest):
    """
    Main chat endpoint. Runs the full RAG pipeline:
    retrieve → augment → generate → critique → (optionally improve) → respond.
    """
    # Auto-generate session ID if not provided
    session_id = request.session_id or generate_session_id()

    try:
        pipeline = get_pipeline()
        result = pipeline.run(
            query=request.message,
            session_id=session_id,
            top_k=request.top_k,
            use_critic=request.use_critic,
            filter_source=request.filter_source,
        )
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    return ChatResponse(
        answer=result.answer,
        session_id=result.session_id,
        sources=[
            SourceChunk(
                chunk_id=s["chunk_id"],
                text=s["text"],
                metadata=s["metadata"],
                score=s["score"],
            )
            for s in result.sources
        ],
        critic_score=result.critic_score,
        critic_passed=result.critic_passed,
        hallucination_detected=result.hallucination_detected,
        improvement_iterations=result.improvement_iterations,
        retrieval_count=result.retrieval_count,
        latency_ms=result.latency_ms,
        mode=result.mode,
    )


@router.post("/clear", summary="Clear conversation history for a session")
async def clear_chat(request: ClearRequest):
    """Reset the conversation history for a given session ID."""
    clear_session(request.session_id)
    return {"message": f"Session '{request.session_id}' cleared.", "session_id": request.session_id}


@router.get("/session/{session_id}", summary="Get session summary")
async def session_info(session_id: str):
    """Return stats about a session (turn count, timestamps, etc.)"""
    return get_session_summary(session_id)
