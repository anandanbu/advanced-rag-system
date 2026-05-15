"""
api/routes/history.py
──────────────────────
GET  /history/{session_id}        — get full conversation history
GET  /history/{session_id}/memory — get persistent memory facts
POST /history/{session_id}/memory — save a fact to persistent memory
GET  /history/sessions            — list all active sessions
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from memory.conversation import (
    get_history_with_timestamps,
    get_session_summary,
    get_all_session_ids,
)
from memory.persistent import PersistentMemory, list_all_sessions
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/history", tags=["Memory & History"])


class FactRequest(BaseModel):
    key: str
    value: str


@router.get("/sessions", summary="List all active session IDs")
async def list_sessions():
    """Returns all session IDs that have conversation history or persistent memory."""
    active = get_all_session_ids()
    persisted = list_all_sessions()
    all_ids = list(set(active + persisted))
    return {"sessions": all_ids, "count": len(all_ids)}


@router.get("/{session_id}", summary="Get conversation history for a session")
async def get_history(session_id: str):
    """Returns the full conversation history with timestamps."""
    history = get_history_with_timestamps(session_id)
    summary = get_session_summary(session_id)
    return {
        "session_id": session_id,
        "summary": summary,
        "history": history,
    }


@router.get("/{session_id}/memory", summary="Get persistent memory for a session")
async def get_memory(session_id: str):
    """Returns all stored facts, preferences, and summaries for a session."""
    mem = PersistentMemory(session_id)
    return {
        "session_id": session_id,
        "facts": mem.get_all_facts(),
        "preferences": mem._data.get("preferences", {}),
        "summaries": mem.get_summaries(),
        "memory_context": mem.build_memory_context(),
    }


@router.post("/{session_id}/memory", summary="Store a fact in persistent memory")
async def save_fact(session_id: str, request: FactRequest):
    """
    Manually save a fact about the user.
    Example: {"key": "profession", "value": "farmer"}
    """
    mem = PersistentMemory(session_id)
    mem.set_fact(request.key, request.value)
    return {
        "message": f"Fact saved: {request.key}={request.value}",
        "session_id": session_id,
        "all_facts": mem.get_all_facts(),
    }
