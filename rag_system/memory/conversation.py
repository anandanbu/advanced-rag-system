"""
memory/conversation.py
──────────────────────
Manages per-session conversation history in memory.

Each session has a list of turns:
  [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

This is passed to the LLM on every call so it can maintain context.
The history is bounded by MAX_HISTORY_TURNS to prevent context overflow.
"""

from datetime import datetime
from collections import defaultdict

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# In-memory store: session_id → list of message dicts
# defaultdict(list) means a new session auto-initializes to []
_sessions: dict[str, list[dict]] = defaultdict(list)


def add_turn(session_id: str, user_message: str, assistant_message: str) -> None:
    """
    Append a user+assistant exchange to the session history.

    Args:
        session_id        : Unique session identifier
        user_message      : What the user said
        assistant_message : What the assistant replied
    """
    history = _sessions[session_id]

    history.append({
        "role": "user",
        "content": user_message,
        "timestamp": datetime.utcnow().isoformat(),
    })
    history.append({
        "role": "assistant",
        "content": assistant_message,
        "timestamp": datetime.utcnow().isoformat(),
    })

    # Trim history to MAX_HISTORY_TURNS (each turn = 2 messages)
    max_messages = settings.max_history_turns * 2
    if len(history) > max_messages:
        # Always drop the oldest turns, never the most recent
        _sessions[session_id] = history[-max_messages:]

    logger.debug(
        f"Session '{session_id[:8]}…' | "
        f"history_length={len(_sessions[session_id])} messages"
    )


def get_history(session_id: str) -> list[dict]:
    """
    Return the conversation history for a session.
    Returns empty list if session doesn't exist yet.

    The returned list is suitable for passing directly to the LLM:
    [{"role": "user"|"assistant", "content": str}, ...]
    Note: timestamp keys are stripped for LLM compatibility.
    """
    raw = _sessions.get(session_id, [])
    # Strip timestamp — LLM APIs only accept role + content
    return [{"role": m["role"], "content": m["content"]} for m in raw]


def get_history_with_timestamps(session_id: str) -> list[dict]:
    """Return full history including timestamps (for API/UI display)."""
    return list(_sessions.get(session_id, []))


def clear_session(session_id: str) -> None:
    """Clear all history for a session (e.g., user clicks 'New Chat')."""
    if session_id in _sessions:
        del _sessions[session_id]
        logger.info(f"Cleared session '{session_id[:8]}…'")


def get_all_session_ids() -> list[str]:
    """Return list of all active session IDs."""
    return list(_sessions.keys())


def get_session_summary(session_id: str) -> dict:
    """Return a summary of a session's stats."""
    history = _sessions.get(session_id, [])
    user_turns = [m for m in history if m["role"] == "user"]
    return {
        "session_id": session_id,
        "total_messages": len(history),
        "total_turns": len(user_turns),
        "first_message_at": history[0].get("timestamp") if history else None,
        "last_message_at": history[-1].get("timestamp") if history else None,
    }
