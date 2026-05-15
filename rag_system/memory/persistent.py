"""
memory/persistent.py
────────────────────
Long-term persistent memory stored as JSON files on disk.

Use cases:
  - Save important facts the user mentions ("I'm a farmer in Tamil Nadu")
  - Remember user preferences ("always respond in bullet points")
  - Store session summaries for future context
  - Foundation for AI companion / agent memory systems

Each session is saved as: data/memory/{session_id}.json
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from config.settings import settings
from utils.logger import get_logger
from utils.helpers import ensure_dir

logger = get_logger(__name__)


class PersistentMemory:
    """
    JSON-file-backed key-value memory store per session.

    Structure of each memory file:
    {
      "session_id": "...",
      "created_at": "...",
      "updated_at": "...",
      "facts": {"key": "value", ...},        ← user-specific facts
      "preferences": {"key": "value", ...},  ← style preferences
      "summaries": ["summary1", ...],        ← past session summaries
      "raw_history": [...]                   ← full conversation turns
    }
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._dir = ensure_dir(settings.memory_dir)
        self._file = self._dir / f"{session_id}.json"
        self._data = self._load()

    # ── Core CRUD ─────────────────────────────────────────────────────────────

    def set_fact(self, key: str, value: Any) -> None:
        """Store a user fact (e.g., set_fact('location', 'Chennai'))."""
        self._data.setdefault("facts", {})[key] = value
        self._save()
        logger.debug(f"[{self.session_id[:8]}] Stored fact: {key}={value}")

    def get_fact(self, key: str, default: Any = None) -> Any:
        """Retrieve a stored fact."""
        return self._data.get("facts", {}).get(key, default)

    def get_all_facts(self) -> dict:
        """Return all stored facts."""
        return dict(self._data.get("facts", {}))

    def set_preference(self, key: str, value: str) -> None:
        """Store a user preference (e.g., 'response_style': 'concise')."""
        self._data.setdefault("preferences", {})[key] = value
        self._save()

    def get_preference(self, key: str, default: str = "") -> str:
        """Retrieve a user preference."""
        return self._data.get("preferences", {}).get(key, default)

    def add_summary(self, summary: str) -> None:
        """Append a session summary string."""
        self._data.setdefault("summaries", []).append({
            "summary": summary,
            "created_at": datetime.utcnow().isoformat(),
        })
        self._save()

    def get_summaries(self) -> list[dict]:
        """Return all session summaries."""
        return list(self._data.get("summaries", []))

    def save_history(self, history: list[dict]) -> None:
        """Persist the full conversation history to disk."""
        self._data["raw_history"] = history
        self._data["updated_at"] = datetime.utcnow().isoformat()
        self._save()

    def get_saved_history(self) -> list[dict]:
        """Load previously saved conversation history."""
        return list(self._data.get("raw_history", []))

    def build_memory_context(self) -> str:
        """
        Build a compact memory string to inject into the system prompt.
        Lets the LLM "remember" facts about the user across sessions.

        Example output:
          Known facts: location=Chennai, profession=farmer
          Preferences: response_style=bullet_points
        """
        parts = []
        facts = self.get_all_facts()
        if facts:
            fact_str = ", ".join(f"{k}={v}" for k, v in facts.items())
            parts.append(f"Known facts about the user: {fact_str}")

        prefs = self._data.get("preferences", {})
        if prefs:
            pref_str = ", ".join(f"{k}={v}" for k, v in prefs.items())
            parts.append(f"User preferences: {pref_str}")

        summaries = self.get_summaries()
        if summaries:
            # Only include the 3 most recent summaries
            recent = [s["summary"] for s in summaries[-3:]]
            parts.append("Past session notes:\n- " + "\n- ".join(recent))

        return "\n".join(parts) if parts else ""

    def clear(self) -> None:
        """Wipe all memory for this session."""
        self._data = self._default_data()
        self._save()
        logger.info(f"Cleared persistent memory for session '{self.session_id[:8]}'")

    # ── File I/O ──────────────────────────────────────────────────────────────

    def _load(self) -> dict:
        """Load from disk, or return default structure if file doesn't exist."""
        if self._file.exists():
            try:
                with open(self._file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.debug(f"Loaded memory from {self._file.name}")
                return data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Corrupt memory file — starting fresh. Error: {e}")
        return self._default_data()

    def _save(self) -> None:
        """Write current state to disk atomically."""
        self._data["updated_at"] = datetime.utcnow().isoformat()
        tmp = self._file.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            tmp.replace(self._file)  # Atomic rename
        except IOError as e:
            logger.error(f"Failed to save memory: {e}")
            if tmp.exists():
                tmp.unlink()

    def _default_data(self) -> dict:
        return {
            "session_id": self.session_id,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "facts": {},
            "preferences": {},
            "summaries": [],
            "raw_history": [],
        }


def list_all_sessions() -> list[str]:
    """Return all session IDs that have persistent memory files."""
    mem_dir = Path(settings.memory_dir)
    if not mem_dir.exists():
        return []
    return [f.stem for f in mem_dir.glob("*.json")]
