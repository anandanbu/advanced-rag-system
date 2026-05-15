"""
utils/helpers.py
────────────────
Shared utility functions used across the project.
Keeps other modules clean by centralizing common operations.
"""

import re
import uuid
import hashlib
from typing import Optional
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False


# ── Token Counting ────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """
    Count tokens in a string.
    Uses tiktoken (OpenAI-compatible) if available, else rough word estimate.
    """
    if _HAS_TIKTOKEN:
        return len(_enc.encode(text))
    # Fallback: ~0.75 tokens per word is a good approximation
    return int(len(text.split()) * 0.75)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to a maximum token count.
    Useful for keeping prompts within context limits.
    """
    if not _HAS_TIKTOKEN:
        words = text.split()
        return " ".join(words[: int(max_tokens / 0.75)])
    tokens = _enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _enc.decode(tokens[:max_tokens])


# ── Text Cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalize whitespace and remove non-printable characters.
    Applied to document text before chunking.
    """
    # Collapse multiple newlines into two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Strip non-printable chars (keep newlines/tabs)
    text = re.sub(r"[^\x09\x0a\x0d\x20-\x7e\u00a0-\uffff]", "", text)
    return text.strip()


def sanitize_filename(filename: str) -> str:
    """
    Make a filename safe for filesystem storage.
    Replaces spaces and special chars with underscores.
    """
    safe = re.sub(r"[^\w.\-]", "_", filename)
    return safe[:200]  # Limit length


# ── ID / Hashing ──────────────────────────────────────────────────────────────

def generate_session_id() -> str:
    """Generate a unique session ID for a new conversation."""
    return str(uuid.uuid4())


def hash_document(text: str) -> str:
    """
    SHA-256 hash of document content.
    Used to detect duplicate uploads without re-embedding.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ── File Utilities ────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist. Returns Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_file_extension(filename: str) -> str:
    """Return lowercase file extension without dot. e.g. 'pdf'"""
    return Path(filename).suffix.lstrip(".").lower()


def is_supported_file(filename: str) -> bool:
    """Check if the file type is supported for ingestion."""
    return get_file_extension(filename) in {"pdf", "txt", "docx", "csv"}


# ── Formatting ────────────────────────────────────────────────────────────────

def format_sources(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a readable sources block.
    Injected into the RAG prompt so the LLM can cite sources.
    """
    if not chunks:
        return "No relevant sources found."
    lines = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", "Unknown")
        page = chunk.get("metadata", {}).get("page", "")
        page_str = f" (page {page})" if page else ""
        lines.append(f"[{i}] Source: {source}{page_str}\n{chunk.get('text', '')}")
    return "\n\n".join(lines)
