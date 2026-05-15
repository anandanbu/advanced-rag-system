"""
tests/conftest.py
──────────────────
Shared pytest configuration and fixtures.

This file is automatically loaded by pytest before any tests run.
Use it for:
  - Setting test environment variables
  - Shared fixtures across all test files
  - Patching global singletons before they're imported
"""

import os
import pytest

# ── Set test environment variables BEFORE any module imports ──────────────────
# This ensures settings.py reads test values instead of requiring a real .env

os.environ.setdefault("GROQ_API_KEY", "test-api-key-for-testing-only")
os.environ.setdefault("GROQ_MODEL", "llama3-8b-8192")
os.environ.setdefault("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
os.environ.setdefault("EMBEDDING_DEVICE", "cpu")
os.environ.setdefault("CHROMA_PERSIST_DIR", "/tmp/test_chroma")
os.environ.setdefault("CHROMA_COLLECTION_NAME", "test_collection")
os.environ.setdefault("MEMORY_DIR", "/tmp/test_memory")
os.environ.setdefault("LOG_LEVEL", "WARNING")  # Quiet logs during tests
os.environ.setdefault("CHUNK_SIZE", "256")
os.environ.setdefault("CHUNK_OVERLAP", "32")
os.environ.setdefault("RETRIEVER_TOP_K", "3")
os.environ.setdefault("CRITIC_SCORE_THRESHOLD", "0.6")
os.environ.setdefault("CRITIC_MAX_RETRIES", "1")
os.environ.setdefault("MAX_HISTORY_TURNS", "5")
os.environ.setdefault("API_HOST", "0.0.0.0")
os.environ.setdefault("API_PORT", "8000")
os.environ.setdefault("API_RELOAD", "false")
os.environ.setdefault("ALLOWED_ORIGINS", "*")


# ── Shared Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_conversation_memory():
    """
    Auto-use fixture: clears in-memory conversation sessions before each test.
    Prevents test pollution between test functions.
    """
    import memory.conversation as conv
    conv._sessions.clear()
    yield
    conv._sessions.clear()


@pytest.fixture
def sample_text():
    return (
        "Retrieval-Augmented Generation (RAG) is a technique that enhances "
        "large language model outputs by incorporating external knowledge. "
        "It retrieves relevant documents from a knowledge base and uses them "
        "to ground the model's response, reducing hallucinations significantly. "
        "RAG combines the parametric knowledge of LLMs with non-parametric "
        "retrieval to produce more accurate and verifiable answers."
    )


@pytest.fixture
def sample_documents():
    return [
        {
            "text": "Photosynthesis converts light energy into chemical energy stored in glucose.",
            "metadata": {"source": "biology.txt", "file_type": "txt"},
        },
        {
            "text": "Chlorophyll is the pigment responsible for absorbing light in plants.",
            "metadata": {"source": "biology.txt", "file_type": "txt"},
        },
        {
            "text": "The Calvin cycle uses CO2 to produce carbohydrates in the stroma.",
            "metadata": {"source": "biology.txt", "file_type": "txt"},
        },
    ]
