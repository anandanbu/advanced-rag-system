"""
config/settings.py
──────────────────
Single source of truth for all configuration.
Pydantic BaseSettings automatically reads from .env file.
Access anywhere: from config.settings import settings
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ────────────────────────────────────────────────────────────────────
    groq_api_key: str = Field(..., description="Groq API key")
    groq_model: str = Field("llama3-70b-8192")
    groq_max_tokens: int = Field(2048)
    groq_temperature: float = Field(0.2)

    # ── Embeddings ─────────────────────────────────────────────────────────────
    embedding_model: str = Field("all-MiniLM-L6-v2")
    embedding_device: str = Field("cpu")

    # ── Vector Store ───────────────────────────────────────────────────────────
    chroma_persist_dir: str = Field("./data/chroma_db")
    chroma_collection_name: str = Field("rag_documents")

    # ── Chunking ───────────────────────────────────────────────────────────────
    chunk_size: int = Field(512)
    chunk_overlap: int = Field(64)

    # ── Retrieval ──────────────────────────────────────────────────────────────
    retriever_top_k: int = Field(5)

    # ── Critic ─────────────────────────────────────────────────────────────────
    critic_score_threshold: float = Field(0.6)
    critic_max_retries: int = Field(2)

    # ── Memory ─────────────────────────────────────────────────────────────────
    memory_dir: str = Field("./data/memory")
    max_history_turns: int = Field(10)

    # ── API ────────────────────────────────────────────────────────────────────
    api_host: str = Field("0.0.0.0")
    api_port: int = Field(8000)
    api_reload: bool = Field(True)
    allowed_origins: str = Field("*")

    # ── Logging ────────────────────────────────────────────────────────────────
    log_level: str = Field("INFO")


@lru_cache()
def get_settings() -> Settings:
    """
    Cached singleton — settings are loaded once and reused.
    lru_cache means .env is read only on first call.
    """
    return Settings()


# Convenience import: from config.settings import settings
settings = get_settings()
