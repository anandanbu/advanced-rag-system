"""
rag/pipeline.py
───────────────
Core RAG orchestrator — optimized for Render free tier (512 MB RAM).

BUGS FIXED IN THIS VERSION:
════════════════════════════════════════════════════════════════

BUG 1 — "summarize this document" → CONVERSATIONAL (empty response)
  Root cause: has_context check used a similarity score threshold (0.3).
  Command-type queries ("summarize", "explain", "overview") embed as
  INSTRUCTIONS, not as document content. They score 0.10–0.22 against
  any real document — always below 0.3 — so the system fell through to
  conversational mode and Groq replied "please paste the text."

  Fix: _is_document_command() detects command keywords. When documents
  exist in the store AND the query is a document command, we fetch ALL
  available chunks regardless of similarity score and force RAG mode.

BUG 2 — General threshold 0.3 too aggressive
  Queries like "what does section 3 say" are specific but score ~0.25
  against their matching chunk due to vocabulary mismatch.

  Fix: Lowered threshold from 0.3 → 0.15. This means we use RAG
  whenever ANY chunk has even weak relevance, which is almost always
  better than falling back to a context-free LLM response.

BUG 3 — Empty store → silent conversational fallback
  After Render restarts (sleeps after 15 min idle), /tmp/chroma_db
  is wiped. All uploaded documents vanish. The pipeline silently
  fell back to conversational, making users think the AI was broken.

  Fix: _check_store_empty() detects this and returns a clear message
  telling the user to re-upload their documents.

BUG 4 — Critic instantiated at pipeline __init__ (wasted memory)
  get_critic() is stateless (creates CriticEvaluator which holds get_llm()).
  Creating it in __init__ means it's always allocated even when use_critic=False.

  Fix: Lazy-create critic only when use_critic=True and score check needed.

MEMORY OPTIMIZATIONS:
  - Lazy critic loading
  - gc.collect() after large operations
  - Smaller context window passed to LLM (max 3000 chars per source)
════════════════════════════════════════════════════════════════
"""

import gc
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from config.settings import settings
from vectorstore.chroma_store import get_vector_store
from llm.groq_client import get_llm
from critic.evaluator import CriticResult
from rag.prompt_templates import (
    RAG_SYSTEM_PROMPT,
    RAG_USER_PROMPT_TEMPLATE,
    CONVERSATIONAL_SYSTEM_PROMPT,
)
from memory.conversation import add_turn, get_history
from memory.persistent import PersistentMemory
from utils.logger import get_logger
from utils.helpers import format_sources

logger = get_logger(__name__)

# ── Threshold ─────────────────────────────────────────────────────────────────
# Lowered from 0.3 → 0.15.
# 0.3 was too strict — command queries and paraphrased questions scored 0.15–0.28
# and were wrongly routed to conversational mode.
# At 0.15 we use RAG whenever there's any weak relevance (almost always better).
SIMILARITY_THRESHOLD = 0.15

# Keywords that signal a document-command query.
# These queries embed as instructions (not document content) so their
# similarity scores are always low regardless of what's in the store.
DOCUMENT_COMMAND_KEYWORDS = {
    "summarize", "summary", "tldr", "overview", "explain",
    "analyze", "analyse", "describe", "outline", "review",
    "what is this", "what does this", "what are the main",
    "key points", "key takeaways", "main points", "main topics",
    "tell me about", "what have i uploaded", "what files",
    "what documents", "list documents", "list files",
    "give me a", "walk me through",
}


# ── Response Schema ───────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    """Structured output of one pipeline run — returned to the API."""
    answer: str
    session_id: str
    query: str
    sources: list[dict] = field(default_factory=list)
    critic_score: float = 0.0
    critic_passed: bool = True
    hallucination_detected: bool = False
    improvement_iterations: int = 0
    retrieval_count: int = 0
    latency_ms: float = 0.0
    mode: str = "rag"
    error: Optional[str] = None


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Main pipeline. Instantiate once, call .run() per query.
    Critic is lazy-loaded — only created when actually needed.
    """

    def __init__(self):
        self._vector_store = get_vector_store()
        self._llm = get_llm()
        self._critic = None  # lazy — created only when use_critic=True

    def run(
        self,
        query: str,
        session_id: str,
        top_k: Optional[int] = None,
        use_critic: bool = True,
        filter_source: Optional[str] = None,
    ) -> RAGResponse:
        start_time = time.time()
        logger.info(f"[{session_id[:8]}] Query: '{query[:80]}'")

        # ── Load memory ───────────────────────────────────────────────────────
        history = get_history(session_id)
        persistent_mem = PersistentMemory(session_id)
        memory_context = persistent_mem.build_memory_context()

        # ── Route the query ───────────────────────────────────────────────────
        store_count = self._vector_store._collection.count()

        # Case A: Store is completely empty — user needs to upload first
        if store_count == 0:
            return self._no_documents_response(
                query, session_id, history, persistent_mem, start_time
            )

        # Case B: Document command query ("summarize", "explain", etc.)
        # These score low on similarity but MUST use document context.
        is_doc_command = _is_document_command(query)

        if is_doc_command:
            logger.info(
                f"[{session_id[:8]}] Document command detected — "
                f"fetching all top chunks regardless of score"
            )
            # Fetch more chunks for summarization tasks (need broader coverage)
            chunks = self._vector_store.similarity_search(
                query=query,
                top_k=min(store_count, (top_k or settings.retriever_top_k) * 2),
                filter_metadata={"source": filter_source} if filter_source else None,
            )
            # For document commands use ALL returned chunks, ignore score
            has_context = bool(chunks)

        else:
            # Case C: Normal question — use similarity threshold
            chunks = self._vector_store.similarity_search(
                query=query,
                top_k=top_k or settings.retriever_top_k,
                filter_metadata={"source": filter_source} if filter_source else None,
            )
            has_context = bool(chunks) and chunks[0]["score"] > SIMILARITY_THRESHOLD

        # ── Conversational fallback ───────────────────────────────────────────
        if not has_context:
            return self._conversational_response(
                query, session_id, history, memory_context, persistent_mem, start_time
            )

        # ── Build RAG context ─────────────────────────────────────────────────
        context = format_sources(chunks)

        # ── Build prompts ─────────────────────────────────────────────────────
        system_prompt = RAG_SYSTEM_PROMPT.format(
            memory_context=f"\n\n{memory_context}" if memory_context else ""
        )
        user_prompt = RAG_USER_PROMPT_TEMPLATE.format(
            context=context,
            question=query,
        )

        # ── Generate answer ───────────────────────────────────────────────────
        answer = self._llm.generate(
            user_message=user_prompt,
            system_prompt=system_prompt,
            history=history,
        )

        # ── Critic evaluation (lazy, optional) ───────────────────────────────
        critic_result = CriticResult(score=0.75, critic_passed=True)  # safe default
        improvement_iterations = 0

        if use_critic:
            if self._critic is None:
                from critic.evaluator import get_critic
                self._critic = get_critic()

            critic_result = self._critic.evaluate(
                question=query,
                context=context,
                response=answer,
            )

            if not critic_result.passed:
                logger.info(
                    f"Critic score {critic_result.score:.2f} below threshold — improving"
                )
                from rag.improver import improve_answer
                answer, critic_result, improvement_iterations = improve_answer(
                    question=query,
                    context=context,
                    initial_answer=answer,
                    initial_critic_result=critic_result,
                    critic=self._critic,
                )

        # ── Save to memory ────────────────────────────────────────────────────
        add_turn(session_id, query, answer)
        persistent_mem.save_history(get_history(session_id))

        # Free temporary objects — important on 512 MB Render
        gc.collect()

        latency = (time.time() - start_time) * 1000
        logger.info(
            f"[{session_id[:8]}] Done | mode=rag | "
            f"score={critic_result.score:.2f} | "
            f"chunks={len(chunks)} | latency={latency:.0f}ms"
        )

        return RAGResponse(
            answer=answer,
            session_id=session_id,
            query=query,
            sources=chunks,
            critic_score=critic_result.score,
            critic_passed=critic_result.passed,
            hallucination_detected=critic_result.hallucination_detected,
            improvement_iterations=improvement_iterations,
            retrieval_count=len(chunks),
            latency_ms=round(latency, 2),
            mode="rag",
        )

    # ── Private response builders ─────────────────────────────────────────────

    def _conversational_response(
        self,
        query: str,
        session_id: str,
        history: list[dict],
        memory_context: str,
        persistent_mem: PersistentMemory,
        start_time: float,
    ) -> RAGResponse:
        """Used when documents exist but query has no relevant match."""
        logger.info(f"[{session_id[:8]}] Low similarity → conversational mode")
        system_prompt = CONVERSATIONAL_SYSTEM_PROMPT.format(
            memory_context=f"\n\n{memory_context}" if memory_context else ""
        )
        answer = self._llm.generate(
            user_message=query,
            system_prompt=system_prompt,
            history=history,
        )
        add_turn(session_id, query, answer)
        persistent_mem.save_history(get_history(session_id))
        latency = (time.time() - start_time) * 1000
        return RAGResponse(
            answer=answer,
            session_id=session_id,
            query=query,
            latency_ms=round(latency, 2),
            mode="conversational",
        )

    def _no_documents_response(
        self,
        query: str,
        session_id: str,
        history: list[dict],
        persistent_mem: PersistentMemory,
        start_time: float,
    ) -> RAGResponse:
        """
        Used when the vector store is empty.
        Tells the user to upload documents instead of silently
        falling back to a context-free LLM answer.
        """
        logger.info(f"[{session_id[:8]}] Store is empty → no-documents response")
        answer = (
            "I don't have any documents to work with yet. "
            "Please upload a PDF, TXT, DOCX, or CSV file using the upload button "
            "in the sidebar, and then ask your question again.\n\n"
            "Note: If you already uploaded documents and I'm asking again, "
            "the server may have restarted (this clears the document store on "
            "the free hosting tier). Please re-upload your file."
        )
        add_turn(session_id, query, answer)
        persistent_mem.save_history(get_history(session_id))
        latency = (time.time() - start_time) * 1000
        return RAGResponse(
            answer=answer,
            session_id=session_id,
            query=query,
            latency_ms=round(latency, 2),
            mode="conversational",
        )


# ── Query Router Helpers ──────────────────────────────────────────────────────

def _is_document_command(query: str) -> bool:
    """
    Returns True if the query is a document-command type.

    These queries embed as instructions, not document content, so their
    cosine similarity against stored chunks is always low (0.10–0.22)
    regardless of what's actually in the store.

    We detect them by keyword matching and bypass the score threshold,
    forcing RAG mode so the LLM always gets document context.
    """
    q = query.lower().strip()
    return any(keyword in q for keyword in DOCUMENT_COMMAND_KEYWORDS)


# ── Ingestion Pipeline ────────────────────────────────────────────────────────

def ingest_file(file_bytes: bytes, filename: str) -> dict:
    """
    Full ingestion: load → chunk → embed → store.
    Called by the upload endpoint via run_in_executor (non-blocking).
    """
    from ingestion.loader import load_from_bytes
    from ingestion.chunker import chunk_documents
    from utils.helpers import is_supported_file

    if not is_supported_file(filename):
        raise ValueError(
            f"Unsupported file: '{filename}'. Supported: PDF, TXT, DOCX, CSV"
        )

    logger.info(f"Ingesting: '{filename}'")

    documents = load_from_bytes(file_bytes, filename)
    if not documents:
        raise ValueError(f"No text could be extracted from '{filename}'")

    chunks = chunk_documents(documents)
    if not chunks:
        raise ValueError(f"No chunks produced from '{filename}'")

    store = get_vector_store()
    added = store.add_chunks(chunks)

    # Free memory after large ingestion
    gc.collect()

    stats = {
        "filename": filename,
        "pages_loaded": len(documents),
        "chunks_created": len(chunks),
        "chunks_added": added,
        "chunks_skipped": len(chunks) - added,
    }
    logger.info(f"Ingestion complete: {stats}")
    return stats
