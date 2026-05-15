"""
rag/pipeline.py
───────────────
The core RAG orchestrator — ties every module together.

Full pipeline for a single user query:
  1.  Load conversation history from memory
  2.  Load long-term user facts from persistent memory
  3.  Embed the query using sentence-transformers
  4.  Retrieve top-k relevant chunks from ChromaDB
  5.  Format context + build the augmented prompt
  6.  Generate response via Groq (Llama 3)
  7.  Evaluate response with the Critic
  8.  If score < threshold → run self-improvement loop
  9.  Save the turn to conversation memory
  10. Optionally persist memory to disk
  11. Return structured RAGResponse

This single file is the "glue" of the entire system.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from config.settings import settings
from vectorstore.chroma_store import get_vector_store
from llm.groq_client import get_llm
from critic.evaluator import get_critic, CriticResult
from rag.improver import improve_answer
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


# ── Response Schema ───────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    """
    The complete, structured output of one RAG pipeline run.
    Returned by the API to the frontend.
    """
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
    mode: str = "rag"               # "rag" or "conversational"
    error: Optional[str] = None


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Main pipeline class. Instantiate once and call .run() per query.
    """

    def __init__(self):
        self._vector_store = get_vector_store()
        self._llm = get_llm()
        self._critic = get_critic()

    def run(
        self,
        query: str,
        session_id: str,
        top_k: Optional[int] = None,
        use_critic: bool = True,
        filter_source: Optional[str] = None,
    ) -> RAGResponse:
        """
        Execute the full RAG pipeline for a user query.

        Args:
            query         : The user's question
            session_id    : Session identifier for memory
            top_k         : Override retrieval count
            use_critic    : Whether to evaluate with critic (disable for speed)
            filter_source : Restrict retrieval to a specific source document

        Returns:
            RAGResponse dataclass with answer and all metadata
        """
        start_time = time.time()
        logger.info(f"[{session_id[:8]}] Query: '{query[:80]}'")

        # ── Step 1: Load memory ───────────────────────────────────────────────
        history = get_history(session_id)
        persistent_mem = PersistentMemory(session_id)
        memory_context = persistent_mem.build_memory_context()

        # ── Step 2: Retrieve relevant chunks ─────────────────────────────────
        filter_dict = {"source": filter_source} if filter_source else None
        chunks = self._vector_store.similarity_search(
            query=query,
            top_k=top_k or settings.retriever_top_k,
            filter_metadata=filter_dict,
        )

        # ── Step 3: Decide pipeline mode ──────────────────────────────────────
        # If no relevant chunks found OR store is empty → conversational fallback
        has_context = bool(chunks) and chunks[0]["score"] > 0.3

        if not has_context:
            return self._conversational_response(
                query=query,
                session_id=session_id,
                history=history,
                memory_context=memory_context,
                persistent_mem=persistent_mem,
                start_time=start_time,
            )

        # ── Step 4: Build context string ──────────────────────────────────────
        context = format_sources(chunks)

        # ── Step 5: Build augmented prompt ────────────────────────────────────
        system_prompt = RAG_SYSTEM_PROMPT.format(
            memory_context=f"\n\n{memory_context}" if memory_context else ""
        )
        user_prompt = RAG_USER_PROMPT_TEMPLATE.format(
            context=context,
            question=query,
        )

        # ── Step 6: Generate answer ───────────────────────────────────────────
        answer = self._llm.generate(
            user_message=user_prompt,
            system_prompt=system_prompt,
            history=history,
        )

        # ── Step 7: Critic evaluation ─────────────────────────────────────────
        critic_result = CriticResult()  # default passing result
        improvement_iterations = 0

        if use_critic:
            critic_result = self._critic.evaluate(
                question=query,
                context=context,
                response=answer,
            )

            # ── Step 8: Self-improvement loop ─────────────────────────────────
            if not critic_result.passed:
                logger.info(
                    f"Critic score {critic_result.score:.2f} below threshold "
                    f"{settings.critic_score_threshold} — running self-improvement"
                )
                answer, critic_result, improvement_iterations = improve_answer(
                    question=query,
                    context=context,
                    initial_answer=answer,
                    initial_critic_result=critic_result,
                    critic=self._critic,
                )

        # ── Step 9: Save to memory ────────────────────────────────────────────
        add_turn(session_id, query, answer)
        persistent_mem.save_history(get_history(session_id))

        latency = (time.time() - start_time) * 1000
        logger.info(
            f"[{session_id[:8]}] RAG complete | "
            f"score={critic_result.score:.2f} | "
            f"chunks={len(chunks)} | "
            f"iterations={improvement_iterations} | "
            f"latency={latency:.0f}ms"
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

    def _conversational_response(
        self,
        query: str,
        session_id: str,
        history: list[dict],
        memory_context: str,
        persistent_mem: PersistentMemory,
        start_time: float,
    ) -> RAGResponse:
        """
        Fallback path when no relevant documents are found.
        Responds using general LLM knowledge + conversation history.
        """
        logger.info(f"[{session_id[:8]}] No relevant docs → conversational mode")

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


# ── Ingestion Pipeline ────────────────────────────────────────────────────────

def ingest_file(file_bytes: bytes, filename: str) -> dict:
    """
    Full document ingestion pipeline:
      load → chunk → embed → store in ChromaDB

    Called by the /upload API endpoint.

    Returns:
        Dict with ingestion stats
    """
    from ingestion.loader import load_from_bytes
    from ingestion.chunker import chunk_documents
    from utils.helpers import is_supported_file

    if not is_supported_file(filename):
        raise ValueError(
            f"Unsupported file: '{filename}'. "
            "Supported: PDF, TXT, DOCX, CSV"
        )

    logger.info(f"Ingesting file: '{filename}'")

    # Load raw text from file
    documents = load_from_bytes(file_bytes, filename)
    if not documents:
        raise ValueError(f"No text could be extracted from '{filename}'")

    # Split into chunks
    chunks = chunk_documents(documents)
    if not chunks:
        raise ValueError(f"No chunks produced from '{filename}'")

    # Store in ChromaDB
    store = get_vector_store()
    added = store.add_chunks(chunks)

    stats = {
        "filename": filename,
        "pages_loaded": len(documents),
        "chunks_created": len(chunks),
        "chunks_added": added,
        "chunks_skipped": len(chunks) - added,
    }

    logger.info(f"Ingestion complete: {stats}")
    return stats
