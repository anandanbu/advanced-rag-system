"""
tools/document_summary.py
──────────────────────────
Generates concise summaries of ingested documents.

Two modes:
  1. Quick summary  — summarize the first N chunks of a document
  2. Map-reduce summary — summarize each chunk, then summarize the summaries
                          (handles very long documents)

Called via:
  - POST /upload response (auto-summary on ingest)
  - GET /upload/summary/{source_name} endpoint
  - Agents via the summarize tool
"""

from vectorstore.chroma_store import get_vector_store
from llm.groq_client import get_llm
from rag.prompt_templates import SUMMARY_PROMPT_TEMPLATE
from utils.logger import get_logger

logger = get_logger(__name__)


def summarize_document(source_name: str, max_chunks: int = 10) -> str:
    """
    Summarize a document that is already ingested in the vector store.

    Args:
        source_name : Exact source name as stored in ChromaDB metadata
        max_chunks  : Max chunks to include (prevents token overflow)

    Returns:
        A 3-5 sentence summary string.
    """
    store = get_vector_store()
    llm = get_llm()

    # Retrieve all chunks for this source
    results = store._collection.get(
        where={"source": source_name},
        include=["documents", "metadatas"],
    )

    docs = results.get("documents", [])
    if not docs:
        return f"No document found with source: '{source_name}'"

    # Use first max_chunks to avoid context overflow
    selected = docs[:max_chunks]
    combined_text = "\n\n".join(selected)

    logger.info(f"Summarizing '{source_name}' ({len(docs)} chunks, using {len(selected)})")

    # If document is short, summarize directly
    if len(combined_text) < 6000:
        return _direct_summary(llm, source_name, combined_text)

    # For long documents: map-reduce
    return _map_reduce_summary(llm, source_name, selected)


def _direct_summary(llm, source_name: str, content: str) -> str:
    """Single-pass summary for shorter documents."""
    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        source_name=source_name,
        content=content[:5000],
    )
    return llm.generate(
        user_message=prompt,
        temperature=0.1,
        max_tokens=400,
    )


def _map_reduce_summary(llm, source_name: str, chunks: list[str]) -> str:
    """
    Map-reduce for long documents:
      1. Summarize each chunk individually (map)
      2. Combine chunk summaries into a final summary (reduce)
    """
    logger.info(f"Using map-reduce for '{source_name}' ({len(chunks)} chunks)")

    # Map: summarize each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        summary = llm.generate(
            user_message=f"Summarize this excerpt in 2 sentences:\n\n{chunk[:1500]}",
            temperature=0.0,
            max_tokens=150,
        )
        chunk_summaries.append(summary)
        logger.debug(f"Chunk {i+1}/{len(chunks)} summarized")

    # Reduce: combine all chunk summaries
    combined = "\n".join(f"- {s}" for s in chunk_summaries)
    final_prompt = (
        f"These are summaries of different sections of '{source_name}':\n\n"
        f"{combined}\n\n"
        f"Write a cohesive 4-5 sentence summary of the entire document."
    )
    return llm.generate(
        user_message=final_prompt,
        temperature=0.1,
        max_tokens=400,
    )


def summarize_text(text: str, max_length: int = 300) -> str:
    """
    Summarize any arbitrary text string.
    Used by agents as a utility tool.
    """
    if len(text) <= max_length:
        return text
    llm = get_llm()
    return llm.generate(
        user_message=f"Summarize in 2-3 sentences:\n\n{text[:3000]}",
        temperature=0.0,
        max_tokens=200,
    )
