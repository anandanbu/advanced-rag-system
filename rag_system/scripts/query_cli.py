"""
scripts/query_cli.py
─────────────────────
Interactive command-line interface for querying the RAG system directly
(without needing the FastAPI server running).

Useful for:
  - Quick local testing
  - Debugging retrieval quality
  - Demonstrations without a UI

Usage:
  python scripts/query_cli.py
  python scripts/query_cli.py --no-critic        # Skip critic evaluation
  python scripts/query_cli.py --session my-sess  # Use a specific session ID
  python scripts/query_cli.py --show-sources      # Print retrieved chunks
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.pipeline import RAGPipeline
from utils.helpers import generate_session_id
from utils.logger import get_logger

logger = get_logger("query_cli")

SEPARATOR = "─" * 60
BANNER = """
╔══════════════════════════════════════════════════════════╗
║           Advanced RAG System — Interactive CLI          ║
║  Type 'quit' or 'exit' to stop | 'clear' to reset chat  ║
║  Type 'stats' for vector store stats                     ║
╚══════════════════════════════════════════════════════════╝
"""


def print_sources(sources: list[dict]):
    """Pretty-print retrieved source chunks."""
    if not sources:
        return
    print(f"\n  📚 Sources ({len(sources)} chunks retrieved):")
    for i, src in enumerate(sources, 1):
        meta = src.get("metadata", {})
        source_name = meta.get("source", "Unknown")
        page = meta.get("page", "")
        page_str = f" p.{page}" if page else ""
        score = src.get("score", 0)
        print(f"  [{i}] {source_name}{page_str}  (score: {score:.3f})")
        # Show first 120 chars of the chunk
        snippet = src.get("text", "")[:120].replace("\n", " ")
        print(f"      \"{snippet}…\"")


def print_critic_info(response):
    """Print critic evaluation results."""
    score = response.critic_score
    passed = response.critic_passed
    hallucinated = response.hallucination_detected
    iterations = response.improvement_iterations

    status = "✅ PASS" if passed else "❌ FAIL"
    hall_flag = "  ⚠️ HALLUCINATION DETECTED" if hallucinated else ""

    print(f"\n  🔍 Critic: {status}  score={score:.2f}{hall_flag}")
    if iterations > 0:
        print(f"  🔁 Self-improved in {iterations} iteration(s)")


def show_stats():
    """Display current vector store statistics."""
    from vectorstore.chroma_store import get_vector_store
    store = get_vector_store()
    stats = store.get_stats()
    print(f"\n{SEPARATOR}")
    print(f"  Vector Store Stats")
    print(SEPARATOR)
    print(f"  Collection  : {stats['collection_name']}")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Embedding   : {stats['embedding_model']} (dim={stats['embedding_dim']})")
    print(f"  Sources ({len(stats['sources'])}):")
    for src in stats["sources"]:
        print(f"    • {src}")
    print(SEPARATOR)


def main():
    parser = argparse.ArgumentParser(description="RAG System Interactive CLI")
    parser.add_argument("--no-critic", action="store_true", help="Skip critic evaluation")
    parser.add_argument("--show-sources", action="store_true", help="Print retrieved source chunks")
    parser.add_argument("--session", default=None, help="Session ID (auto-generated if not set)")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve")
    args = parser.parse_args()

    session_id = args.session or generate_session_id()
    use_critic = not args.no_critic
    show_sources = args.show_sources

    print(BANNER)
    print(f"  Session ID : {session_id}")
    print(f"  Critic     : {'enabled' if use_critic else 'disabled'}")
    print(f"  Show sources: {show_sources}")
    print(SEPARATOR)

    # Initialize pipeline (loads models — takes a moment on first run)
    print("\n  Initializing pipeline (loading models)...", end=" ", flush=True)
    try:
        pipeline = RAGPipeline()
        print("✅ Ready!\n")
    except Exception as e:
        print(f"\n  ❌ Failed to initialize: {e}")
        print("  Make sure your .env file is configured correctly.")
        sys.exit(1)

    # Show initial stats
    show_stats()

    # ── REPL loop ─────────────────────────────────────────────────────────────
    while True:
        try:
            query = input("\n  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye! 👋")
            break

        if not query:
            continue

        # ── Commands ──────────────────────────────────────────────────────────
        if query.lower() in ("quit", "exit", "q"):
            print("\n  Goodbye! 👋")
            break

        if query.lower() == "clear":
            from memory.conversation import clear_session
            clear_session(session_id)
            print("  ✅ Conversation history cleared.")
            continue

        if query.lower() == "stats":
            show_stats()
            continue

        if query.lower() == "history":
            from memory.conversation import get_history_with_timestamps
            history = get_history_with_timestamps(session_id)
            if not history:
                print("  (No history yet)")
            else:
                print(f"\n  Conversation History ({len(history)} messages):")
                for msg in history:
                    role = "You" if msg["role"] == "user" else "AI"
                    print(f"  [{role}] {msg['content'][:100]}")
            continue

        # ── Run pipeline ──────────────────────────────────────────────────────
        print(f"\n  🤔 Thinking...", end="", flush=True)
        try:
            result = pipeline.run(
                query=query,
                session_id=session_id,
                use_critic=use_critic,
                top_k=args.top_k,
            )
        except Exception as e:
            print(f"\n  ❌ Error: {e}")
            logger.error(f"Pipeline error: {e}", exc_info=True)
            continue

        # ── Display answer ────────────────────────────────────────────────────
        mode_label = "📄 RAG" if result.mode == "rag" else "💬 Chat"
        latency = f"{result.latency_ms:.0f}ms"

        print(f"\r{SEPARATOR}")
        print(f"  {mode_label}  [{latency}]")
        print(SEPARATOR)
        print()

        # Word-wrap the answer at 65 chars
        words = result.answer.split()
        line = "  "
        for word in words:
            if len(line) + len(word) + 1 > 67:
                print(line)
                line = "  " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)

        print()

        if use_critic and result.mode == "rag":
            print_critic_info(result)

        if show_sources and result.sources:
            print_sources(result.sources)

        print(SEPARATOR)


if __name__ == "__main__":
    main()
