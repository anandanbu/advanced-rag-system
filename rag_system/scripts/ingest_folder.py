"""
scripts/ingest_folder.py
─────────────────────────
CLI script to bulk ingest all supported documents from a folder.

Usage:
  python scripts/ingest_folder.py --folder ./my_documents
  python scripts/ingest_folder.py --folder ./data --recursive
  python scripts/ingest_folder.py --folder ./docs --clear-first

This is useful for:
  - Initial knowledge base population
  - Batch document updates
  - CI/CD pipelines that re-ingest updated docs
"""

import sys
import argparse
import time
from pathlib import Path

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.loader import load_document
from ingestion.chunker import chunk_documents
from vectorstore.chroma_store import get_vector_store
from utils.helpers import is_supported_file, get_file_extension
from utils.logger import get_logger

logger = get_logger("ingest_folder")

SUPPORTED_EXTENSIONS = {"pdf", "txt", "docx", "csv"}


def find_documents(folder: str, recursive: bool = False) -> list[Path]:
    """Find all supported documents in a folder."""
    folder_path = Path(folder)
    if not folder_path.exists():
        logger.error(f"Folder not found: {folder}")
        sys.exit(1)

    files = []
    pattern = "**/*" if recursive else "*"
    for f in folder_path.glob(pattern):
        if f.is_file() and get_file_extension(f.name) in SUPPORTED_EXTENSIONS:
            files.append(f)

    return sorted(files)


def ingest_file(file_path: Path, store) -> dict:
    """Ingest a single file and return stats."""
    try:
        docs = load_document(str(file_path), original_filename=file_path.name)
        chunks = chunk_documents(docs)
        added = store.add_chunks(chunks)
        return {
            "file": file_path.name,
            "status": "success",
            "pages": len(docs),
            "chunks": len(chunks),
            "added": added,
            "skipped": len(chunks) - added,
        }
    except Exception as e:
        logger.error(f"Failed to ingest '{file_path.name}': {e}")
        return {
            "file": file_path.name,
            "status": "error",
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Bulk ingest documents into the RAG vector store"
    )
    parser.add_argument(
        "--folder", "-f",
        required=True,
        help="Path to folder containing documents",
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search subfolders recursively",
    )
    parser.add_argument(
        "--clear-first",
        action="store_true",
        help="WARNING: Clears the entire vector store before ingesting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be ingested without actually doing it",
    )
    args = parser.parse_args()

    print("\n" + "═" * 55)
    print("  RAG System — Bulk Document Ingestion")
    print("═" * 55)

    # Find documents
    files = find_documents(args.folder, recursive=args.recursive)
    if not files:
        print(f"\n⚠  No supported documents found in: {args.folder}")
        print(f"   Supported formats: {', '.join(SUPPORTED_EXTENSIONS).upper()}")
        sys.exit(0)

    print(f"\n📂 Folder  : {args.folder}")
    print(f"📄 Found   : {len(files)} document(s)")
    print(f"🔁 Recursive: {args.recursive}")
    print()

    for f in files:
        print(f"  • {f.name}")

    if args.dry_run:
        print("\n[DRY RUN] No files were ingested.")
        return

    # Initialize vector store
    store = get_vector_store()

    # Optionally clear first
    if args.clear_first:
        print("\n⚠  --clear-first: Deleting entire collection…")
        confirm = input("  Type 'yes' to confirm: ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            sys.exit(0)
        store.delete_collection()
        print("  Collection cleared.\n")
        # Re-initialize after clearing
        store = get_vector_store()

    # Ingest files
    print(f"\n🚀 Starting ingestion...\n{'─' * 40}")
    start_time = time.time()

    results = []
    success_count = 0
    total_chunks_added = 0

    for i, file_path in enumerate(files, 1):
        print(f"[{i:02d}/{len(files):02d}] {file_path.name}...", end=" ", flush=True)
        result = ingest_file(file_path, store)
        results.append(result)

        if result["status"] == "success":
            success_count += 1
            total_chunks_added += result["added"]
            print(
                f"✓  {result['pages']} page(s) → "
                f"{result['chunks']} chunks "
                f"({result['added']} added, {result['skipped']} skipped)"
            )
        else:
            print(f"✗  ERROR: {result.get('error', 'Unknown error')}")

    elapsed = time.time() - start_time

    # Final summary
    print(f"\n{'═' * 55}")
    print("  Ingestion Summary")
    print(f"{'═' * 55}")
    print(f"  Files processed  : {len(files)}")
    print(f"  Succeeded        : {success_count}")
    print(f"  Failed           : {len(files) - success_count}")
    print(f"  Chunks added     : {total_chunks_added}")
    print(f"  Time elapsed     : {elapsed:.1f}s")

    final_stats = store.get_stats()
    print(f"  Total in store   : {final_stats['total_chunks']} chunks")
    print(f"{'═' * 55}\n")

    if success_count < len(files):
        sys.exit(1)  # Non-zero exit for CI/CD pipelines


if __name__ == "__main__":
    main()
