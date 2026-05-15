"""
scripts/reset_db.py
────────────────────
Utility script to reset the ChromaDB vector store and/or memory files.

Use when:
  - You want to start fresh with a new document set
  - The database becomes corrupted
  - You want to change embedding models (requires re-embedding everything)

Usage:
  python scripts/reset_db.py                    # Reset vector store only
  python scripts/reset_db.py --memory           # Also clear all memory files
  python scripts/reset_db.py --all              # Reset everything
  python scripts/reset_db.py --force            # Skip confirmation prompt
"""

import sys
import shutil
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from utils.logger import get_logger

logger = get_logger("reset_db")


def reset_vector_store(force: bool = False) -> bool:
    """Delete the ChromaDB persistence directory."""
    db_path = Path(settings.chroma_persist_dir)

    if not db_path.exists():
        print(f"  Vector store not found at: {db_path} — nothing to delete.")
        return True

    if not force:
        confirm = input(
            f"  Delete vector store at '{db_path}'? "
            f"This cannot be undone. Type 'yes': "
        ).strip().lower()
        if confirm != "yes":
            print("  Aborted.")
            return False

    try:
        shutil.rmtree(db_path)
        print(f"  ✅ Vector store deleted: {db_path}")
        return True
    except Exception as e:
        print(f"  ❌ Failed to delete vector store: {e}")
        return False


def reset_memory(force: bool = False) -> bool:
    """Delete all persistent memory JSON files."""
    mem_path = Path(settings.memory_dir)

    if not mem_path.exists():
        print(f"  Memory directory not found at: {mem_path} — nothing to delete.")
        return True

    json_files = list(mem_path.glob("*.json"))
    if not json_files:
        print("  No memory files found.")
        return True

    if not force:
        confirm = input(
            f"  Delete {len(json_files)} memory file(s) in '{mem_path}'? "
            f"Type 'yes': "
        ).strip().lower()
        if confirm != "yes":
            print("  Aborted.")
            return False

    deleted = 0
    for f in json_files:
        try:
            f.unlink()
            deleted += 1
        except Exception as e:
            print(f"  ⚠  Could not delete {f.name}: {e}")

    print(f"  ✅ Deleted {deleted}/{len(json_files)} memory file(s).")
    return True


def reset_uploads(force: bool = False) -> bool:
    """Delete all uploaded files from the uploads directory."""
    uploads_path = Path("./data/uploads")

    if not uploads_path.exists():
        print("  Uploads directory not found — nothing to delete.")
        return True

    files = [f for f in uploads_path.iterdir() if f.is_file()]
    if not files:
        print("  No uploaded files found.")
        return True

    if not force:
        confirm = input(
            f"  Delete {len(files)} uploaded file(s)? Type 'yes': "
        ).strip().lower()
        if confirm != "yes":
            print("  Aborted.")
            return False

    for f in files:
        f.unlink()
    print(f"  ✅ Deleted {len(files)} uploaded file(s).")
    return True


def main():
    parser = argparse.ArgumentParser(description="Reset RAG System databases")
    parser.add_argument("--memory", action="store_true", help="Also reset memory files")
    parser.add_argument("--uploads", action="store_true", help="Also delete uploaded files")
    parser.add_argument("--all", action="store_true", help="Reset everything")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompts")
    args = parser.parse_args()

    print("\n" + "═" * 55)
    print("  RAG System — Reset Utility")
    print("═" * 55 + "\n")

    success = True

    # Always reset vector store
    print("📦 Vector Store:")
    success &= reset_vector_store(force=args.force)

    # Optional: memory
    if args.memory or args.all:
        print("\n🧠 Persistent Memory:")
        success &= reset_memory(force=args.force)

    # Optional: uploads
    if args.uploads or args.all:
        print("\n📁 Uploaded Files:")
        success &= reset_uploads(force=args.force)

    print("\n" + "═" * 55)
    if success:
        print("  ✅ Reset complete. Restart the server to reinitialize.")
    else:
        print("  ⚠  Reset incomplete. Check errors above.")
    print("═" * 55 + "\n")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
