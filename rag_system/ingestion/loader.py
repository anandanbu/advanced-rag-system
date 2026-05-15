"""
ingestion/loader.py
───────────────────
Loads documents from multiple formats (PDF, TXT, DOCX, CSV) into
a normalized list of {text, metadata} dicts.

Each returned document dict has:
  - text     : raw extracted string
  - metadata : {source, file_type, page (if applicable), row (for CSV)}

This normalized structure means the rest of the pipeline never needs
to know what format the source was.
"""

import csv
import io
from pathlib import Path
from typing import Optional

from utils.logger import get_logger
from utils.helpers import clean_text, get_file_extension

logger = get_logger(__name__)


def load_document(file_path: str, original_filename: Optional[str] = None) -> list[dict]:
    """
    Main entry point. Dispatches to format-specific loaders.

    Args:
        file_path: Path to the file on disk.
        original_filename: Original name (used for metadata). Defaults to file_path stem.

    Returns:
        List of dicts: [{text: str, metadata: dict}, ...]
    """
    path = Path(file_path)
    ext = get_file_extension(path.name)
    source_name = original_filename or path.name

    logger.info(f"Loading document: {source_name} (type={ext})")

    loaders = {
        "pdf":  _load_pdf,
        "txt":  _load_txt,
        "docx": _load_docx,
        "csv":  _load_csv,
    }

    loader_fn = loaders.get(ext)
    if loader_fn is None:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            f"Supported types: {list(loaders.keys())}"
        )

    docs = loader_fn(str(path), source_name)
    logger.info(f"Loaded {len(docs)} page(s)/section(s) from '{source_name}'")
    return docs


# ── Format-Specific Loaders ───────────────────────────────────────────────────

def _load_pdf(file_path: str, source_name: str) -> list[dict]:
    """
    Extract text page-by-page from a PDF.
    Uses pypdf — no external services needed.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("Install pypdf: pip install pypdf")

    reader = PdfReader(file_path)
    docs = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = clean_text(text)
        if len(text.strip()) < 20:
            # Skip near-empty pages (often cover images, page numbers)
            continue
        docs.append({
            "text": text,
            "metadata": {
                "source": source_name,
                "file_type": "pdf",
                "page": page_num,
                "total_pages": len(reader.pages),
            },
        })

    return docs


def _load_txt(file_path: str, source_name: str) -> list[dict]:
    """
    Load a plain text file as a single document.
    Tries UTF-8 first, falls back to latin-1 for legacy files.
    """
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                text = f.read()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Cannot decode file: {file_path}")

    text = clean_text(text)
    return [{
        "text": text,
        "metadata": {"source": source_name, "file_type": "txt"},
    }]


def _load_docx(file_path: str, source_name: str) -> list[dict]:
    """
    Extract text from a DOCX file paragraph by paragraph.
    Groups paragraphs into logical sections to preserve structure.
    """
    try:
        from docx import Document
    except ImportError:
        raise ImportError("Install python-docx: pip install python-docx")

    doc = Document(file_path)
    full_text = "\n".join(
        para.text for para in doc.paragraphs if para.text.strip()
    )
    text = clean_text(full_text)

    return [{
        "text": text,
        "metadata": {"source": source_name, "file_type": "docx"},
    }]


def _load_csv(file_path: str, source_name: str) -> list[dict]:
    """
    Load a CSV where each row becomes its own document.
    Row text is formatted as "column: value | column: value ..."
    so the LLM can read structured data naturally.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Install pandas: pip install pandas")

    df = pd.read_csv(file_path, dtype=str).fillna("")
    docs = []

    for row_idx, row in df.iterrows():
        # Format row as "col1: val1 | col2: val2 ..."
        row_text = " | ".join(
            f"{col}: {val}" for col, val in row.items() if val.strip()
        )
        row_text = clean_text(row_text)
        if not row_text:
            continue
        docs.append({
            "text": row_text,
            "metadata": {
                "source": source_name,
                "file_type": "csv",
                "row": int(row_idx) + 1,
                "columns": list(df.columns),
            },
        })

    return docs


def load_from_bytes(
    file_bytes: bytes,
    filename: str,
) -> list[dict]:
    """
    Load a document from raw bytes (used by the upload API endpoint).
    Writes to a temp file, loads it, then cleans up.
    """
    import tempfile
    import os

    ext = get_file_extension(filename)
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        return load_document(tmp_path, original_filename=filename)
    finally:
        os.unlink(tmp_path)  # Always clean up temp file
