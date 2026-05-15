"""
tests/test_ingestion.py
────────────────────────
Unit tests for document loading and chunking modules.

Run: pytest tests/ -v
"""

import os
import tempfile
import pytest

from ingestion.loader import load_document, load_from_bytes
from ingestion.chunker import chunk_documents


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_txt_file(tmp_path):
    """Create a temporary .txt file for testing."""
    content = (
        "Artificial intelligence is transforming industries worldwide.\n\n"
        "Machine learning, a subset of AI, enables systems to learn from data.\n\n"
        "Deep learning uses neural networks with many layers.\n\n"
        "Natural language processing allows computers to understand human language.\n\n"
        "Retrieval-Augmented Generation combines search with language models."
    )
    f = tmp_path / "test_doc.txt"
    f.write_text(content, encoding="utf-8")
    return str(f)


@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a temporary .csv file for testing."""
    content = "crop,soil_type,water_need,season\nRice,Clay,High,Kharif\nWheat,Loam,Medium,Rabi\nCorn,Sandy,Medium,Kharif"
    f = tmp_path / "crops.csv"
    f.write_text(content, encoding="utf-8")
    return str(f)


@pytest.fixture
def sample_chunks():
    """Pre-built chunk list for testing downstream functions."""
    return [
        {
            "chunk_id": "abc123",
            "text": "Artificial intelligence enables machines to simulate human thinking.",
            "metadata": {"source": "test.txt", "file_type": "txt", "chunk_index": 0},
        },
        {
            "chunk_id": "def456",
            "text": "Machine learning is a method of teaching computers using data examples.",
            "metadata": {"source": "test.txt", "file_type": "txt", "chunk_index": 1},
        },
    ]


# ── Loader Tests ──────────────────────────────────────────────────────────────

class TestDocumentLoader:

    def test_load_txt_returns_documents(self, sample_txt_file):
        docs = load_document(sample_txt_file)
        assert isinstance(docs, list)
        assert len(docs) >= 1
        assert "text" in docs[0]
        assert "metadata" in docs[0]

    def test_load_txt_text_not_empty(self, sample_txt_file):
        docs = load_document(sample_txt_file)
        for doc in docs:
            assert len(doc["text"].strip()) > 0

    def test_load_txt_metadata_has_source(self, sample_txt_file):
        docs = load_document(sample_txt_file, original_filename="my_doc.txt")
        assert docs[0]["metadata"]["source"] == "my_doc.txt"
        assert docs[0]["metadata"]["file_type"] == "txt"

    def test_load_csv_returns_rows(self, sample_csv_file):
        docs = load_document(sample_csv_file)
        # Should have one doc per non-empty row
        assert len(docs) >= 3
        # Each row should contain column data
        assert "crop" in docs[0]["text"].lower() or "Rice" in docs[0]["text"]

    def test_load_csv_metadata_has_row(self, sample_csv_file):
        docs = load_document(sample_csv_file)
        assert "row" in docs[0]["metadata"]
        assert docs[0]["metadata"]["file_type"] == "csv"

    def test_unsupported_file_raises(self, tmp_path):
        f = tmp_path / "bad.xyz"
        f.write_text("test")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document(str(f))

    def test_load_from_bytes_txt(self):
        content = b"Hello world. This is a test document for RAG."
        docs = load_from_bytes(content, "test.txt")
        assert len(docs) >= 1
        assert "Hello world" in docs[0]["text"]

    def test_load_from_bytes_csv(self):
        content = b"name,age\nAlice,30\nBob,25"
        docs = load_from_bytes(content, "people.csv")
        assert len(docs) >= 2


# ── Chunker Tests ─────────────────────────────────────────────────────────────

class TestChunker:

    def test_chunk_documents_returns_chunks(self, sample_txt_file):
        docs = load_document(sample_txt_file)
        chunks = chunk_documents(docs)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_chunks_have_required_fields(self, sample_txt_file):
        docs = load_document(sample_txt_file)
        chunks = chunk_documents(docs)
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "text" in chunk
            assert "metadata" in chunk
            assert len(chunk["text"]) > 0

    def test_chunk_ids_are_unique(self, sample_txt_file):
        docs = load_document(sample_txt_file)
        chunks = chunk_documents(docs)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_chunk_size_respected(self, sample_txt_file):
        docs = load_document(sample_txt_file)
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        for chunk in chunks:
            # Allow slight overage due to word boundaries
            assert len(chunk["text"]) <= 200, f"Chunk too long: {len(chunk['text'])}"

    def test_chunk_metadata_inherited(self, sample_txt_file):
        docs = load_document(sample_txt_file, original_filename="inherit_test.txt")
        chunks = chunk_documents(docs)
        for chunk in chunks:
            assert chunk["metadata"]["source"] == "inherit_test.txt"

    def test_empty_documents_returns_empty(self):
        chunks = chunk_documents([])
        assert chunks == []

    def test_chunk_index_in_metadata(self, sample_txt_file):
        docs = load_document(sample_txt_file)
        chunks = chunk_documents(docs)
        for chunk in chunks:
            assert "chunk_index" in chunk["metadata"]
            assert isinstance(chunk["metadata"]["chunk_index"], int)

    def test_custom_chunk_size(self, sample_txt_file):
        docs = load_document(sample_txt_file)
        small_chunks = chunk_documents(docs, chunk_size=50)
        large_chunks = chunk_documents(docs, chunk_size=500)
        # Smaller chunk size should produce more chunks
        assert len(small_chunks) >= len(large_chunks)
