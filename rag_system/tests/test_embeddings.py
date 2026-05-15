"""
tests/test_embeddings.py
─────────────────────────
Unit tests for the embedding module.
These tests run offline — no API keys needed.

Run: pytest tests/test_embeddings.py -v
"""

import pytest
import math


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def embedder():
    """Load the embedder once for all tests in this module."""
    from embeddings.embedder import Embedder
    return Embedder(model_name="all-MiniLM-L6-v2", device="cpu")


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestEmbedder:

    def test_embed_text_returns_list(self, embedder):
        vector = embedder.embed_text("Hello world")
        assert isinstance(vector, list)

    def test_embed_text_correct_dimension(self, embedder):
        vector = embedder.embed_text("Test sentence for embedding")
        assert len(vector) == 384  # all-MiniLM-L6-v2 dimension

    def test_embed_text_returns_floats(self, embedder):
        vector = embedder.embed_text("Numbers should be floats")
        assert all(isinstance(v, float) for v in vector)

    def test_embed_text_normalized(self, embedder):
        """L2 norm of a normalized vector should be ~1.0."""
        vector = embedder.embed_text("Normalized vector test")
        norm = math.sqrt(sum(v ** 2 for v in vector))
        assert abs(norm - 1.0) < 0.01, f"Norm should be ~1.0, got {norm}"

    def test_embed_text_empty_raises(self, embedder):
        with pytest.raises(ValueError):
            embedder.embed_text("")

    def test_embed_text_whitespace_raises(self, embedder):
        with pytest.raises(ValueError):
            embedder.embed_text("   ")

    def test_embed_batch_returns_list_of_lists(self, embedder):
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        vectors = embedder.embed_batch(texts)
        assert len(vectors) == 3
        assert all(len(v) == 384 for v in vectors)

    def test_embed_batch_empty_returns_empty(self, embedder):
        assert embedder.embed_batch([]) == []

    def test_similar_texts_have_higher_similarity(self, embedder):
        """Semantically similar texts should have higher cosine similarity."""
        v1 = embedder.embed_text("The cat sat on the mat")
        v2 = embedder.embed_text("A cat is sitting on a rug")  # Similar
        v3 = embedder.embed_text("The stock market crashed today")  # Different

        def cosine_sim(a, b):
            return sum(x * y for x, y in zip(a, b))  # Already normalized

        sim_similar = cosine_sim(v1, v2)
        sim_different = cosine_sim(v1, v3)
        assert sim_similar > sim_different, (
            f"Expected similar texts to score higher: "
            f"similar={sim_similar:.3f}, different={sim_different:.3f}"
        )

    def test_dimension_property(self, embedder):
        assert embedder.dimension == 384

    def test_singleton_returns_same_instance(self):
        from embeddings.embedder import get_embedder
        e1 = get_embedder()
        e2 = get_embedder()
        assert e1 is e2, "get_embedder() should return the same instance"
