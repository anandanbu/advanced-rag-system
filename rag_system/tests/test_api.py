"""
tests/test_api.py
──────────────────
Integration tests for FastAPI endpoints using TestClient.
Mocks external services (LLM, vector store) so no API keys needed.

Run: pytest tests/test_api.py -v
"""

import pytest
import io
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ── App Fixture ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """
    Create a TestClient with mocked external dependencies.
    We mock the RAGPipeline so no LLM or vector store is needed.
    """
    with patch("embeddings.embedder.get_embedder") as mock_embedder_factory, \
         patch("vectorstore.chroma_store.get_vector_store") as mock_vs_factory:

        # Mock embedder
        mock_embedder = MagicMock()
        mock_embedder.model_name = "all-MiniLM-L6-v2"
        mock_embedder.dimension = 384
        mock_embedder_factory.return_value = mock_embedder

        # Mock vector store
        mock_vs = MagicMock()
        mock_vs.get_stats.return_value = {
            "total_chunks": 5,
            "collection_name": "test",
            "sources": ["test.txt"],
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dim": 384,
        }
        mock_vs_factory.return_value = mock_vs

        from main import create_app
        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ── Health Endpoint ───────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status_ok(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_has_model_info(self, client):
        response = client.get("/health")
        data = response.json()
        assert "model" in data
        assert "embedding_model" in data

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200


# ── Chat Endpoint ─────────────────────────────────────────────────────────────

class TestChatEndpoint:

    @pytest.fixture
    def mock_pipeline_response(self):
        from rag.pipeline import RAGResponse
        return RAGResponse(
            answer="Photosynthesis converts sunlight into energy.",
            session_id="test-session",
            query="What is photosynthesis?",
            sources=[{
                "chunk_id": "c1",
                "text": "Plants use sunlight.",
                "metadata": {"source": "bio.pdf"},
                "score": 0.92,
            }],
            critic_score=0.88,
            critic_passed=True,
            hallucination_detected=False,
            improvement_iterations=0,
            retrieval_count=1,
            latency_ms=450.0,
            mode="rag",
        )

    def test_chat_valid_request(self, client, mock_pipeline_response):
        with patch("api.routes.chat.get_pipeline") as mock_get_pipeline:
            mock_pipeline = MagicMock()
            mock_pipeline.run.return_value = mock_pipeline_response
            mock_get_pipeline.return_value = mock_pipeline

            response = client.post("/chat", json={
                "message": "What is photosynthesis?",
                "session_id": "sess-001",
            })

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] != ""
        assert data["session_id"] == "test-session"
        assert data["mode"] == "rag"

    def test_chat_response_has_all_fields(self, client, mock_pipeline_response):
        with patch("api.routes.chat.get_pipeline") as mock_get_pipeline:
            mock_pipeline = MagicMock()
            mock_pipeline.run.return_value = mock_pipeline_response
            mock_get_pipeline.return_value = mock_pipeline

            response = client.post("/chat", json={"message": "Test question"})

        data = response.json()
        required_fields = [
            "answer", "session_id", "sources", "critic_score",
            "critic_passed", "hallucination_detected",
            "improvement_iterations", "retrieval_count", "latency_ms", "mode"
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_chat_empty_message_rejected(self, client):
        response = client.post("/chat", json={"message": ""})
        assert response.status_code == 422  # Validation error

    def test_chat_missing_message_rejected(self, client):
        response = client.post("/chat", json={})
        assert response.status_code == 422

    def test_chat_auto_generates_session_id(self, client, mock_pipeline_response):
        with patch("api.routes.chat.get_pipeline") as mock_get_pipeline:
            mock_pipeline = MagicMock()
            mock_pipeline.run.return_value = mock_pipeline_response
            mock_get_pipeline.return_value = mock_pipeline

            response = client.post("/chat", json={"message": "Hello"})

        assert response.status_code == 200
        assert response.json()["session_id"] != ""

    def test_chat_clear_session(self, client):
        response = client.post("/chat/clear", json={"session_id": "sess-to-clear"})
        assert response.status_code == 200
        assert "cleared" in response.json()["message"].lower()


# ── Upload Endpoint ───────────────────────────────────────────────────────────

class TestUploadEndpoint:

    def test_upload_txt_file(self, client):
        with patch("api.routes.upload.ingest_file") as mock_ingest:
            mock_ingest.return_value = {
                "filename": "test.txt",
                "pages_loaded": 1,
                "chunks_created": 5,
                "chunks_added": 5,
                "chunks_skipped": 0,
            }
            file_content = b"This is a test document for RAG ingestion testing."
            response = client.post(
                "/upload",
                files={"file": ("test.txt", io.BytesIO(file_content), "text/plain")},
            )

        assert response.status_code == 200
        data = response.json()
        assert "stats" in data
        assert data["stats"]["chunks_added"] == 5

    def test_upload_unsupported_format_rejected(self, client):
        response = client.post(
            "/upload",
            files={"file": ("test.xyz", io.BytesIO(b"content"), "application/octet-stream")},
        )
        assert response.status_code == 400

    def test_upload_empty_file_rejected(self, client):
        response = client.post(
            "/upload",
            files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
        )
        assert response.status_code == 400

    def test_upload_stats_endpoint(self, client):
        response = client.get("/upload/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_chunks" in data


# ── History Endpoint ──────────────────────────────────────────────────────────

class TestHistoryEndpoint:

    def test_list_sessions(self, client):
        response = client.get("/history/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert "count" in data

    def test_get_history_for_session(self, client):
        response = client.get("/history/new-session-xyz")
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "history" in data

    def test_save_fact_to_memory(self, client):
        response = client.post(
            "/history/sess-mem-test/memory",
            json={"key": "location", "value": "Chennai"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["all_facts"]["location"] == "Chennai"

    def test_retrieve_saved_fact(self, client):
        # Save a fact
        client.post(
            "/history/sess-fact-test/memory",
            json={"key": "profession", "value": "engineer"},
        )
        # Retrieve memory
        response = client.get("/history/sess-fact-test/memory")
        assert response.status_code == 200
        data = response.json()
        assert "profession" in data["facts"]
        assert data["facts"]["profession"] == "engineer"
