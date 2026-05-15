"""
tests/test_rag_pipeline.py
───────────────────────────
Unit tests for the RAG pipeline using mocks.
We mock the LLM and vector store so no API keys are needed.

Run: pytest tests/test_rag_pipeline.py -v
"""

import pytest
from unittest.mock import MagicMock, patch


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_chunks():
    return [
        {
            "chunk_id": "c1",
            "text": "Photosynthesis is the process by which plants convert sunlight into energy.",
            "metadata": {"source": "biology.pdf", "page": 1},
            "score": 0.91,
        },
        {
            "chunk_id": "c2",
            "text": "Chlorophyll is the green pigment in plants that absorbs light.",
            "metadata": {"source": "biology.pdf", "page": 2},
            "score": 0.85,
        },
    ]


@pytest.fixture
def mock_vector_store(mock_chunks):
    store = MagicMock()
    store.similarity_search.return_value = mock_chunks
    store.get_stats.return_value = {"total_chunks": 10, "sources": ["biology.pdf"]}
    return store


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate.return_value = (
        "Photosynthesis is the process by which plants use sunlight, "
        "water, and CO2 to produce oxygen and energy in the form of sugar. "
        "According to biology.pdf, chlorophyll is the key pigment involved."
    )
    return llm


@pytest.fixture
def mock_critic_pass():
    from critic.evaluator import CriticResult
    result = CriticResult(
        score=0.88,
        faithfulness=0.90,
        completeness=0.85,
        hallucination_detected=False,
        issues=[],
        improvement_suggestion="",
    )
    critic = MagicMock()
    critic.evaluate.return_value = result
    return critic


@pytest.fixture
def mock_critic_fail():
    from critic.evaluator import CriticResult
    result = CriticResult(
        score=0.40,
        faithfulness=0.45,
        completeness=0.35,
        hallucination_detected=True,
        issues=["Contains information not in context"],
        improvement_suggestion="Stick strictly to provided context.",
    )
    critic = MagicMock()
    critic.evaluate.return_value = result
    return critic


# ── Pipeline Tests ────────────────────────────────────────────────────────────

class TestRAGPipeline:

    @patch("rag.pipeline.get_vector_store")
    @patch("rag.pipeline.get_llm")
    @patch("rag.pipeline.get_critic")
    def test_pipeline_returns_rag_response(
        self, mock_get_critic, mock_get_llm, mock_get_store,
        mock_vector_store, mock_llm, mock_critic_pass
    ):
        mock_get_store.return_value = mock_vector_store
        mock_get_llm.return_value = mock_llm
        mock_get_critic.return_value = mock_critic_pass

        from rag.pipeline import RAGPipeline
        pipeline = RAGPipeline()
        result = pipeline.run(
            query="What is photosynthesis?",
            session_id="test-session-001",
        )

        assert result.answer != ""
        assert result.session_id == "test-session-001"
        assert result.mode == "rag"

    @patch("rag.pipeline.get_vector_store")
    @patch("rag.pipeline.get_llm")
    @patch("rag.pipeline.get_critic")
    def test_pipeline_retrieves_chunks(
        self, mock_get_critic, mock_get_llm, mock_get_store,
        mock_vector_store, mock_llm, mock_critic_pass
    ):
        mock_get_store.return_value = mock_vector_store
        mock_get_llm.return_value = mock_llm
        mock_get_critic.return_value = mock_critic_pass

        from rag.pipeline import RAGPipeline
        pipeline = RAGPipeline()
        result = pipeline.run("What is photosynthesis?", "sess-002")

        mock_vector_store.similarity_search.assert_called_once()
        assert result.retrieval_count == 2

    @patch("rag.pipeline.get_vector_store")
    @patch("rag.pipeline.get_llm")
    @patch("rag.pipeline.get_critic")
    def test_pipeline_records_critic_score(
        self, mock_get_critic, mock_get_llm, mock_get_store,
        mock_vector_store, mock_llm, mock_critic_pass
    ):
        mock_get_store.return_value = mock_vector_store
        mock_get_llm.return_value = mock_llm
        mock_get_critic.return_value = mock_critic_pass

        from rag.pipeline import RAGPipeline
        pipeline = RAGPipeline()
        result = pipeline.run("What is photosynthesis?", "sess-003")

        assert result.critic_score == 0.88
        assert result.critic_passed is True
        assert result.hallucination_detected is False

    @patch("rag.pipeline.get_vector_store")
    @patch("rag.pipeline.get_llm")
    @patch("rag.pipeline.get_critic")
    def test_pipeline_fallback_when_no_chunks(
        self, mock_get_critic, mock_get_llm, mock_get_store,
        mock_llm, mock_critic_pass
    ):
        """When no relevant chunks found, pipeline should fall back to conversational."""
        empty_store = MagicMock()
        empty_store.similarity_search.return_value = []  # No results
        mock_get_store.return_value = empty_store
        mock_get_llm.return_value = mock_llm
        mock_get_critic.return_value = mock_critic_pass

        from rag.pipeline import RAGPipeline
        pipeline = RAGPipeline()
        result = pipeline.run("Hello, how are you?", "sess-004")

        assert result.mode == "conversational"
        assert result.retrieval_count == 0

    @patch("rag.pipeline.get_vector_store")
    @patch("rag.pipeline.get_llm")
    @patch("rag.pipeline.get_critic")
    def test_pipeline_sources_in_response(
        self, mock_get_critic, mock_get_llm, mock_get_store,
        mock_vector_store, mock_llm, mock_critic_pass
    ):
        mock_get_store.return_value = mock_vector_store
        mock_get_llm.return_value = mock_llm
        mock_get_critic.return_value = mock_critic_pass

        from rag.pipeline import RAGPipeline
        pipeline = RAGPipeline()
        result = pipeline.run("What is photosynthesis?", "sess-005")

        assert len(result.sources) == 2
        assert result.sources[0]["score"] == 0.91

    @patch("rag.pipeline.get_vector_store")
    @patch("rag.pipeline.get_llm")
    @patch("rag.pipeline.get_critic")
    def test_pipeline_no_critic_skips_evaluation(
        self, mock_get_critic, mock_get_llm, mock_get_store,
        mock_vector_store, mock_llm, mock_critic_pass
    ):
        mock_get_store.return_value = mock_vector_store
        mock_get_llm.return_value = mock_llm
        mock_get_critic.return_value = mock_critic_pass

        from rag.pipeline import RAGPipeline
        pipeline = RAGPipeline()
        result = pipeline.run(
            "What is photosynthesis?", "sess-006",
            use_critic=False  # Skip critic
        )

        mock_critic_pass.evaluate.assert_not_called()


# ── Prompt Template Tests ─────────────────────────────────────────────────────

class TestPromptTemplates:

    def test_rag_user_prompt_contains_context(self):
        from rag.prompt_templates import RAG_USER_PROMPT_TEMPLATE
        prompt = RAG_USER_PROMPT_TEMPLATE.format(
            context="Plants use sunlight for energy.",
            question="What do plants use?",
        )
        assert "Plants use sunlight" in prompt
        assert "What do plants use?" in prompt

    def test_rag_system_prompt_formats_memory(self):
        from rag.prompt_templates import RAG_SYSTEM_PROMPT
        prompt = RAG_SYSTEM_PROMPT.format(memory_context="User is a farmer.")
        assert "farmer" in prompt

    def test_rag_system_prompt_empty_memory(self):
        from rag.prompt_templates import RAG_SYSTEM_PROMPT
        prompt = RAG_SYSTEM_PROMPT.format(memory_context="")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_critic_prompt_contains_all_fields(self):
        from rag.prompt_templates import CRITIC_USER_PROMPT_TEMPLATE
        prompt = CRITIC_USER_PROMPT_TEMPLATE.format(
            context="Context text",
            question="The question",
            response="The response",
        )
        assert "Context text" in prompt
        assert "The question" in prompt
        assert "The response" in prompt


# ── Improver Tests ────────────────────────────────────────────────────────────

class TestImprover:

    def test_improve_answer_returns_tuple(self):
        from rag.improver import improve_answer
        from critic.evaluator import CriticResult

        failing_result = CriticResult(
            score=0.3,
            hallucination_detected=True,
            issues=["Hallucinated facts"],
            improvement_suggestion="Be more accurate.",
        )

        mock_critic = MagicMock()
        # Second evaluation passes
        passing_result = CriticResult(score=0.85, hallucination_detected=False)
        mock_critic.evaluate.return_value = passing_result

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Improved answer based on context."

        with patch("rag.improver.get_llm", return_value=mock_llm):
            answer, result, iterations = improve_answer(
                question="What is X?",
                context="Context about X.",
                initial_answer="Wrong answer.",
                initial_critic_result=failing_result,
                critic=mock_critic,
            )

        assert isinstance(answer, str)
        assert isinstance(result, CriticResult)
        assert isinstance(iterations, int)
        assert iterations >= 1
