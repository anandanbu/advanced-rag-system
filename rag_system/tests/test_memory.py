"""
tests/test_memory.py
─────────────────────
Unit tests for conversation and persistent memory modules.
No external API calls needed.

Run: pytest tests/test_memory.py -v
"""

import pytest
import os
import json
import tempfile


# ── Conversation Memory Tests ─────────────────────────────────────────────────

class TestConversationMemory:

    def setup_method(self):
        """Clear sessions before each test."""
        import memory.conversation as conv
        conv._sessions.clear()

    def test_add_turn_creates_session(self):
        from memory.conversation import add_turn, get_history
        add_turn("sess1", "Hello", "Hi there!")
        history = get_history("sess1")
        assert len(history) == 2

    def test_history_has_correct_roles(self):
        from memory.conversation import add_turn, get_history
        add_turn("sess2", "What is AI?", "AI stands for Artificial Intelligence.")
        history = get_history("sess2")
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_history_has_correct_content(self):
        from memory.conversation import add_turn, get_history
        add_turn("sess3", "My question", "My answer")
        history = get_history("sess3")
        assert history[0]["content"] == "My question"
        assert history[1]["content"] == "My answer"

    def test_multiple_turns_accumulate(self):
        from memory.conversation import add_turn, get_history
        add_turn("sess4", "Q1", "A1")
        add_turn("sess4", "Q2", "A2")
        add_turn("sess4", "Q3", "A3")
        history = get_history("sess4")
        assert len(history) == 6

    def test_empty_session_returns_empty(self):
        from memory.conversation import get_history
        history = get_history("nonexistent_session")
        assert history == []

    def test_clear_session_removes_history(self):
        from memory.conversation import add_turn, get_history, clear_session
        add_turn("sess5", "Hello", "Hi")
        clear_session("sess5")
        assert get_history("sess5") == []

    def test_history_stripped_of_timestamp_for_llm(self):
        from memory.conversation import add_turn, get_history
        add_turn("sess6", "Q", "A")
        history = get_history("sess6")
        for msg in history:
            assert "timestamp" not in msg
            assert "role" in msg
            assert "content" in msg

    def test_session_summary(self):
        from memory.conversation import add_turn, get_session_summary
        add_turn("sess7", "Q1", "A1")
        add_turn("sess7", "Q2", "A2")
        summary = get_session_summary("sess7")
        assert summary["total_turns"] == 2
        assert summary["total_messages"] == 4


# ── Persistent Memory Tests ───────────────────────────────────────────────────

class TestPersistentMemory:

    @pytest.fixture
    def mem(self, tmp_path, monkeypatch):
        """Create a PersistentMemory with a temp directory."""
        monkeypatch.setattr("config.settings.settings.memory_dir", str(tmp_path))
        from memory.persistent import PersistentMemory
        return PersistentMemory("test-session-123")

    def test_set_and_get_fact(self, mem):
        mem.set_fact("location", "Chennai")
        assert mem.get_fact("location") == "Chennai"

    def test_get_missing_fact_returns_default(self, mem):
        assert mem.get_fact("nonexistent") is None
        assert mem.get_fact("nonexistent", "fallback") == "fallback"

    def test_get_all_facts(self, mem):
        mem.set_fact("name", "Alice")
        mem.set_fact("profession", "farmer")
        facts = mem.get_all_facts()
        assert facts["name"] == "Alice"
        assert facts["profession"] == "farmer"

    def test_set_and_get_preference(self, mem):
        mem.set_preference("response_style", "concise")
        assert mem.get_preference("response_style") == "concise"

    def test_add_summary(self, mem):
        mem.add_summary("User asked about crop diseases")
        summaries = mem.get_summaries()
        assert len(summaries) == 1
        assert "crop diseases" in summaries[0]["summary"]

    def test_persistence_across_instances(self, tmp_path, monkeypatch):
        """Test that data survives creating a new instance (disk persistence)."""
        monkeypatch.setattr("config.settings.settings.memory_dir", str(tmp_path))
        from memory.persistent import PersistentMemory

        mem1 = PersistentMemory("persist-test")
        mem1.set_fact("crop", "rice")

        mem2 = PersistentMemory("persist-test")
        assert mem2.get_fact("crop") == "rice"

    def test_build_memory_context_empty(self, mem):
        context = mem.build_memory_context()
        assert context == ""

    def test_build_memory_context_with_facts(self, mem):
        mem.set_fact("location", "Tamil Nadu")
        mem.set_fact("crop", "rice")
        context = mem.build_memory_context()
        assert "Tamil Nadu" in context
        assert "rice" in context

    def test_clear_wipes_memory(self, mem):
        mem.set_fact("key", "value")
        mem.clear()
        assert mem.get_fact("key") is None
        assert mem.get_all_facts() == {}

    def test_save_and_load_history(self, mem):
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        mem.save_history(history)
        loaded = mem.get_saved_history()
        assert len(loaded) == 2
        assert loaded[0]["content"] == "Hello"
