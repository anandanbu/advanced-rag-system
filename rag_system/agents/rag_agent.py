"""
agents/rag_agent.py
────────────────────
A ReAct-style agent that can use multiple tools to answer complex queries.

Available tools:
  - search_documents : semantic search in ChromaDB
  - summarize        : summarize a retrieved document
  - calculate        : simple math expressions
  - recall_memory    : look up stored user facts

The agent reasons step-by-step, choosing tools as needed,
then synthesizes a final answer from all observations.

This is the bridge between the basic RAG pipeline and a fully
autonomous AI assistant.
"""

import re
import json
from typing import Any, Optional

from agents.base_agent import BaseAgent, AgentResult, AgentStep
from vectorstore.chroma_store import get_vector_store
from llm.groq_client import get_llm
from memory.persistent import PersistentMemory
from utils.logger import get_logger
from utils.helpers import format_sources

logger = get_logger(__name__)


# ── Agent Prompt ──────────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """You are a ReAct agent — you Reason and Act step by step.

Available tools:
- search_documents(query: str) → Search the knowledge base for relevant information
- summarize(text: str)         → Summarize a long text
- calculate(expression: str)   → Evaluate a math expression (e.g., "25 * 4 + 10")
- recall_memory(key: str)      → Look up a stored fact about the user

STRICT FORMAT — respond ONLY in this JSON format:
{
  "thought": "What I'm thinking about what to do next",
  "action": "tool_name OR null if ready to answer",
  "action_input": "input for the tool OR null",
  "final_answer": "Complete answer if action is null, else null"
}

Rules:
1. Always start with a thought about what information you need
2. Use search_documents for any factual questions
3. Only provide final_answer when you have enough information
4. final_answer must be comprehensive and cite sources
5. If you cannot find an answer after searching, say so honestly"""


class RAGAgent(BaseAgent):
    """
    ReAct agent with access to RAG retrieval and utility tools.
    Uses the LLM to iteratively reason and act until it can answer.
    """

    def __init__(self):
        super().__init__(name="RAGAgent")
        self._llm = get_llm()
        self._vector_store = get_vector_store()

    def run(self, query: str, session_id: str) -> AgentResult:
        """
        Run the agent loop on a query.

        Loop:
          1. Ask LLM: what should I do next?
          2. Parse action + input
          3. Execute tool
          4. Feed observation back to LLM
          5. Repeat until LLM gives a final_answer or MAX_STEPS reached
        """
        self._reset()
        self._session_id = session_id

        # Build conversation for the agent (accumulates thoughts + observations)
        agent_history = []
        final_answer = None

        logger.info(f"[RAGAgent] Starting on query: '{query[:80]}'")

        for step_num in range(1, self.MAX_STEPS + 1):
            # Build the current prompt
            current_prompt = self._build_agent_prompt(query, agent_history)

            # Ask the LLM what to do
            try:
                raw_response = self._llm.generate(
                    user_message=current_prompt,
                    system_prompt=AGENT_SYSTEM_PROMPT,
                    temperature=0.1,
                    max_tokens=1024,
                )
            except Exception as e:
                logger.error(f"LLM call failed at step {step_num}: {e}")
                break

            # Parse the JSON response
            parsed = self._parse_agent_response(raw_response)
            if parsed is None:
                logger.warning(f"Failed to parse agent response at step {step_num}")
                break

            thought = parsed.get("thought", "")
            action = parsed.get("action")
            action_input = parsed.get("action_input")
            final_answer = parsed.get("final_answer")

            # ── Final answer reached ──────────────────────────────────────────
            if final_answer and (action is None or action == "null"):
                self._add_step(thought=thought, observation="[Final answer provided]")
                logger.info(f"[RAGAgent] Final answer reached at step {step_num}")
                break

            # ── Execute tool ──────────────────────────────────────────────────
            if action and action != "null":
                observation = self._execute_tool(action, action_input)
                self._add_step(
                    thought=thought,
                    action=action,
                    action_input=action_input,
                    observation=observation,
                )
                # Append this step to agent history for next iteration
                agent_history.append({
                    "thought": thought,
                    "action": action,
                    "action_input": action_input,
                    "observation": observation,
                })
            else:
                # No action and no final answer — shouldn't happen, but handle it
                self._add_step(thought=thought)
                break

        # If no final_answer was given, synthesize from steps
        if not final_answer:
            final_answer = self._synthesize_fallback(query, agent_history)

        return AgentResult(
            answer=final_answer,
            steps=list(self._steps),
            total_steps=len(self._steps),
            success=bool(final_answer),
            metadata={"session_id": session_id, "query": query},
        )

    def _get_tools(self) -> dict[str, callable]:
        return {
            "search_documents": self._tool_search,
            "summarize":        self._tool_summarize,
            "calculate":        self._tool_calculate,
            "recall_memory":    self._tool_recall_memory,
        }

    # ── Tools ─────────────────────────────────────────────────────────────────

    def _tool_search(self, query: str) -> str:
        """Search the vector store and return formatted chunks."""
        chunks = self._vector_store.similarity_search(query=str(query), top_k=4)
        if not chunks:
            return "No relevant documents found for this query."
        return format_sources(chunks)

    def _tool_summarize(self, text: str) -> str:
        """Summarize a long text using the LLM."""
        if len(text) < 200:
            return text  # Already short
        summary = self._llm.generate(
            user_message=f"Summarize this in 3 sentences:\n\n{text[:3000]}",
            temperature=0.1,
            max_tokens=300,
        )
        return summary

    def _tool_calculate(self, expression: str) -> str:
        """Safely evaluate a math expression."""
        try:
            # Only allow safe math characters
            safe_expr = re.sub(r"[^0-9+\-*/().\s%]", "", str(expression))
            if not safe_expr.strip():
                return "Error: Invalid expression"
            result = eval(safe_expr, {"__builtins__": {}})
            return f"{expression} = {result}"
        except Exception as e:
            return f"Calculation error: {e}"

    def _tool_recall_memory(self, key: str) -> str:
        """Look up a fact from persistent memory."""
        mem = PersistentMemory(self._session_id)
        value = mem.get_fact(str(key))
        if value:
            return f"Memory: {key} = {value}"
        return f"No memory found for key: '{key}'"

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_agent_prompt(self, query: str, history: list[dict]) -> str:
        """Build the prompt including the original query + all previous steps."""
        lines = [f"User Query: {query}"]
        if history:
            lines.append("\nPrevious steps:")
            for i, step in enumerate(history, 1):
                lines.append(f"\nStep {i}:")
                lines.append(f"  Thought: {step['thought']}")
                lines.append(f"  Action: {step['action']}({step['action_input']})")
                lines.append(f"  Observation: {step['observation'][:500]}")
        lines.append("\nWhat is your next step? Respond in the required JSON format.")
        return "\n".join(lines)

    def _parse_agent_response(self, raw: str) -> Optional[dict]:
        """Parse the agent's JSON response, handling markdown fences."""
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None

    def _synthesize_fallback(self, query: str, history: list[dict]) -> str:
        """If no final_answer was given, synthesize one from observations."""
        if not history:
            return "I was unable to find relevant information to answer your question."
        observations = "\n".join(
            f"- {s['observation'][:300]}" for s in history if s.get("observation")
        )
        return self._llm.generate(
            user_message=(
                f"Based on these research notes, answer: {query}\n\n"
                f"Notes:\n{observations}"
            ),
            temperature=0.2,
        )
