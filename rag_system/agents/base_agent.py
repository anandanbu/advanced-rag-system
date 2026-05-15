"""
agents/base_agent.py
─────────────────────
Base class for all agents in the system.

An "agent" is an LLM that can:
  1. Decide WHAT to do (choose a tool)
  2. Execute the tool
  3. Observe the result
  4. Repeat until it has enough info to answer

This is the foundation for:
  - Agriculture AI agent (soil analysis, crop advice)
  - Research agent (multi-document synthesis)
  - AI companion (long-term memory + emotional context)
  - Autonomous task agent (web search + code execution)

Design pattern: ReAct (Reasoning + Acting)
  Thought → Action → Observation → Thought → ... → Final Answer
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AgentStep:
    """A single step in an agent's reasoning chain."""
    step_number: int
    thought: str                        # What the agent is thinking
    action: Optional[str] = None        # Tool name chosen
    action_input: Optional[Any] = None  # Input passed to the tool
    observation: Optional[str] = None   # Tool output / result
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class AgentResult:
    """Final result returned by an agent after completing its task."""
    answer: str
    steps: list[AgentStep]
    total_steps: int
    success: bool
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def reasoning_trace(self) -> str:
        """Human-readable trace of all agent steps for debugging."""
        lines = []
        for step in self.steps:
            lines.append(f"\n── Step {step.step_number} ──")
            lines.append(f"Thought: {step.thought}")
            if step.action:
                lines.append(f"Action: {step.action}({step.action_input})")
            if step.observation:
                lines.append(f"Observation: {step.observation[:300]}")
        lines.append(f"\n── Final Answer ──\n{self.answer}")
        return "\n".join(lines)


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Subclasses must implement:
      - run(query, session_id) → AgentResult
      - _get_tools() → dict of tool_name: callable

    Provides:
      - Step tracking
      - Max iteration safety limit
      - Logging
    """

    MAX_STEPS = 10  # Safety: never loop more than this

    def __init__(self, name: str):
        self.name = name
        self._steps: list[AgentStep] = []
        logger.info(f"Agent '{name}' initialized")

    @abstractmethod
    def run(self, query: str, session_id: str) -> AgentResult:
        """Execute the agent on a query. Must be implemented by subclasses."""
        ...

    @abstractmethod
    def _get_tools(self) -> dict[str, callable]:
        """Return dict of available tools: {name: function}"""
        ...

    def _add_step(
        self,
        thought: str,
        action: Optional[str] = None,
        action_input: Optional[Any] = None,
        observation: Optional[str] = None,
    ) -> AgentStep:
        """Record a reasoning step."""
        step = AgentStep(
            step_number=len(self._steps) + 1,
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation,
        )
        self._steps.append(step)
        logger.debug(
            f"[{self.name}] Step {step.step_number}: "
            f"thought='{thought[:60]}' | action={action}"
        )
        return step

    def _reset(self):
        """Clear step history for a new run."""
        self._steps = []

    def _execute_tool(self, tool_name: str, tool_input: Any) -> str:
        """
        Safely execute a tool and return its string output.
        Catches exceptions so a tool failure doesn't crash the agent.
        """
        tools = self._get_tools()
        if tool_name not in tools:
            return f"Error: Unknown tool '{tool_name}'. Available: {list(tools.keys())}"
        try:
            result = tools[tool_name](tool_input)
            return str(result)
        except Exception as e:
            logger.error(f"Tool '{tool_name}' failed: {e}")
            return f"Tool error: {str(e)}"
