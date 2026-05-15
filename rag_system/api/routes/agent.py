"""
api/routes/agent.py
────────────────────
POST /agent  — run the ReAct agent on a complex query

Unlike /chat (single retrieval + generate), the agent:
  - Plans multi-step reasoning
  - Chooses which tools to use
  - Iterates until it has a confident answer
  - Returns the full reasoning trace

Best for: complex, multi-part questions that need multiple lookups.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agents.rag_agent import RAGAgent
from utils.helpers import generate_session_id
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/agent", tags=["Agent"])


class AgentRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = Field(None)
    include_trace: bool = Field(False, description="Include step-by-step reasoning trace")


class AgentStepResponse(BaseModel):
    step_number: int
    thought: str
    action: Optional[str]
    action_input: Optional[str]
    observation: Optional[str]


class AgentResponse(BaseModel):
    answer: str
    session_id: str
    total_steps: int
    success: bool
    steps: Optional[list[AgentStepResponse]] = None


@router.post("", response_model=AgentResponse, summary="Run the ReAct agent")
async def run_agent(request: AgentRequest):
    """
    Run the multi-step ReAct agent.
    The agent iteratively reasons, searches, and synthesizes an answer.
    Use for complex questions that may require multiple document lookups.
    """
    session_id = request.session_id or generate_session_id()

    try:
        agent = RAGAgent()
        result = agent.run(query=request.query, session_id=session_id)
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    steps = None
    if request.include_trace:
        steps = [
            AgentStepResponse(
                step_number=s.step_number,
                thought=s.thought,
                action=s.action,
                action_input=str(s.action_input) if s.action_input else None,
                observation=s.observation,
            )
            for s in result.steps
        ]

    return AgentResponse(
        answer=result.answer,
        session_id=session_id,
        total_steps=result.total_steps,
        success=result.success,
        steps=steps,
    )
