"""
critic/evaluator.py
───────────────────
Evaluates RAG-generated answers for quality, faithfulness, and hallucinations.

The Critic uses the LLM itself as a judge — a technique called
"LLM-as-a-judge" (popularized by papers like Constitutional AI, RLHF, and RAGAS).

Flow:
  1. Receive: question + retrieved context + generated answer
  2. Send a structured evaluation prompt to the LLM
  3. Parse the JSON response into a CriticResult
  4. Return score + actionable feedback

The RAG pipeline uses this score to decide whether to regenerate.
"""

import json
import re
from dataclasses import dataclass, field

from config.settings import settings
from llm.groq_client import get_llm
from rag.prompt_templates import CRITIC_SYSTEM_PROMPT, CRITIC_USER_PROMPT_TEMPLATE
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CriticResult:
    """
    Structured output from the evaluator.
    All fields have safe defaults so partial JSON parses don't crash.
    """
    score: float = 0.5
    faithfulness: float = 0.5
    completeness: float = 0.5
    hallucination_detected: bool = False
    issues: list[str] = field(default_factory=list)
    improvement_suggestion: str = ""
    raw_response: str = ""           # Raw LLM output (for debugging)
    parse_error: bool = False        # True if JSON parsing failed

    @property
    def passed(self) -> bool:
        """True if the answer quality meets the configured threshold."""
        return self.score >= settings.critic_score_threshold

    @property
    def summary(self) -> str:
        """Human-readable one-liner for logging."""
        status = "✓ PASS" if self.passed else "✗ FAIL"
        flags = " [HALLUCINATION]" if self.hallucination_detected else ""
        return (
            f"{status}{flags} | score={self.score:.2f} | "
            f"faithful={self.faithfulness:.2f} | complete={self.completeness:.2f}"
        )


class CriticEvaluator:
    """
    Uses an LLM to evaluate the quality of a generated answer.
    """

    def __init__(self):
        self._llm = get_llm()

    def evaluate(
        self,
        question: str,
        context: str,
        response: str,
    ) -> CriticResult:
        """
        Evaluate a generated answer against the retrieval context.

        Args:
            question : The original user question
            context  : The concatenated retrieved chunk texts (ground truth)
            response : The LLM's generated answer to evaluate

        Returns:
            CriticResult with score, flags, and improvement suggestion
        """
        if not response.strip():
            return CriticResult(score=0.0, issues=["Empty response"], parse_error=False)

        prompt = CRITIC_USER_PROMPT_TEMPLATE.format(
            context=context[:3000],   # Limit context size for the critic call
            question=question,
            response=response,
        )

        logger.debug("Running critic evaluation…")

        try:
            raw = self._llm.generate(
                user_message=prompt,
                system_prompt=CRITIC_SYSTEM_PROMPT,
                temperature=0.0,    # Deterministic for evaluation
                max_tokens=512,
            )
            result = self._parse_critic_response(raw)
            result.raw_response = raw
            logger.info(f"Critic: {result.summary}")
            return result

        except Exception as e:
            logger.error(f"Critic evaluation failed: {e}")
            # On failure, return a neutral passing score to avoid blocking the pipeline
            return CriticResult(
                score=0.65,
                issues=[f"Critic unavailable: {str(e)}"],
                parse_error=True,
            )

    def _parse_critic_response(self, raw: str) -> CriticResult:
        """
        Parse the LLM's JSON evaluation response.
        Handles markdown fences and extra text around the JSON.
        """
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()

        # Find first { ... } block in case of preamble
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            logger.warning(f"Critic returned non-JSON: {raw[:200]}")
            return CriticResult(score=0.5, parse_error=True, raw_response=raw)

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError as e:
            logger.warning(f"Critic JSON parse error: {e} | raw={raw[:200]}")
            return CriticResult(score=0.5, parse_error=True, raw_response=raw)

        return CriticResult(
            score=float(data.get("score", 0.5)),
            faithfulness=float(data.get("faithfulness", 0.5)),
            completeness=float(data.get("completeness", 0.5)),
            hallucination_detected=bool(data.get("hallucination_detected", False)),
            issues=list(data.get("issues", [])),
            improvement_suggestion=str(data.get("improvement_suggestion", "")),
        )


def get_critic() -> CriticEvaluator:
    """Factory — creates a new CriticEvaluator (stateless, cheap to create)."""
    return CriticEvaluator()
