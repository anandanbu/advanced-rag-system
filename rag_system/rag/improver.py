"""
rag/improver.py
───────────────
Self-improvement loop: if the critic scores an answer below threshold,
regenerate it using the critic's feedback as guidance.

This is a simplified version of the "Self-RAG" and "CRITIC" paper concepts:
  - Generate an answer
  - Evaluate it with a critic
  - If below threshold: regenerate with feedback → re-evaluate
  - Repeat up to MAX_RETRIES times
  - Return the best answer produced

The loop never runs more than settings.critic_max_retries times to
prevent infinite loops and control API costs.
"""

from config.settings import settings
from critic.evaluator import CriticEvaluator, CriticResult
from llm.groq_client import get_llm
from rag.prompt_templates import (
    IMPROVEMENT_SYSTEM_PROMPT,
    IMPROVEMENT_USER_PROMPT_TEMPLATE,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def improve_answer(
    question: str,
    context: str,
    initial_answer: str,
    initial_critic_result: CriticResult,
    critic: CriticEvaluator,
) -> tuple[str, CriticResult, int]:
    """
    Attempt to improve an answer that failed the critic's threshold.

    Args:
        question             : Original user question
        context              : Retrieved context string
        initial_answer       : First-pass LLM answer
        initial_critic_result: CriticResult for the initial answer
        critic               : CriticEvaluator instance

    Returns:
        Tuple of (best_answer, best_critic_result, iterations_used)
    """
    llm = get_llm()
    max_retries = settings.critic_max_retries

    best_answer = initial_answer
    best_result = initial_critic_result
    iterations = 0

    logger.info(
        f"Starting self-improvement loop | "
        f"initial_score={initial_critic_result.score:.2f} | "
        f"max_retries={max_retries}"
    )

    for attempt in range(1, max_retries + 1):
        iterations = attempt

        # Build the improvement prompt with critic feedback embedded
        improvement_prompt = IMPROVEMENT_USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
            previous_answer=best_answer,
            score=round(best_result.score, 2),
            issues=", ".join(best_result.issues) if best_result.issues else "vague or incomplete",
            suggestion=best_result.improvement_suggestion or "Be more specific and grounded.",
        )

        logger.info(f"Improvement attempt {attempt}/{max_retries}…")

        try:
            improved_answer = llm.generate(
                user_message=improvement_prompt,
                system_prompt=IMPROVEMENT_SYSTEM_PROMPT,
                temperature=0.1,    # Low temp for focused improvement
            )
        except Exception as e:
            logger.error(f"Improvement LLM call failed: {e}")
            break

        # Evaluate the improved answer
        new_result = critic.evaluate(
            question=question,
            context=context,
            response=improved_answer,
        )

        logger.info(
            f"Attempt {attempt} result: {new_result.summary}"
        )

        # Keep the improved answer if it's better (even if still below threshold)
        if new_result.score > best_result.score:
            best_answer = improved_answer
            best_result = new_result

        # Stop early if we've hit a passing score
        if new_result.passed:
            logger.info(f"Self-improvement succeeded on attempt {attempt} ✓")
            break

    if not best_result.passed:
        logger.warning(
            f"Self-improvement exhausted ({iterations} attempts) | "
            f"best_score={best_result.score:.2f} — returning best available answer"
        )

    return best_answer, best_result, iterations
