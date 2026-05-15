"""
llm/groq_client.py
──────────────────
Wraps the Groq API for fast Llama 3 inference.

Why Groq?
  - Free tier: 14,400 requests/day, 30 req/min
  - Fastest LLM inference available (LPU hardware)
  - Supports: llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768

Features:
  - Retry with exponential backoff on rate limits / transient errors
  - Structured message format (system + history + user)
  - Optional streaming support
"""

from functools import lru_cache
from typing import Optional, Generator

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class GroqClient:
    """
    Groq API client for chat completions.
    Handles message formatting, retries, and error logging.
    """

    def __init__(self, api_key: str, model: str):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("pip install groq")

        self._client = Groq(api_key=api_key)
        self.model = model
        logger.info(f"Groq client initialized — model='{model}'")

    # ── Main Generate Method ──────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def generate(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        history: Optional[list[dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            user_message  : The current user query (with RAG context injected)
            system_prompt : Instructions for how the model should behave
            history       : Previous conversation turns as
                            [{"role": "user"|"assistant", "content": str}, ...]
            temperature   : Creativity (0.0=deterministic, 1.0=creative)
            max_tokens    : Max tokens in the response

        Returns:
            The model's response as a plain string.
        """
        messages = _build_messages(user_message, system_prompt, history)

        logger.debug(
            f"Calling Groq [{self.model}] | "
            f"messages={len(messages)} | "
            f"user_msg_len={len(user_message)}"
        )

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else settings.groq_temperature,
            max_tokens=max_tokens or settings.groq_max_tokens,
            stream=False,
        )

        result = response.choices[0].message.content.strip()
        usage = response.usage

        logger.debug(
            f"Groq response received | "
            f"prompt_tokens={usage.prompt_tokens} | "
            f"completion_tokens={usage.completion_tokens}"
        )

        return result

    def generate_streaming(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        history: Optional[list[dict]] = None,
    ) -> Generator[str, None, None]:
        """
        Stream the LLM response token by token.
        Useful for real-time UI updates (e.g., Streamlit or WebSocket).

        Yields:
            String tokens as they arrive from the API.
        """
        messages = _build_messages(user_message, system_prompt, history)

        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=settings.groq_temperature,
            max_tokens=settings.groq_max_tokens,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def is_available(self) -> bool:
        """Quick health check — sends a tiny request to verify API key works."""
        try:
            self.generate("Say OK", max_tokens=5)
            return True
        except Exception as e:
            logger.error(f"Groq health check failed: {e}")
            return False


# ── Message Builder ───────────────────────────────────────────────────────────

def _build_messages(
    user_message: str,
    system_prompt: Optional[str],
    history: Optional[list[dict]],
) -> list[dict]:
    """
    Build the messages array for the Groq API.

    Format: [system?, ...history_turns, user_message]
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if history:
        # Include only the last N turns to avoid context overflow
        max_turns = settings.max_history_turns
        recent_history = history[-(max_turns * 2):]  # *2 because each turn = 2 messages
        messages.extend(recent_history)

    messages.append({"role": "user", "content": user_message})
    return messages


@lru_cache(maxsize=1)
def get_llm() -> GroqClient:
    """Singleton factory for the Groq client."""
    return GroqClient(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
    )
