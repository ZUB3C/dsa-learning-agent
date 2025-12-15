"""
LLM fallback strategies.
"""

import logging
import time
from collections.abc import Callable
from functools import wraps

from langchain_core.language_models import BaseChatModel

from src.exceptions import LLMUnavailableError

logger = logging.getLogger(__name__)


class LLMFallbackHandler:
    """
    Handle LLM failures with fallback strategies.

    Strategies:
    1. Retry with exponential backoff
    2. Switch to alternative model
    3. Use cached response (if available)
    4. Return rule-based fallback
    """

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0) -> None:
        """
        Initialize fallback handler.

        Args:
            max_retries: Maximum retry attempts
            base_delay: Base delay for exponential backoff (seconds)
        """
        self.max_retries = max_retries
        self.base_delay = base_delay

    async def call_with_fallback(
        self,
        primary_llm: BaseChatModel,
        fallback_llm: BaseChatModel | None,
        prompt: str,
        rule_based_fallback: Callable | None = None,
        **kwargs,
    ) -> str:
        """
        Call LLM with automatic fallback.

        Args:
            primary_llm: Primary LLM to use
            fallback_llm: Fallback LLM (optional)
            prompt: Prompt text
            rule_based_fallback: Rule-based function (optional)
            **kwargs: Additional arguments for LLM

        Returns:
            Response text

        Raises:
            LLMUnavailableError: If all fallbacks fail
        """

        # Try primary LLM with retries
        for attempt in range(self.max_retries):
            try:
                response = await primary_llm.ainvoke(prompt, **kwargs)
                return response.content

            except Exception as e:
                logger.warning(
                    f"⚠️ Primary LLM failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = self.base_delay * (2**attempt)
                    logger.info(f"⏳ Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.exception("❌ Primary LLM exhausted all retries")

        # Try fallback LLM
        if fallback_llm:
            logger.info("🔄 Switching to fallback LLM...")
            try:
                response = await fallback_llm.ainvoke(prompt, **kwargs)
                logger.info("✅ Fallback LLM succeeded")
                return response.content
            except Exception as e:
                logger.exception(f"❌ Fallback LLM also failed: {e}")

        # Try rule-based fallback
        if rule_based_fallback:
            logger.info("🔄 Using rule-based fallback...")
            try:
                result = rule_based_fallback(prompt)
                logger.warning("⚠️ Using rule-based fallback (degraded quality)")
                return result
            except Exception as e:
                logger.exception(f"❌ Rule-based fallback failed: {e}")

        # All fallbacks exhausted
        raise LLMUnavailableError(
            model_name=primary_llm.__class__.__name__, details="All fallback strategies exhausted"
        )


def with_llm_fallback(
    fallback_llm: BaseChatModel | None = None, rule_based: Callable | None = None
):
    """
    Decorator for automatic LLM fallback.

    Usage:
        @with_llm_fallback(fallback_llm=cheap_model, rule_based=simple_function)
        async def my_llm_call(llm, prompt):
            return await llm.ainvoke(prompt)
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            handler = LLMFallbackHandler()

            # Extract primary LLM from args
            primary_llm = args[0] if args else kwargs.get("llm")
            prompt = args[1] if len(args) > 1 else kwargs.get("prompt", "")

            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"⚠️ LLM call failed: {e}, using fallback")
                return await handler.call_with_fallback(
                    primary_llm=primary_llm,
                    fallback_llm=fallback_llm,
                    prompt=prompt,
                    rule_based_fallback=rule_based,
                )

        return wrapper

    return decorator
