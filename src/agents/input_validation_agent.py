"""
Input Validation Agent using GigaChat3.
"""

import logging
from typing import Any

from src.agents.chains.output_parsers import ValidationParser
from src.config import get_settings
from src.core.llm import TaskType, get_llm_router
from src.exceptions import InvalidInputError, PromptInjectionError
from src.prompts.content_guard_prompts import INJECTION_PATTERNS
from src.prompts.validation_prompts import INPUT_VALIDATION_PROMPT

logger = logging.getLogger(__name__)


class InputValidationAgent:
    """
    Agent for validating user input.

    Checks:
    1. Prompt injection
    2. SQL injection
    3. Content relevance
    4. Input length
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.llm_router = get_llm_router()
        self.llm = self.llm_router.get_model_for_task(TaskType.INPUT_VALIDATION)
        self.parser = ValidationParser()

    async def validate(self, user_input: str) -> dict[str, Any]:
        """
        Validate user input.

        Args:
            user_input: Raw user input

        Returns:
            Dict with validation result

        Raises:
            InvalidInputError: If input is invalid
            PromptInjectionError: If injection detected
        """

        if not self.settings.validation.validation_enabled:
            logger.info("⚠️ Validation disabled")
            return {"is_valid": True, "sanitized_input": user_input, "detected_issues": []}

        logger.info(f"🔍 Validating input: '{user_input[:50]}...'")

        # ═══════════════════════════════════════════════════════════
        # STEP 1: RULE-BASED CHECKS
        # ═══════════════════════════════════════════════════════════

        # Check length
        if len(user_input) < self.settings.validation.validation_min_input_length:
            msg = "Input too short"
            raise InvalidInputError(msg)

        if len(user_input) > self.settings.validation.validation_max_input_length:
            msg = "Input too long"
            raise InvalidInputError(msg)

        # Check injection patterns
        input_lower = user_input.lower()

        for pattern in INJECTION_PATTERNS:
            if pattern.lower() in input_lower:
                logger.warning(f"⚠️ Injection pattern detected: {pattern}")
                raise PromptInjectionError(pattern)

        # ═══════════════════════════════════════════════════════════
        # STEP 2: LLM VALIDATION
        # ═══════════════════════════════════════════════════════════

        try:
            result = await self._validate_with_llm(user_input)

            if not result["is_valid"]:
                if "prompt_injection" in result["detected_issues"]:
                    raise PromptInjectionError(result["reason"])
                raise InvalidInputError(result["reason"])

            logger.info("✅ Input validated successfully")

            return result

        except (InvalidInputError, PromptInjectionError):
            raise
        except Exception as e:
            logger.warning(f"⚠️ LLM validation failed: {e}, using rule-based only")

            # Fallback: Assume valid if rule-based passed
            return {
                "is_valid": True,
                "sanitized_input": user_input,
                "detected_issues": [],
                "reason": "LLM validation unavailable",
            }

    async def _validate_with_llm(self, user_input: str) -> dict[str, Any]:
        """Validate using LLM."""

        prompt = INPUT_VALIDATION_PROMPT.format(user_input=user_input)

        response = await self.llm.ainvoke(
            prompt, config={"timeout": self.settings.validation.validation_timeout_s}
        )

        return self.parser.parse(response.content)
