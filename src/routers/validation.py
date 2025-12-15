"""
Validation router for testing input validation.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.agents.input_validation_agent import InputValidationAgent
from src.exceptions import InvalidInputError, PromptInjectionError

logger = logging.getLogger(__name__)
router = APIRouter()


class ValidateInputRequest(BaseModel):
    """Request for input validation."""

    input_text: str = Field(..., description="Text to validate")
    check_injection: bool = Field(default=True, description="Check for prompt injection")
    check_length: bool = Field(default=True, description="Check length constraints")


class ValidateInputResponse(BaseModel):
    """Response from input validation."""

    is_valid: bool
    sanitized_input: str
    issues: list[str] = []
    warnings: list[str] = []
    metadata: dict[str, Any] = {}


@router.post("/validate", response_model=ValidateInputResponse)
async def validate_input(request: ValidateInputRequest):
    """
    Validate user input for safety and quality.

    Useful for testing validation rules.
    """

    logger.info(f"🔍 Validating input: {request.input_text[:50]}...")

    validation_agent = InputValidationAgent()

    try:
        result = await validation_agent.validate(request.input_text)

        return ValidateInputResponse(
            is_valid=result["is_valid"],
            sanitized_input=result["sanitized_input"],
            issues=result["detected_issues"],
            warnings=[],
            metadata={
                "reason": result.get("reason", ""),
                "original_length": len(request.input_text),
                "sanitized_length": len(result["sanitized_input"]),
            },
        )

    except PromptInjectionError as e:
        logger.warning(f"⚠️ Prompt injection detected: {e}")
        return ValidateInputResponse(
            is_valid=False,
            sanitized_input="",
            issues=["prompt_injection"],
            warnings=[str(e)],
            metadata={"injection_pattern": str(e)},
        )

    except InvalidInputError as e:
        logger.warning(f"⚠️ Invalid input: {e}")
        return ValidateInputResponse(
            is_valid=False,
            sanitized_input="",
            issues=["invalid_input"],
            warnings=[str(e)],
            metadata={},
        )

    except Exception as e:
        logger.exception(f"❌ Validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validation-rules")
async def get_validation_rules():
    """
    Get current validation rules and thresholds.
    """

    from src.config import get_settings

    settings = get_settings()

    return {
        "enabled": settings.validation.validation_enabled,
        "rules": {
            "min_length": settings.validation.validation_min_input_length,
            "max_length": settings.validation.validation_max_input_length,
            "timeout_s": settings.validation.validation_timeout_s,
        },
        "injection_patterns": [
            "ignore previous",
            "disregard all",
            "forget instructions",
            "new instructions",
            "system prompt",
        ],
    }
