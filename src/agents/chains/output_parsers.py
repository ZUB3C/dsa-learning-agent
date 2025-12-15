"""
JSON output parsers for LLM responses.
"""

import json
import logging
import re
from typing import Any

from src.exceptions import ToolExecutionError

logger = logging.getLogger(__name__)


class OutputParser:
    """Base output parser."""

    @staticmethod
    def extract_json(text: str) -> dict | None:
        """
        Extract JSON from text (handles markdown code blocks).

        Args:
            text: Text containing JSON

        Returns:
            Parsed dict or None
        """

        # Try to find JSON in markdown code blocks
        json_pattern = r"``````"
        matches = re.findall(json_pattern, text, re.DOTALL)

        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass

        # Try direct JSON parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in text
        json_obj_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        matches = re.findall(json_obj_pattern, text, re.DOTALL)

        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        return None


class ThoughtGenerationParser(OutputParser):
    """
    Parser for thought generation responses.

    Expected format:
    {
      "thoughts": [
        {
          "reasoning": "...",
          "tool_name": "...",
          "tool_params": {...},
          "explanation": "..."
        }
      ]
    }
    """

    @staticmethod
    def parse(response_text: str) -> list[dict[str, Any]]:
        """
        Parse thought generation response.

        Args:
            response_text: LLM response text

        Returns:
            List of thought dicts

        Raises:
            ToolExecutionError: If parsing fails
        """

        parsed = OutputParser.extract_json(response_text)

        if not parsed:
            logger.error(f"Failed to parse JSON from: {response_text[:200]}...")
            msg = "thought_generation"
            raise ToolExecutionError(msg, "Failed to parse JSON response", 0)

        thoughts = parsed.get("thoughts", [])

        if not thoughts:
            logger.warning("No thoughts in response, using empty list")
            return []

        # Validate each thought
        validated_thoughts = []
        for thought in thoughts:
            if not isinstance(thought, dict):
                logger.warning(f"Invalid thought format: {thought}")
                continue

            # Required fields
            if "tool_name" not in thought:
                logger.warning(f"Thought missing tool_name: {thought}")
                continue

            # Set defaults
            thought.setdefault("reasoning", "")
            thought.setdefault("tool_params", {})
            thought.setdefault("explanation", "")

            validated_thoughts.append(thought)

        return validated_thoughts


class PromiseEvaluationParser(OutputParser):
    """
    Parser for promise evaluation responses.

    Expected format:
    {
      "promise_score": 0.85,
      "reasoning": "...",
      "breakdown": {...}
    }
    """

    @staticmethod
    def parse(response_text: str) -> float:
        """
        Parse promise score.

        Args:
            response_text: LLM response text

        Returns:
            Promise score (0-1)
        """

        parsed = OutputParser.extract_json(response_text)

        if not parsed:
            logger.warning("Failed to parse promise evaluation, using default 0.5")
            return 0.5

        score = parsed.get("promise_score", 0.5)

        # Validate range
        return max(0.0, min(1.0, float(score)))


class NodeEvaluationParser(OutputParser):
    """
    Parser for post-execution node evaluation.

    Expected format:
    {
      "completeness_score": 0.75,
      "relevance_score": 0.88,
      "quality_score": 0.82,
      "should_continue": true,
      "reasoning": "..."
    }
    """

    @staticmethod
    def parse(response_text: str) -> dict[str, Any]:
        """
        Parse node evaluation.

        Args:
            response_text: LLM response text

        Returns:
            Dict with evaluation scores
        """

        parsed = OutputParser.extract_json(response_text)

        if not parsed:
            logger.warning("Failed to parse node evaluation, using defaults")
            return {
                "completeness_score": 0.5,
                "relevance_score": 0.5,
                "quality_score": 0.5,
                "should_continue": True,
                "reasoning": "Failed to parse evaluation",
            }

        # Validate and bound scores
        return {
            "completeness_score": max(0.0, min(1.0, float(parsed.get("completeness_score", 0.5)))),
            "relevance_score": max(0.0, min(1.0, float(parsed.get("relevance_score", 0.5)))),
            "quality_score": max(0.0, min(1.0, float(parsed.get("quality_score", 0.5)))),
            "should_continue": parsed.get("should_continue", True),
            "reasoning": parsed.get("reasoning", ""),
        }


class ValidationParser(OutputParser):
    """
    Parser for input validation responses.

    Expected format:
    {
      "is_valid": true,
      "reason": "...",
      "sanitized_input": "...",
      "detected_issues": []
    }
    """

    @staticmethod
    def parse(response_text: str) -> dict[str, Any]:
        """
        Parse validation response.

        Args:
            response_text: LLM response text

        Returns:
            Dict with validation result
        """

        parsed = OutputParser.extract_json(response_text)

        if not parsed:
            logger.warning("Failed to parse validation, assuming valid")
            return {
                "is_valid": True,
                "reason": "Failed to parse validation response",
                "sanitized_input": "",
                "detected_issues": [],
            }

        return {
            "is_valid": parsed.get("is_valid", True),
            "reason": parsed.get("reason", ""),
            "sanitized_input": parsed.get("sanitized_input", ""),
            "detected_issues": parsed.get("detected_issues", []),
        }
