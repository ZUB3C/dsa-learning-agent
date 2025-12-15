"""
Policy Compliance Checker using GigaChat3.
"""

import json
import logging

from src.config import get_settings
from src.core.llm import TaskType, get_llm_router
from src.models.content_guard_schemas import PolicyCheckResult
from src.prompts.content_guard_prompts import POLICY_COMPLIANCE_CHECK_PROMPT

logger = logging.getLogger(__name__)


class PolicyChecker:
    """
    Check document compliance with GigaChat policies.

    Fallback: Assume compliant if check fails (log for manual review)
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.llm_router = get_llm_router()
        self.llm = self.llm_router.get_model_for_task(TaskType.POLICY_CHECK)

    async def check_compliance(self, document: str) -> PolicyCheckResult:
        """
        Check document policy compliance.

        Args:
            document: Document text

        Returns:
            PolicyCheckResult
        """

        if not self.settings.content_guard.policy_check_enabled:
            # Policy check disabled, assume compliant
            return PolicyCheckResult(compliant=True, violations=[], confidence=1.0)

        try:
            return await self._check_with_llm(document)

        except Exception as e:
            logger.warning(f"⚠️ Policy check failed: {e}, assuming compliant (REVIEW NEEDED)")

            # Fallback: Assume compliant but log for manual review
            return PolicyCheckResult(
                compliant=True,
                violations=[],
                confidence=0.0,  # Low confidence = needs review
            )

    async def _check_with_llm(self, document: str) -> PolicyCheckResult:
        """Check compliance using GigaChat3."""

        # Truncate if too long
        max_length = 2000
        doc_snippet = document[:max_length]
        if len(document) > max_length:
            doc_snippet += "..."

        prompt = POLICY_COMPLIANCE_CHECK_PROMPT.format(document_content=doc_snippet)

        response = await self.llm.ainvoke(
            prompt, config={"timeout": self.settings.content_guard.policy_check_timeout_s}
        )

        # Parse response
        result = json.loads(response.content)

        return PolicyCheckResult(
            compliant=result["compliant"],
            violations=result.get("violations", []),
            confidence=result.get("confidence", 0.95),
        )
