"""
Toxicity Checker using GigaChat3.
"""

import json
import logging

from src.config import get_settings
from src.core.llm import TaskType, get_llm_router
from src.models.content_guard_schemas import ToxicityBatchResult, ToxicityCheckResult
from src.prompts.content_guard_prompts import BLACKLIST_WORDS_RU, TOXICITY_CHECK_PROMPT

logger = logging.getLogger(__name__)


class ToxicityChecker:
    """
    Check documents for toxic content using GigaChat3.

    Fallback chain:
    1. GigaChat3 batch check
    2. Individual checks
    3. Rule-based filter (blacklist)
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.llm_router = get_llm_router()
        self.llm = self.llm_router.get_model_for_task(TaskType.TOXICITY_CHECK)

    async def check_batch(self, documents: list[str]) -> ToxicityBatchResult:
        """
        Check batch of documents for toxicity.

        Args:
            documents: List of document texts

        Returns:
            ToxicityBatchResult with scores for each document
        """

        batch_size = self.settings.content_guard.toxicity_batch_size
        all_results = []

        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            try:
                batch_results = await self._check_batch_llm(batch, start_id=i)
                all_results.extend(batch_results)

            except Exception as e:
                logger.warning(f"⚠️ LLM batch check failed: {e}, using fallback")

                # Fallback: Individual checks
                for doc_id, doc in enumerate(batch, start=i):
                    try:
                        result = await self._check_individual(doc, doc_id)
                        all_results.append(result)
                    except Exception:
                        # Final fallback: Rule-based
                        result = self._check_rule_based(doc, doc_id)
                        all_results.append(result)

        # Calculate aggregates
        filtered_count = sum(1 for r in all_results if not r.is_safe)
        avg_toxicity = (
            sum(r.toxicity_score for r in all_results) / len(all_results) if all_results else 0.0
        )

        return ToxicityBatchResult(
            results=all_results, avg_toxicity=avg_toxicity, filtered_count=filtered_count
        )

    async def _check_batch_llm(
        self, documents: list[str], start_id: int = 0
    ) -> list[ToxicityCheckResult]:
        """Check batch using GigaChat3."""

        # Format documents for prompt
        docs_text = ""
        for idx, doc in enumerate(documents):
            snippet = doc[:500]  # First 500 chars
            docs_text += f"{idx + 1}. {snippet}\n\n"

        prompt = TOXICITY_CHECK_PROMPT.format(documents_batch=docs_text)

        response = await self.llm.ainvoke(
            prompt, config={"timeout": self.settings.content_guard.toxicity_timeout_s}
        )

        # Parse response
        result = json.loads(response.content)

        # Convert to ToxicityCheckResult objects
        return [
            ToxicityCheckResult(
                doc_id=start_id + r["doc_id"] - 1,  # Adjust ID
                is_safe=r["is_safe"],
                toxicity_score=r["toxicity_score"],
                issues=r.get("issues", []),
            )
            for r in result["results"]
        ]

    async def _check_individual(self, document: str, doc_id: int) -> ToxicityCheckResult:
        """Check individual document (fallback)."""

        batch_result = await self._check_batch_llm([document], start_id=doc_id)
        return batch_result[0]

    def _check_rule_based(self, document: str, doc_id: int) -> ToxicityCheckResult:
        """
        Rule-based toxicity check (final fallback).

        Uses:
        - Blacklist words
        - Pattern matching
        """

        doc_lower = document.lower()
        issues = []
        toxicity_score = 0.0

        # Check blacklist words
        for word in BLACKLIST_WORDS_RU:
            if word in doc_lower:
                issues.append(f"Содержит запрещенное слово: {word}")
                toxicity_score += 0.3

        # Cap at 1.0
        toxicity_score = min(toxicity_score, 1.0)

        threshold = self.settings.content_guard.toxicity_threshold
        is_safe = toxicity_score < threshold

        return ToxicityCheckResult(
            doc_id=doc_id, is_safe=is_safe, toxicity_score=toxicity_score, issues=issues
        )
