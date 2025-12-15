"""
Quality Gate for final document validation.
"""

import logging
import re

from src.config import get_settings
from src.models.content_guard_schemas import QualityGateResult

logger = logging.getLogger(__name__)


class QualityGate:
    """
    Final quality checks for documents.

    Checks:
    1. Minimum content length
    2. Maximum content length
    3. Minimum sentence count
    4. Content type (not just URLs/metadata)
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def check(self, content: str) -> QualityGateResult:
        """
        Perform quality checks.

        Args:
            content: Document content

        Returns:
            QualityGateResult
        """

        # Check 1: Length
        length_ok = self._check_length(content)

        # Check 2: Sentence count
        sentence_count_ok = self._check_sentence_count(content)

        # Check 3: Content type
        content_type_ok = self._check_content_type(content)

        # Overall pass
        passed = length_ok and sentence_count_ok and content_type_ok

        # Build reason if failed
        reason = None
        if not passed:
            reasons = []
            if not length_ok:
                reasons.append(
                    f"length not in range [{self.settings.content_guard.min_content_length}, {self.settings.content_guard.max_content_length}]"
                )
            if not sentence_count_ok:
                reasons.append(
                    f"sentence count < {self.settings.content_guard.min_sentence_count}"
                )
            if not content_type_ok:
                reasons.append("content is mostly URLs or metadata")

            reason = "; ".join(reasons)

        return QualityGateResult(
            passed=passed,
            length_ok=length_ok,
            sentence_count_ok=sentence_count_ok,
            content_type_ok=content_type_ok,
            reason=reason,
        )

    def _check_length(self, content: str) -> bool:
        """Check if content length is within acceptable range."""
        length = len(content)

        min_length = self.settings.content_guard.min_content_length
        max_length = self.settings.content_guard.max_content_length

        return min_length <= length <= max_length

    def _check_sentence_count(self, content: str) -> bool:
        """Check if content has minimum sentence count."""
        # Simple sentence splitting
        sentences = re.split(r"[.!?]+", content)
        sentences = [s.strip() for s in sentences if s.strip()]

        min_sentences = self.settings.content_guard.min_sentence_count

        return len(sentences) >= min_sentences

    def _check_content_type(self, content: str) -> bool:
        """
        Check if content is meaningful (not just URLs/metadata).

        Heuristic: If > 30% of content is URLs, reject
        """

        # Count URLs
        url_pattern = r"https?://[^\s]+"
        urls = re.findall(url_pattern, content)

        if not urls:
            return True  # No URLs, good

        # Calculate URL ratio
        url_chars = sum(len(url) for url in urls)
        total_chars = len(content)

        url_ratio = url_chars / total_chars if total_chars > 0 else 0

        # Reject if > 30% URLs
        return url_ratio < 0.3
