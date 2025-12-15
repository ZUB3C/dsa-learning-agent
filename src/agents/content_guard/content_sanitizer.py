"""
Content Sanitizer (rule-based).
"""

import logging
import re

from src.config import get_settings
from src.models.content_guard_schemas import ContentSanitizationResult

logger = logging.getLogger(__name__)


class ContentSanitizer:
    """
    Rule-based content sanitization.

    Steps:
    1. Remove HTML tags
    2. Remove suspicious URLs
    3. Remove emails (privacy)
    4. Normalize whitespace
    5. Truncate to max length
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def sanitize(self, content: str, source_type: str = "unknown") -> ContentSanitizationResult:
        """
        Sanitize content.

        Args:
            content: Original content
            source_type: "rag" or "web"

        Returns:
            ContentSanitizationResult
        """

        original_length = len(content)
        removed_elements = []

        sanitized = content

        # 1. Remove HTML tags (if from web)
        if source_type == "web":
            html_removed = self._remove_html_tags(sanitized)
            if html_removed != sanitized:
                removed_elements.append("html_tags")
                sanitized = html_removed

        # 2. Remove suspicious URLs
        if self.settings.content_guard.sanitize_remove_urls:
            url_removed = self._remove_suspicious_urls(sanitized)
            if url_removed != sanitized:
                removed_elements.append("suspicious_urls")
                sanitized = url_removed

        # 3. Remove emails
        if self.settings.content_guard.sanitize_remove_emails:
            email_removed = self._remove_emails(sanitized)
            if email_removed != sanitized:
                removed_elements.append("emails")
                sanitized = email_removed

        # 4. Normalize whitespace
        sanitized = self._normalize_whitespace(sanitized)

        # 5. Truncate to max length
        max_length = self.settings.content_guard.sanitize_max_length_per_doc
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "..."
            removed_elements.append("truncated")

        return ContentSanitizationResult(
            original_length=original_length,
            sanitized_length=len(sanitized),
            removed_elements=removed_elements,
            sanitized_content=sanitized,
        )

    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML tags."""
        # Simple regex-based removal
        return re.sub(r"<[^>]+>", "", text)

    def _remove_suspicious_urls(self, text: str) -> str:
        """Remove suspicious URLs (redirects, executables)."""

        # Patterns for suspicious URLs
        suspicious_patterns = [
            r"bit\.ly/\S+",
            r"tinyurl\.com/\S+",
            r"goo\.gl/\S+",
            r"\S+\.exe",
            r"\S+\.bat",
            r"\S+\.sh",
        ]

        for pattern in suspicious_patterns:
            text = re.sub(pattern, "[URL removed]", text)

        return text

    def _remove_emails(self, text: str) -> str:
        """Remove email addresses."""
        # Simple email pattern
        return re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[email removed]", text
        )

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace."""
        # Replace multiple spaces with single space
        text = re.sub(r" +", " ", text)

        # Replace multiple newlines with double newline
        text = re.sub(r"\n\n+", "\n\n", text)

        # Strip leading/trailing whitespace
        return text.strip()
