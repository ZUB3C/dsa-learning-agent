"""
Text processing utilities.
"""

import logging
import re

logger = logging.getLogger(__name__)


class TextProcessor:
    """Text processing utilities."""

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text (remove extra whitespace, normalize).

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        # Remove extra spaces
        text = re.sub(r" +", " ", text)

        # Remove extra newlines
        text = re.sub(r"\n\n+", "\n\n", text)

        # Strip
        return text.strip()

    @staticmethod
    def truncate(text: str, max_length: int, suffix: str = "...") -> str:
        """
        Truncate text to max length.

        Args:
            text: Input text
            max_length: Maximum length
            suffix: Suffix to add if truncated

        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text

        return text[: max_length - len(suffix)] + suffix

    @staticmethod
    def extract_sentences(text: str) -> list[str]:
        """
        Extract sentences from text.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Simple sentence splitting
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def extract_keywords(text: str, stopwords: set[str] | None = None) -> list[str]:
        """
        Extract keywords (simple word frequency).

        Args:
            text: Input text
            stopwords: Set of stopwords to exclude

        Returns:
            List of keywords
        """
        stopwords = stopwords or {
            "и",
            "в",
            "на",
            "с",
            "по",
            "для",
            "как",
            "что",
            "это",
            "не",
            "the",
            "a",
            "an",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "is",
            "it",
        }

        # Tokenize
        words = re.findall(r"\b\w+\b", text.lower())

        # Filter stopwords and short words
        keywords = [w for w in words if len(w) > 3 and w not in stopwords]

        # Count frequency
        from collections import Counter

        word_freq = Counter(keywords)

        # Return top words
        return [word for word, count in word_freq.most_common(20)]

    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """
        Calculate Jaccard similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        # Tokenize
        words1 = set(re.findall(r"\b\w+\b", text1.lower()))
        words2 = set(re.findall(r"\b\w+\b", text2.lower()))

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize all whitespace to single spaces."""
        return " ".join(text.split())

    @staticmethod
    def remove_urls(text: str) -> str:
        """Remove URLs from text."""
        return re.sub(r"https?://\S+", "", text)

    @staticmethod
    def remove_emails(text: str) -> str:
        """Remove email addresses from text."""
        return re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)
