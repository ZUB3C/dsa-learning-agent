"""
DeepEval metrics for LLM quality assessment.
"""

import logging

logger = logging.getLogger(__name__)


class DeepEvalMetrics:
    """
    Custom metrics for evaluating generated materials.

    Note: This is a simplified implementation. For production,
    integrate with DeepEval library: https://docs.confident-ai.com/
    """

    @staticmethod
    def calculate_answer_relevance(query: str, answer: str) -> float:
        """
        Calculate answer relevance score.

        Simple implementation using keyword overlap.
        In production, use DeepEval's AnswerRelevancyMetric.

        Args:
            query: User query
            answer: Generated answer

        Returns:
            Relevance score (0-1)
        """

        # Extract keywords from query
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())

        # Calculate Jaccard similarity
        intersection = len(query_words & answer_words)
        union = len(query_words | answer_words)

        score = intersection / union if union > 0 else 0.0

        return min(score, 1.0)

    @staticmethod
    def calculate_faithfulness(context: str, answer: str) -> float:
        """
        Calculate faithfulness score (how well answer is grounded in context).

        Args:
            context: Source documents
            answer: Generated answer

        Returns:
            Faithfulness score (0-1)
        """

        # Simple check: what percentage of answer content appears in context
        answer_sentences = answer.split(".")

        faithful_count = 0

        for sentence in answer_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check if key words from sentence appear in context
            words = set(sentence.lower().split())
            context_words = set(context.lower().split())

            overlap = len(words & context_words) / len(words) if words else 0

            if overlap > 0.5:  # At least 50% overlap
                faithful_count += 1

        score = faithful_count / len(answer_sentences) if answer_sentences else 0.0

        return min(score, 1.0)

    @staticmethod
    def calculate_contextual_relevance(query: str, context: str) -> float:
        """
        Calculate how relevant retrieved context is to query.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Relevance score (0-1)
        """

        query_words = set(query.lower().split())
        context_words = set(context.lower().split())

        intersection = len(query_words & context_words)

        score = intersection / len(query_words) if query_words else 0.0

        return min(score, 1.0)

    @staticmethod
    def calculate_coherence(text: str) -> float:
        """
        Calculate text coherence.

        Simple heuristic based on sentence length variation and structure.

        Args:
            text: Text to evaluate

        Returns:
            Coherence score (0-1)
        """

        sentences = [s.strip() for s in text.split(".") if s.strip()]

        if len(sentences) < 2:
            return 0.5  # Not enough data

        # Calculate sentence length variation
        lengths = [len(s.split()) for s in sentences]
        avg_length = sum(lengths) / len(lengths)

        # Ideal: 10-20 words per sentence
        ideal_min, ideal_max = 10, 20

        score = 1.0

        if avg_length < ideal_min:
            score -= 0.3
        elif avg_length > ideal_max:
            score -= 0.2

        return max(0.0, min(score, 1.0))

    @staticmethod
    def evaluate_material(query: str, material: str, context: str) -> dict[str, float]:
        """
        Comprehensive evaluation of generated material.

        Args:
            query: User query
            material: Generated material
            context: Source context

        Returns:
            Dict with evaluation scores
        """

        return {
            "answer_relevance": DeepEvalMetrics.calculate_answer_relevance(query, material),
            "faithfulness": DeepEvalMetrics.calculate_faithfulness(context, material),
            "contextual_relevance": DeepEvalMetrics.calculate_contextual_relevance(query, context),
            "coherence": DeepEvalMetrics.calculate_coherence(material),
        }
