"""
Cost calculation utilities.
"""

import logging

logger = logging.getLogger(__name__)


class CostCalculator:
    """
    Calculate estimated costs for LLM usage.

    Pricing (approximate, as of 2025):
    - GigaChat-2-Max: $0.002 per call
    - GigaChat3: $0.0005 per call
    - OpenAI embeddings: $0.0001 per 1K tokens
    """

    # Pricing per call (USD)
    PRICING = {"gigachat2_max": 0.002, "gigachat3": 0.0005, "openai_embeddings_per_1k": 0.0001}

    @classmethod
    def calculate_llm_cost(cls, llm_usage: dict[str, int]) -> float:
        """
        Calculate total LLM cost.

        Args:
            llm_usage: Dict with LLM call counts, e.g., {"gigachat2": 5, "gigachat3": 20}

        Returns:
            Total cost in USD
        """

        total_cost = 0.0

        # GigaChat-2-Max
        gigachat2_calls = llm_usage.get("gigachat2", 0)
        total_cost += gigachat2_calls * cls.PRICING["gigachat2_max"]

        # GigaChat3
        gigachat3_calls = llm_usage.get("gigachat3", 0)
        total_cost += gigachat3_calls * cls.PRICING["gigachat3"]

        return round(total_cost, 4)

    @classmethod
    def calculate_embedding_cost(cls, token_count: int) -> float:
        """
        Calculate embedding cost.

        Args:
            token_count: Number of tokens

        Returns:
            Cost in USD
        """

        cost = (token_count / 1000) * cls.PRICING["openai_embeddings_per_1k"]
        return round(cost, 4)

    @classmethod
    def estimate_total_cost(
        cls, gigachat2_calls: int, gigachat3_calls: int, embedding_tokens: int = 0
    ) -> dict[str, float]:
        """
        Estimate total cost breakdown.

        Args:
            gigachat2_calls: Number of GigaChat-2-Max calls
            gigachat3_calls: Number of GigaChat3 calls
            embedding_tokens: Number of embedding tokens

        Returns:
            Dict with cost breakdown
        """

        gigachat2_cost = gigachat2_calls * cls.PRICING["gigachat2_max"]
        gigachat3_cost = gigachat3_calls * cls.PRICING["gigachat3"]
        embedding_cost = cls.calculate_embedding_cost(embedding_tokens)

        total_cost = gigachat2_cost + gigachat3_cost + embedding_cost

        return {
            "gigachat2_max_cost": round(gigachat2_cost, 4),
            "gigachat3_cost": round(gigachat3_cost, 4),
            "embedding_cost": round(embedding_cost, 4),
            "total_cost": round(total_cost, 4),
        }

    @classmethod
    def cost_per_generation(cls, tot_iterations: int, branching_factor: int = 3) -> float:
        """
        Estimate cost per ToT generation.

        Assumptions:
        - 1 GigaChat-2-Max call per iteration (thought generation)
        - 3-5 GigaChat3 calls per iteration (promise eval + node eval)

        Args:
            tot_iterations: Number of ToT iterations
            branching_factor: Branching factor (candidates per iteration)

        Returns:
            Estimated cost in USD
        """

        # GigaChat-2-Max: 1 call per iteration
        gigachat2_calls = tot_iterations

        # GigaChat3: promise eval (branching_factor) + node eval (1) per iteration
        gigachat3_calls = tot_iterations * (branching_factor + 1)

        llm_usage = {"gigachat2": gigachat2_calls, "gigachat3": gigachat3_calls}

        return cls.calculate_llm_cost(llm_usage)
