"""
Reasoning chain for thought generation.
"""

import logging

from langchain_core.language_models import BaseChatModel

from src.agents.chains.output_parsers import ThoughtGenerationParser
from src.core.memory.memory_schemas import MemoryContext
from src.exceptions import LLMUnavailableError
from src.models.react_schemas import TreeNode
from src.prompts.thought_generation_prompts import (
    THOUGHT_GENERATION_PROMPT,
)

logger = logging.getLogger(__name__)


class ReasoningChain:
    """
    Chain for generating candidate thoughts in ToT.

    Uses: GigaChat-2-Max for high-quality reasoning
    """

    def __init__(self, llm: BaseChatModel) -> None:
        """
        Initialize reasoning chain.

        Args:
            llm: LLM instance (should be GigaChat-2-Max)
        """
        self.llm = llm
        self.parser = ThoughtGenerationParser()

    async def generate_thoughts(
        self,
        current_node: TreeNode,
        query: str,
        user_level: str,
        memory_context: MemoryContext,
        branching_factor: int = 3,
    ) -> list[TreeNode]:
        """
        Generate candidate thoughts for next step.

        Args:
            current_node: Current ToT node
            query: User query
            user_level: User level
            memory_context: Memory context
            branching_factor: Number of thoughts to generate

        Returns:
            List of TreeNode candidates
        """

        logger.info(f"💭 Generating {branching_factor} thoughts at depth {current_node.depth}")

        # Format collected info summary
        collected_info_summary = self._summarize_collected(current_node.collected_info)

        # Build prompt
        prompt = THOUGHT_GENERATION_PROMPT.format(
            query=query,
            user_level=user_level,
            depth=current_node.depth,
            completeness=current_node.completeness_score,
            collected_info_summary=collected_info_summary,
            memory_hints=memory_context.procedural_hints,
            branching_factor=branching_factor,
        )

        try:
            # Call LLM
            response = await self.llm.ainvoke(prompt)

            # Parse response
            thoughts_data = self.parser.parse(response.content)

            if not thoughts_data:
                logger.warning("No thoughts generated, using fallback")
                msg = "Empty thoughts list"
                raise ValueError(msg)

            # Convert to TreeNode objects
            candidates = []
            for thought_data in thoughts_data[:branching_factor]:
                node = TreeNode(
                    parent_id=current_node.node_id,
                    depth=current_node.depth + 1,
                    thought=thought_data.get("reasoning", ""),
                    reasoning=thought_data.get("reasoning", ""),
                    planned_action={
                        "tool_name": thought_data.get("tool_name"),
                        "tool_params": thought_data.get("tool_params", {}),
                    },
                    collected_info=current_node.collected_info.copy(),  # Inherit
                )
                candidates.append(node)

            logger.info(f"✅ Generated {len(candidates)} candidate thoughts")

            return candidates

        except Exception as e:
            logger.exception(f"❌ Thought generation failed: {e}")
            msg = "GigaChat-2-Max"
            raise LLMUnavailableError(msg, str(e))

    def _summarize_collected(self, documents: list) -> str:
        """
        Summarize collected documents.

        Args:
            documents: List of Document objects

        Returns:
            Summary string
        """

        if not documents:
            return "Нет собранной информации"

        # Count by source
        rag_count = sum(1 for doc in documents if "rag" in doc.metadata.get("source", "").lower())
        web_count = sum(1 for doc in documents if "web" in doc.metadata.get("source", "").lower())

        summary = f"{len(documents)} документов (RAG: {rag_count}, Web: {web_count})"

        # Add snippets
        if documents:
            summary += "\n\nПоследние документы:\n"
            for i, doc in enumerate(documents[-3:], 1):  # Last 3
                snippet = doc.page_content[:100].replace("\n", " ")
                summary += f"{i}. {snippet}...\n"

        return summary
