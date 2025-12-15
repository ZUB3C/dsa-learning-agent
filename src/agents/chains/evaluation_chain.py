"""
Evaluation chain for promise and completeness scoring.
"""

import logging

from langchain_core.language_models import BaseChatModel

from src.agents.chains.output_parsers import NodeEvaluationParser, PromiseEvaluationParser
from src.models.react_schemas import NodeEvaluation, TreeNode
from src.prompts.evaluation_prompts import (
    POST_EXECUTION_EVALUATION_PROMPT,
    PROMISE_EVALUATION_PROMPT,
)

logger = logging.getLogger(__name__)


class EvaluationChain:
    """
    Chain for evaluating nodes (promise and post-execution).

    Uses: GigaChat3 for fast evaluation
    """

    def __init__(self, llm: BaseChatModel) -> None:
        """
        Initialize evaluation chain.

        Args:
            llm: LLM instance (should be GigaChat3)
        """
        self.llm = llm
        self.promise_parser = PromiseEvaluationParser()
        self.node_parser = NodeEvaluationParser()

    async def evaluate_promise(
        self, candidate: TreeNode, current_state: TreeNode, query: str
    ) -> float:
        """
        Evaluate promise score for a candidate.

        Args:
            candidate: Candidate node
            current_state: Current node state
            query: User query

        Returns:
            Promise score (0-1)
        """

        logger.debug(f"📊 Evaluating promise for {candidate.node_id}")

        # Build prompt
        prompt = PROMISE_EVALUATION_PROMPT.format(
            query=query,
            collected_so_far=len(current_state.collected_info),
            proposed_action=candidate.planned_action,
            reasoning=candidate.reasoning,
        )

        try:
            # Call LLM
            response = await self.llm.ainvoke(prompt, config={"timeout": 5.0})

            # Parse score
            score = self.promise_parser.parse(response.content)

            logger.debug(f"📊 Promise score: {score:.2f}")

            return score

        except Exception as e:
            logger.warning(f"⚠️ Promise evaluation failed: {e}, using heuristic")

            # Fallback: Heuristic
            return self._heuristic_promise(candidate)

    async def evaluate_node(self, node: TreeNode, query: str) -> NodeEvaluation:
        """
        Evaluate node after action execution.

        Args:
            node: Node to evaluate
            query: User query

        Returns:
            NodeEvaluation with scores
        """

        logger.debug(f"📊 Evaluating node {node.node_id}")

        # Summarize latest documents
        latest_docs_summary = self._summarize_docs(node.collected_info[-3:])

        # Build prompt
        prompt = POST_EXECUTION_EVALUATION_PROMPT.format(
            query=query,
            total_docs=len(node.collected_info),
            latest_docs_summary=latest_docs_summary,
        )

        try:
            # Call LLM
            response = await self.llm.ainvoke(prompt, config={"timeout": 5.0})

            # Parse evaluation
            eval_data = self.node_parser.parse(response.content)

            evaluation = NodeEvaluation(
                completeness=eval_data["completeness_score"],
                relevance=eval_data["relevance_score"],
                quality=eval_data["quality_score"],
                should_continue=eval_data["should_continue"],
            )

            logger.debug(
                f"📊 Node evaluation: completeness={evaluation.completeness:.2f}, "
                f"relevance={evaluation.relevance:.2f}, quality={evaluation.quality:.2f}"
            )

            return evaluation

        except Exception as e:
            logger.warning(f"⚠️ Node evaluation failed: {e}, using heuristic")

            # Fallback: Heuristic
            completeness = self._heuristic_completeness(node)

            return NodeEvaluation(
                completeness=completeness,
                relevance=0.8,  # Assume good
                quality=0.8,  # Assume good
                should_continue=completeness < 0.85,
            )

    def _heuristic_promise(self, candidate: TreeNode) -> float:
        """
        Heuristic promise scoring (fallback).

        Logic:
        - adaptive_rag: high priority (0.9)
        - corrective_check: medium (0.7)
        - web_search: medium (0.6)
        - other: low (0.5)
        """

        tool_name = candidate.planned_action.get("tool_name", "")

        tool_priorities = {
            "adaptive_rag_search": 0.9,
            "corrective_check": 0.7,
            "web_search": 0.6,
            "web_scraper": 0.5,
            "extract_concepts": 0.6,
            "memory_retrieval": 0.8,
        }

        return tool_priorities.get(tool_name, 0.5)

    def _heuristic_completeness(self, node: TreeNode) -> float:
        """
        Heuristic completeness scoring (fallback).

        Formula: 0.15 * num_documents (capped at 1.0)
        """

        num_docs = len(node.collected_info)
        return min(1.0, 0.15 * num_docs)

    def _summarize_docs(self, documents: list) -> str:
        """Summarize documents for prompt."""

        if not documents:
            return "Нет документов"

        summary = ""
        for i, doc in enumerate(documents, 1):
            snippet = doc.page_content[:200].replace("\n", " ")
            source = doc.metadata.get("source", "unknown")
            summary += f"{i}. [{source}] {snippet}...\n\n"

        return summary
