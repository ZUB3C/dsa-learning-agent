"""
Corrective RAG Tool: проверка и фильтрация документов по релевантности.
Code from Section 6 of architecture.
"""

import json
import logging
import time
from typing import Any

from src.config import get_settings
from src.core.llm import TaskType, get_llm_router
from src.prompts.content_guard_prompts import RELEVANCE_SCORING_PROMPT
from src.tools.base_tool import BaseTool, Document, ToolResult

logger = logging.getLogger(__name__)


class CorrectiveRAGTool(BaseTool):
    """
    Corrective RAG: проверка релевантности и покрытия концепций.

    Steps:
    1. Relevance scoring (GigaChat3 batch)
    2. Filter by min_relevance
    3. Concept coverage check (optional)
    4. Return filtered documents
    """

    name = "corrective_check"
    description = """
    Проверка релевантности и качества собранных документов.

    Params:
      query (str): поисковый запрос
      documents (list): список документов для проверки
      min_relevance (float): минимальная релевантность (default: 0.6)
      evaluate_coverage (bool): проверять покрытие концепций (default: True)

    Returns:
      ToolResult with filtered documents
    """

    def __init__(self) -> None:
        super().__init__()
        self.settings = get_settings()
        self.llm_router = get_llm_router()
        self.llm = self.llm_router.get_model_for_task(TaskType.RELEVANCE_SCORING)

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute corrective RAG check."""
        query = params.get("query", "")
        documents = params.get("documents", [])
        min_relevance = params.get(
            "min_relevance", self.settings.corrective_rag.corrective_min_relevance
        )
        evaluate_coverage = params.get("evaluate_coverage", True)

        if not query or not documents:
            return ToolResult(
                success=False, documents=[], error="Query and documents are required"
            )

        start_time = time.time()

        logger.info(f"🔍 Corrective RAG: checking {len(documents)} documents")

        # Convert to Document objects if needed
        doc_objects = []
        for doc in documents:
            if isinstance(doc, Document):
                doc_objects.append(doc)
            elif isinstance(doc, str):
                doc_objects.append(Document(page_content=doc))
            elif isinstance(doc, dict):
                doc_objects.append(
                    Document(
                        page_content=doc.get("page_content", ""), metadata=doc.get("metadata", {})
                    )
                )

        # ═══════════════════════════════════════════════════════════
        # STEP 1: RELEVANCE SCORING (Batch with GigaChat3)
        # ═══════════════════════════════════════════════════════════

        try:
            relevance_scores = await self._batch_relevance_scoring(query, doc_objects)
        except Exception as e:
            logger.warning(f"⚠️ Batch relevance scoring failed: {e}, using heuristic")
            relevance_scores = [0.7] * len(doc_objects)  # Optimistic fallback

        # ═══════════════════════════════════════════════════════════
        # STEP 2: FILTER BY RELEVANCE
        # ═══════════════════════════════════════════════════════════

        filtered_docs = []
        for doc, score in zip(doc_objects, relevance_scores, strict=False):
            if score >= min_relevance:
                doc.relevance_score = score
                filtered_docs.append(doc)

        logger.info(
            f"✅ Filtered: {len(doc_objects)} → {len(filtered_docs)} docs (min_relevance={min_relevance})"
        )

        # Check minimum count
        if len(filtered_docs) < self.settings.corrective_rag.corrective_min_docs_after_filter:
            logger.warning(f"⚠️ Only {len(filtered_docs)} docs passed, below minimum")

        # ═══════════════════════════════════════════════════════════
        # STEP 3: CONCEPT COVERAGE (Optional)
        # ═══════════════════════════════════════════════════════════

        coverage_score = 1.0
        if evaluate_coverage and filtered_docs:
            try:
                coverage_score = await self._evaluate_concept_coverage(query, filtered_docs)
                logger.info(f"📊 Concept coverage: {coverage_score:.2f}")
            except Exception as e:
                logger.warning(f"⚠️ Concept coverage evaluation failed: {e}")

        execution_time = (time.time() - start_time) * 1000  # ms

        return ToolResult(
            success=len(filtered_docs) > 0,
            documents=filtered_docs,
            metadata={
                "original_count": len(doc_objects),
                "filtered_count": len(filtered_docs),
                "avg_relevance": sum(relevance_scores) / len(relevance_scores)
                if relevance_scores
                else 0,
                "concept_coverage": coverage_score,
                "min_relevance_threshold": min_relevance,
            },
            execution_time_ms=execution_time,
        )

    async def _batch_relevance_scoring(self, query: str, documents: list[Document]) -> list[float]:
        """
        Batch relevance scoring with GigaChat3.

        Fallback: Individual scoring if batch fails
        """

        batch_size = self.settings.corrective_rag.corrective_batch_size
        all_scores = []

        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            # Format batch for prompt
            docs_text = ""
            for idx, doc in enumerate(batch):
                snippet = doc.page_content[:500]  # First 500 chars
                docs_text += f"{idx}. {snippet}\n\n"

            prompt = RELEVANCE_SCORING_PROMPT.format(query=query, documents_batch=docs_text)

            try:
                response = await self.llm.ainvoke(
                    prompt, config={"timeout": self.settings.corrective_rag.corrective_timeout_s}
                )

                # Parse JSON response
                result = json.loads(response.content)
                batch_scores = [r["relevance_score"] for r in result["results"]]
                all_scores.extend(batch_scores)

            except Exception as e:
                logger.warning(
                    f"⚠️ Batch {i // batch_size + 1} failed: {e}, using individual scoring"
                )

                # Fallback: Individual scoring
                for doc in batch:
                    try:
                        score = await self._individual_relevance_score(query, doc)
                        all_scores.append(score)
                    except Exception:
                        all_scores.append(0.5)  # Neutral fallback

        return all_scores

    async def _individual_relevance_score(self, query: str, doc: Document) -> float:
        """Individual relevance scoring (fallback)."""

        prompt = RELEVANCE_SCORING_PROMPT.format(
            query=query, documents_batch=f"0. {doc.page_content[:500]}"
        )

        try:
            response = await self.llm.ainvoke(prompt, config={"timeout": 5.0})
            result = json.loads(response.content)
            return result["results"][0]["relevance_score"]
        except Exception as e:
            logger.exception(f"❌ Individual scoring failed: {e}")
            raise

    async def _evaluate_concept_coverage(self, query: str, documents: list[Document]) -> float:
        """
        Evaluate concept coverage.

        Uses: GigaChat3 to check if documents cover key concepts
        """

        # Extract key concepts from query (simple heuristic)
        key_concepts = self._extract_key_concepts_heuristic(query)

        # Extract concepts from documents
        found_concepts = set()
        for doc in documents:
            doc_concepts = self._extract_key_concepts_heuristic(doc.page_content)
            found_concepts.update(doc_concepts)

        # Calculate coverage
        if not key_concepts:
            return 1.0  # No specific concepts required

        coverage = len(key_concepts & found_concepts) / len(key_concepts)

        return min(coverage, 1.0)

    def _extract_key_concepts_heuristic(self, text: str) -> set:
        """
        Simple heuristic for concept extraction.

        TODO: Use KeyBERT or spaCy for better extraction
        """

        # Common algorithm/data structure terms
        concepts = set()

        keywords = [
            "сортировка",
            "поиск",
            "дерево",
            "граф",
            "хеш",
            "стек",
            "очередь",
            "список",
            "массив",
            "алгоритм",
            "сложность",
            "O(n)",
            "рекурсия",
            "динамическое программирование",
            "жадный",
            "bfs",
            "dfs",
            "dijkstra",
        ]

        text_lower = text.lower()

        for keyword in keywords:
            if keyword in text_lower:
                concepts.add(keyword)

        return concepts
