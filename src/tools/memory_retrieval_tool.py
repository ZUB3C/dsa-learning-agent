"""
Memory Retrieval Tool for accessing procedural memory patterns.
"""

import logging
from typing import Any

from src.config import get_settings
from src.tools.base_tool import BaseTool, Document, ToolResult

logger = logging.getLogger(__name__)


class MemoryRetrievalTool(BaseTool):
    """
    Retrieve successful patterns from procedural memory.

    Helps agent learn from past successful generations.
    """

    name = "memory_retrieval"
    description = """
    Поиск успешных стратегий в памяти агента.

    Params:
      query (str): запрос для поиска паттернов
      memory_type (str): "working" | "procedural" | "all"
      limit (int): количество паттернов (default: 3)
      min_success_score (float): минимальный успех (default: 0.8)

    Returns:
      ToolResult with memory patterns as documents
    """

    def __init__(self) -> None:
        super().__init__()
        self.settings = get_settings()
        self._memory_manager = None

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute memory retrieval."""
        query = params.get("query", "")
        memory_type = params.get("memory_type", "procedural")
        limit = params.get("limit", 3)
        min_success_score = params.get(
            "min_success_score", self.settings.memory.memory_procedural_min_success_score
        )

        if not query:
            return ToolResult(success=False, documents=[], error="Query parameter is required")

        logger.info(
            f"🧠 Memory Retrieval: query='{query[:50]}...', type={memory_type}, limit={limit}"
        )

        # Lazy load memory manager
        if not self._memory_manager:
            from src.core.memory_manager import MemoryManager

            self._memory_manager = MemoryManager()

        # Retrieve patterns
        try:
            if memory_type == "procedural":
                patterns = await self._retrieve_procedural(query, limit, min_success_score)
            elif memory_type == "working":
                patterns = await self._retrieve_working(query, limit)
            else:
                # All: combine both
                procedural = await self._retrieve_procedural(
                    query, limit // 2 + 1, min_success_score
                )
                working = await self._retrieve_working(query, limit // 2 + 1)
                patterns = procedural + working

            logger.info(f"✅ Retrieved {len(patterns)} memory patterns")

        except Exception as e:
            logger.exception(f"❌ Memory retrieval failed: {e}")
            return ToolResult(success=False, documents=[], error=str(e))

        # Convert patterns to documents
        documents = self._patterns_to_documents(patterns)

        return ToolResult(
            success=len(documents) > 0,
            documents=documents,
            metadata={
                "memory_type": memory_type,
                "patterns_found": len(patterns),
                "min_success_score": min_success_score,
            },
        )

    async def _retrieve_procedural(
        self, query: str, limit: int, min_success_score: float
    ) -> list[dict[str, Any]]:
        """
        Retrieve from procedural memory.

        Returns successful patterns similar to query.
        """

        try:
            return await self._memory_manager.procedural_memory.find_similar_patterns(
                query=query, limit=limit, min_success_score=min_success_score
            )

        except Exception as e:
            logger.warning(f"⚠️ Procedural memory unavailable: {e}")
            return []

    async def _retrieve_working(self, query: str, limit: int) -> list[dict[str, Any]]:
        """
        Retrieve from working memory.

        Returns recent reasoning steps (usually not needed for new requests).
        """

        # Working memory is session-specific, usually empty for new requests
        # This is more for debugging/analysis

        logger.debug("Working memory retrieval not implemented yet")
        return []

    def _patterns_to_documents(self, patterns: list[dict[str, Any]]) -> list[Document]:
        """Convert memory patterns to documents."""

        documents = []

        for pattern in patterns:
            # Format pattern as readable text
            content = self._format_pattern(pattern)

            doc = Document(
                page_content=content,
                metadata={
                    "pattern_id": pattern.get("pattern_id", ""),
                    "category": pattern.get("topic_category", ""),
                    "success_score": pattern.get("success_score", 0.0),
                    "usage_count": pattern.get("usage_count", 0),
                    "source": "procedural_memory",
                },
                source="memory",
            )

            documents.append(doc)

        return documents

    def _format_pattern(self, pattern: dict[str, Any]) -> str:
        """Format pattern as readable text."""

        return f"""# Успешная стратегия: {pattern.get("topic_category", "Unknown")}

**Уровень**: {pattern.get("user_level", "intermediate")}
**Успешность**: {pattern.get("success_score", 0.0):.2f}
**Использований**: {pattern.get("usage_count", 0)}
**Среднее количество итераций**: {pattern.get("avg_iterations", 0):.1f}

## Последовательность инструментов
{" → ".join(pattern.get("tools_sequence", []))}

## Паттерн рассуждений
{pattern.get("reasoning_pattern", "Нет описания")}

---
*Этот паттерн был успешен в прошлых генерациях и может быть полезен для текущего запроса.*
"""
