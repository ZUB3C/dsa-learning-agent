"""
Memory Manager for Working and Procedural memory.
Code from Section 8.2 of architecture.
"""

import logging
import uuid

from src.config import get_settings
from src.core.memory.memory_schemas import MemoryContext, ProceduralPattern
from src.core.memory.procedural_memory import ProceduralMemoryStore
from src.core.memory.working_memory import WorkingMemoryStore
from src.models.react_schemas import ToTResult

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Unified manager for Working and Procedural memory.
    Code from Section 8.2 of architecture.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.working_memory = WorkingMemoryStore()
        self.procedural_memory = ProceduralMemoryStore()

    async def load_context(self, user_id: str, query: str) -> MemoryContext:
        """
        Load memory context for new request.

        Args:
            user_id: User ID
            query: User query

        Returns:
            MemoryContext with session_id and procedural hints
        """

        # Create new session
        session_id = f"sess_{uuid.uuid4().hex[:12]}"

        logger.info(f"🧠 Loading memory context for session {session_id}")

        # Load procedural patterns (if available)
        procedural_hints = "No prior patterns available"
        patterns = []

        if self.settings.features.feature_procedural_memory_enabled:
            try:
                self._detect_category(query)
                patterns = await self.procedural_memory.find_similar_patterns(query=query, limit=3)

                if patterns:
                    procedural_hints = self._format_hints(patterns)
                    logger.info(f"📚 Loaded {len(patterns)} procedural patterns")
                else:
                    logger.info("📚 No procedural patterns found")

            except Exception as e:
                logger.warning(f"⚠️ Procedural memory unavailable: {e}")

        return MemoryContext(
            session_id=session_id,
            user_id=user_id,
            procedural_hints=procedural_hints,
            patterns=patterns,
        )

    async def save_successful_generation(
        self, session_id: str, tot_result: ToTResult, query: str, user_level: str
    ) -> None:
        """
        Save successful generation to procedural memory.

        Args:
            session_id: Session ID
            tot_result: ToT search result
            query: User query
            user_level: User level
        """

        if not self.settings.features.feature_procedural_memory_enabled:
            return

        # Check if generation was successful enough
        if (
            tot_result.final_completeness
            < self.settings.memory.memory_procedural_min_success_score
        ):
            logger.info(
                f"⚠️ Completeness {tot_result.final_completeness:.2f} < {self.settings.memory.memory_procedural_min_success_score}, not saving"
            )
            return

        # Extract tool sequence from best path
        tools_sequence = [
            node.planned_action.get("tool_name")
            for node in tot_result.best_path
            if node.planned_action and node.planned_action.get("tool_name")
        ]

        # Detect category
        category = self._detect_category(query)

        # Create pattern
        pattern = ProceduralPattern(
            pattern_id=f"pat_{uuid.uuid4().hex[:12]}",
            topic_category=category,
            user_level=user_level,
            tools_sequence=tools_sequence,
            avg_iterations=len(tot_result.best_path),
            success_score=tot_result.final_completeness,
            reasoning_pattern=self._extract_reasoning(tot_result.best_path),
        )

        try:
            await self.procedural_memory.save_pattern(pattern)
            logger.info(f"✅ Saved procedural pattern: {category} / {user_level}")
        except Exception as e:
            logger.exception(f"❌ Failed to save procedural pattern: {e}")

    def _detect_category(self, query: str) -> str:
        """
        Detect topic category from query.

        Args:
            query: User query

        Returns:
            Category string
        """

        categories = {
            "sorting": ["сортировка", "quicksort", "mergesort", "heapsort", "bubble sort"],
            "graphs": ["граф", "дейкстра", "bfs", "dfs", "кратчайший путь", "поиск в ширину"],
            "dynamic_programming": ["динамическое программирование", "мемоизация", "рюкзак"],
            "data_structures": ["структура данных", "дерево", "хеш", "стек", "очередь"],
            "complexity": ["сложность", "big O", "время выполнения", "асимптотика"],
            "recursion": ["рекурсия", "рекурсивный"],
            "greedy": ["жадный алгоритм", "greedy"],
        }

        query_lower = query.lower()

        for category, keywords in categories.items():
            if any(kw in query_lower for kw in keywords):
                return category

        return "general"

    def _format_hints(self, patterns: list) -> str:
        """
        Format patterns as hints for orchestrator.

        Args:
            patterns: List of pattern dicts

        Returns:
            Formatted hints string
        """

        if not patterns:
            return "No prior patterns available"

        hints = "## Успешные стратегии из памяти:\n\n"

        for i, pattern in enumerate(patterns, 1):
            hints += f"{i}. **{pattern.get('topic_category', 'Unknown')}** (успех: {pattern.get('success_score', 0):.2f})\n"
            hints += f"   - Инструменты: {' → '.join(pattern.get('tools_sequence', []))}\n"
            hints += f"   - Итераций: ~{pattern.get('avg_iterations', 0):.0f}\n"
            hints += f"   - Стратегия: {pattern.get('reasoning_pattern', '')[:100]}...\n\n"

        return hints

    def _extract_reasoning(self, path: list) -> str:
        """
        Extract reasoning pattern from best path.

        Args:
            path: List of TreeNode objects

        Returns:
            Reasoning description
        """

        if not path:
            return "No reasoning available"

        reasoning_steps = [node.thought[:100] for node in path if node.thought]

        return " → ".join(reasoning_steps)
