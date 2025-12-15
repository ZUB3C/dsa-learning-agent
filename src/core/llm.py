"""
LLM provider management with model routing.
Code from Section 2.2 of architecture.
"""

import logging
from enum import StrEnum

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.config import get_settings
from src.exceptions import LLMUnavailableError

logger = logging.getLogger(__name__)


class TaskType(StrEnum):
    """Types of tasks for LLM routing."""

    # High-cost tasks (GigaChat-2-Max)
    THOUGHT_GENERATION = "thought_generation"
    FINAL_GENERATION = "final_generation"

    # Low-cost tasks (GigaChat3)
    INPUT_VALIDATION = "input_validation"
    PROMISE_EVALUATION = "promise_evaluation"
    RELEVANCE_SCORING = "relevance_scoring"
    CONTENT_GUARD = "content_guard"
    CONCEPT_EXTRACTION = "concept_extraction"
    COMPLETENESS_CHECK = "completeness_check"
    POLICY_CHECK = "policy_check"
    TOXICITY_CHECK = "toxicity_check"


class LLMRouter:
    """
    Роутер для выбора модели на основе типа задачи.
    Code from Section 2.2 of architecture.
    """

    # High-cost tasks that require GigaChat-2-Max
    HIGH_COST_TASKS = {TaskType.THOUGHT_GENERATION, TaskType.FINAL_GENERATION}

    def __init__(self) -> None:
        """Initialize LLM router."""
        self.settings = get_settings()
        self._gigachat_max_instance = None
        self._gigachat3_instance = None

    def get_model_for_task(self, task_type: TaskType) -> BaseChatModel:
        """
        Выбирает модель на основе типа задачи.

        High-cost tasks (GigaChat-2-Max):
        - thought_generation: Генерация мыслей в ToT
        - final_generation: Финальная генерация материала

        Low-cost tasks (GigaChat3):
        - input_validation: Проверка ввода
        - promise_evaluation: Оценка перспективности
        - relevance_scoring: Оценка релевантности
        - content_guard: Проверка контента
        - concept_extraction: Извлечение концепций
        - completeness_check: Оценка полноты

        Args:
            task_type: Type of task

        Returns:
            BaseChatModel instance

        Raises:
            LLMUnavailableError: If model initialization fails
        """

        if task_type in self.HIGH_COST_TASKS:
            logger.debug(f"Using GigaChat-2-Max for {task_type.value}")
            return self.gigachat_max
        logger.debug(f"Using GigaChat3 for {task_type.value}")
        return self.gigachat3

    @property
    def gigachat_max(self) -> BaseChatModel:
        """
        Get GigaChat-2-Max instance (cached).

        Returns:
            GigaChat-2-Max model instance
        """
        if self._gigachat_max_instance is None:
            try:
                self._gigachat_max_instance = ChatOpenAI(
                    api_key=self.settings.llm.gigachat_api_key,  # pyright: ignore[reportArgumentType]
                    base_url=self.settings.llm.gigachat_base_url,
                    model=self.settings.llm.gigachat_model,
                    temperature=self.settings.llm.gigachat_temperature,
                    # max_tokens=self.settings.llm.gigachat_max_tokens,
                    timeout=self.settings.llm.gigachat_timeout_s,
                    max_retries=2,
                )
                logger.info("✅ GigaChat-2-Max initialized")
            except Exception as e:
                logger.exception(f"❌ Failed to initialize GigaChat-2-Max: {e}")
                raise LLMUnavailableError(model="GigaChat-2-Max", message=str(e))

        return self._gigachat_max_instance

    @property
    def gigachat3(self) -> BaseChatModel:
        """
        Get GigaChat3 instance (cached).

        Returns:
            GigaChat3 model instance
        """
        if self._gigachat3_instance is None:
            try:
                self._gigachat3_instance = ChatOpenAI(
                    api_key=self.settings.llm.gigachat_api_key,  # pyright: ignore[reportArgumentType]
                    base_url=self.settings.llm.gigachat_base_url,
                    model=self.settings.llm.gigachat3_model,
                    temperature=self.settings.llm.gigachat3_temperature,
                    # max_tokens=self.settings.llm.gigachat3_max_tokens,
                    timeout=self.settings.llm.gigachat3_timeout_s,
                    max_retries=2,
                )
                logger.info("✅ GigaChat3 initialized")
            except Exception as e:
                logger.exception(f"❌ Failed to initialize GigaChat3: {e}")
                raise LLMUnavailableError(model="GigaChat3", message=str(e))

        return self._gigachat3_instance

    def _get_gigachat_max(self) -> BaseChatModel:
        """Get GigaChat-2-Max instance (legacy method)."""
        return self.gigachat_max

    def _get_gigachat3(self) -> BaseChatModel:
        """Get GigaChat3 instance (legacy method)."""
        return self.gigachat3


# Global router instance
_router = None


def get_llm_router() -> LLMRouter:
    """Get global LLM router instance."""
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router


def get_llm(use_gigachat3: bool = False) -> BaseChatModel:
    """
    Legacy function for backward compatibility.

    Args:
        use_gigachat3: If True, return GigaChat3, else GigaChat-2-Max

    Returns:
        BaseChatModel instance
    """
    router = get_llm_router()

    if use_gigachat3:
        return router.gigachat3
    return router.gigachat_max
