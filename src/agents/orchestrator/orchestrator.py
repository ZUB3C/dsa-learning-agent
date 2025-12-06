# src/agents/orchestrator/orchestrator.py

from __future__ import annotations

import time

from ...models.orchestrator_schemas import ResolveRequest, ResolveResponse
from .classifier import RequestClassifier
from .executor import Executor
from .aggregator import Aggregator


class Orchestrator:
    """Высокоуровневый оркестратор комплексных запросов.

    Задачи:
    - классифицировать запрос с помощью LLM (task_type, include_support, topic, question, user_answer)
    - выбрать и запустить нужных воркеров (основной + при необходимости support)
    - собрать единый ответ для API.
    """

    def __init__(self) -> None:
        self._classifier = RequestClassifier()
        self._executor = Executor()
        self._aggregator = Aggregator()

    async def resolve(self, request: ResolveRequest) -> ResolveResponse:
        """Обработать комплексный запрос пользователя."""
        start = time.time()

        # 1. LLM-классификация текста (intent + slots)
        classification = await self._classifier.classify(request.message)

        # 2. Запуск соответствующих воркеров (HTTP вызовы к существующим эндпоинтам)
        exec_result = await self._executor.execute(request, classification)

        # 3. Агрегация результатов в финальный ответ
        elapsed_ms = int((time.time() - start) * 1000)
        response = self._aggregator.aggregate(exec_result, classification, elapsed_ms)

        return response


# Готовый экземпляр для использования в роутере
orchestrator = Orchestrator()
