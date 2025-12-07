from __future__ import annotations

import time

from ...models.orchestrator_schemas import ResolveRequest, ResolveResponse
from .classifier import RequestClassifier
from .executor import Executor
from .aggregator import Aggregator


class Orchestrator:
    def __init__(self) -> None:
        self._classifier = RequestClassifier()
        self._executor = Executor()
        self._aggregator = Aggregator()

    async def resolve(self, request: ResolveRequest) -> ResolveResponse:
        start = time.time()

        classification = await self._classifier.classify(request.message)
        exec_result = await self._executor.execute(request, classification)

        elapsed_ms = int((time.time() - start) * 1000)
        response = self._aggregator.aggregate(exec_result, classification, elapsed_ms)

        return response


orchestrator = Orchestrator()
