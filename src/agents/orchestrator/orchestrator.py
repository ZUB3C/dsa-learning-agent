from __future__ import annotations

import time
from typing import TYPE_CHECKING

from .aggregator import Aggregator
from .classifier import RequestClassifier
from .executor import Executor

if TYPE_CHECKING:
    from ...models.orchestrator_schemas import ResolveRequest, ResolveResponse


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
        return aggregate(exec_result, classification, elapsed_ms)


orchestrator = Orchestrator()
