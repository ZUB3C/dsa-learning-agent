import time

from ...models.orchestrator_schemas import ResolveRequest, ResolveResponse
from .aggregator import Aggregator, aggregate
from .classifier import RequestClassifier
from .executor import Executor


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
