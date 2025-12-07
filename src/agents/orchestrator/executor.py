from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ...models.orchestrator_schemas import ClassificationResult, ResolveRequest, TaskType
from .workers.base_worker import WorkerResult
from .workers.materials_worker import MaterialsWorker
from .workers.support_worker import SupportWorker
from .workers.test_worker import TestWorker
from .workers.verification_worker import VerificationWorker


@dataclass
class ExecutionResult:
    main_result: WorkerResult
    support_result: WorkerResult | None
    agents_used: list[str]


class Executor:
    def _get_main_worker(self, task_type: TaskType):
        if task_type == TaskType.MATERIALS:
            return "materials", MaterialsWorker()
        if task_type == TaskType.TEST:
            return "test", TestWorker()
        if task_type == TaskType.VERIFICATION:
            return "verification", VerificationWorker()
        if task_type == TaskType.SUPPORT:
            return "support", SupportWorker()
        return "materials", MaterialsWorker()

    async def execute(
        self,
        request: ResolveRequest,
        cls: ClassificationResult,
    ) -> ExecutionResult:
        main_name, main_worker = self._get_main_worker(cls.task_type)
        agents_used = [main_name]

        if cls.task_type == TaskType.SUPPORT:
            main_res = await main_worker.run(
                user_id=request.user_id,
                message=request.message,
            )
            return ExecutionResult(
                main_result=main_res, support_result=None, agents_used=agents_used
            )

        async def run_main() -> WorkerResult:
            if cls.task_type == TaskType.MATERIALS:
                return await main_worker.run(
                    user_id=request.user_id,
                    topic=cls.topic,
                    user_level=request.user_level,
                )
            if cls.task_type == TaskType.TEST:
                return await main_worker.run(
                    user_id=request.user_id,
                    topic=cls.topic,
                    user_level=request.user_level,
                )
            if cls.task_type == TaskType.VERIFICATION:
                return await main_worker.run(
                    user_id=request.user_id,
                    question=cls.question,
                    user_answer=cls.user_answer,
                )
            return await main_worker.run(
                user_id=request.user_id,
                topic=cls.topic,
                user_level=request.user_level,
            )

        if not cls.include_support:
            main_res = await run_main()
            return ExecutionResult(
                main_result=main_res, support_result=None, agents_used=agents_used
            )

        support_worker = SupportWorker()
        agents_used.append("support")

        async def run_support() -> WorkerResult:
            return await support_worker.run(
                user_id=request.user_id,
                message=request.message,
                emotional_state="stressed",
            )

        main_res, supp_res = await asyncio.gather(run_main(), run_support())
        return ExecutionResult(
            main_result=main_res, support_result=supp_res, agents_used=agents_used
        )
