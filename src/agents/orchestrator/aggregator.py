from __future__ import annotations

from ...models.orchestrator_schemas import ClassificationResult, ResolveResponse, SupportBlock
from .executor import ExecutionResult


class Aggregator:
    def aggregate(
        self,
        exec_result: ExecutionResult,
        cls: ClassificationResult,
        execution_time_ms: int,
    ) -> ResolveResponse:
        support_block: SupportBlock | None = None
        if exec_result.support_result and exec_result.support_result.ok:
            support_block = SupportBlock(
                message=exec_result.support_result.content,
                recommendations=[
                    "Делайте перерывы каждые 25–30 минут.",
                    "Разбивайте сложную тему на небольшие части.",
                ],
            )

        main_content = exec_result.main_result.content

        return ResolveResponse(
            status="success" if exec_result.main_result.ok else "partial",
            main_content=main_content,
            task_type=cls.task_type,
            support=support_block,
            agents_used=exec_result.agents_used,
            execution_time_ms=execution_time_ms,
        )
