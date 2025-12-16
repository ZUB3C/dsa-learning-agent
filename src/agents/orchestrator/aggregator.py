from __future__ import annotations
from ...models.orchestrator_schemas import (
    ClassificationResult,
    ResolveResponse,
    SupportBlock,
)
from .executor import ExecutionResult


class Aggregator:
    """Агрегирует результаты воркеров в финальный ответ оркестратора."""

    def _build_support_block(self, support_raw: dict | None, fallback_text: str) -> SupportBlock:
        """Сформировать SupportBlock из ответа support-сервиса.

        Если support-эндпоинт отдаёт структурированный JSON с полями
        вроде support_message/recommendations, используем их.
        Иначе — берём текст fallback и добавляем дефолтные рекомендации.
        """
        if support_raw and isinstance(support_raw, dict):
            msg = (
                support_raw.get("support_message")
                or support_raw.get("message")
                or support_raw.get("content")
                or fallback_text
            )
            rec_list = support_raw.get("recommendations") or []
            if isinstance(rec_list, list):
                recommendations = [str(r) for r in rec_list]
            else:
                recommendations = [str(rec_list)]
        else:
            msg = fallback_text
            recommendations = [
                "Делайте короткие перерывы каждые 25–30 минут.",
                "Разбивайте сложные темы на небольшие подзадачи.",
            ]

        return SupportBlock(message=str(msg), recommendations=recommendations)

    def _choose_main_content(self, main_raw: dict | None, fallback_text: str) -> str:
        """Выбрать основной контент из ответа доменного сервиса.

        Если доменный сервис уже вернул content/material/answer в JSON,
        используем это; иначе — текст, который заполнил BaseWorker.
        """
        if main_raw and isinstance(main_raw, dict):
            text = (
                main_raw.get("content")
                or main_raw.get("material")
                or main_raw.get("answer")
                or main_raw.get("message")
            )
            if text:
                return str(text)
        return fallback_text

    def aggregate(
        self,
        exec_result: ExecutionResult,
        cls: ClassificationResult,
        execution_time_ms: int,
    ) -> ResolveResponse:
        """Построить финальный ResolveResponse из результатов воркеров."""

        main_res = exec_result.main_result
        supp_res = exec_result.support_result

        # 1. Основной контент
        main_content = self._choose_main_content(main_res.raw, main_res.content)

        # 2. Статус
        if main_res.ok:
            # если основной воркер ок, но support (если был) упал — считаем partial
            if supp_res is None or supp_res.ok:
                status = "success"
            else:
                status = "partial"
        else:
            status = "error"

        # 3. Блок поддержки
        support_block: SupportBlock | None = None
        if supp_res is not None and supp_res.ok:
            support_block = self._build_support_block(supp_res.raw, supp_res.content)

        # 4. Финальный ответ оркестратора
        return ResolveResponse(
            status=status,
            main_content=main_content,
            task_type=cls.task_type,
            support=support_block,
            agents_used=exec_result.agents_used,
            execution_time_ms=execution_time_ms,
        )
def aggregate(
    exec_result: ExecutionResult,
    cls: ClassificationResult,
    execution_time_ms: int,
) -> ResolveResponse:
    """Функция-обёртка вокруг Aggregator для удобного импорта."""
    return Aggregator().aggregate(exec_result, cls, execution_time_ms)
