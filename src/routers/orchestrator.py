from fastapi import APIRouter, HTTPException

from ..agents.orchestrator.orchestrator import orchestrator
from ..models.orchestrator_schemas import ResolveRequest, ResolveResponse

router = APIRouter(prefix="/api/v1/orchestrator", tags=["Orchestrator"])


@router.post("/resolve")
async def resolve_complex_task(request: ResolveRequest) -> ResolveResponse:
    """
    Обрабатывает комплексный запрос студента.

    Пример использования:
    - "Объясни быструю сортировку"
    - "Не понимаю рекурсию, голова кругом"
    - "Сгенерируй тест по графам"
    - "Хочу бросить программирование"
    """
    try:
        return await orchestrator.resolve(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
