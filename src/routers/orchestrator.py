from fastapi import APIRouter

from ..agents.orchestrator.orchestrator import orchestrator
from ..models.orchestrator_schemas import ResolveRequest, ResolveResponse


router = APIRouter(prefix="/api/v1/orchestrator", tags=["Orchestrator"])


@router.post("/resolve", response_model=ResolveResponse)
async def resolve_complex_task(request: ResolveRequest) -> ResolveResponse:
    return await orchestrator.resolve(request)
