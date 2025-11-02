import datetime

from fastapi import APIRouter

from ..models.schemas import HealthCheckResponse

router = APIRouter(prefix="/health", tags=["System"])


@router.get("/")
def health_check() -> HealthCheckResponse:
    """Health check endpoint."""
    return HealthCheckResponse(
        status="ok",
        time=datetime.datetime.utcnow().isoformat(),  # noqa: DTZ003
    )
