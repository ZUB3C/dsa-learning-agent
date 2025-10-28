import datetime

from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["System"])


@router.get("/")
def health_check():
    return {"status": "ok", "time": datetime.datetime.utcnow().isoformat()}  # noqa: DTZ003
