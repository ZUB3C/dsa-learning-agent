import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from .core.database import init_database
from .models.schemas import RootResponse
from .routers import assessment, health, llm_router, materials, support, tests, verification

app = FastAPI(
    title="АиСД Learning Platform API",
    description="API для платформы изучения алгоритмов и структур данных с агентными системами",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def latency_header(request: Request, call_next: RequestResponseEndpoint) -> Response:
    start = time.time()
    resp = await call_next(request)
    resp.headers["X-Process-Time"] = f"{time.time() - start:.3f}"
    return resp


@app.on_event("startup")
async def startup_event() -> None:  # noqa: RUF029
    """Инициализация при запуске."""
    init_database()


# Подключение роутеров
app.include_router(health.router)
app.include_router(verification.router)
app.include_router(assessment.router)
app.include_router(materials.router)
app.include_router(tests.router)
app.include_router(llm_router.router)
app.include_router(support.router)


@app.get("/")
def root() -> RootResponse:
    """Root endpoint."""
    return RootResponse(message="DSA Learning Platform API", version="1.0.0", docs="/docs")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
