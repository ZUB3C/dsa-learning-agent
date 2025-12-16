"""
FastAPI application main entry point.
Updated to include v2 routes.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from src.config import get_settings
from src.core.database import init_db
from src.core.logging_handler import get_db_handler, setup_database_logging
from src.routers import (
    health,
    materials,  # v1 (legacy)
    materials_v2,  # v2 (new)
)

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


# Middleware для установки контекста логирования
@app.middleware("http")
async def logging_context_middleware(request: Request, call_next):
    """Set logging context for each request."""
    import uuid

    request_id = str(uuid.uuid4())
    user_id = request.headers.get("X-User-ID")
    session_id = request.headers.get("X-Session-ID")

    # Set context in database handler
    db_handler = get_db_handler()
    if db_handler:
        db_handler.set_context(
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
        )

    try:
        return await call_next(request)
    finally:
        # Clear context after request
        if db_handler:
            db_handler.clear_context()


@app.on_event("startup")
async def startup_event() -> None:
    """Инициализация при запуске."""
    await init_db()
    # Setup database logging
    setup_database_logging(level=logging.INFO)
    logger.info("Database logging initialized")


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events."""
    # Startup
    logger.info("🚀 Starting Materials Agent v2...")
    logger.info(f"Environment: {settings.project.environment}")
    logger.info(f"Version: {settings.project.version}")

    # Initialize database
    await init_db()

    yield

    # Shutdown
    logger.info("🛑 Shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.project.project_name,
    version=settings.project.version,
    description="Materials Agent with Tree-of-Thoughts",
    lifespan=lifespan,
)

# CORS
if settings.api.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=settings.api.cors_methods,
        allow_headers=[settings.api.cors_headers],
    )

# Register routers
app.include_router(health.router, prefix="/api/v2", tags=["health"])
app.include_router(materials.router, prefix="/api/v1/materials", tags=["materials-v1"])
app.include_router(materials_v2.router, prefix="/api/v2/materials", tags=["materials-v2"])
app.include_router(health.router)
app.include_router(verification.router)
app.include_router(assessment.router)
app.include_router(materials.router)
app.include_router(tests.router)
app.include_router(llm_router.router)
app.include_router(support.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.project.project_name,
        "version": settings.project.version,
        "status": "running",
        "features": {
            "tot_enabled": settings.features.feature_tot_enabled,
            "content_guard_enabled": settings.features.feature_content_guard_enabled,
            "adaptive_rag_enabled": settings.features.feature_adaptive_rag_enabled,
            "web_search_enabled": settings.features.feature_web_search_enabled,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.api.api_host,
        port=settings.api.api_port,
        workers=settings.api.api_workers,
        reload=settings.project.debug,
    )
