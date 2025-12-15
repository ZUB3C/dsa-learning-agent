"""
Health check router (updated for v2).
"""

import logging
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.core.database import MaterialGeneration, get_db
from src.core.llm import get_llm_router
from src.core.vector_store import vector_store_manager
from src.models.schemas import HealthCheckResponse, HealthCheckV2Response, SystemMetricsResponse

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


# ════════════════════════════════════════════════════════════════
# V1 HEALTH CHECK (backwards compatible)
# ════════════════════════════════════════════════════════════════


@router.get("/health", response_model=HealthCheckResponse)
async def health_check_v1():
    """V1 health check (legacy)."""
    return HealthCheckResponse(status="ok", time=datetime.now().isoformat())


# ════════════════════════════════════════════════════════════════
# V2 ENHANCED HEALTH CHECK
# ════════════════════════════════════════════════════════════════


@router.get("/health/detailed", response_model=HealthCheckV2Response)
async def health_check_v2():
    """
    Enhanced health check with component status.

    Checks:
    - LLM availability (GigaChat-2-Max, GigaChat3)
    - ChromaDB availability
    - Database availability
    - Feature flags
    """

    components = {}
    metrics = {}
    overall_status = "healthy"

    # ───────────────────────────────────────────────────────────
    # CHECK: GigaChat-2-Max
    # ───────────────────────────────────────────────────────────

    try:
        llm_router = get_llm_router()
        gigachat2 = llm_router.gigachat_max

        # Simple ping test
        await gigachat2.ainvoke("ping", config={"timeout": 3.0})

        components["gigachat2_max"] = {
            "status": "healthy",
            "available": True,
            "latency_ms": 0,  # Could measure actual latency
        }
    except Exception as e:
        logger.warning(f"GigaChat-2-Max unhealthy: {e}")
        components["gigachat2_max"] = {"status": "unhealthy", "available": False, "error": str(e)}
        overall_status = "degraded"

    # ───────────────────────────────────────────────────────────
    # CHECK: GigaChat3
    # ───────────────────────────────────────────────────────────

    try:
        llm_router = get_llm_router()
        gigachat3 = llm_router.gigachat3

        await gigachat3.ainvoke("ping", config={"timeout": 3.0})

        components["gigachat3"] = {"status": "healthy", "available": True, "latency_ms": 0}
    except Exception as e:
        logger.warning(f"GigaChat3 unhealthy: {e}")
        components["gigachat3"] = {"status": "unhealthy", "available": False, "error": str(e)}
        overall_status = "degraded"

    # ───────────────────────────────────────────────────────────
    # CHECK: ChromaDB
    # ───────────────────────────────────────────────────────────

    try:
        # Simple collection check
        collections = vector_store_manager.client.list_collections()

        components["chromadb"] = {
            "status": "healthy",
            "available": True,
            "collections": len(collections),
        }
    except Exception as e:
        logger.warning(f"ChromaDB unhealthy: {e}")
        components["chromadb"] = {"status": "unhealthy", "available": False, "error": str(e)}
        overall_status = "degraded"

    # ───────────────────────────────────────────────────────────
    # METRICS
    # ───────────────────────────────────────────────────────────

    metrics = {
        "uptime_seconds": 0.0,  # TODO: Track actual uptime
        "memory_usage_mb": 0.0,  # TODO: Track memory
        "active_sessions": 0,  # TODO: Track sessions
    }

    # ───────────────────────────────────────────────────────────
    # FEATURES
    # ───────────────────────────────────────────────────────────

    features = {
        "tot_enabled": settings.features.feature_tot_enabled,
        "content_guard_enabled": settings.features.feature_content_guard_enabled,
        "adaptive_rag_enabled": settings.features.feature_adaptive_rag_enabled,
        "web_search_enabled": settings.features.feature_web_search_enabled,
        "procedural_memory_enabled": settings.features.feature_procedural_memory_enabled,
    }

    return HealthCheckV2Response(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        version=settings.project.version,
        components=components,
        metrics=metrics,
        features=features,
    )


# ════════════════════════════════════════════════════════════════
# SYSTEM METRICS
# ════════════════════════════════════════════════════════════════


@router.get("/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics(db: Annotated[AsyncSession, Depends(get_db)]):
    """
    Get system metrics (last hour).
    """

    from datetime import timedelta

    # Time window
    one_hour_ago = datetime.now() - timedelta(hours=1)

    # Query generations (last hour)
    result = await db.execute(
        select(MaterialGeneration).where(MaterialGeneration.created_at >= one_hour_ago)
    )
    generations = result.scalars().all()

    # Calculate metrics
    total_requests = len(generations)
    successful_requests = sum(1 for g in generations if g.success)
    failed_requests = total_requests - successful_requests

    avg_response_time = (
        sum(g.generation_time_seconds for g in generations) / total_requests
        if total_requests > 0
        else 0.0
    )

    # LLM usage
    gigachat2_calls = sum(g.gigachat2_max_calls for g in generations)
    gigachat3_calls = sum(g.gigachat3_calls for g in generations)
    total_cost = sum(g.estimated_cost_usd for g in generations)

    # Tool usage
    tool_usage = {}
    for g in generations:
        for tool, count in g.tool_call_counts.items():
            tool_usage[tool] = tool_usage.get(tool, 0) + count

    # Content Guard
    documents_filtered = sum(g.content_guard_filtered for g in generations)
    filter_rate = (
        documents_filtered / sum(g.documents_collected for g in generations)
        if sum(g.documents_collected for g in generations) > 0
        else 0.0
    )

    return SystemMetricsResponse(
        total_requests=total_requests,
        successful_requests=successful_requests,
        failed_requests=failed_requests,
        avg_response_time=avg_response_time,
        gigachat2_max_calls=gigachat2_calls,
        gigachat3_calls=gigachat3_calls,
        total_cost_usd=total_cost,
        tool_usage=tool_usage,
        documents_filtered=documents_filtered,
        filter_rate=filter_rate,
        procedural_patterns_stored=0,  # TODO: Query from procedural memory
        working_memory_sessions=0,  # TODO: Query from working memory
    )
