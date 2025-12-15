"""
Health check service with detailed component status.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class HealthService:
    """
    Service for checking system health.
    """

    @staticmethod
    async def check_gigachat2_max() -> dict[str, Any]:
        """Check GigaChat-2-Max availability."""

        try:
            from src.core.llm import get_llm_router

            llm_router = get_llm_router()
            gigachat2 = llm_router.gigachat_max

            # Simple ping test
            start = asyncio.get_event_loop().time()
            await gigachat2.ainvoke("test", config={"timeout": 3.0})
            latency = (asyncio.get_event_loop().time() - start) * 1000

            return {
                "status": "healthy",
                "available": True,
                "latency_ms": round(latency, 2),
                "last_check": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.exception(f"GigaChat-2-Max health check failed: {e}")
            return {
                "status": "unhealthy",
                "available": False,
                "error": str(e),
                "last_check": datetime.now().isoformat(),
            }

    @staticmethod
    async def check_gigachat3() -> dict[str, Any]:
        """Check GigaChat3 availability."""

        try:
            from src.core.llm import get_llm_router

            llm_router = get_llm_router()
            gigachat3 = llm_router.gigachat3

            start = asyncio.get_event_loop().time()
            await gigachat3.ainvoke("test", config={"timeout": 3.0})
            latency = (asyncio.get_event_loop().time() - start) * 1000

            return {
                "status": "healthy",
                "available": True,
                "latency_ms": round(latency, 2),
                "last_check": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.exception(f"GigaChat3 health check failed: {e}")
            return {
                "status": "unhealthy",
                "available": False,
                "error": str(e),
                "last_check": datetime.now().isoformat(),
            }

    @staticmethod
    async def check_chromadb() -> dict[str, Any]:
        """Check ChromaDB availability."""

        try:
            from src.core.vector_store import vector_store_manager

            # List collections
            collections = vector_store_manager.client.list_collections()

            return {
                "status": "healthy",
                "available": True,
                "collections": len(collections),
                "last_check": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.exception(f"ChromaDB health check failed: {e}")
            return {
                "status": "unhealthy",
                "available": False,
                "error": str(e),
                "last_check": datetime.now().isoformat(),
            }

    @staticmethod
    async def check_database() -> dict[str, Any]:
        """Check database availability."""

        try:
            from sqlalchemy import text

            from src.core.database import AsyncSessionLocal

            async with AsyncSessionLocal() as session:
                result = await session.execute(text("SELECT 1"))
                result.scalar()

            return {
                "status": "healthy",
                "available": True,
                "last_check": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.exception(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "available": False,
                "error": str(e),
                "last_check": datetime.now().isoformat(),
            }

    @staticmethod
    async def check_redis() -> dict[str, Any]:
        """Check Redis availability."""

        try:
            from src.core.cache.redis_cache import RedisCache

            cache = RedisCache()

            if not cache.enabled:
                return {
                    "status": "disabled",
                    "available": False,
                    "last_check": datetime.now().isoformat(),
                }

            # Ping Redis
            await cache.redis.ping()

            return {
                "status": "healthy",
                "available": True,
                "last_check": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.exception(f"Redis health check failed: {e}")
            return {
                "status": "unhealthy",
                "available": False,
                "error": str(e),
                "last_check": datetime.now().isoformat(),
            }

    @staticmethod
    async def check_all() -> dict[str, Any]:
        """Check all components."""

        results = await asyncio.gather(
            HealthService.check_gigachat2_max(),
            HealthService.check_gigachat3(),
            HealthService.check_chromadb(),
            HealthService.check_database(),
            HealthService.check_redis(),
            return_exceptions=True,
        )

        components = {
            "gigachat2_max": results[0]
            if not isinstance(results[0], Exception)
            else {"status": "error", "error": str(results[0])},
            "gigachat3": results[1]
            if not isinstance(results[1], Exception)
            else {"status": "error", "error": str(results[1])},
            "chromadb": results[2]
            if not isinstance(results[2], Exception)
            else {"status": "error", "error": str(results[2])},
            "database": results[3]
            if not isinstance(results[3], Exception)
            else {"status": "error", "error": str(results[3])},
            "redis": results[4]
            if not isinstance(results[4], Exception)
            else {"status": "error", "error": str(results[4])},
        }

        # Determine overall status
        unhealthy_count = sum(
            1 for comp in components.values() if comp.get("status") in {"unhealthy", "error"}
        )

        if unhealthy_count == 0:
            overall_status = "healthy"
        elif unhealthy_count <= 1:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"

        return {
            "overall_status": overall_status,
            "components": components,
            "timestamp": datetime.now().isoformat(),
        }
