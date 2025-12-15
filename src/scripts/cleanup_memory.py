"""
Cleanup script for working memory (TTL expired sessions).
"""

import asyncio
import logging

from src.config import get_settings
from src.core.memory.working_memory import WorkingMemoryStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def cleanup_expired_sessions() -> None:
    """
    Cleanup expired working memory sessions.

    Should be run as a cron job (e.g., every hour).
    """

    settings = get_settings()
    working_memory = WorkingMemoryStore()

    logger.info("🧹 Starting working memory cleanup...")
    logger.info(f"   TTL: {settings.memory.memory_working_ttl_hours} hours")

    try:
        await working_memory.cleanup_old_sessions()
        logger.info("✅ Cleanup complete")

    except Exception as e:
        logger.exception(f"❌ Cleanup failed: {e}")


if __name__ == "__main__":
    asyncio.run(cleanup_expired_sessions())
