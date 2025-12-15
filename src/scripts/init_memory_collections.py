"""
Initialize ChromaDB collections for memory.
Code from Section 10.2 of architecture.
"""

import logging

import chromadb
from chromadb.config import Settings

from src.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_memory_collections() -> None:
    """
    Initialize ChromaDB collections for memory system.
    """

    settings = get_settings()

    logger.info("🔧 Initializing ChromaDB collections...")

    # Create client
    client = chromadb.Client(
        Settings(
            persist_directory=settings.memory.chroma_persist_directory, anonymized_telemetry=False
        )
    )

    # ═══════════════════════════════════════════════════════════════
    # COLLECTION: agent_working_memory
    # ═══════════════════════════════════════════════════════════════

    try:
        client.get_or_create_collection(
            name=settings.memory.chroma_working_memory_collection,
            metadata={
                "description": "Short-term memory for ToT reasoning steps",
                "ttl_hours": settings.memory.memory_working_ttl_hours,
            },
        )
        logger.info(f"✅ Created collection: {settings.memory.chroma_working_memory_collection}")
    except Exception as e:
        logger.exception(f"❌ Failed to create working memory collection: {e}")

    # ═══════════════════════════════════════════════════════════════
    # COLLECTION: agent_procedural_memory
    # ═══════════════════════════════════════════════════════════════

    try:
        client.get_or_create_collection(
            name=settings.memory.chroma_procedural_memory_collection,
            metadata={
                "description": "Long-term memory of successful strategies",
                "max_patterns": settings.memory.memory_procedural_max_patterns,
            },
        )
        logger.info(
            f"✅ Created collection: {settings.memory.chroma_procedural_memory_collection}"
        )
    except Exception as e:
        logger.exception(f"❌ Failed to create procedural memory collection: {e}")

    # ═══════════════════════════════════════════════════════════════
    # COLLECTION: aisd_materials (existing RAG collection)
    # ═══════════════════════════════════════════════════════════════

    try:
        client.get_or_create_collection(
            name=settings.memory.chroma_rag_collection,
            metadata={
                "description": "AISD knowledge base (PDFs, web content)",
                "chunk_size": 1000,
                "chunk_overlap": 200,
            },
        )
        logger.info(f"✅ Created/verified collection: {settings.memory.chroma_rag_collection}")
    except Exception as e:
        logger.exception(f"❌ Failed to create RAG collection: {e}")

    logger.info("🎉 Memory collections initialization complete!")


if __name__ == "__main__":
    init_memory_collections()
