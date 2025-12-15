"""
Procedural Memory Store.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any

from src.config import get_settings
from src.core.memory.memory_schemas import ProceduralPattern
from src.exceptions import MemoryError

logger = logging.getLogger(__name__)


class ProceduralMemoryStore:
    """
    Procedural memory for storing successful patterns.

    Storage: ChromaDB (with SQLite backup)
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.chromadb_available = True

        # Try to initialize ChromaDB
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self.client = chromadb.Client(
                ChromaSettings(
                    persist_directory=self.settings.memory.chroma_persist_directory,
                    anonymized_telemetry=False,
                )
            )

            # Get or create collection with embeddings
            self.collection = self.client.get_or_create_collection(
                name=self.settings.memory.chroma_procedural_memory_collection,
                metadata={"description": "Procedural memory for successful patterns"},
            )

            logger.info("✅ Procedural Memory (ChromaDB) initialized")

        except Exception as e:
            logger.exception(f"❌ ChromaDB unavailable for procedural memory: {e}")
            self.chromadb_available = False

    async def save_pattern(self, pattern: ProceduralPattern) -> None:
        """
        Save successful pattern to procedural memory.

        Args:
            pattern: ProceduralPattern to save
        """

        if not self.chromadb_available:
            logger.warning("⚠️ ChromaDB unavailable, pattern not saved")
            # TODO: Implement SQLite backup
            return

        try:
            # Generate ID if not provided
            if not pattern.pattern_id:
                pattern.pattern_id = f"pat_{uuid.uuid4().hex[:12]}"

            # Serialize pattern
            pattern_json = pattern.model_dump_json()

            # Store in ChromaDB (with embedding on reasoning_pattern)
            self.collection.upsert(
                documents=[pattern.reasoning_pattern],  # This will be embedded
                ids=[pattern.pattern_id],
                metadatas=[
                    {
                        "pattern_json": pattern_json,
                        "category": pattern.topic_category,
                        "level": pattern.user_level,
                        "success_score": pattern.success_score,
                        "created_at": pattern.created_at.isoformat(),
                    }
                ],
            )

            logger.info(
                f"✅ Saved pattern {pattern.pattern_id}: {pattern.topic_category}/{pattern.user_level}"
            )

        except Exception as e:
            logger.exception(f"❌ Failed to save pattern: {e}")
            msg = "save_pattern"
            raise MemoryError(msg, str(e))

    async def find_similar_patterns(
        self, query: str, limit: int = 3, min_success_score: float = 0.8
    ) -> list[dict[str, Any]]:
        """
        Find similar patterns based on query.

        Args:
            query: Query string (will be embedded)
            limit: Number of patterns to return
            min_success_score: Minimum success score threshold

        Returns:
            List of pattern dicts
        """

        if not self.chromadb_available:
            logger.warning("⚠️ ChromaDB unavailable, returning empty patterns")
            return []

        try:
            # Search by semantic similarity
            results = self.collection.query(
                query_texts=[query],
                n_results=limit * 2,  # Get more for filtering
                where={"success_score": {"$gte": min_success_score}},
            )

            if not results or not results["metadatas"]:
                logger.info(f"No patterns found for query: {query[:50]}...")
                return []

            # Parse patterns
            patterns = []
            for metadata in results["metadatas"][0]:  # First query results
                pattern_json = metadata.get("pattern_json")
                if pattern_json:
                    pattern_dict = json.loads(pattern_json)
                    patterns.append(pattern_dict)

            logger.info(f"📚 Found {len(patterns)} similar patterns")

            return patterns[:limit]

        except Exception as e:
            logger.exception(f"❌ Failed to find patterns: {e}")
            return []

    async def increment_usage(self, pattern_id: str) -> None:
        """
        Increment usage count for a pattern.

        Args:
            pattern_id: Pattern ID
        """

        if not self.chromadb_available:
            return

        try:
            # Get current pattern
            result = self.collection.get(ids=[pattern_id])

            if not result or not result["metadatas"]:
                logger.warning(f"⚠️ Pattern {pattern_id} not found")
                return

            # Parse pattern
            metadata = result["metadatas"][0]
            pattern_json = metadata.get("pattern_json")

            if not pattern_json:
                return

            pattern_dict = json.loads(pattern_json)

            # Increment usage
            pattern_dict["usage_count"] = pattern_dict.get("usage_count", 0) + 1
            pattern_dict["last_used"] = datetime.now().isoformat()

            # Update
            pattern = ProceduralPattern(**pattern_dict)
            await self.save_pattern(pattern)

            logger.debug(f"📈 Incremented usage for {pattern_id}")

        except Exception as e:
            logger.exception(f"❌ Failed to increment usage: {e}")
