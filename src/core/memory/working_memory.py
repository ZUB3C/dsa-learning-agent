"""
Working Memory Store.
Code from Section 8.3 of architecture.
"""

import json
import logging

from src.config import get_settings

logger = logging.getLogger(__name__)


class WorkingMemoryStore:
    """
    Working memory with ChromaDB → in-memory fallback.
    Code from Section 8.3 of architecture.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.chromadb_available = True
        self.in_memory_fallback: dict[str, list[dict]] = {}  # {session_id: list[steps]}

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

            self.collection = self.client.get_or_create_collection(
                name=self.settings.memory.chroma_working_memory_collection,
                metadata={"description": "Working memory for ToT reasoning"},
            )

            logger.info("✅ Working Memory (ChromaDB) initialized")

        except Exception as e:
            logger.exception(f"❌ ChromaDB unavailable: {e}")
            self.chromadb_available = False

    async def append_step(self, session_id: str, step_data: dict) -> None:
        """
        Append reasoning step to working memory.

        Args:
            session_id: Session ID
            step_data: Step data (dict)
        """

        if self.chromadb_available:
            try:
                # Store in ChromaDB
                doc_id = f"{session_id}_{step_data['iteration']}"

                self.collection.add(
                    documents=[json.dumps(step_data)],
                    ids=[doc_id],
                    metadatas=[{"session_id": session_id, "iteration": step_data["iteration"]}],
                )

                logger.debug(f"📝 Stored step {step_data['iteration']} for {session_id}")
                return

            except Exception as e:
                logger.warning(f"⚠️ ChromaDB write failed: {e}, using in-memory")
                self.chromadb_available = False

        # FALLBACK: In-memory storage
        if session_id not in self.in_memory_fallback:
            self.in_memory_fallback[session_id] = []

        self.in_memory_fallback[session_id].append(step_data)
        logger.debug(f"📝 In-memory: stored step for {session_id}")

    async def get_session_context(self, session_id: str) -> list[dict]:
        """
        Retrieve all steps for session.

        Args:
            session_id: Session ID

        Returns:
            List of step data
        """

        if self.chromadb_available:
            try:
                results = self.collection.get(where={"session_id": session_id})

                if results and results["documents"]:
                    steps = [json.loads(doc) for doc in results["documents"]]
                    # Sort by iteration
                    steps.sort(key=lambda x: x.get("iteration", 0))
                    return steps

            except Exception as e:
                logger.warning(f"⚠️ ChromaDB read failed: {e}, using in-memory")

        # FALLBACK: In-memory
        return self.in_memory_fallback.get(session_id, [])

    async def clear_session(self, session_id: str) -> None:
        """
        Clear session after completion.

        Args:
            session_id: Session ID
        """

        if self.chromadb_available:
            try:
                self.collection.delete(where={"session_id": session_id})
                logger.info(f"🗑️ Cleared session {session_id} from ChromaDB")
            except Exception as e:
                logger.warning(f"⚠️ ChromaDB delete failed: {e}")

        # FALLBACK: In-memory
        if session_id in self.in_memory_fallback:
            del self.in_memory_fallback[session_id]
            logger.debug(f"🗑️ Cleared session {session_id} from in-memory")

    async def cleanup_old_sessions(self) -> None:
        """
        Cleanup old sessions (TTL expired).

        Should be called periodically (cron job).
        """

        from datetime import datetime, timedelta

        ttl_hours = self.settings.memory.memory_working_ttl_hours
        cutoff_time = datetime.now() - timedelta(hours=ttl_hours)

        logger.info(f"🧹 Cleaning up sessions older than {ttl_hours} hours")

        if self.chromadb_available:
            try:
                # ChromaDB doesn't support time-based queries easily
                # For now, skip automatic cleanup (manual via admin)
                logger.info("⚠️ Automatic cleanup not implemented for ChromaDB")

            except Exception as e:
                logger.exception(f"❌ Cleanup failed: {e}")

        # In-memory cleanup
        sessions_to_delete = []
        for session_id, steps in self.in_memory_fallback.items():
            if not steps:
                sessions_to_delete.append(session_id)
                continue

            # Check last step timestamp
            last_step = steps[-1]
            timestamp_str = last_step.get("timestamp")
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if timestamp < cutoff_time:
                        sessions_to_delete.append(session_id)
                except Exception:
                    pass

        for session_id in sessions_to_delete:
            del self.in_memory_fallback[session_id]

        logger.info(f"🧹 Cleaned up {len(sessions_to_delete)} in-memory sessions")
