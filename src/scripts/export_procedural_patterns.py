"""
Export procedural patterns to JSON for backup.
"""

import asyncio
import json
import logging
import pathlib
from datetime import datetime

from src.core.memory.procedural_memory import ProceduralMemoryStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def export_patterns(output_file: str = "procedural_patterns_backup.json") -> None:
    """
    Export procedural patterns to JSON.

    Args:
        output_file: Output JSON file path
    """

    logger.info(f"📤 Exporting procedural patterns to {output_file}...")

    procedural_memory = ProceduralMemoryStore()

    if not procedural_memory.chromadb_available:
        logger.error("❌ ChromaDB not available")
        return

    try:
        # Get all patterns
        # Note: This is a simplified version - in production you'd paginate
        all_results = procedural_memory.collection.get()

        if not all_results or not all_results["metadatas"]:
            logger.warning("⚠️ No patterns found")
            return

        patterns = []

        for metadata in all_results["metadatas"]:
            pattern_json = metadata.get("pattern_json")
            if pattern_json:
                pattern = json.loads(pattern_json)
                patterns.append(pattern)

        # Export
        backup = {
            "exported_at": datetime.now().isoformat(),
            "total_patterns": len(patterns),
            "patterns": patterns,
        }

        with pathlib.Path(output_file).open("w", encoding="utf-8") as f:
            json.dump(backup, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Exported {len(patterns)} patterns")

    except Exception as e:
        logger.exception(f"❌ Export failed: {e}")


if __name__ == "__main__":
    import sys

    output = sys.argv[1] if len(sys.argv) > 1 else "procedural_patterns_backup.json"

    asyncio.run(export_patterns(output))
