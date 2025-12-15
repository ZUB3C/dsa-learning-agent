"""
Database fallback strategies.
"""

import json
import logging
from pathlib import Path
from typing import Any

from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class DatabaseFallbackHandler:
    """
    Handle database failures with fallback to JSON files.
    """

    def __init__(self, fallback_dir: str = "./data/fallback") -> None:
        """
        Initialize database fallback handler.

        Args:
            fallback_dir: Directory for fallback JSON files
        """
        self.fallback_dir = Path(fallback_dir)
        self.fallback_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Database fallback directory: {self.fallback_dir}")

    async def save_with_fallback(self, db: AsyncSession, record: Any, fallback_name: str) -> bool:
        """
        Save record with fallback to JSON.

        Args:
            db: Database session
            record: SQLAlchemy model instance
            fallback_name: Name for fallback file

        Returns:
            True if saved successfully (to DB or fallback)
        """

        try:
            # Try database first
            db.add(record)
            await db.commit()
            logger.debug(f"✅ Saved to database: {fallback_name}")
            return True

        except (OperationalError, IntegrityError) as e:
            logger.warning(f"⚠️ Database save failed: {e}, using fallback")

            # Rollback
            await db.rollback()

            # Fallback to JSON
            return self._save_to_json(record, fallback_name)

    def _save_to_json(self, record: Any, fallback_name: str) -> bool:
        """Save record to JSON file."""

        try:
            # Convert SQLAlchemy model to dict
            record_dict = {}
            for column in record.__table__.columns:
                value = getattr(record, column.name)
                # Handle special types
                if hasattr(value, "isoformat"):
                    value = value.isoformat()
                record_dict[column.name] = value

            # Save to JSON
            fallback_file = self.fallback_dir / f"{fallback_name}.json"

            # Append to existing file if it exists
            existing_data = []
            if fallback_file.exists():
                with Path(fallback_file).open(encoding="utf-8") as f:
                    existing_data = json.load(f)

            existing_data.append(record_dict)

            with Path(fallback_file).open("w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=2, default=str)

            logger.info(f"✅ Saved to fallback JSON: {fallback_file}")
            return True

        except Exception as e:
            logger.exception(f"❌ Fallback JSON save failed: {e}")
            return False

    async def load_from_fallback(self, fallback_name: str) -> list[dict[str, Any]]:
        """
        Load records from fallback JSON.

        Args:
            fallback_name: Name of fallback file

        Returns:
            List of record dicts
        """

        fallback_file = self.fallback_dir / f"{fallback_name}.json"

        if not fallback_file.exists():
            logger.debug(f"📂 No fallback file: {fallback_file}")
            return []

        try:
            with Path(fallback_file).open(encoding="utf-8") as f:
                data = json.load(f)

            logger.info(f"✅ Loaded {len(data)} records from fallback: {fallback_file}")
            return data

        except Exception as e:
            logger.exception(f"❌ Failed to load fallback: {e}")
            return []

    async def migrate_fallback_to_db(
        self, db: AsyncSession, fallback_name: str, model_class: type
    ) -> int:
        """
        Migrate fallback JSON records to database.

        Args:
            db: Database session
            fallback_name: Name of fallback file
            model_class: SQLAlchemy model class

        Returns:
            Number of migrated records
        """

        records = await self.load_from_fallback(fallback_name)

        if not records:
            return 0

        migrated = 0

        for record_dict in records:
            try:
                # Create model instance
                record = model_class(**record_dict)
                db.add(record)
                migrated += 1

            except Exception as e:
                logger.warning(f"⚠️ Failed to migrate record: {e}")
                continue

        try:
            await db.commit()
            logger.info(f"✅ Migrated {migrated}/{len(records)} records")

            # Delete fallback file after successful migration
            fallback_file = self.fallback_dir / f"{fallback_name}.json"
            fallback_file.unlink()
            logger.info(f"🗑️ Deleted fallback file: {fallback_file}")

        except Exception as e:
            logger.exception(f"❌ Migration commit failed: {e}")
            await db.rollback()

        return migrated
