"""
Migration script from v1 to v2 database schema.
"""

import asyncio
import logging
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import AsyncSessionLocal, init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def check_v1_tables_exist(db: AsyncSession) -> bool:
    """
    Check if v1 tables exist.

    Args:
        db: Database session

    Returns:
        True if v1 tables exist
    """

    try:
        # Try to query v1 table (assuming it was named 'generations' or similar)
        result = await db.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        tables = [row[0] for row in result.fetchall()]

        logger.info(f"📋 Found tables: {tables}")

        # Check for common v1 table names
        v1_tables = ["generations", "materials", "users"]
        return any(table in tables for table in v1_tables)

    except Exception as e:
        logger.exception(f"❌ Failed to check v1 tables: {e}")
        return False


async def migrate_generations_table(db: AsyncSession) -> None:
    """
    Migrate v1 generations table to v2 schema.

    Args:
        db: Database session
    """

    logger.info("🔄 Migrating generations table...")

    try:
        # Check if old table exists
        result = await db.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='generations'")
        )

        if not result.fetchone():
            logger.info("⚠️ No v1 'generations' table found, skipping")
            return

        # Get columns from old table
        result = await db.execute(text("PRAGMA table_info(generations)"))
        old_columns = {row[1] for row in result.fetchall()}

        logger.info(f"📋 Old columns: {old_columns}")

        # Get v1 data
        result = await db.execute(text("SELECT * FROM generations"))
        old_records = result.fetchall()

        logger.info(f"📊 Found {len(old_records)} old records")

        # Rename old table
        await db.execute(text("ALTER TABLE generations RENAME TO generations_v1_backup"))
        await db.commit()

        logger.info("✅ Old table backed up as 'generations_v1_backup'")

        # Create new v2 tables
        from src.core.database import Base, engine

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("✅ New v2 tables created")

        # Migrate data (if old records exist)
        if old_records:
            logger.info("🔄 Migrating old records to v2 schema...")

            from src.core.database import MaterialGeneration

            migrated_count = 0

            for record in old_records:
                try:
                    # Map old columns to new schema
                    # This is a template - adjust based on your v1 schema

                    new_record = MaterialGeneration(
                        generation_id=f"migrated_{record[0]}",  # Adjust indices
                        user_id="migrated_user",
                        topic=record[1] if len(record) > 1 else "Unknown",
                        user_level="intermediate",
                        tot_iterations=0,
                        tot_explored_nodes=0,
                        tot_dead_end_nodes=0,
                        tot_best_path_depth=0,
                        tools_used=[],
                        tool_call_counts={},
                        gigachat2_max_calls=0,
                        gigachat3_calls=0,
                        estimated_cost_usd=0.0,
                        success=True,
                        final_completeness_score=0.0,
                        documents_collected=0,
                        material_length=0,
                        generation_time_seconds=0.0,
                        created_at=datetime.now(),
                    )

                    db.add(new_record)
                    migrated_count += 1

                except Exception as e:
                    logger.warning(f"⚠️ Failed to migrate record: {e}")
                    continue

            await db.commit()

            logger.info(f"✅ Migrated {migrated_count}/{len(old_records)} records")

        logger.info("✅ Migration complete!")

    except Exception as e:
        logger.exception(f"❌ Migration failed: {e}")
        await db.rollback()
        raise


async def cleanup_v1_backup(db: AsyncSession) -> None:
    """
    Clean up v1 backup tables.

    WARNING: This will permanently delete v1 data!

    Args:
        db: Database session
    """

    logger.warning("⚠️ CLEANUP: This will DELETE v1 backup tables!")

    response = input("Are you sure? Type 'yes' to confirm: ")

    if response.lower() != "yes":
        logger.info("❌ Cleanup cancelled")
        return

    try:
        await db.execute(text("DROP TABLE IF EXISTS generations_v1_backup"))
        await db.commit()

        logger.info("✅ Cleanup complete")

    except Exception as e:
        logger.exception(f"❌ Cleanup failed: {e}")


async def run_migration() -> None:
    """Run full migration."""

    logger.info("=" * 80)
    logger.info("MIGRATION TO V2")
    logger.info("=" * 80)
    logger.info("")

    # Initialize database (creates v2 tables if they don't exist)
    await init_db()

    async with AsyncSessionLocal() as db:
        # Check v1 tables
        has_v1 = await check_v1_tables_exist(db)

        if not has_v1:
            logger.info("✅ No v1 tables found, v2 schema is ready")
            return

        # Migrate
        await migrate_generations_table(db)

    logger.info("")
    logger.info("=" * 80)
    logger.info("MIGRATION COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Verify migrated data")
    logger.info("2. Test v2 API endpoints")
    logger.info("3. (Optional) Run cleanup: python src/scripts/migrate_to_v2.py --cleanup")


async def main() -> None:
    """Main entry point."""

    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--cleanup":
        # Cleanup mode
        async with AsyncSessionLocal() as db:
            await cleanup_v1_backup(db)
    else:
        # Migration mode
        await run_migration()


if __name__ == "__main__":
    asyncio.run(main())
