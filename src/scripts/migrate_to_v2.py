"""
Migration script from v1 to v2 database schema.
"""

import asyncio
import logging

from sqlalchemy import text

from src.core.database import AsyncSessionLocal, init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def check_v1_tables_exist(db: "AsyncSession") -> bool:  # noqa: F821
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


async def backup_v1_tables(db: "AsyncSession") -> None:  # noqa: F821
    """
    Backup v1 tables to v1_backup_ prefix.

    Args:
        db: Database session
    """

    logger.info("💾 Backing up v1 tables...")

    try:
        # Get list of tables to backup
        result = await db.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        tables = [row[0] for row in result.fetchall()]

        v1_tables = [t for t in tables if not t.startswith("v1_backup_")]

        for table in v1_tables:
            backup_name = f"v1_backup_{table}"
            logger.info(f"   Backing up {table} -> {backup_name}")

            await db.execute(text(f"DROP TABLE IF EXISTS {backup_name}"))
            await db.execute(text(f"CREATE TABLE {backup_name} AS SELECT * FROM {table}"))

        await db.commit()
        logger.info("✅ Backup complete")

    except Exception as e:
        logger.exception(f"❌ Backup failed: {e}")
        await db.rollback()
        raise


async def create_v2_tables() -> None:
    """Create v2 tables."""

    logger.info("🏗️  Creating v2 tables...")

    try:
        # Create new v2 tables
        from sqlalchemy.ext.asyncio import create_async_engine

        from src.config import get_settings
        from src.core.database import Base

        settings = get_settings()
        engine = create_async_engine(settings.database.database_url, echo=False)

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("✅ V2 tables created")

    except Exception as e:
        logger.exception(f"❌ Failed to create v2 tables: {e}")
        raise


async def migrate_data() -> None:
    """Migrate data from v1 to v2 schema."""

    logger.info("🔄 Migrating data from v1 to v2...")

    try:
        async with AsyncSessionLocal() as db:
            # Check if v1 tables exist
            v1_exists = await check_v1_tables_exist(db)

            if not v1_exists:
                logger.info("ℹ️  No v1 tables found, skipping migration")
                return

            # Backup v1 tables first
            await backup_v1_tables(db)

            # Migrate data
            logger.info("🔄 Migrating old records to v2 schema...")

            migrated_count = 0

            # Here you would write the actual migration logic
            # Example:
            # result = await db.execute(text("SELECT * FROM v1_backup_generations"))
            # for row in result:
            #     new_record = MaterialGeneration(...)
            #     db.add(new_record)
            #     migrated_count += 1

            await db.commit()

            logger.info(f"✅ Migrated {migrated_count} records")

    except Exception as e:
        logger.exception(f"❌ Migration failed: {e}")
        raise


async def verify_migration() -> None:
    """Verify that migration was successful."""

    logger.info("🔍 Verifying migration...")

    try:
        async with AsyncSessionLocal() as db:
            # Check v2 tables exist and have data
            result = await db.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result.fetchall()]

            logger.info(f"📋 Tables after migration: {tables}")

            v2_tables = ["material_generations", "tot_node_logs", "tool_usage_stats"]

            for table in v2_tables:
                if table in tables:
                    count_result = await db.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = count_result.scalar()
                    logger.info(f"   {table}: {count} records")
                else:
                    logger.warning(f"   ⚠️ {table} not found")

        logger.info("✅ Verification complete")

    except Exception as e:
        logger.exception(f"❌ Verification failed: {e}")


async def cleanup_backup_tables() -> None:
    """
    Cleanup v1 backup tables.

    WARNING: This will permanently delete v1 backup data!
    """

    logger.warning("⚠️ CLEANUP: This will DELETE v1 backup tables!")

    response = input("Are you sure? Type 'yes' to confirm: ")

    if response.lower() != "yes":
        logger.info("Cleanup cancelled")
        return

    logger.info("🧹 Cleaning up backup tables...")

    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result.fetchall()]

            backup_tables = [t for t in tables if t.startswith("v1_backup_")]

            for table in backup_tables:
                logger.info(f"   Dropping {table}")
                await db.execute(text(f"DROP TABLE {table}"))

            await db.commit()

        logger.info("✅ Cleanup complete")

    except Exception as e:
        logger.exception(f"❌ Cleanup failed: {e}")


async def main() -> None:
    """Main entry point."""

    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--cleanup":
        await cleanup_backup_tables()
        return

    logger.info("=" * 80)
    logger.info("DATABASE MIGRATION: V1 -> V2")
    logger.info("=" * 80)
    logger.info("")

    # Initialize database
    await init_db()

    # Create v2 tables
    await create_v2_tables()

    # Migrate data
    await migrate_data()

    # Verify
    await verify_migration()

    logger.info("")
    logger.info("=" * 80)
    logger.info("MIGRATION COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Verify data integrity")
    logger.info("2. Test application")
    logger.info("3. Run cleanup: python -m src.scripts.migrate_to_v2 --cleanup")


if __name__ == "__main__":
    asyncio.run(main())
