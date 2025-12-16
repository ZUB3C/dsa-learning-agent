"""Модуль для работы с базой данных через SQLAlchemy."""

from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    """Базовый класс для всех моделей."""


class User(Base):
    """Таблица пользователей."""

    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class AssessmentSession(Base):
    """Таблица сессий оценки."""

    __tablename__ = "assessment_sessions"

    session_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    questions: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class Assessment(Base):
    """Таблица результатов оценки."""

    __tablename__ = "assessments"

    assessment_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    session_id: Mapped[str] = mapped_column(String, nullable=False)
    level: Mapped[str] = mapped_column(String, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    knowledge_areas: Mapped[str] = mapped_column(Text, nullable=False)
    recommendations: Mapped[str] = mapped_column(Text, nullable=False)
    completed_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class Test(Base):
    """Таблица тестов."""

    __tablename__ = "tests"

    test_id: Mapped[str] = mapped_column(String, primary_key=True)
    topic: Mapped[str] = mapped_column(String, nullable=False)
    difficulty: Mapped[str] = mapped_column(String, nullable=False)
    questions: Mapped[str] = mapped_column(Text, nullable=False)
    expected_duration: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class TestResult(Base):
    """Таблица результатов тестов."""

    __tablename__ = "test_results"

    result_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    test_id: Mapped[str] = mapped_column(String, nullable=False)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    answers: Mapped[str] = mapped_column(Text, nullable=False)
    submitted_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class Verification(Base):
    """Таблица верификаций без баллов"""

    __tablename__ = "verifications"

    verification_id: Mapped[str] = mapped_column(String, primary_key=True)
    test_id: Mapped[str] = mapped_column(String, nullable=False)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    user_answer: Mapped[str] = mapped_column(Text, nullable=False)

    # Только булевы значения, без баллов
    is_correct: Mapped[bool] = mapped_column(Boolean, nullable=False)

    feedback: Mapped[str] = mapped_column(Text, nullable=False)

    # Вторичная проверка
    secondary_is_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    agree_with_primary: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    verification_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class CustomTopic(Base):
    """Таблица пользовательских тем."""

    __tablename__ = "custom_topics"

    topic_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    topic_name: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class SupportSession(Base):
    """Таблица сессий поддержки."""

    __tablename__ = "support_sessions"

    session_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    user_message: Mapped[str] = mapped_column(Text, nullable=False)
    emotional_state: Mapped[str] = mapped_column(String, nullable=False)
    response_content: Mapped[str] = mapped_column(Text, nullable=False)
    recommendations: Mapped[str] = mapped_column(Text, nullable=False)
    helpful: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    comments: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


"""
Database models and initialization.
Updated with Section 10.1 tables.
"""

import logging
from datetime import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base

from src.config import get_settings

logger = logging.getLogger(__name__)

Base = declarative_base()


# ════════════════════════════════════════════════════════════════
# TABLE: material_generations
# ════════════════════════════════════════════════════════════════


class MaterialGeneration(Base):
    """Material generation log."""

    __tablename__ = "material_generations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    generation_id = Column(String(50), unique=True, nullable=False, index=True)
    user_id = Column(String(100), nullable=False, index=True)

    # Request data
    topic = Column(Text, nullable=False, index=True)
    user_level = Column(String(20), nullable=False)

    # ToT metrics
    tot_iterations = Column(Integer, nullable=False)
    tot_explored_nodes = Column(Integer, nullable=False)
    tot_dead_end_nodes = Column(Integer, nullable=False)
    tot_best_path_depth = Column(Integer, nullable=False)

    # Tool usage
    tools_used = Column(JSON, nullable=False)  # ["adaptive_rag", "web_search"]
    tool_call_counts = Column(JSON, nullable=False)  # {"adaptive_rag": 1, ...}

    # LLM usage
    gigachat2_max_calls = Column(Integer, nullable=False)
    gigachat3_calls = Column(Integer, nullable=False)
    estimated_cost_usd = Column(Float, nullable=False)

    # Results
    success = Column(Boolean, nullable=False, index=True)
    final_completeness_score = Column(Float, nullable=False)
    documents_collected = Column(Integer, nullable=False)
    material_length = Column(Integer, nullable=False)
    material_content = Column(Text, nullable=True)

    # Performance
    generation_time_seconds = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.now, index=True)

    # Metadata
    memory_hints_used = Column(Boolean, default=False)
    content_guard_filtered = Column(Integer, default=0)
    fallbacks_triggered = Column(JSON, nullable=True)


# ════════════════════════════════════════════════════════════════
# TABLE: tot_node_logs
# ════════════════════════════════════════════════════════════════


class ToTNodeLog(Base):
    """Detailed ToT node logs."""

    __tablename__ = "tot_node_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    generation_id = Column(String(50), nullable=False, index=True)

    # Node identification
    node_id = Column(String(50), nullable=False)
    parent_node_id = Column(String(50), nullable=True)
    depth = Column(Integer, nullable=False)

    # Node data
    thought = Column(Text, nullable=False)
    reasoning = Column(Text, nullable=False)
    planned_action = Column(JSON, nullable=False)

    # Scores
    promise_score = Column(Float, nullable=False)
    completeness_score = Column(Float, nullable=False)
    relevance_score = Column(Float, nullable=False)
    quality_score = Column(Float, nullable=False)

    # Status
    status = Column(String(20), nullable=False, index=True)

    # Performance
    execution_time_ms = Column(Float, nullable=False)
    llm_calls = Column(JSON, nullable=False)

    created_at = Column(DateTime, default=datetime.now)


# ════════════════════════════════════════════════════════════════
# TABLE: tool_usage_stats
# ════════════════════════════════════════════════════════════════


class ToolUsageStats(Base):
    """Aggregated tool usage statistics."""

    __tablename__ = "tool_usage_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)

    tool_name = Column(String(50), nullable=False)
    date = Column(String(10), nullable=False)  # YYYY-MM-DD

    # Counts
    total_calls = Column(Integer, default=0)
    successful_calls = Column(Integer, default=0)
    failed_calls = Column(Integer, default=0)

    # Performance
    avg_execution_time_ms = Column(Float, default=0.0)
    min_execution_time_ms = Column(Float, default=0.0)
    max_execution_time_ms = Column(Float, default=0.0)

    # Strategy distribution (for Adaptive RAG)
    strategy_tfidf_count = Column(Integer, default=0)
    strategy_semantic_count = Column(Integer, default=0)
    strategy_hybrid_count = Column(Integer, default=0)

    # Fallbacks
    fallback_triggered_count = Column(Integer, default=0)

    updated_at = Column(DateTime, default=datetime.now)

    __table_args__ = (Index("idx_tool_date", "tool_name", "date", unique=True),)


# ════════════════════════════════════════════════════════════════
# TABLE: procedural_patterns (SQLite backup)
# ════════════════════════════════════════════════════════════════


class ProceduralPatternDB(Base):
    """Procedural patterns backup (for ChromaDB)."""

    __tablename__ = "procedural_patterns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pattern_id = Column(String(50), unique=True, nullable=False)

    topic_category = Column(String(50), nullable=False, index=True)
    user_level = Column(String(20), nullable=False, index=True)

    tools_sequence = Column(JSON, nullable=False)
    avg_iterations = Column(Float, nullable=False)
    success_score = Column(Float, nullable=False, index=True)
    usage_count = Column(Integer, default=1)

    reasoning_pattern = Column(Text, nullable=False)

    created_at = Column(DateTime, default=datetime.now)
    last_used = Column(DateTime, default=datetime.now)


# ════════════════════════════════════════════════════════════════
# TABLE: content_guard_logs
# ════════════════════════════════════════════════════════════════


class ContentGuardLog(Base):
    """Content Guard filtering logs."""

    __tablename__ = "content_guard_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    generation_id = Column(String(50), nullable=False, index=True)

    # Document info
    document_source = Column(String(200), nullable=False)
    document_snippet = Column(Text, nullable=False)

    # Checks
    toxicity_score = Column(Float, nullable=False)
    policy_compliant = Column(Boolean, nullable=False)
    quality_passed = Column(Boolean, nullable=False)

    # Result
    filtered = Column(Boolean, nullable=False, index=True)
    filter_reason = Column(String(100), nullable=True)

    created_at = Column(DateTime, default=datetime.now)


# ════════════════════════════════════════════════════════════════
# TABLE: system_health_metrics
# ════════════════════════════════════════════════════════════════


class SystemHealthMetric(Base):
    """System health metrics."""

    __tablename__ = "system_health_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.now, index=True)

    # Component availability (0-1)
    gigachat2_max_available = Column(Float, default=1.0)
    gigachat3_available = Column(Float, default=1.0)
    chromadb_available = Column(Float, default=1.0)
    fourget_available = Column(Float, default=1.0)
    redis_available = Column(Float, default=1.0)

    # Latencies (ms)
    gigachat2_max_avg_latency = Column(Float, default=0.0)
    gigachat3_avg_latency = Column(Float, default=0.0)
    chromadb_avg_latency = Column(Float, default=0.0)

    # Rates (per minute)
    request_rate = Column(Float, default=0.0)
    success_rate = Column(Float, default=0.0)
    error_rate = Column(Float, default=0.0)

    # Costs (last hour)
    estimated_cost_last_hour = Column(Float, default=0.0)


# ════════════════════════════════════════════════════════════════
# Database setup
# ════════════════════════════════════════════════════════════════

settings = get_settings()

# Create async engine (rename to async_engine)
async_engine = create_async_engine(
    settings.database.database_url, echo=settings.database.database_echo, future=True
)

# Create async session factory
AsyncSessionLocal = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

# Create sync engine (rename to sync_engine)
sync_engine = create_engine(settings.database.database_url, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Контекстный менеджер для работы с БД через SQLAlchemy."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_database() -> None:
    """Инициализация базы данных - создание всех таблиц."""
    Base.metadata.create_all(bind=sync_engine)


def get_or_create_user(user_id: str) -> User:
    """Получить или создать пользователя."""
    with get_db_session() as session:
        user = session.query(User).filter(User.user_id == user_id).first()
        if not user:
            user = User(user_id=user_id)
            session.add(user)
            session.commit()
        return user


async def init_db() -> None:
    """Initialize database (create tables)."""
    settings = get_settings()

    # Create engine in function scope
    engine = create_async_engine(
        settings.database.database_url, echo=settings.database.database_echo, future=True
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Dispose before function returns
    await engine.dispose()

    logger.info("✅ Database initialized")


async def get_db() -> AsyncSession:
    """Get database session (dependency)."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
