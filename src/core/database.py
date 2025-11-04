"""Модуль для работы с базой данных через SQLAlchemy."""

from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import Boolean, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from ..config import settings


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
    response: Mapped[str] = mapped_column(Text, nullable=False)
    recommendations: Mapped[str] = mapped_column(Text, nullable=False)
    helpful: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    comments: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


# Создание движка и сессии
engine = create_engine(settings.database_url, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db_session() -> Generator[Session]:
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
    Base.metadata.create_all(bind=engine)


def get_or_create_user(user_id: str) -> User:
    """Получить или создать пользователя."""
    with get_db_session() as session:
        user = session.query(User).filter(User.user_id == user_id).first()
        if not user:
            user = User(user_id=user_id)
            session.add(user)
            session.commit()
        return user
