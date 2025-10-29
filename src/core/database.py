"""
Модуль для работы с SQLite базой данных
"""

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

DATABASE_PATH = Path(__file__).parent.parent.parent / "app_data.db"


@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection]:
    """Контекстный менеджер для работы с БД"""
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database() -> None:
    """Инициализация базы данных"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Таблица пользователей
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Таблица сессий оценки
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assessment_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                questions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)

        # Таблица результатов оценки
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assessments (
                assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                session_id TEXT,
                level TEXT,
                score REAL,
                knowledge_areas TEXT,
                recommendations TEXT,
                completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)

        # Таблица тестов
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tests (
                test_id TEXT PRIMARY KEY,
                topic TEXT,
                difficulty TEXT,
                questions TEXT,
                expected_duration INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Таблица результатов тестов
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT,
                user_id TEXT,
                answers TEXT,
                submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (test_id) REFERENCES tests(test_id),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)

        # Таблица проверок
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS verifications (
                verification_id TEXT PRIMARY KEY,
                test_id TEXT,
                user_id TEXT,
                question TEXT,
                user_answer TEXT,
                is_correct BOOLEAN,
                score REAL,
                feedback TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (test_id) REFERENCES tests(test_id),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)

        # Таблица пользовательских тем
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS custom_topics (
                topic_id TEXT PRIMARY KEY,
                user_id TEXT,
                topic_name TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)

        # Таблица сессий поддержки
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS support_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                user_message TEXT,
                emotional_state TEXT,
                response TEXT,
                recommendations TEXT,
                helpful BOOLEAN,
                comments TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)


def get_or_create_user(user_id: str) -> None:
    """Получить или создать пользователя"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,))
