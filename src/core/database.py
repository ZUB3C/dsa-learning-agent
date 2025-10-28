"""
Модуль для работы с SQLite базой данных
"""
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Таблица первичной оценки
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assessments (
                assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT UNIQUE NOT NULL,
                level TEXT NOT NULL,
                score REAL NOT NULL,
                knowledge_areas TEXT,
                recommendations TEXT,
                completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)

        # Таблица сессий оценки
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assessment_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                questions TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)

        # Таблица сгенерированных тестов
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tests (
                test_id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                questions TEXT NOT NULL,
                expected_duration INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Таблица пользовательских тем
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS custom_topics (
                topic_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                topic_name TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)

        # Таблица результатов тестов
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                answers TEXT NOT NULL,
                submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (test_id) REFERENCES tests(test_id),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)

        # Таблица проверок
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS verifications (
                verification_id TEXT PRIMARY KEY,
                test_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                question TEXT NOT NULL,
                user_answer TEXT NOT NULL,
                expected_answer TEXT,
                is_correct BOOLEAN NOT NULL,
                score REAL NOT NULL,
                feedback TEXT NOT NULL,
                verification_details TEXT,
                language TEXT DEFAULT 'ru',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)

        # Таблица сессий поддержки
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS support_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                message TEXT NOT NULL,
                emotional_state TEXT NOT NULL,
                support_message TEXT NOT NULL,
                helpful BOOLEAN,
                comments TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)

        # Индексы для ускорения запросов
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_assessments_user ON assessments(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tests_topic ON tests(topic)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_test_results_user ON test_results(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_verifications_user ON verifications(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_support_user ON support_sessions(user_id)")


def get_or_create_user(user_id: str) -> dict[str, Any]:
    """Получить или создать пользователя"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Проверяем существование
        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        user = cursor.fetchone()

        if user:
            return dict(user)

        # Создаем нового
        cursor.execute(
            "INSERT INTO users (user_id) VALUES (?)",
            (user_id,)
        )

        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        return dict(cursor.fetchone())
