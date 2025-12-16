import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import aiohttp
from pydantic import BaseModel, Field


# Модели данных
class TestQuestion(BaseModel):
    """Модель вопроса теста."""
    question_id: int
    difficulty: Literal["easy", "medium", "hard"]
    question_text: str
    expected_answer: str
    user_answer: str = ""
    key_points: list[str]
    is_correct: bool = False


class Topic(BaseModel):
    """Модель темы с вопросами."""
    topic_id: str
    topic_name: str
    questions: list[TestQuestion]


class TestData(BaseModel):
    """Модель полного теста."""
    creation_date: str
    total_questions: int
    topics_count: int
    topics: list[Topic]


class GenerateTestRequest(BaseModel):
    """Запрос на генерацию теста."""
    topic: str
    difficulty: Literal["easy", "medium", "hard"]
    questioncount: int = Field(default=5, ge=1, le=20)
    language: str = "ru"


class TestQuestionResponse(BaseModel):
    """Ответ с вопросом от API."""
    questionid: int
    questiontext: str
    expectedanswer: str
    keypoints: list[str]


class GenerateTestResponse(BaseModel):
    """Ответ на запрос генерации теста."""
    testid: str
    questions: list[TestQuestionResponse]
    expectedduration: int


# Конфигурация тем и их распределение
TOPICS_CONFIG: list[dict[str, Any]] = [
    {
        "topic_id": "topic_1",
        "topic_name": "Сложность алгоритмов и Big O",
        "description": "Временная и пространственная сложность алгоритмов",
        "questions_count": 10,
        "difficulty_distribution": {"easy": 3, "medium": 5, "hard": 2}
    },
    {
        "topic_id": "topic_2",
        "topic_name": "Деревья и сбалансированные структуры данных",
        "description": "BST, AVL, красно-черные деревья, B-деревья",
        "questions_count": 10,
        "difficulty_distribution": {"easy": 3, "medium": 5, "hard": 2}
    },
    {
        "topic_id": "topic_3",
        "topic_name": "Графы и алгоритмы обхода",
        "description": "DFS, BFS, кратчайшие пути, топологическая сортировка",
        "questions_count": 13,
        "difficulty_distribution": {"easy": 3, "medium": 6, "hard": 4}
    },
    {
        "topic_id": "topic_4",
        "topic_name": "Хеш-таблицы и хеширование",
        "description": "Хеш-функции, коллизии, методы разрешения",
        "questions_count": 11,
        "difficulty_distribution": {"easy": 3, "medium": 5, "hard": 3}
    },
    {
        "topic_id": "topic_5",
        "topic_name": "Динамическое программирование",
        "description": "Мемоизация, табуляция, классические задачи ДП",
        "questions_count": 12,
        "difficulty_distribution": {"easy": 2, "medium": 5, "hard": 5}
    },
    {
        "topic_id": "topic_6",
        "topic_name": "Линейные структуры данных и куча",
        "description": "Стек, очередь, связный список, бинарная куча",
        "questions_count": 13,
        "difficulty_distribution": {"easy": 3, "medium": 6, "hard": 4}
    },
    {
        "topic_id": "topic_7",
        "topic_name": "Строковые алгоритмы и продвинутые структуры",
        "description": "Trie, KMP, суффиксное дерево, Ахо-Корасик",
        "questions_count": 6,
        "difficulty_distribution": {"easy": 0, "medium": 2, "hard": 4}
    }
]


class TestGenerator:
    def __init__(
            self,
            base_url: str = "http://127.0.0.1:8000",
            timeout: int = 60
    ) -> None:
        """
        Инициализация генератора.

        Args:
            base_url: Базовый URL API
            timeout: Таймаут запросов в секундах
        """
        self.base_url: str = base_url.rstrip("/")
        self.timeout: int = timeout
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "TestGenerator":
        """Создание сессии при входе в контекстный менеджер."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Закрытие сессии при выходе из контекстного менеджера."""
        if self.session:
            await self.session.close()

    async def generate_questions_for_topic(
            self,
            topic_config: dict[str, Any]
    ) -> Topic:
        """
        Генерация вопросов для одной темы.

        Args:
            topic_config: Конфигурация темы

        Returns:
            Topic с сгенерированными вопросами
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'async with' context manager.")

        print(f"\n⚙️  Генерация вопросов для темы: {topic_config['topic_name']}")

        all_questions: list[TestQuestion] = []
        question_id_counter: int = 1

        for difficulty, count in topic_config["difficulty_distribution"].items():
            if count == 0:
                continue

            print(f"  📝 Генерация {count} вопросов уровня '{difficulty}'...")

            batch_size: int = min(count, 20)
            remaining: int = count

            while remaining > 0:
                current_batch: int = min(remaining, batch_size)

                request_data: dict[str, Any] = GenerateTestRequest(
                    topic=topic_config["description"],
                    difficulty=difficulty,
                    questioncount=current_batch,
                    language="ru"
                ).model_dump()

                try:
                    async with self.session.post(
                            f"{self.base_url}/api/v1/tests/generate",
                            json=request_data
                    ) as response:
                        if response.status != 200:
                            error_text: str = await response.text()
                            print(f"  ❌ Ошибка API: {response.status} - {error_text}")
                            remaining -= current_batch
                            continue

                        response_data: dict[str, Any] = await response.json()
                        test_response: GenerateTestResponse = GenerateTestResponse(
                            **response_data
                        )

                        # Преобразуем вопросы в нужный формат
                        for q in test_response.questions:
                            test_question: TestQuestion = TestQuestion(
                                question_id=question_id_counter,
                                difficulty=difficulty,
                                question_text=q.questiontext,
                                expected_answer=q.expectedanswer,
                                user_answer="",
                                key_points=q.keypoints,
                                is_correct=False
                            )
                            all_questions.append(test_question)
                            question_id_counter += 1

                        print(f"    ✅ Получено {len(test_response.questions)} вопросов")
                        remaining -= current_batch

                        # Небольшая задержка между запросами
                        await asyncio.sleep(0.5)

                except TimeoutError:
                    print(f"  ⏱️  Таймаут при генерации вопросов (difficulty={difficulty})")
                    remaining -= current_batch
                except Exception as e:
                    print(f"  ❌ Ошибка: {e!s}")
                    remaining -= current_batch

        return Topic(
            topic_id=topic_config["topic_id"],
            topic_name=topic_config["topic_name"],
            questions=all_questions
        )

    async def generate_full_test(self) -> TestData:
        """Генерация полного теста со всеми темами."""
        print("=" * 80)
        print("🚀 Начало генерации теста")
        print("=" * 80)

        topics: list[Topic] = []

        for topic_config in TOPICS_CONFIG:
            topic: Topic = await self.generate_questions_for_topic(topic_config)
            topics.append(topic)
            print(f"  ✓ Тема завершена: {len(topic.questions)} вопросов сгенерировано")

        # Подсчитываем общее количество вопросов
        total_questions: int = sum(len(topic.questions) for topic in topics)

        test_data: TestData = TestData(
            creation_date=datetime.now().strftime("%Y-%m-%d"),
            total_questions=total_questions,
            topics_count=len(topics),
            topics=topics
        )

        print("\n" + "=" * 80)
        print(f"✅ Генерация завершена! Всего вопросов: {total_questions}")
        print("=" * 80)

        return test_data


async def main() -> None:
    output_file: Path = Path("test_data.json")

    try:
        async with TestGenerator() as generator:
            test_data: TestData = await generator.generate_full_test()

            output_file.write_text(
                json.dumps(
                    test_data.model_dump(),
                    ensure_ascii=False,
                    indent=2
                ),
                encoding="utf-8"
            )

            print(f"\n💾 Тест сохранен в файл: {output_file}")
            print("📊 Статистика:")
            print(f"   - Всего вопросов: {test_data.total_questions}")
            print(f"   - Количество тем: {test_data.topics_count}")

            # Вывод статистики по темам
            for topic in test_data.topics:
                easy: int = sum(1 for q in topic.questions if q.difficulty == "easy")
                medium: int = sum(1 for q in topic.questions if q.difficulty == "medium")
                hard: int = sum(1 for q in topic.questions if q.difficulty == "hard")
                print(f"   - {topic.topic_name}: "
                      f"{len(topic.questions)} вопросов "
                      f"(easy: {easy}, medium: {medium}, hard: {hard})")

    except KeyboardInterrupt:
        print("\n⚠️  Генерация прервана пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e!s}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
