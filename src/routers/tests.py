import json
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException

from ..agents.llm_router_agent import LLMRouter
from ..agents.registry import load_agent
from ..core.database import get_db_connection, get_or_create_user
from ..models.schemas import GenerateTestRequest, GenerateTestResponse, TestQuestion

router = APIRouter(prefix="/api/v1/tests", tags=["Tests"])


@router.post("/generate")
async def generate_test(request: GenerateTestRequest) -> GenerateTestResponse:
    """Сгенерировать тест по теме"""

    try:
        agent = load_agent("test-generation", language=request.language)

        result = await agent.ainvoke({
            "topic": request.topic,
            "difficulty": request.difficulty,
            "question_count": request.question_count
        })

        try:
            test_data = json.loads(result)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Failed to parse generated test")

        questions = [
            TestQuestion(
                question_id=q["question_id"],
                question_text=q["question_text"],
                expected_answer=q["expected_answer"],
                key_points=q.get("key_points", [])
            )
            for q in test_data.get("questions", [])
        ]

        expected_duration = len(questions) * 3
        test_id = test_data.get("test_id", str(uuid.uuid4()))

        # Сохраняем тест в БД
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO tests (test_id, topic, difficulty, questions, expected_duration)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    test_id,
                    request.topic,
                    request.difficulty,
                    json.dumps([q.dict() for q in questions]),
                    expected_duration
                )
            )

        return GenerateTestResponse(
            test_id=test_id,
            questions=questions,
            expected_duration=expected_duration
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating test: {e!s}")


@router.post("/generate-task")
async def generate_task(
    topic: str, difficulty: str, task_type: str, language: str = "ru"
) -> dict[str, Any]:
    """Сгенерировать задачу"""

    try:
        # Создаем роутер напрямую для определения модели
        router_instance = LLMRouter(language=language)
        selected_model = router_instance.get_model_name(language)

        # Генерируем задачу через агента генерации тестов
        task_agent = load_agent("test-generation", language=language)

        task_result = await task_agent.ainvoke({
            "topic": topic,
            "difficulty": difficulty,
            "task_type": task_type,
            "question_count": 1,
        })

        try:
            task_data = json.loads(task_result)

            # Извлекаем первый вопрос как задачу
            questions = task_data.get("questions", [])
            if questions:
                task_question = questions[0]
                task = {
                    "task_id": task_question.get("question_id"),
                    "description": task_question.get("question_text"),
                    "topic": topic,
                    "difficulty": difficulty,
                    "task_type": task_type,
                    "expected_answer": task_question.get("expected_answer"),
                }

                # Генерируем подсказки
                hints = []
                key_points = task_question.get("key_points", [])
                for idx, point in enumerate(key_points[:3], 1):
                    hints.append({"hint_level": idx, "hint_text": point})

                # Сохраняем задачу в БД
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """INSERT INTO tests (test_id, topic, difficulty, questions, expected_duration)
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            str(task["task_id"]),
                            topic,
                            difficulty,
                            json.dumps([task]),
                            10,  # 10 минут на задачу
                        ),
                    )

                return {"task": task, "solution_hints": hints, "model_used": selected_model}

            msg = "No questions generated"
            raise ValueError(msg)

        except (json.JSONDecodeError, ValueError):
            # Fallback: создаем простую задачу
            return {
                "task": {
                    "task_id": f"task_{hash(topic + difficulty)}",
                    "description": f"Решите задачу по теме '{topic}' уровня сложности '{difficulty}'",
                    "topic": topic,
                    "difficulty": difficulty,
                    "task_type": task_type,
                },
                "solution_hints": [
                    {"hint_level": 1, "hint_text": f"Изучите основы темы: {topic}"},
                    {"hint_level": 2, "hint_text": "Разбейте задачу на подзадачи"},
                ],
                "model_used": selected_model,
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating task: {e!s}")


@router.post("/submit-for-verification")
async def submit_test_for_verification(
        test_id: str,
        user_id: str,
        answers: list[dict[str, Any]]
) -> dict[str, str]:
    """Отправить тест на проверку"""

    get_or_create_user(user_id)

    verification_id = str(uuid.uuid4())

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO test_results (test_id, user_id, answers)
               VALUES (?, ?, ?)""",
            (test_id, user_id, json.dumps(answers))
        )

    return {
        "verification_id": verification_id,
        "status": "pending"
    }


@router.get("/{test_id}")
async def get_test(test_id: str) -> dict[str, Any]:
    """Получить тест по ID"""

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT test_id, topic, difficulty, questions, expected_duration, created_at
               FROM tests WHERE test_id = ?""",
            (test_id,)
        )
        test = cursor.fetchone()

        if not test:
            raise HTTPException(status_code=404, detail="Test not found")

        return {
            "test": {
                "test_id": test["test_id"],
                "topic": test["topic"],
                "difficulty": test["difficulty"],
                "questions": json.loads(test["questions"]),
                "expected_duration": test["expected_duration"]
            },
            "metadata": {
                "created_at": test["created_at"]
            }
        }


@router.get("/user/{user_id}/completed")
async def get_completed_tests(user_id: str) -> dict[str, Any]:
    """Получить завершенные тесты пользователя"""

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT tr.result_id, tr.test_id, t.topic, t.difficulty, tr.submitted_at
               FROM test_results tr
               JOIN tests t ON tr.test_id = t.test_id
               WHERE tr.user_id = ?
               ORDER BY tr.submitted_at DESC""",
            (user_id,)
        )
        results = cursor.fetchall()

        completed_tests = [
            {
                "result_id": r["result_id"],
                "test_id": r["test_id"],
                "topic": r["topic"],
                "difficulty": r["difficulty"],
                "submitted_at": r["submitted_at"]
            }
            for r in results
        ]

        return {
            "completed_tests": completed_tests,
            "statistics": {
                "total_completed": len(completed_tests)
            }
        }
