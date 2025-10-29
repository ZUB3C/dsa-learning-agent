import json
import uuid

from fastapi import APIRouter, HTTPException

from ..agents.llm_router_agent import LLMRouter
from ..agents.registry import load_agent
from ..core.database import get_db_connection, get_or_create_user
from ..models.schemas import (
    CompletedTestInfo,
    GenerateTaskRequest,
    GenerateTaskResponse,
    GenerateTestRequest,
    GenerateTestResponse,
    GetCompletedTestsResponse,
    GetTestResponse,
    SubmitTestRequest,
    SubmitTestResponse,
    Task,
    TaskHint,
    TestQuestion,
)

router = APIRouter(prefix="/api/v1/tests", tags=["Tests"])


@router.post("/generate")
async def generate_test(request: GenerateTestRequest) -> GenerateTestResponse:
    """Сгенерировать тест по теме"""

    try:
        # Создаем роутер для определения модели
        router_instance = LLMRouter(language=request.language)
        router_instance.get_model_name(request.language)

        # Генерируем тест через агента
        test_agent = load_agent("test-generation", language=request.language)

        test_content = await test_agent.ainvoke({
            "topic": request.topic,
            "difficulty": request.difficulty,
            "question_count": request.question_count
        })

        # Парсим результат
        try:
            test_data = json.loads(test_content)
            questions_data = test_data.get("questions", [])
        except json.JSONDecodeError:
            questions_data = []

        # Создаем объекты вопросов
        questions = [
            TestQuestion(
                question_id=q.get("question_id", idx),
                question_text=q.get("question_text", ""),
                expected_answer=q.get("expected_answer", ""),
                key_points=q.get("key_points", [])
            )
            for idx, q in enumerate(questions_data, 1)
        ]

        # Генерируем ID теста и сохраняем
        test_id = str(uuid.uuid4())
        expected_duration = request.question_count * 5  # 5 минут на вопрос

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO tests (test_id, topic, difficulty, questions, expected_duration)
                   VALUES (?, ?, ?, ?, ?)""",
                (test_id, request.topic, request.difficulty, json.dumps([q.dict() for q in questions]), expected_duration)
            )

        return GenerateTestResponse(
            test_id=test_id,
            questions=questions,
            expected_duration=expected_duration
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating test: {e!s}")


@router.post("/generate-task")
async def generate_task(request: GenerateTaskRequest) -> GenerateTaskResponse:
    """Сгенерировать задачу"""

    try:
        # Создаем роутер напрямую для определения модели
        router_instance = LLMRouter(language=request.language)
        selected_model = router_instance.get_model_name(request.language)

        # Генерируем задачу через агента генерации тестов
        task_agent = load_agent("test-generation", language=request.language)

        task_result = await task_agent.ainvoke({
            "topic": request.topic,
            "difficulty": request.difficulty,
            "task_type": request.task_type,
            "question_count": 1,
        })

        try:
            task_data = json.loads(task_result)

            # Извлекаем первый вопрос как задачу
            questions = task_data.get("questions", [])
            if questions:
                task_question = questions[0]
                task = Task(
                    task_id=task_question.get("question_id"),
                    description=task_question.get("question_text"),
                    topic=request.topic,
                    difficulty=request.difficulty,
                    task_type=request.task_type,
                    expected_answer=task_question.get("expected_answer")
                )

                # Генерируем подсказки
                hints: list[TaskHint] = []
                key_points = task_question.get("key_points", [])
                for idx, point in enumerate(key_points[:3], 1):
                    hints.append(TaskHint(hint_level=idx, hint_text=point))

                # Сохраняем задачу в БД
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """INSERT INTO tests (test_id, topic, difficulty, questions, expected_duration)
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            str(task.task_id),
                            request.topic,
                            request.difficulty,
                            json.dumps([task.dict()]),
                            10,  # 10 минут на задачу
                        ),
                    )

                return GenerateTaskResponse(
                    task=task,
                    solution_hints=hints,
                    model_used=selected_model
                )

            msg = "No questions generated"
            raise ValueError(msg)

        except (json.JSONDecodeError, ValueError):
            # Fallback: создаем простую задачу
            return GenerateTaskResponse(
                task=Task(
                    task_id=f"task_{hash(request.topic + request.difficulty)}",
                    description=f"Решите задачу по теме '{request.topic}' уровня сложности '{request.difficulty}'",
                    topic=request.topic,
                    difficulty=request.difficulty,
                    task_type=request.task_type
                ),
                solution_hints=[
                    TaskHint(hint_level=1, hint_text=f"Изучите основы темы: {request.topic}"),
                    TaskHint(hint_level=2, hint_text="Разбейте задачу на подзадачи")
                ],
                model_used=selected_model
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating task: {e!s}")


@router.post("/submit-for-verification")
async def submit_test_for_verification(request: SubmitTestRequest) -> SubmitTestResponse:
    """Отправить тест на проверку"""

    get_or_create_user(request.user_id)

    verification_id = str(uuid.uuid4())

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO test_results (test_id, user_id, answers)
               VALUES (?, ?, ?)""",
            (request.test_id, request.user_id, json.dumps(request.answers))
        )

    return SubmitTestResponse(
        verification_id=verification_id,
        status="pending"
    )


@router.get("/{test_id}")
async def get_test(test_id: str) -> GetTestResponse:
    """Получить тест по ID"""

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT test_id, topic, difficulty, questions, expected_duration, created_at
               FROM tests
               WHERE test_id = ?""",
            (test_id,)
        )
        test = cursor.fetchone()

        if not test:
            raise HTTPException(status_code=404, detail="Test not found")

        return GetTestResponse(
            test={
                "test_id": test["test_id"],
                "topic": test["topic"],
                "difficulty": test["difficulty"],
                "questions": json.loads(test["questions"]),
                "expected_duration": test["expected_duration"]
            },
            metadata={
                "created_at": test["created_at"]
            }
        )


@router.get("/user/{user_id}/completed")
async def get_completed_tests(user_id: str) -> GetCompletedTestsResponse:
    """Получить завершенные тесты пользователя"""

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT r.result_id, r.test_id, t.topic, t.difficulty, r.submitted_at
               FROM test_results r
               JOIN tests t ON r.test_id = t.test_id
               WHERE r.user_id = ?
               ORDER BY r.submitted_at DESC""",
            (user_id,)
        )
        results = cursor.fetchall()

        completed_tests: list[CompletedTestInfo] = [
            CompletedTestInfo(
                result_id=r["result_id"],
                test_id=r["test_id"],
                topic=r["topic"],
                difficulty=r["difficulty"],
                submitted_at=r["submitted_at"]
            )
            for r in results
        ]

        return GetCompletedTestsResponse(
            completed_tests=completed_tests,
            statistics={
                "total_completed": len(completed_tests)
            }
        )
