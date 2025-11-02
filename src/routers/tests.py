import json
import uuid

from fastapi import APIRouter, HTTPException

from ..agents.llm_router_agent import LLMRouter
from ..agents.registry import load_agent
from ..core.database import Test, TestResult, get_db_session, get_or_create_user
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
    """Сгенерировать тест по теме."""
    try:
        # Создаем роутер для определения модели
        router_instance = LLMRouter(language=request.language)
        router_instance.get_model_name(request.language)

        # Генерируем тест через агента
        test_agent = load_agent("test-generation", language=request.language)

        test_content = await test_agent.ainvoke({
            "topic": request.topic,
            "difficulty": request.difficulty,
            "question_count": request.question_count,
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
                key_points=q.get("key_points", []),
            )
            for idx, q in enumerate(questions_data, 1)
        ]

        # Генерируем ID теста и сохраняем
        test_id = str(uuid.uuid4())
        expected_duration = request.question_count * 5  # 5 минут на вопрос

        with get_db_session() as session:
            test = Test(
                test_id=test_id,
                topic=request.topic,
                difficulty=request.difficulty,
                questions=json.dumps([q.dict() for q in questions]),
                expected_duration=expected_duration,
            )
            session.add(test)

        return GenerateTestResponse(
            test_id=test_id, questions=questions, expected_duration=expected_duration
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating test: {e!s}")


@router.post("/generate-task")
async def generate_task(request: GenerateTaskRequest) -> GenerateTaskResponse:
    """Сгенерировать задачу."""
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
                task_id = uuid.uuid4()
                task = Task(
                    task_id=task_id,
                    description=task_question.get("question_text"),
                    topic=request.topic,
                    difficulty=request.difficulty,
                    task_type=request.task_type,
                    expected_answer=task_question.get("expected_answer"),
                )

                # Генерируем подсказки
                hints: list[TaskHint] = []
                key_points = task_question.get("key_points", [])
                for idx, point in enumerate(key_points[:3], 1):
                    hints.append(TaskHint(hint_level=idx, hint_text=point))

                # Сохраняем задачу в БД
                with get_db_session() as session:
                    test = Test(
                        test_id=task_id,
                        topic=request.topic,
                        difficulty=request.difficulty,
                        questions=json.dumps([task.dict()]),
                        expected_duration=10,
                    )
                    session.add(test)

                return GenerateTaskResponse(
                    task=task, solution_hints=hints, model_used=selected_model
                )

            raise HTTPException(status_code=500, detail=f"No questions generated {request=}")

        except (json.JSONDecodeError, ValueError):
            # Fallback: создаем простую задачу
            return GenerateTaskResponse(
                task=Task(
                    task_id=f"task_{hash(request.topic + request.difficulty)}",
                    description=f"Решите задачу по теме '{request.topic}' уровня сложности '{request.difficulty}'",  # noqa: E501
                    topic=request.topic,
                    difficulty=request.difficulty,
                    task_type=request.task_type,
                ),
                solution_hints=[
                    TaskHint(hint_level=1, hint_text=f"Изучите основы темы: {request.topic}"),
                    TaskHint(hint_level=2, hint_text="Разбейте задачу на подзадачи"),
                ],
                model_used=selected_model,
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating task: {e!s}")


@router.post("/submit-for-verification")
async def submit_test_for_verification(request: SubmitTestRequest) -> SubmitTestResponse:
    """Отправить тест на проверку."""
    get_or_create_user(request.user_id)

    verification_id = str(uuid.uuid4())

    with get_db_session() as session:
        test_result = TestResult(
            test_id=request.test_id, user_id=request.user_id, answers=json.dumps(request.answers)
        )
        session.add(test_result)

    return SubmitTestResponse(verification_id=verification_id, status="pending")


@router.get("/{test_id}")
async def get_test(test_id: str) -> GetTestResponse:
    """Получить тест по ID."""
    with get_db_session() as session:
        test = session.query(Test).filter(Test.test_id == test_id).first()

        if not test:
            raise HTTPException(status_code=404, detail="Test not found")

        return GetTestResponse(
            test={
                "test_id": test.test_id,
                "topic": test.topic,
                "difficulty": test.difficulty,
                "questions": json.loads(test.questions),
                "expected_duration": test.expected_duration,
            },
            metadata={"created_at": test.created_at.isoformat()},
        )


@router.get("/user/{user_id}/completed")
async def get_completed_tests(user_id: str) -> GetCompletedTestsResponse:
    """Получить завершенные тесты пользователя."""
    with get_db_session() as session:
        results = (
            session.query(TestResult, Test)
            .join(Test, TestResult.test_id == Test.test_id)
            .filter(TestResult.user_id == user_id)
            .order_by(TestResult.submitted_at.desc())
            .all()
        )

        completed_tests: list[CompletedTestInfo] = [
            CompletedTestInfo(
                result_id=r.TestResult.result_id,
                test_id=r.TestResult.test_id,
                topic=r.Test.topic,
                difficulty=r.Test.difficulty,
                submitted_at=r.TestResult.submitted_at.isoformat(),
            )
            for r in results
        ]

        return GetCompletedTestsResponse(
            completed_tests=completed_tests, statistics={"total_completed": len(completed_tests)}
        )
