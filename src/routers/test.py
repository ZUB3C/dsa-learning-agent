import json
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException

from ..agents.registry import load_agent
from ..models.schemas import GenerateTestRequest, GenerateTestResponse, TestQuestion

router = APIRouter(prefix="/api/v1/tests", tags=["Tests"])


@router.post("/generate")
async def generate_test(request: GenerateTestRequest) -> GenerateTestResponse:
    """Сгенерировать тест по теме"""

    try:
        # Загружаем агента для генерации тестов
        agent = load_agent("test-generation", language=request.language)

        # Генерируем тест
        result = await agent.ainvoke({
            "topic": request.topic,
            "difficulty": request.difficulty,
            "question_count": request.question_count
        })

        # Парсим JSON результат
        try:
            test_data = json.loads(result)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Failed to parse generated test")

        # Формируем ответ
        questions = [
            TestQuestion(
                question_id=q["question_id"],
                question_text=q["question_text"],
                expected_answer=q["expected_answer"],
                key_points=q.get("key_points", [])
            )
            for q in test_data.get("questions", [])
        ]

        # Оценка времени: ~3 минуты на вопрос
        expected_duration = len(questions) * 3

        test_id = test_data.get("test_id", str(uuid.uuid4()))

        # TODO: Сохранить тест в БД

        return GenerateTestResponse(
            test_id=test_id,
            questions=questions,
            expected_duration=expected_duration
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating test: {e!s}")


@router.post("/submit-for-verification")
async def submit_test_for_verification(test_id: str, user_id: str, answers: list[dict]) ->\
        dict[str, str]:
    """Отправить тест на проверку"""
    # TODO: Сохранить ответы и отправить на проверку в модуль 1
    return {
        "verification_id": str(uuid.uuid4()),
        "status": "pending"
    }


@router.get("/{test_id}")
async def get_test(test_id: str) -> dict[str, Any]:
    """Получить тест по ID"""
    # TODO: Получить из БД
    return {
        "test": {},
        "metadata": {}
    }


@router.get("/user/{user_id}/completed")
async def get_completed_tests(user_id: str) -> dict[str, Any]:
    """Получить завершенные тесты пользователя"""
    # TODO: Получить из БД
    return {
        "completed_tests": [],
        "statistics": {}
    }
