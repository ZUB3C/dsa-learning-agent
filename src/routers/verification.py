import json
from typing import Any

from fastapi import APIRouter, HTTPException

from ..agents.registry import load_agent
from ..models.schemas import TestVerificationRequest, TestVerificationResponse

router = APIRouter(prefix="/api/v1/verification", tags=["Verification"])


@router.post("/check-test")
async def check_test(request: TestVerificationRequest) -> TestVerificationResponse:
    """Проверка ответа на тест с двойной верификацией"""

    try:
        # Первичная проверка
        primary_agent = load_agent("verification", language=request.language)
        primary_result = await primary_agent.ainvoke({
            "question": request.question,
            "expected_answer": request.expected_answer or "",
            "user_answer": request.user_answer
        })

        # Парсим JSON ответ
        try:
            primary_eval = json.loads(primary_result)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Failed to parse primary evaluation")

        # Вторичная проверка
        secondary_agent = load_agent("verification-secondary", language=request.language)
        secondary_result = await secondary_agent.ainvoke({
            "primary_evaluation": json.dumps(primary_eval, ensure_ascii=False),
            "question": request.question,
            "user_answer": request.user_answer
        })

        try:
            secondary_eval = json.loads(secondary_result)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Failed to parse secondary evaluation")

        # Формируем итоговый ответ
        return TestVerificationResponse(
            is_correct=secondary_eval.get("final_score", 0) >= 70,  # noqa: PLR2004
            score=secondary_eval.get("final_score", 0),
            feedback=secondary_eval.get("final_feedback", ""),
            verification_details={
                "primary_score": primary_eval.get("score", 0),
                "secondary_score": secondary_eval.get("final_score", 0),
                "agree_with_primary": secondary_eval.get("agree_with_primary", True),
                "verification_notes": secondary_eval.get("verification_notes", "")
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification error: {e!s}")


@router.get("/history/{user_id}")
async def get_verification_history(user_id: str) -> dict[str, Any]:
    """Получить историю проверок пользователя"""
    # TODO: Реализовать получение из БД
    return {
        "tests": [],
        "average_score": 0.0,
        "total_tests": 0
    }
