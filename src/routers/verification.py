import json
import uuid

from fastapi import APIRouter, HTTPException

from ..agents.registry import load_agent
from ..core.database import TestResult, Verification, get_db_session, get_or_create_user
from ..models.schemas import (
    GetVerificationHistoryResponse,
    TestVerificationRequest,
    TestVerificationResponse,
    VerificationDetails,
    VerificationHistoryItem,
)

router = APIRouter(prefix="/api/v1/verification", tags=["Verification"])


@router.post("/check-test")
async def check_test(request: TestVerificationRequest) -> TestVerificationResponse:
    """Проверка ответа с опциональной двойной верификацией"""

    try:
        # Первичная проверка
        primary_agent = load_agent("verification", language=request.language)
        primary_result = await primary_agent.ainvoke({
            "question": request.question,
            "expected_answer": request.expected_answer or "",
            "user_answer": request.user_answer,
        })

        try:
            primary_eval = json.loads(primary_result)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Failed to parse primary evaluation")

        # Генерируем ID верификации
        verification_id = str(uuid.uuid4())

        # Инициализируем переменные для вторичной проверки
        secondary_eval = None
        is_correct = primary_eval.get("is_correct", False)
        final_score = primary_eval.get("score", 0)
        final_feedback = primary_eval.get("feedback", "")

        # Выполняем вторичную проверку только если включена
        if request.secondary_check:
            secondary_agent = load_agent("verification-secondary", language=request.language)
            secondary_result = await secondary_agent.ainvoke({
                "primary_evaluation": json.dumps(primary_eval, ensure_ascii=False),
                "question": request.question,
                "user_answer": request.user_answer,
            })

        try:
            secondary_eval = json.loads(secondary_result)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Failed to parse secondary evaluation")

        # Получаем user_id из тестового результата или используем "unknown"
        with get_db_session() as session:
            test_result = (
                session.query(TestResult).filter(TestResult.test_id == request.test_id).first()
            )
            user_id = test_result.userid if test_result else "unknown"
            get_or_create_user(user_id)

        # Сохраняем результат в БД
        with get_db_session() as session:
            verification = Verification(
                verification_id=verification_id,
                test_id=request.test_id,
                user_id=user_id,
                question=request.question,
                user_answer=request.user_answer,
                is_correct=is_correct,
                score=final_score,
                feedback=final_feedback,
            )
            session.add(verification)

        # Формируем детали верификации
        verification_details = VerificationDetails(
            verification_id=verification_id,
            primary_score=primary_eval.get("score", 0),
            secondary_score=secondary_eval.get("final_score") if secondary_eval else None,
            agree_with_primary=secondary_eval.get("agree_with_primary")
            if secondary_eval
            else None,
            verification_notes=secondary_eval.get("verification_notes", "")
            if secondary_eval
            else None,
        )

        return TestVerificationResponse(
            is_correct=is_correct,
            score=final_score,
            feedback=final_feedback,
            verification_details=verification_details,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification error: {e!s}")


@router.get("/history/{user_id}")
async def get_verification_history(user_id: str) -> GetVerificationHistoryResponse:
    """Получить историю проверок пользователя"""

    with get_db_session() as session:
        verifications = (
            session.query(Verification)
            .filter(Verification.user_id == user_id)
            .order_by(Verification.created_at.desc())
            .all()
        )

        tests_list: list[VerificationHistoryItem] = [
            VerificationHistoryItem(
                verification_id=v.verification_id,
                test_id=v.test_id,
                question=v.question,
                score=v.score,
                is_correct=v.is_correct,
                created_at=v.created_at.isoformat(),
            )
            for v in verifications
        ]

        average_score = sum(t.score for t in tests_list) / len(tests_list) if tests_list else 0.0

        return GetVerificationHistoryResponse(
            tests=tests_list, average_score=average_score, total_tests=len(tests_list)
        )
