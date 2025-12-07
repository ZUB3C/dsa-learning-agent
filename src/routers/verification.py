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
    """Проверка ответа с опциональной двойной верификацией."""
    try:
        user_id = "unknown"
        verification_id = str(uuid.uuid4())

        # Первичная проверка - теперь возвращает {"verdict": true/false}
        primary_agent = load_agent("verification", language=request.language)
        primary_result = await primary_agent.ainvoke({
            "question": request.question,
            "expected_answer": request.expected_answer or "",
            "user_answer": request.user_answer,
        })

        try:
            primary_eval = json.loads(primary_result)
            # НОВЫЙ ФОРМАТ: {"verdict": true/false}
            # Сохраняем обратную совместимость со старым форматом
            if "verdict" in primary_eval:
                is_correct = primary_eval["verdict"]
            elif "is_correct" in primary_eval:
                is_correct = primary_eval["is_correct"]
            else:
                is_correct = False
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Failed to parse primary evaluation")

        feedback = ""  # В новом формате первичная проверка не дает фидбек
        primary_verdict = is_correct  # Сохраняем для передачи вторичной проверке

        # Вторичная проверка
        secondary_eval = None
        if request.secondary_check:
            secondary_agent = load_agent("verification-secondary", language=request.language)
            secondary_result = await secondary_agent.ainvoke({
                "primary_verdict": primary_verdict,  # Передаем только булево значение
                "question": request.question,
                "user_answer": request.user_answer,
            })

            try:
                secondary_eval = json.loads(secondary_result)
            except json.JSONDecodeError:
                raise HTTPException(status_code=500, detail="Failed to parse secondary evaluation")

        # Финальные значения (если есть вторичная проверка, берем от нее)
        if secondary_eval:
            # Ожидаем формат: {"agree_with_primary": bool, "verdict": bool, "feedback": str, ...}
            if "verdict" in secondary_eval:
                is_correct = secondary_eval["verdict"]
            elif "is_correct" in secondary_eval:
                is_correct = secondary_eval["is_correct"]
            else:
                is_correct = primary_verdict

            feedback = secondary_eval.get("feedback", "")

        # Сохранение в БД
        with get_db_session() as session:
            # Определяем значения для полей secondary_is_correct и agree_with_primary
            secondary_is_correct = None
            agree_with_primary = None
            verification_notes = ""

            if secondary_eval:
                # Получаем вердикт вторичной проверки
                if "verdict" in secondary_eval:
                    secondary_is_correct = secondary_eval["verdict"]
                elif "is_correct" in secondary_eval:
                    secondary_is_correct = secondary_eval["is_correct"]

                agree_with_primary = secondary_eval.get("agree_with_primary")
                verification_notes = secondary_eval.get("verification_notes", "")

            verification = Verification(
                verification_id=verification_id,
                test_id=request.test_id,
                user_id=user_id,
                question=request.question,
                user_answer=request.user_answer,
                is_correct=is_correct,
                feedback=feedback,
                secondary_is_correct=secondary_is_correct,
                agree_with_primary=agree_with_primary,
                verification_notes=verification_notes,
            )
            session.add(verification)

        # Создаем детали верификации для ответа
        verification_details = VerificationDetails(
            verification_id=verification_id,
            primary_is_correct=primary_verdict,
            secondary_is_correct=secondary_is_correct,
            agree_with_primary=agree_with_primary,
            verification_notes=verification_notes,
        )

        return TestVerificationResponse(
            is_correct=is_correct,
            feedback=feedback,
            verification_details=verification_details,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification error: {e!s}")


@router.get("/history/{user_id}")
async def get_verification_history(user_id: str) -> GetVerificationHistoryResponse:
    """Получить историю проверок пользователя."""
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
                is_correct=v.is_correct,
                created_at=v.created_at.isoformat(),
            )
            for v in verifications
        ]

        # Процент правильных ответов
        correct_count = sum(1 for t in tests_list if t.is_correct)
        accuracy_rate = (correct_count / len(tests_list) * 100) if tests_list else 0.0

        return GetVerificationHistoryResponse(
            tests=tests_list, accuracy_rate=accuracy_rate, total_tests=len(tests_list)
        )
