import json
import uuid

from fastapi import APIRouter, HTTPException

from ..agents.registry import load_agent
from ..core.database import get_db_connection, get_or_create_user
from ..models.schemas import (
    GetVerificationHistoryResponse,
    TestVerificationRequest,
    TestVerificationResponse,
    VerificationHistoryItem,
)

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
            "user_answer": request.user_answer,
        })

        try:
            primary_eval = json.loads(primary_result)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Failed to parse primary evaluation")

        # Вторичная проверка (перекрёстная верификация другой моделью)
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

        verification_id = str(uuid.uuid4())
        is_correct = secondary_eval.get("final_score", 0) >= 70

        # Извлекаем user_id из test_results
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT user_id FROM test_results WHERE test_id = ? LIMIT 1", (request.test_id,)
            )
            test_result = cursor.fetchone()
            user_id = test_result["user_id"] if test_result else "unknown"

        get_or_create_user(user_id)

        # Сохраняем результат с деталями двойной проверки
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO verifications
                   (verification_id, test_id, user_id, question, user_answer, expected_answer,
                    is_correct, score, feedback, verification_details, language)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    verification_id,
                    request.test_id,
                    user_id,
                    request.question,
                    request.user_answer,
                    request.expected_answer,
                    is_correct,
                    secondary_eval.get("final_score", 0),
                    secondary_eval.get("final_feedback", ""),
                    json.dumps({
                        "primary_score": primary_eval.get("score", 0),
                        "secondary_score": secondary_eval.get("final_score", 0),
                        "agree_with_primary": secondary_eval.get("agree_with_primary", True),
                        "verification_notes": secondary_eval.get("verification_notes", ""),
                    }),
                    request.language,
                ),
            )

        return TestVerificationResponse(
            is_correct=is_correct,
            score=secondary_eval.get("final_score", 0),
            feedback=secondary_eval.get("final_feedback", ""),
            verification_details={
                "verification_id": verification_id,
                "primary_score": primary_eval.get("score", 0),
                "secondary_score": secondary_eval.get("final_score", 0),
                "agree_with_primary": secondary_eval.get("agree_with_primary", True),
                "verification_notes": secondary_eval.get("verification_notes", ""),
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification error: {e!s}")


@router.get("/history/{user_id}")
async def get_verification_history(user_id: str) -> GetVerificationHistoryResponse:
    """Получить историю проверок пользователя"""

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT verification_id, test_id, question, score, is_correct, created_at
               FROM verifications
               WHERE user_id = ?
               ORDER BY created_at DESC""",
            (user_id,),
        )
        verifications = cursor.fetchall()

        tests_list: list[VerificationHistoryItem] = [
            VerificationHistoryItem(
                verification_id=v["verification_id"],
                test_id=v["test_id"],
                question=v["question"],
                score=v["score"],
                is_correct=bool(v["is_correct"]),
                created_at=v["created_at"],
            )
            for v in verifications
        ]

        average_score = sum(t.score for t in tests_list) / len(tests_list) if tests_list else 0.0

        return GetVerificationHistoryResponse(
            tests=tests_list, average_score=average_score, total_tests=len(tests_list)
        )
