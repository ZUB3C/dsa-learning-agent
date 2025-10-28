import uuid
from typing import Any

from fastapi import APIRouter, HTTPException

from ..agents.registry import load_agent
from ..core.database import get_db_connection, get_or_create_user
from ..models.schemas import SupportRequest, SupportResponse

router = APIRouter(prefix="/api/v1/support", tags=["Support"])


@router.post("/request")
async def request_support(request: SupportRequest) -> SupportResponse:
    """Запросить психологическую поддержку"""

    try:
        get_or_create_user(request.user_id)

        agent = load_agent("support", language=request.language)

        support_message = await agent.ainvoke({
            "emotional_state": request.emotional_state,
            "message": request.message
        })

        recommendations = _get_recommendations_by_state(request.emotional_state)
        resources = _get_support_resources(request.language)

        # Сохраняем сессию поддержки
        session_id = str(uuid.uuid4())
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO support_sessions
                   (session_id, user_id, message, emotional_state, support_message)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, request.user_id, request.message,
                 request.emotional_state, support_message)
            )

        return SupportResponse(
            support_message=support_message,
            recommendations=recommendations,
            resources=resources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error providing support: {e!s}")


def _get_recommendations_by_state(emotional_state: str) -> list[str]:
    """Получить рекомендации в зависимости от эмоционального состояния"""

    recommendations_map = {
        "stressed": [
            "Делайте регулярные перерывы во время обучения (техника Помодоро)",
            "Практикуйте дыхательные упражнения для снятия стресса",
            "Разбивайте сложные темы на маленькие части"
        ],
        "confused": [
            "Не стесняйтесь задавать вопросы по непонятным моментам",
            "Попробуйте объяснить материал своими словами",
            "Решайте больше практических задач для закрепления"
        ],
        "unmotivated": [
            "Ставьте маленькие достижимые цели на каждый день",
            "Отмечайте свой прогресс и успехи",
            "Найдите партнера по обучению для взаимной поддержки"
        ],
        "frustrated": [
            "Помните, что трудности - это нормальная часть обучения",
            "Попробуйте подойти к задаче с другой стороны",
            "Сделайте перерыв и вернитесь к задаче позже"
        ]
    }

    return recommendations_map.get(emotional_state, [
        "Продолжайте учиться в своем темпе",
        "Не забывайте отдыхать",
        "Верьте в свои силы"
    ])


def _get_support_resources(language: str) -> list[dict[str, str]]:
    """Получить ресурсы поддержки"""

    if language == "ru":
        return [
            {
                "title": "Техника Помодоро",
                "description": "Метод управления временем для повышения продуктивности",
                "url": "https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%BF%D0%BE%D0%BC%D0%B8%D0%B4%D0%BE%D1%80%D0%B0"
            },
            {
                "title": "Визуализация алгоритмов",
                "description": "Интерактивные визуализации для лучшего понимания",
                "url": "https://visualgo.net"
            }
        ]
    return [
        {
            "title": "Pomodoro Technique",
            "description": "Time management method to boost productivity",
            "url": "https://en.wikipedia.org/wiki/Pomodoro_Technique"
        },
        {
            "title": "Algorithm Visualizations",
            "description": "Interactive visualizations for better understanding",
            "url": "https://visualgo.net"
        }
    ]


@router.get("/resources")
async def get_support_resources() -> dict[str, Any]:
    """Получить ресурсы психологической поддержки"""

    return {
        "articles": [
            {"title": "Как справиться со стрессом при изучении алгоритмов", "url": "#"},
            {"title": "Техники запоминания сложных концепций", "url": "#"}
        ],
        "exercises": [
            {"name": "Дыхательная гимнастика 4-7-8", "duration": "5 минут"},
            {"name": "Медитация осознанности", "duration": "10 минут"}
        ],
        "tips": [
            "Учитесь регулярно, но небольшими порциями",
            "Практикуйте активное вспоминание",
            "Объясняйте материал другим"
        ]
    }


@router.post("/feedback")
async def submit_feedback(session_id: str, helpful: bool, comments: str = "") -> dict[str, str]:
    """Отправить обратную связь о сессии поддержки"""

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """UPDATE support_sessions
               SET helpful = ?, comments = ?
               WHERE session_id = ?""",
            (helpful, comments, session_id)
        )

    return {"status": "received"}
