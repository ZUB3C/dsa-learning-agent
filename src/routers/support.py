from typing import Any

from fastapi import APIRouter, HTTPException

from ..agents.registry import load_agent
from ..models.schemas import SupportRequest, SupportResponse

router = APIRouter(prefix="/api/v1/support", tags=["Support"])


@router.post("/request")
async def request_support(request: SupportRequest) -> SupportResponse:
    """Запросить психологическую поддержку"""

    try:
        # Загружаем агента поддержки
        agent = load_agent("support", language=request.language)

        # Генерируем ответ
        support_message = await agent.ainvoke({
            "emotional_state": request.emotional_state,
            "message": request.message
        })

        # Формируем рекомендации
        recommendations = _get_recommendations_by_state(request.emotional_state)

        # Формируем ресурсы
        resources = _get_support_resources(request.language)

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
                "url": "https://example.com/pomodoro"
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
            "url": "https://example.com/pomodoro"
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
async def submit_feedback(session_id: str, helpful: bool, comments: str = "") -> dict[str, str]:  # noqa: FBT001
    """Отправить обратную связь о сессии поддержки"""
    # TODO: Сохранить в БД
    return {"status": "received"}
