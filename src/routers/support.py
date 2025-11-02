import json
import uuid

from fastapi import APIRouter, HTTPException

from ..agents.registry import load_agent
from ..core.database import SupportSession, get_db_session, get_or_create_user
from ..models.schemas import (
    GetSupportResourcesResponse,
    SubmitFeedbackRequest,
    SubmitFeedbackResponse,
    SupportRequest,
    SupportResponse,
)

router = APIRouter(prefix="/api/v1/support", tags=["Support"])


@router.post("/get-support")
async def get_support(request: SupportRequest) -> SupportResponse:
    """Получить психологическую поддержку."""
    try:
        get_or_create_user(request.user_id)

        # Загружаем агента поддержки
        support_agent = load_agent("support", language=request.language)

        # Генерируем ответ
        response = await support_agent.ainvoke({
            "message": request.message,
            "emotional_state": request.emotional_state,
            "user_id": request.user_id,
        })

        # Парсим результат
        try:
            support_data = json.loads(response)
            support_message = support_data.get("message", response)
            recommendations = support_data.get("recommendations", [])
        except json.JSONDecodeError:
            support_message = response
            recommendations = _generate_default_recommendations(request.emotional_state)

        # Получаем ресурсы
        resources = _get_support_resources(request.language)

        # Сохраняем в БД
        session_id = str(uuid.uuid4())
        with get_db_session() as session:
            support_session = SupportSession(
                session_id=session_id,
                user_id=request.user_id,
                user_message=request.message,
                emotional_state=request.emotional_state,
                response=support_message,
                recommendations=json.dumps(recommendations),
            )
            session.add(support_session)

        return SupportResponse(
            support_message=support_message, recommendations=recommendations, resources=resources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Support error: {e!s}")


def _generate_default_recommendations(emotional_state: str) -> list[str]:
    """Сгенерировать рекомендации по умолчанию."""
    return _get_recommendations_by_state(emotional_state)


def _get_recommendations_by_state(emotional_state: str) -> list[str]:
    """Получить рекомендации в зависимости от эмоционального состояния."""
    recommendations_map: dict[str, list[str]] = {
        "stressed": [
            "Делайте регулярные перерывы во время обучения (техника Помодоро)",
            "Практикуйте дыхательные упражнения для снятия стресса",
            "Разбивайте сложные темы на маленькие части",
        ],
        "confused": [
            "Не стесняйтесь задавать вопросы по непонятным моментам",
            "Попробуйте объяснить материал своими словами",
            "Решайте больше практических задач для закрепления",
        ],
        "unmotivated": [
            "Ставьте маленькие достижимые цели на каждый день",
            "Отмечайте свой прогресс и успехи",
            "Найдите партнера по обучению для взаимной поддержки",
        ],
        "frustrated": [
            "Помните, что трудности - это нормальная часть обучения",
            "Попробуйте подойти к задаче с другой стороны",
            "Сделайте перерыв и вернитесь к задаче позже",
        ],
    }
    return recommendations_map.get(emotional_state, ["Продолжайте работать в своем темпе"])


def _get_support_resources(language: str) -> list[dict[str, str]]:
    """Получить ресурсы поддержки."""
    if language == "ru":
        return [
            {
                "title": "Техника Помодоро",
                "description": "Метод управления временем для повышения продуктивности",
                "url": "https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%BF%D0%BE%D0%BC%D0%B8%D0%B4%D0%BE%D1%80%D0%B0",
            },
            {
                "title": "Визуализация алгоритмов",
                "description": "Интерактивные визуализации для лучшего понимания",
                "url": "https://visualgo.net",
            },
        ]
    return [
        {
            "title": "Pomodoro Technique",
            "description": "Time management method to boost productivity",
            "url": "https://en.wikipedia.org/wiki/Pomodoro_Technique",
        },
        {
            "title": "Algorithm Visualizations",
            "description": "Interactive visualizations for better understanding",
            "url": "https://visualgo.net",
        },
    ]


@router.get("/resources")
async def get_support_resources() -> GetSupportResourcesResponse:
    """Получить ресурсы психологической поддержки."""
    return GetSupportResourcesResponse(
        articles=[
            {"title": "Как справиться со стрессом при изучении алгоритмов", "url": "#"},
            {"title": "Техники запоминания сложных концепций", "url": "#"},
        ],
        exercises=[
            {"name": "Дыхательная гимнастика 4-7-8", "duration": "5 минут"},
            {"name": "Медитация осознанности", "duration": "10 минут"},
        ],
        tips=[
            "Учитесь регулярно, но небольшими порциями",
            "Практикуйте активное вспоминание",
            "Объясняйте материал другим",
        ],
    )


@router.post("/feedback")
async def submit_feedback(request: SubmitFeedbackRequest) -> SubmitFeedbackResponse:
    """Отправить обратную связь о сессии поддержки."""
    with get_db_session() as session:
        support_session = (
            session.query(SupportSession)
            .filter(SupportSession.session_id == request.session_id)
            .first()
        )

        if support_session:
            support_session.helpful = request.helpful
            support_session.comments = request.comments

    return SubmitFeedbackResponse(status="received")
