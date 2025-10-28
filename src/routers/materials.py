
from fastapi import APIRouter, HTTPException

from ..agents.materials_agent import format_retrieved_materials, retrieve_materials
from ..agents.registry import load_agent
from ..models.schemas import (
    AskQuestionRequest,
    AskQuestionResponse,
    GetMaterialsRequest,
    GetMaterialsResponse,
)

router = APIRouter(prefix="/api/v1/materials", tags=["Materials"])


@router.post("/get-topic")
async def get_topic_materials(request: GetMaterialsRequest) -> GetMaterialsResponse:
    """Получить материалы по теме с адаптацией под уровень пользователя"""

    try:
        # Получаем материалы через RAG
        documents = retrieve_materials(request.topic, request.user_level)
        retrieved_text = format_retrieved_materials(documents)

        # Загружаем агента для адаптации материалов
        agent = load_agent("materials", language=request.language)

        # Генерируем адаптированный контент
        adapted_content = await agent.ainvoke({
            "topic": request.topic,
            "user_level": request.user_level,
            "retrieved_materials": retrieved_text
        })

        sources = [doc.metadata.get("source", "unknown") for doc in documents]

        return GetMaterialsResponse(
            content=adapted_content,
            sources=sources,
            adapted_for_level=request.user_level
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving materials: {e!s}")


@router.post("/ask-question")
async def ask_question(request: AskQuestionRequest) -> AskQuestionResponse:
    """Задать вопрос по материалам"""

    try:
        # Получаем контекст через RAG
        documents = retrieve_materials(request.context_topic, request.user_level)
        retrieved_text = format_retrieved_materials(documents)

        # Загружаем агента для ответов на вопросы
        agent = load_agent("question-answering", language=request.language)

        # Генерируем ответ
        answer = await agent.ainvoke({
            "topic": request.context_topic,
            "user_level": request.user_level,
            "retrieved_materials": retrieved_text,
            "question": request.question
        })

        # Извлекаем связанные концепции из метаданных документов
        related_concepts = list({
            concept
            for doc in documents
            for concept in doc.metadata.get("concepts", [])
        })

        return AskQuestionResponse(
            answer=answer,
            related_concepts=related_concepts[:5]  # Топ-5
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {e!s}")


@router.post("/add-custom-topic")
async def add_custom_topic(topic_name: str, user_id: str, content: str):
    """Добавить пользовательскую тему"""
    # TODO: Реализовать добавление в векторное хранилище
    return {"topic_id": "custom_123", "status": "added"}


@router.get("/topics")
async def get_topics():
    """Получить список доступных тем"""
    predefined_topics = [
        "Временная сложность",
        "Пространственная сложность",
        "Массивы и списки",
        "Стеки и очереди",
        "Связные списки",
        "Деревья",
        "Графы",
        "Хеш-таблицы",
        "Сортировка",
        "Поиск",
        "Рекурсия",
        "Динамическое программирование",
        "Жадные алгоритмы"
    ]

    return {
        "predefined_topics": predefined_topics,
        "custom_topics": []
    }


@router.post("/search")
async def search_materials(query: str, filters: dict | None = None):
    """Поиск материалов"""
    # TODO: Реализовать поиск через RAG
    return {
        "results": [],
        "relevance_scores": []
    }
