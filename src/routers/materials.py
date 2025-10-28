import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from langchain_core.documents import Document

from ..agents.materials_agent import format_retrieved_materials, retrieve_materials
from ..agents.registry import load_agent
from ..core.database import get_db_connection, get_or_create_user
from ..core.vector_store import vector_store_manager
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
        documents = retrieve_materials(request.topic, request.user_level)
        retrieved_text = format_retrieved_materials(documents)

        agent = load_agent("materials", language=request.language)

        adapted_content = await agent.ainvoke({
            "topic": request.topic,
            "user_level": request.user_level,
            "retrieved_materials": retrieved_text
        })

        sources = list({doc.metadata.get("source", "unknown") for doc in documents})

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
        documents = retrieve_materials(request.context_topic, request.user_level)
        retrieved_text = format_retrieved_materials(documents)

        agent = load_agent("question-answering", language=request.language)

        answer = await agent.ainvoke({
            "topic": request.context_topic,
            "user_level": request.user_level,
            "retrieved_materials": retrieved_text,
            "question": request.question
        })

        related_concepts = []
        for doc in documents:
            concepts_str = doc.metadata.get("concepts", "")
            if concepts_str:
                related_concepts.extend(concepts_str.split(", "))

        related_concepts = list(set(related_concepts))[:5]

        return AskQuestionResponse(
            answer=answer,
            related_concepts=related_concepts
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {e!s}")


@router.post("/add-custom-topic")
async def add_custom_topic(topic_name: str, user_id: str, content: str) -> dict[str, str]:
    """Добавить пользовательскую тему"""

    get_or_create_user(user_id)
    topic_id = f"custom_{uuid.uuid4()}"

    # Сохраняем в БД
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO custom_topics (topic_id, user_id, topic_name, content)
               VALUES (?, ?, ?, ?)""",
            (topic_id, user_id, topic_name, content)
        )

    # Добавляем в векторное хранилище
    try:
        document = Document(
            page_content=content,
            metadata={
                "source": f"custom_topic_{topic_id}",
                "title": topic_name,
                "user_id": user_id,
                "type": "custom"
            }
        )
        vector_store_manager.add_documents([document])
    except Exception as e:
        print(f"Warning: Failed to add to vector store: {e}")

    return {"topic_id": topic_id, "status": "added"}


@router.get("/topics")
async def get_topics() -> dict[str, Any]:
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

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT topic_id, topic_name, user_id FROM custom_topics")
        custom = cursor.fetchall()

        custom_topics = [
            {
                "topic_id": t["topic_id"],
                "topic_name": t["topic_name"],
                "user_id": t["user_id"]
            }
            for t in custom
        ]

    return {
        "predefined_topics": predefined_topics,
        "custom_topics": custom_topics
    }


@router.post("/search")
async def search_materials(query: str, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Поиск материалов"""

    try:
        documents = vector_store_manager.similarity_search(
            query=query,
            k=10,
            filter_dict=filters
        )

        results = [
            {
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata
            }
            for doc in documents
        ]

        relevance_scores = [1.0 / (i + 1) for i in range(len(results))]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {e!s}")
    else:
        return {
            "results": results,
            "relevance_scores": relevance_scores
        }
