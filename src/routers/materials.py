import uuid

from fastapi import APIRouter, HTTPException
from langchain_core.documents import Document

from ..agents.llm_router_agent import LLMRouter
from ..agents.materials_agent import retrieve_materials
from ..agents.registry import load_agent
from ..core.database import CustomTopic, get_db_session, get_or_create_user
from ..core.vector_store import vector_store_manager
from ..models.schemas import (
    AddCustomTopicRequest,
    AddCustomTopicResponse,
    AskQuestionRequest,
    AskQuestionResponse,
    GenerateMaterialRequest,
    GenerateMaterialResponse,
    GetMaterialsRequest,
    GetMaterialsResponse,
    GetTopicsResponse,
    MaterialSearchResult,
    SearchMaterialsRequest,
    SearchMaterialsResponse,
    TopicInfo,
)

router = APIRouter(prefix="/api/v1/materials", tags=["Materials"])


@router.post("/get-materials")
async def get_materials(request: GetMaterialsRequest) -> GetMaterialsResponse:
    """Получить адаптированные учебные материалы."""
    try:
        # Поиск в векторном хранилище
        documents = vector_store_manager.similarity_search(query=request.topic, k=5)

        retrieved_content = "\n\n".join([doc.page_content for doc in documents])

        # Используем агента для адаптации материала
        materials_agent = load_agent("materials", language=request.language)

        adapted_content = await materials_agent.ainvoke({
            "topic": request.topic,
            "user_level": request.user_level,
            "retrieved_materials": retrieved_content,
        })

        return GetMaterialsResponse(
            content=adapted_content,
            sources=[doc.metadata.get("source", "unknown") for doc in documents],
            adapted_for_level=request.user_level,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving materials: {e!s}")


@router.post("/ask-question")
async def ask_question(request: AskQuestionRequest) -> AskQuestionResponse:
    """Задать вопрос по материалу."""
    try:
        materials_agent = load_agent("materials", language=request.language)

        answer = await materials_agent.ainvoke({
            "topic": request.context_topic,
            "user_level": request.user_level,
            "retrieved_materials": retrieve_materials(
                topic=request.context_topic, user_level=request.user_level
            ),
            "question": request.question,
        })

        return AskQuestionResponse(answer=answer, related_concepts=[request.context_topic])

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {e!s}")


@router.post("/generate-material")
async def generate_material(request: GenerateMaterialRequest) -> GenerateMaterialResponse:
    """Сгенерировать учебный материал."""
    try:
        # Создаем роутер напрямую
        router_instance = LLMRouter(language=request.language)
        selected_model = router_instance.get_model_name(request.language)

        # Используем materials agent для генерации контента
        materials_agent = load_agent("materials", language=request.language)

        # Определяем целевой уровень в зависимости от длины
        level_map = {"short": "beginner", "medium": "intermediate", "long": "advanced"}
        user_level = level_map.get(request.length.lower(), "intermediate")

        # Генерируем материал
        material_content = await materials_agent.ainvoke({
            "topic": request.topic,
            "user_level": user_level,
            "retrieved_materials": f"Создайте {request.format} материал по теме '{request.topic}' длиной {request.length}",  # noqa: E501
            "format": request.format,
        })

        # Подсчет слов
        word_count = len(material_content.split())

        # Форматируем в зависимости от типа
        if request.format.lower() == "summary":
            formatted_material = f"# Краткое содержание: {request.topic}\n\n{material_content}"
        elif request.format.lower() == "detailed":
            formatted_material = f"# Подробный материал: {request.topic}\n\n{material_content}"
        elif request.format.lower() == "example":
            formatted_material = f"# Примеры по теме: {request.topic}\n\n{material_content}"
        else:
            formatted_material = material_content

        # Сохраняем сгенерированный материал
        topic_id = f"generated_{uuid.uuid4()}"

        with get_db_session() as session:
            custom_topic = CustomTopic(
                topic_id=topic_id,
                user_id="system",
                topic_name=f"{request.format}_{request.topic}",
                content=formatted_material,
            )
            session.add(custom_topic)

        return GenerateMaterialResponse(
            material=formatted_material,
            format=request.format,
            word_count=word_count,
            model_used=selected_model,
            topic_id=topic_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating material: {e!s}")


@router.post("/add-custom-topic")
async def add_custom_topic(request: AddCustomTopicRequest) -> AddCustomTopicResponse:
    """Добавить пользовательскую тему."""
    get_or_create_user(request.user_id)
    topic_id = f"custom_{uuid.uuid4()}"

    # Сохраняем в БД
    with get_db_session() as session:
        custom_topic = CustomTopic(
            topic_id=topic_id,
            user_id=request.user_id,
            topic_name=request.topic_name,
            content=request.content,
        )
        session.add(custom_topic)

    # Добавляем в векторное хранилище
    try:
        document = Document(
            page_content=request.content,
            metadata={
                "source": f"custom_topic_{topic_id}",
                "title": request.topic_name,
                "user_id": request.user_id,
                "type": "custom",
            },
        )
        vector_store_manager.add_documents([document])
    except Exception as e:
        print(f"Warning: Failed to add to vector store: {e}")

    return AddCustomTopicResponse(topic_id=topic_id, status="added")


@router.get("/topics")
async def get_topics() -> GetTopicsResponse:
    """Получить список доступных тем."""
    predefined_topics = [
        "Временная сложность",
        "Пространственная сложность",
        "Массивы",
        "Связные списки",
        "Стеки и очереди",
        "Деревья",
        "Графы",
        "Хеш-таблицы",
        "Сортировки",
        "Поиск",
        "Рекурсия",
        "Динамическое программирование",
    ]

    with get_db_session() as session:
        custom = session.query(CustomTopic).all()

        custom_topics: list[TopicInfo] = [
            TopicInfo(topic_id=t.topic_id, topic_name=t.topic_name, user_id=t.user_id)
            for t in custom
        ]

    return GetTopicsResponse(predefined_topics=predefined_topics, custom_topics=custom_topics)


@router.post("/search")
async def search_materials(request: SearchMaterialsRequest) -> SearchMaterialsResponse:
    """Поиск материалов."""
    try:
        documents = vector_store_manager.similarity_search(
            query=request.query, k=10, filter_dict=request.filters
        )

        results: list[MaterialSearchResult] = [
            MaterialSearchResult(content=doc.page_content[:200] + "...", metadata=doc.metadata)
            for doc in documents
        ]

        return SearchMaterialsResponse(results=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {e!s}")
