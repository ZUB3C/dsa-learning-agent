from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ..config import settings
from ..core.llm import get_llm_by_language
from ..core.vector_store import vector_store_manager

MATERIALS_SYSTEM_PROMPT = """Ты - преподаватель по алгоритмам и структурам данных.

Твоя задача - объяснить материал по теме "{topic}" с учетом уровня знаний студента: {user_level}.

Найденные материалы из базы знаний:
{retrieved_materials}

Требования к ответу:
- Адаптируй сложность изложения под уровень студента
- Если уровень начальный - давай больше определений и примеров
- Если уровень продвинутый - можно использовать специфичные термины
- Структурируй материал логично
- Приводи примеры кода, если это уместно

Изложи материал понятно и структурировано."""


QUESTION_SYSTEM_PROMPT = """Ты - преподаватель по алгоритмам и структурам данных.

Студент задал вопрос по теме "{topic}". Уровень студента: {user_level}.

Контекст из базы знаний:
{retrieved_materials}

Вопрос студента: {question}

Ответь на вопрос четко и понятно, учитывая уровень студента."""


def build_materials_agent(language: str = "ru") -> Runnable:
    """Агент для подбора и адаптации материалов."""
    llm = get_llm_by_language(language)

    prompt = ChatPromptTemplate.from_messages([
        ("system", MATERIALS_SYSTEM_PROMPT),
        ("human", "Объясни материал."),
    ])

    return prompt | llm | StrOutputParser()


def build_question_answering_agent(language: str = "ru") -> Runnable:
    """Агент для ответов на вопросы по материалам."""
    llm = get_llm_by_language(language)

    prompt = ChatPromptTemplate.from_messages([
        ("system", QUESTION_SYSTEM_PROMPT),
        ("human", "Ответь на вопрос."),
    ])

    return prompt | llm | StrOutputParser()


def retrieve_materials(topic: str, user_level: str) -> list[Document]:
    """Получить материалы из RAG по теме."""
    # Формируем запрос с учетом темы и уровня
    query = f"Тема: {topic}. Уровень: {user_level}"

    # Получаем топ-K документов
    return vector_store_manager.similarity_search(
        query=query, k=settings.rag_top_k, filter_dict={"topic": topic} if topic else None
    )


def format_retrieved_materials(documents: list[Document]) -> str:
    """Форматировать полученные документы для промпта."""
    if not documents:
        return "Материалы не найдены в базе знаний."

    formatted = []
    for i, doc in enumerate(documents, 1):
        formatted.append(f"--- Материал {i} ---\n{doc.page_content}\n")

    return "\n".join(formatted)
