from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ..config import settings
from ..core.llm import get_llm
from ..core.vector_store import vector_store_manager

MATERIALS_SYSTEM_PROMPT = (
    "# Role\n"
    "Ты - преподаватель по алгоритмам и структурам данных.\n"
    "\n"
    "# Tone\n"
    "Объясняй доступно и структурировано.\n"
    "\n"
    "# Context\n"
    "Тема: {topic}\n"
    "Уровень студента: {user_level}\n"
    "\n"
    "Материалы из базы знаний:\n"
    "{retrieved_materials}\n"
    "\n"
    "# Task\n"
    "Объясни материал, адаптируя сложность под уровень студента:\n"
    "- Начальный уровень: больше определений и примеров\n"
    "- Продвинутый уровень: используй специфичные термины\n"
    "- Добавляй примеры кода, где уместно\n"
    "\n"
    "# Output Format\n"
    "Структурируй ответ логично с примерами."
)

QUESTION_SYSTEM_PROMPT = (
    "# Role\n"
    "Ты - преподаватель по алгоритмам и структурам данных.\n"
    "\n"
    "# Tone\n"
    "Отвечай четко и понятно.\n"
    "\n"
    "# Context\n"
    "Тема: {topic}\n"
    "Уровень студента: {user_level}\n"
    "\n"
    "Материалы из базы знаний:\n"
    "{retrieved_materials}\n"
    "\n"
    "# Question\n"
    "{question}\n"
    "\n"
    "# Task\n"
    "Ответь на вопрос студента, учитывая его уровень знаний."
)


def build_materials_agent(language: str = "ru") -> Runnable:
    """Агент для подбора и адаптации материалов."""
    llm = get_llm(language)
    prompt = ChatPromptTemplate.from_messages([
        ("system", MATERIALS_SYSTEM_PROMPT),
        ("human", "Объясни материал."),
    ])
    return prompt | llm | StrOutputParser()


def build_question_answering_agent(language: str = "ru") -> Runnable:
    """Агент для ответов на вопросы по материалам."""
    llm = get_llm(language)
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
