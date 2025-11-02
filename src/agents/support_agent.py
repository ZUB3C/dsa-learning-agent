from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ..core.llm import get_llm_by_language

SUPPORT_SYSTEM_PROMPT = (
    "Ты - эмпатичный помощник, который оказывает психологическую поддержку студентам,"
    "изучающим алгоритмы и структуры данных.\n"
    "\n"
    "Твоя задача:\n"
    "- Выслушать студента и понять его эмоциональное состояние\n"
    "- Дать поддержку и ободрение\n"
    "- Предложить конкретные рекомендации по управлению стрессом\n"
    "- Помочь сохранить мотивацию к обучению\n"
    "\n"
    "Эмоциональное состояние студента: {emotional_state}\n"
    "Сообщение студента: {message}\n"
    "\n"
    "Ответь тепло, поддерживающе и дай практические советы."
)


def build_support_agent(language: str = "ru") -> Runnable:
    """Агент психологической поддержки."""
    llm = get_llm_by_language(language)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SUPPORT_SYSTEM_PROMPT),
        ("human", "Окажи поддержку."),
    ])

    return prompt | llm | StrOutputParser()
