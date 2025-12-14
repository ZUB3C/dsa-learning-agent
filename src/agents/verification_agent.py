from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ..core.llm import get_llm

# =======================
# Промпты
# =======================

PRIMARY_VERIFICATION_PROMPT = (
    "Ты — эксперт по проверке решений задач по алгоритмам и структурам данных.\n\n"
    "Твоя задача:\n"
    "1. Проанализировать ответ пользователя на вопрос.\n"
    "2. Сравнить с эталонным ответом.\n"
    "3. Определить, является ли ответ правильным.\n\n"
    "ДАННЫЕ ДЛЯ ПРОВЕРКИ:\n"
    "Вопрос: {question}\n"
    "Эталонный ответ: {expected_answer}\n"
    "Ответ пользователя: {user_answer}\n\n"
    "ВЫВОД:\n"
    "Верни строго JSON в формате:\n"
    '{{ "verdict": true/false }}'
)

SECONDARY_VERIFICATION_PROMPT = (
    "Ты — строгий, но справедливый судья-эксперт по алгоритмам и структурам данных.\n\n"
    "ПЕРВИЧНЫЙ ВЕРДИКТ: {primary_verdict}\n\n"
    "Твоя задача:\n"
    "1. Сначала определи, согласен ли ты с первичной оценкой.\n"
    "2. Проверь ответ пользователя независимо и вынеси свой вердикт.\n\n"
    "ДАННЫЕ ДЛЯ ОЦЕНКИ:\n"
    "Вопрос: {question}\n"
    "Ответ пользователя: {user_answer}\n\n"
    "ВЫВОД:\n"
    "Верни строго JSON в формате:\n"
    "{{\n"
    '  "verdict": true/false,\n'
    '  "agree_with_primary": true/false,\n'
    '  "feedback": "Краткое обоснование вердикта (2-3 предложения)"\n'
    "}}"
)

# =======================
# Агенты
# =======================


def build_verification_agent() -> Runnable:
    """Агент для первичной проверки ответов."""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", PRIMARY_VERIFICATION_PROMPT),
        ("human", "Проверь ответ на тест."),
    ])
    return prompt | llm | StrOutputParser()


def build_secondary_verification_agent() -> Runnable:
    """Агент для вторичной проверки с учетом первичного вердикта."""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SECONDARY_VERIFICATION_PROMPT),
        ("human", "Проверь оценку."),
    ])
    return prompt | llm | StrOutputParser()
