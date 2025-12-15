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
    "Ты — независимый, строгий и справедливый судья-эксперт по проверке ответов.\n\n"
    "ПЕРВИЧНЫЙ ВЕРДИКТ: {primary_verdict}\n\n"
    "ТВОЯ ЗАДАЧА:\n"
    "1. Самостоятельно определить, каким должен быть правильный ответ на вопрос.\n"
    "2. Сравнить ответ пользователя со своим пониманием правильного ответа.\n"
    "3. Вынести собственный вердикт (true/false).\n"
    "4. Определить, согласен ли ты с первичной проверкой.\n\n"
    "КРИТЕРИИ ОЦЕНКИ:\n"
    "- verdict = true: если ответ пользователя по СУТИ верный, даже если формулировка отличается\n"
    "- verdict = false: если есть фактические ошибки, логические противоречия или ответ неполный\n"
    "- Оценивай СМЫСЛ, а не стиль, допускай перефразирование.\n"
    "- Игнорируй мелкие опечатки.\n\n"
    "ДАННЫЕ ДЛЯ ОЦЕНКИ:\n"
    "Вопрос: {question}\n"
    "Ответ пользователя: {user_answer}\n\n"
    "ВЫВОД:\n"
    "Верни СТРОГО валидный JSON без дополнительного текста:\n"
    "{{\n"
    '  "verdict": true/false,\n'
    '  "agree_with_primary": true/false,\n'
    '  "feedback": "Краткое обоснование вердикта (1–3 предложения)",\n'
    '  "verification_notes": "Краткая причина решения для аналитики"\n'
    "}}"
)

# =======================
# Агенты
# =======================


def build_verification_agent() -> Runnable:
    """Агент для первичной проверки ответов."""
    llm = get_llm(use_gigachat3=True)
    prompt = ChatPromptTemplate.from_messages([
        ("system", PRIMARY_VERIFICATION_PROMPT),
        ("human", "Проверь ответ на тест."),
    ])
    return prompt | llm | StrOutputParser()


def build_secondary_verification_agent() -> Runnable:
    """Агент для вторичной проверки с учетом первичного вердикта."""
    llm = get_llm(use_gigachat3=False)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SECONDARY_VERIFICATION_PROMPT),
        ("human", "Проверь оценку."),
    ])
    return prompt | llm | StrOutputParser()
