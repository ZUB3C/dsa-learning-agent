from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ..core.llm import get_deepseek_llm, get_gigachat_llm

VERIFICATION_SYSTEM_PROMPT = """Ты - эксперт по проверке решений задач по алгоритмам и структурам данных.

Твоя задача:
1. Проанализировать правильность ответа пользователя на вопрос
2. Оценить полноту и корректность решения
3. Дать конструктивную обратную связь

Вопрос: {question}
Эталонный ответ (если есть): {expected_answer}
Ответ пользователя: {user_answer}

Оцени ответ по шкале от 0 до 100 и дай развернутую обратную связь.
Формат ответа должен быть JSON:
{{
  "score": <число от 0 до 100>,
  "is_correct": <true/false>,
  "feedback": "<подробная обратная связь>"
}}"""


def build_verification_agent(language: str = "ru") -> Runnable:
    """Агент для первичной проверки ответов"""

    if language.lower() in {"ru", "russian", "русский"}:
        llm = get_gigachat_llm()
    else:
        llm = get_deepseek_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", VERIFICATION_SYSTEM_PROMPT),
        ("human", "Проверь ответ."),
    ])

    return prompt | llm | StrOutputParser()


def build_secondary_verification_agent(language: str = "ru") -> Runnable:
    """Агент для вторичной проверки (другой провайдер для снижения галлюцинаций)"""

    # Используем противоположную модель для перекрестной проверки
    if language.lower() in {"ru", "russian", "русский"}:
        llm = get_deepseek_llm()
    else:
        llm = get_gigachat_llm()

    secondary_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Ты - независимый эксперт по проверке оценок ответов на задачи по АиСД.

Твоя задача - проверить корректность первичной оценки другой модели.

Первичная оценка: {primary_evaluation}
Вопрос: {question}
Ответ пользователя: {user_answer}

Проанализируй, согласен ли ты с первичной оценкой. Если нет - укажи почему.
Формат ответа JSON:
{{
  "agree_with_primary": <true/false>,
  "final_score": <число от 0 до 100>,
  "final_feedback": "<итоговая обратная связь>",
  "verification_notes": "<замечания по первичной проверке, если есть>"
}}""",
        ),
        ("human", "Проверь оценку."),
    ])

    return secondary_prompt | llm | StrOutputParser()
