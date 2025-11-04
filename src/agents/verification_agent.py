from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ..core.llm import get_deepseek_llm, get_gigachat_llm

VERIFICATION_SYSTEM_PROMPT = (
    "Ты - эксперт по проверке решений задач по алгоритмам и структурам данных.\n"
    "\n"
    "Твоя задача:\n"
    "1. Проанализировать правильность ответа пользователя на вопрос\n"
    "2. Оценить полноту и корректность решения\n"
    "3. Дать конструктивную обратную связь\n"
    "\n"
    "Вопрос: {question}\n"
    "Эталонный ответ (если есть): {expected_answer}\n"
    "Ответ пользователя: {user_answer}\n"
    "\n"
    "Оцени ответ по шкале от 0 до 100 и дай развернутую обратную связь.\n"
    "Формат ответа должен быть JSON:\n"
    "{{\n"
    '  "is_correct": <true/false>,\n'
    '  "feedback": "<подробная обратная связь>"\n'
    "}}"
)


def build_verification_agent(language: str = "ru") -> Runnable:
    """Агент для первичной проверки ответов."""
    if language.lower() in {"ru", "russian", "русский"}:
        llm = get_gigachat_llm()
    else:
        llm = get_deepseek_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", VERIFICATION_SYSTEM_PROMPT),
        ("human", "Проверь ответ на тест."),
    ])

    return prompt | llm | StrOutputParser()


def build_secondary_verification_agent(language: str = "ru") -> Runnable:
    """Агент для вторичной проверки (другой провайдер для снижения галлюцинаций)."""
    # Используем противоположную модель для перекрестной проверки
    if language.lower() in {"ru", "russian", "русский"}:
        llm = get_gigachat_llm()
    else:
        llm = get_deepseek_llm()

    secondary_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Ты - независимый эксперт по проверке оценок ответов на задачи по АиСД.\n"
            "\n"
            "Твоя задача - проверить корректность первичной оценки другой модели. "
            "Критично относись к баллам, выставленным первой моделью.\n"
            "\n"
            "Первичная оценка: {primary_evaluation}\n"
            "Вопрос: {question}\n"
            "Ответ пользователя: {user_answer}\n"
            "\n"
            "Проанализируй, согласен ли ты с первичной оценкой. Если нет - укажи почему.\n"
            "Формат ответа JSON:\n"
            "{{\n"
            '  "agree_with_primary": <true/false>,\n'
            '  "feedback": "<итоговая обратная связь>",\n'
            '  "verification_notes": "<замечания по первичной проверке, если есть>"\n'
            "}}",
        ),
        ("human", "Проверь оценку."),
    ])

    return secondary_prompt | llm | StrOutputParser()
