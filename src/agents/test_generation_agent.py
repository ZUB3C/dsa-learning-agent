from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ..core.llm import get_llm_by_language

TEST_GENERATION_SYSTEM_PROMPT = (
    "Ты - эксперт по созданию тестовых заданий по алгоритмам и структурам данных.\n"
    "\n"
    'Твоя задача - создать {question_count} вопросов с открытым ответом по теме "{topic}".\n'
    "\n"
    "Требования:\n"
    "- Сложность: {difficulty}\n"
    "- Вопросы должны проверять понимание концепций, а не только запоминание\n"
    "- Каждый вопрос должен иметь эталонный ответ для проверки\n"
    "- Вопросы должны быть разнообразными\n"
    "\n"
    "Формат ответа JSON:\n"
    "{{\n"
    '  "test_id": "<уникальный ID теста>",\n'
    '  "topic": "{topic}",\n'
    '  "difficulty": "{difficulty}",\n'
    '  "questions": [\n'
    "    {{\n"
    '      "question_id": 1,\n'
    '      "question_text": "<текст вопроса>",\n'
    '      "expected_answer": "<эталонный ответ>",\n'
    '      "key_points": ["<ключевой момент 1>", "<ключевой момент 2>"]\n'
    "    }}\n"
    "  ]\n"
    "}}"
)


def build_test_generation_agent(language: str = "ru") -> Runnable:
    """Агент для генерации тестов."""
    llm = get_llm_by_language(language)

    prompt = ChatPromptTemplate.from_messages([
        ("system", TEST_GENERATION_SYSTEM_PROMPT),
        ("human", "Создай тест."),
    ])

    return prompt | llm | StrOutputParser()
