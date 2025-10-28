from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ..core.llm import get_llm_by_language

TEST_GENERATION_SYSTEM_PROMPT = """Ты - эксперт по созданию тестовых заданий по алгоритмам и структурам данных.

Твоя задача - создать {question_count} вопросов с открытым ответом по теме "{topic}".

Требования:
- Сложность: {difficulty}
- Вопросы должны проверять понимание концепций, а не только запоминание
- Каждый вопрос должен иметь эталонный ответ для проверки
- Вопросы должны быть разнообразными

Формат ответа JSON:
{{
  "test_id": "<уникальный ID теста>",
  "topic": "{topic}",
  "difficulty": "{difficulty}",
  "questions": [
    {{
      "question_id": 1,
      "question_text": "<текст вопроса>",
      "expected_answer": "<эталонный ответ>",
      "key_points": ["<ключевой момент 1>", "<ключевой момент 2>"]
    }}
  ]
}}"""


def build_test_generation_agent(language: str = "ru") -> Runnable:
    """Агент для генерации тестов"""

    llm = get_llm_by_language(language)

    prompt = ChatPromptTemplate.from_messages([
        ("system", TEST_GENERATION_SYSTEM_PROMPT),
        ("human", "Создай тест.")
    ])

    return prompt | llm | StrOutputParser()
