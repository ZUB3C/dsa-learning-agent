from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ...core.llm import get_llm
from ...models.orchestrator_schemas import ClassificationResult, TaskType


_SYSTEM_PROMPT = """
Ты — классификатор запросов для образовательной платформы по алгоритмам и структурам данных.

Тебе даётся одно поле: message — свободный текст запроса студента.

Твоя задача:
1. Определи task_type:
   - "materials"      — студент хочет объяснение темы, примеры, теорию
   - "test"           — студент просит сгенерировать тест или задачи
   - "verification"   — студент просит проверить свой ответ
   - "support"        — студент просит ТОЛЬКО психологическую поддержку

2. Определи include_support (true/false):
   - true, если в тексте есть стресс, выгорание, тревога, желание всё бросить.

3. Извлеки topic (строкой) ИЛИ null.

4. При возможности извлеки question и user_answer.

Ответ верни СТРОГО в формате JSON:
{
  "task_type": "materials" | "test" | "verification" | "support",
  "include_support": true | false,
  "topic": "string | null",
  "question": "string | null",
  "user_answer": "string | null",
  "reasoning": "string"
}
""".strip()


class RequestClassifier:
    """LLM-классификатор пользовательских запросов."""

    def __init__(self) -> None:
        llm = get_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", "USER MESSAGE:\n{message}"),
        ])
        self._chain = prompt | llm | StrOutputParser()

    async def classify(self, message: str) -> ClassificationResult:
        raw = await self._chain.ainvoke({"message": message})
        try:
            import json

            data: dict[str, Any] = json.loads(raw)
        except Exception:
            data = {
                "task_type": "materials",
                "include_support": False,
                "topic": None,
                "question": None,
                "user_answer": None,
                "reasoning": "fallback: invalid JSON from LLM",
            }

        task_type = TaskType(data.get("task_type", "materials"))
        include_support = bool(data.get("include_support", False))

        return ClassificationResult(
            task_type=task_type,
            include_support=include_support,
            topic=data.get("topic"),
            question=data.get("question"),
            user_answer=data.get("user_answer"),
            reasoning=data.get("reasoning"),
        )
