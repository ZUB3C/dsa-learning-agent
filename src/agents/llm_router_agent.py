from typing import Any, Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ..core.llm import get_deepseek_llm, get_gigachat_llm, get_llm_by_language

RequestType = Literal["material", "task", "test", "question", "support"]


def build_router_agent() -> "LLMRouter":
    """Создать агент-роутер для выбора подходящей LLM"""
    return LLMRouter()


class LLMRouter:
    """Роутер для выбора подходящей LLM в зависимости от языка и типа запроса"""

    def select_llm(self, language: str, request_type: RequestType) -> Runnable:
        """Выбрать подходящую LLM"""

        # Определяем базовую модель по языку
        base_llm = get_llm_by_language(language)

        # Можно добавить логику выбора в зависимости от типа запроса
        # Например, для задач использовать модель с большей температурой
        if request_type in {"task", "test"}:
            # Для генерации задач можем использовать чуть большую температуру
            if language.lower() in {"ru", "russian", "русский"}:
                return get_gigachat_llm(temperature=0.4)
            return get_deepseek_llm(temperature=0.4)

        return base_llm

    def get_model_name(self, language: str) -> str:
        """Получить название используемой модели"""
        if language.lower() in {"ru", "russian", "русский"}:
            return "GigaChat"
        return "DeepSeek"

    async def generate_content(
            self,
            request_type: RequestType,
            content: str,
            language: str,
            system_prompt: str,
            parameters: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Генерировать контент с помощью выбранной LLM"""

        llm = self.select_llm(language, request_type)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", content)
        ])

        chain = prompt | llm | StrOutputParser()

        result = await chain.ainvoke({"input": content, **(parameters or {})})

        return {
            "generated_content": result,
            "model_used": self.get_model_name(language),
            "request_type": request_type,
            "language": language
        }
