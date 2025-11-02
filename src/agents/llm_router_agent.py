import json
from typing import Any, Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ..core.llm import get_deepseek_llm, get_gigachat_llm, get_llm_by_language

RequestType = Literal["material", "task", "test", "question", "support"]


def build_router_agent(language: str = "ru") -> "LLMRouter":
    """Создать агент-роутер для выбора подходящей LLM."""
    return LLMRouter(language=language)


class LLMRouter:
    """Роутер для выбора подходящей LLM в зависимости от языка и типа запроса."""

    def __init__(self, language: str = "ru") -> None:
        """Инициализация роутера с языком по умолчанию."""
        self.default_language = language

    def select_llm(
        self, language: str | None = None, request_type: RequestType | None = None
    ) -> Runnable:
        """Выбрать подходящую LLM."""
        lang = language or self.default_language

        # Определяем базовую модель по языку
        base_llm = get_llm_by_language(lang)

        # Можно добавить логику выбора в зависимости от типа запроса
        # Например, для задач использовать модель с большей температурой
        if request_type in {"task", "test"}:
            # Для генерации задач можем использовать чуть большую температуру
            if lang.lower() in {"ru", "russian", "русский"}:
                return get_gigachat_llm(temperature=0.4)
            return get_deepseek_llm(temperature=0.4)

        return base_llm

    def get_model_name(self, language: str | None = None) -> str:
        """Получить название используемой модели."""
        lang = language or self.default_language
        if lang.lower() in {"ru", "russian", "русский"}:
            return "GigaChat"
        return "DeepSeek"

    async def ainvoke(self, inputs: dict[str, Any]) -> str:
        """Совместимость с интерфейсом агентов LangChain."""
        request_type = inputs.get("task_type", "material")
        language = inputs.get("language", self.default_language)

        # Простой выбор модели и возврат JSON с результатом
        model_name = self.get_model_name(language)

        result = {
            "selected_model": model_name,
            "reasoning": f"Selected {model_name} for {request_type} in {language}",
            "confidence": 0.9,
            "alternative_models": [],
        }

        return json.dumps(result, ensure_ascii=False)

    async def generate_content(
        self,
        request_type: RequestType,
        content: str,
        language: str | None = None,
        system_prompt: str = "",
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Генерировать контент с помощью выбранной LLM."""
        lang = language or self.default_language
        llm = self.select_llm(lang, request_type)

        if not system_prompt:
            system_prompt = (
                "You are a helpful AI assistant specialized in algorithms and data structures."
            )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        chain = prompt | llm | StrOutputParser()

        result = await chain.ainvoke({"input": content, **(parameters or {})})

        return {
            "generated_content": result,
            "model_used": self.get_model_name(lang),
            "request_type": request_type,
            "language": lang,
        }
