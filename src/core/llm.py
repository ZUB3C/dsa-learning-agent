from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from ..config import settings


def get_gigachat_llm(
    model: str | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
) -> BaseChatModel:
    """Получить LLM для GigaChat (русский язык)."""
    return ChatGoogleGenerativeAI(
        api_key=settings.gigachat_api_key or None,
        model=model or settings.gigachat_model,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        timeout=timeout or settings.timeout_s,
    )


def get_deepseek_llm(
    model: str | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
) -> BaseChatModel:
    """Получить LLM для DeepSeek (английский язык)."""
    return ChatOpenAI(
        api_key=settings.deepseek_api_key or None,
        model=model or settings.deepseek_model,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        timeout=timeout or settings.timeout_s,
        base_url=settings.deepseek_base_url or None,
    )


def get_llm_by_language(language: str) -> BaseChatModel:
    """Получить подходящую LLM в зависимости от языка."""
    # if language.lower() in {"ru", "russian", "русский"}:
    #     return get_gigachat_llm()
    # if language.lower() in {"en", "english", "английский"}:
    #     return get_deepseek_llm()
    # # По умолчанию русский
    return get_gigachat_llm()


def simple_chain(system_msg: str, language: str = "ru") -> ChatPromptTemplate:
    """Создать простую цепочку с системным промптом."""
    prompt = ChatPromptTemplate.from_messages([("system", system_msg), ("human", "{input}")])
    llm = get_llm_by_language(language)
    return prompt | llm | StrOutputParser()
