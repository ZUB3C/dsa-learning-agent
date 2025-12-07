from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..config import settings


from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..config import settings


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
) -> BaseChatModel:
    """LLM GigaChat (единый провайдер)."""
    return ChatOpenAI(
        api_key=settings.gigachat_api_key,
        model=model or settings.gigachat_model,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        timeout=timeout or settings.timeout_s,
        base_url=settings.gigachat_base_url,
    )


def simple_chain(system_msg: str):
    """Простая цепочка: prompt | llm | parser."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "{input}"),
    ])
    llm = get_llm()
    return prompt | llm | StrOutputParser()
