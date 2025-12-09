from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..config import settings


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
    use_gigachat3: bool = False,
) -> BaseChatModel:
    """LLM GigaChat (единый провайдер с поддержкой двух моделей)."""
    # Выбираем модель: GigaChat3 если указано, иначе основную
    selected_model = None
    if use_gigachat3:
        selected_model = settings.gigachat3_model
    elif model:
        selected_model = model
    else:
        selected_model = settings.gigachat_model

    return ChatOpenAI(
        api_key=settings.gigachat_api_key,
        model=selected_model,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        timeout=timeout or settings.timeout_s,
        base_url=settings.gigachat_base_url,
    )


def simple_chain(system_msg: str, use_gigachat3: bool = False):  # noqa: ANN201
    """Простая цепочка: prompt | llm | parser."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "{input}"),
    ])
    llm = get_llm(use_gigachat3=use_gigachat3)
    return prompt | llm | StrOutputParser()


# Удобные функции-обертки
def create_gigachat_chain(system_msg: str):
    """Создать цепь с основной моделью GigaChat."""
    return simple_chain(system_msg, use_gigachat3=False)


def create_gigachat3_chain(system_msg: str):
    """Создать цепь с моделью GigaChat3-10B-A1.8B."""
    return simple_chain(system_msg, use_gigachat3=True)
