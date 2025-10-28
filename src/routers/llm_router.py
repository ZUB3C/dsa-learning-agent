from typing import Any

from fastapi import APIRouter, HTTPException

from ..agents.llm_router_agent import LLMRouter
from ..agents.registry import load_agent
from ..models.schemas import LLMRouterRequest, LLMRouterResponse

router = APIRouter(prefix="/api/v1/llm-router", tags=["LLM Router"])


@router.post("/select-and-generate")
async def select_and_generate(request: LLMRouterRequest) -> LLMRouterResponse:
    """Выбрать LLM и сгенерировать контент"""

    try:
        router_agent = load_agent("llm-router")

        # Определяем системный промпт в зависимости от типа запроса
        system_prompts = {
            "material": "Ты - преподаватель. Создай учебный материал по теме.",
            "task": "Ты - создатель задач. Создай практическую задачу.",
            "test": "Ты - эксперт по тестированию. Создай тестовое задание.",
            "question": "Ты - преподаватель. Ответь на вопрос студента.",
            "support": "Ты - помощник. Окажи психологическую поддержку."
        }

        system_prompt = system_prompts.get(request.request_type, "Ты - помощник.")

        result = await router_agent.generate_content(
            request_type=request.request_type,
            content=request.content,
            language=request.language,
            system_prompt=system_prompt,
            parameters=request.parameters
        )

        return LLMRouterResponse(
            generated_content=result["generated_content"],
            model_used=result["model_used"],
            metadata={
                "request_type": result["request_type"],
                "language": result["language"]
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in LLM routing: {e!s}")


@router.get("/available-models")
async def get_available_models() -> dict[str, Any]:
    """Получить список доступных моделей"""
    return {
        "models": [
            {"name": "GigaChat", "language": "ru", "provider": "Sber"},
            {"name": "DeepSeek", "language": "en", "provider": "DeepSeek"}
        ],
        "capabilities": {
            "material_generation": True,
            "task_generation": True,
            "test_generation": True,
            "question_answering": True
        }
    }


@router.post("/route-request")
async def route_request(
        request_type: str,
        content: str,
        context: dict[str, Any] | None = None,
        language: str = "ru"
) -> dict[str, Any]:
    """Маршрутизация запроса к подходящей модели"""

    try:
        import json

        router_instance = LLMRouter(language=language)

        result = await router_instance.ainvoke({
            "request_type": request_type,
            "content": content,
            "context": json.dumps(context or {}),
            "language": language
        })

        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {
                "selected_model": router_instance.get_model_name(language),
                "reasoning": "Default model selection",
                "confidence": 0.5,
                "alternative_models": []
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error routing request: {e!s}")
