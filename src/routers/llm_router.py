
import json

from fastapi import APIRouter, HTTPException

from ..agents.llm_router_agent import LLMRouter
from ..agents.registry import load_agent
from ..models.schemas import (
    GetAvailableModelsResponse,
    LLMRouterRequest,
    LLMRouterResponse,
    ModelInfo,
    RouteRequestRequest,
    RouteRequestResponse,
)

router = APIRouter(prefix="/api/v1/llm-router", tags=["LLM Router"])


@router.post("/select-and-generate")
async def select_and_generate(request: LLMRouterRequest) -> LLMRouterResponse:
    """Выбрать подходящую модель и сгенерировать контент"""

    try:
        # Создаем роутер для выбора модели
        router_agent = LLMRouter(language=request.language)
        selected_model = router_agent.get_model_name(request.language)

        # Загружаем соответствующего агента
        agent_map = {
            "material": "materials",
            "task": "test-generation",
            "test": "test-generation",
            "question": "materials",
            "support": "support"
        }

        agent_type = agent_map.get(request.request_type, "materials")
        agent = load_agent(agent_type, language=request.language)

        # Генерируем контент
        result = await agent.ainvoke({
            "content": request.content,
            **request.parameters
        })

        return LLMRouterResponse(
            generated_content=result,
            model_used=selected_model,
            metadata={
                "request_type": request.request_type,
                "agent_type": agent_type
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {e!s}")


@router.get("/available-models")
async def get_available_models() -> GetAvailableModelsResponse:
    """Получить список доступных моделей"""
    return GetAvailableModelsResponse(
        models=[
            ModelInfo(name="GigaChat", language="ru", provider="Sber"),
            ModelInfo(name="DeepSeek", language="en", provider="DeepSeek"),
        ],
        capabilities={
            "material_generation": True,
            "task_generation": True,
            "test_generation": True,
            "question_answering": True
        }
    )


@router.post("/route-request")
async def route_request(request: RouteRequestRequest) -> RouteRequestResponse:
    """Маршрутизация запроса к подходящей модели"""

    try:
        router_instance = LLMRouter(language=request.language)

        result = await router_instance.ainvoke({
            "request_type": request.request_type,
            "content": request.content,
            "context": json.dumps(request.context or {}),
            "language": request.language
        })

        try:
            parsed_result = json.loads(result)
            return RouteRequestResponse(**parsed_result)
        except json.JSONDecodeError:
            return RouteRequestResponse(
                selected_model=router_instance.get_model_name(request.language),
                reasoning="Default model selection",
                confidence=0.5,
                alternative_models=[]
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error routing request: {e!s}")
