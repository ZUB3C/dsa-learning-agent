import json

from fastapi import APIRouter, HTTPException

from ..agents.llm_router_agent import LLMRouter
from ..agents.materials_agent import format_retrieved_materials, retrieve_materials
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
    """Выбрать подходящую модель и сгенерировать контент."""
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
            "support": "support",
        }

        agent_type = agent_map.get(request.request_type, "materials")
        agent = load_agent(agent_type, language=request.language)

        # Подготавливаем параметры для каждого типа агента
        agent_params = {}

        if agent_type == "materials":
            # Materials agent требует: topic, user_level, retrieved_materials
            # Question-answering требует дополнительно: question
            topic = request.parameters.get("topic", "")
            user_level = request.parameters.get(
                "user_level", request.parameters.get("user_level", "intermediate")
            )

            # Получаем материалы из RAG
            documents = retrieve_materials(topic=topic, user_level=user_level)
            retrieved_materials = format_retrieved_materials(documents)

            agent_params = {
                "topic": topic,
                "user_level": user_level,
                "retrieved_materials": retrieved_materials,
            }

            # Если это вопрос, добавляем параметр question
            if request.request_type == "question":
                agent_params["question"] = request.parameters.get("question", request.content)

        elif agent_type == "test-generation":
            # Test-generation agent требует: question_count, topic, difficulty
            agent_params = {
                "question_count": request.parameters.get(
                    "question_count", request.parameters.get("question_count", 5)
                ),
                "topic": request.parameters.get("topic", ""),
                "difficulty": request.parameters.get("difficulty", "medium"),
            }

            # Если это task, добавляем task_type
            if request.request_type == "task":
                agent_params["task_type"] = request.parameters.get(
                    "task_type", request.parameters.get("task_type", "coding")
                )

        elif agent_type == "support":
            # Support agent требует: emotional_state, message, user_id
            agent_params = {
                "emotional_state": request.parameters.get(
                    "emotional_state", request.parameters.get("emotional_state", "neutral")
                ),
                "message": request.parameters.get("message", request.content),
                "user_id": request.parameters.get(
                    "user_id", request.parameters.get("user_id", "anonymous")
                ),
            }

        # Генерируем контент
        result = await agent.ainvoke(agent_params)

        return LLMRouterResponse(
            generated_content=result,
            model_used=selected_model,
            metadata={
                "request_type": request.request_type,
                "agent_type": agent_type,
                "parameters_used": agent_params,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {e!s}")


@router.get("/available-models")
async def get_available_models() -> GetAvailableModelsResponse:
    """Получить список доступных LLM моделей."""
    models = [
        ModelInfo(name="GigaChat", language="ru", provider="Sber"),
        ModelInfo(name="DeepSeek", language="en", provider="DeepSeek"),
    ]

    capabilities = {
        "material_generation": True,
        "task_generation": True,
        "test_generation": True,
        "question_answering": True,
    }

    return GetAvailableModelsResponse(models=models, capabilities=capabilities)


@router.post("/route-request")
async def route_request(request: RouteRequestRequest) -> RouteRequestResponse:
    """Маршрутизация запроса к подходящей модели."""
    try:
        router_instance = LLMRouter(language=request.language)
        result = await router_instance.ainvoke({
            "request_type": request.request_type,
            "content": request.content,
            "context": json.dumps(request.context or {}),
            "language": request.language,
        })

        try:
            parsed_result = json.loads(result)
            return RouteRequestResponse(**parsed_result)
        except json.JSONDecodeError:
            return RouteRequestResponse(
                selected_model=router_instance.get_model_name(request.language),
                reasoning="Default model selection",
                confidence=0.5,
                alternative_models=[],
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error routing request: {e!s}")
