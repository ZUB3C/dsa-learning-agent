from collections.abc import Callable
from typing import Any

from . import (
    llm_router_agent,
    materials_agent,
    support_agent,
    test_generation_agent,
    verification_agent,
)

_REGISTRY: dict[str, Callable[..., Any]] = {
    "verification": verification_agent.build_verification_agent,
    "verification-secondary": verification_agent.build_secondary_verification_agent,
    "materials": materials_agent.build_materials_agent,
    "question-answering": materials_agent.build_question_answering_agent,
    "test-generation": test_generation_agent.build_test_generation_agent,
    "llm-router": llm_router_agent.build_router_agent,
    "support": support_agent.build_support_agent,
}


def list_agents() -> list[str]:
    """Список доступных агентов."""
    return sorted(_REGISTRY.keys())


def load_agent(name: str, **kwargs: Any) -> Any:
    """Загрузить агент по имени."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown agent '{name}'. Available: {list_agents()}")
    return _REGISTRY[name](**kwargs)
