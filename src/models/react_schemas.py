"""
Pydantic schemas for ReAct and Tree-of-Thoughts.
Code from Section 4.1 of architecture.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from src.tools.base_tool import Document


class NodeStatus(StrEnum):
    """Status of a ToT node."""

    PENDING = "pending"  # Еще не выполнен
    EXECUTING = "executing"  # В процессе выполнения
    EXECUTED = "executed"  # Выполнен успешно
    PROMISING = "promising"  # Оценен как перспективный
    DEAD_END = "dead_end"  # Тупиковая ветвь
    GOAL_REACHED = "goal_reached"  # Достигнута цель


@dataclass
class TreeNode:
    """
    Узел дерева рассуждений.
    Code from Section 4.1 of architecture.
    """

    # Идентификация
    node_id: str = field(default_factory=lambda: f"node_{__import__('uuid').uuid4().hex[:8]}")
    parent_id: str | None = None
    depth: int = 0

    # Рассуждение (от GigaChat-2-Max)
    thought: str = ""  # Мысль агента на этом шаге
    reasoning: str = ""  # Объяснение выбора действия
    planned_action: dict[str, Any] = field(default_factory=dict)  # {tool_name, tool_params}

    # Результаты выполнения (от Tools)
    action_result: Any | None = None
    collected_info: list[Document] = field(default_factory=list)  # Аккумулятор документов

    # Оценки (от GigaChat3)
    promise_score: float = 0.0  # 0-1: перспективность ветви
    completeness_score: float = 0.0  # 0-1: полнота материала
    relevance_score: float = 0.0  # 0-1: релевантность собранного
    quality_score: float = 0.0  # 0-1: качество документов

    # Состояние
    status: NodeStatus = NodeStatus.PENDING
    children: list["TreeNode"] = field(default_factory=list)
    visited: bool = False

    # Метаданные
    created_at: datetime = field(default_factory=datetime.now)
    execution_time_ms: float = 0.0
    llm_calls: dict[str, int] = field(default_factory=dict)  # {"gigachat2": 1, "gigachat3": 3}

    def __hash__(self):
        return hash(self.node_id)

    def __eq__(self, other):
        if not isinstance(other, TreeNode):
            return False
        return self.node_id == other.node_id


class NodeEvaluation(BaseModel):
    """Post-execution evaluation result."""

    completeness: float = Field(..., ge=0.0, le=1.0, description="Completeness score")
    relevance: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    quality: float = Field(..., ge=0.0, le=1.0, description="Quality score")
    should_continue: bool = Field(..., description="Should continue search")


class ToTResult(BaseModel):
    """Final result from ToT search."""

    best_path: list[TreeNode] = Field(..., description="Best path from root to solution")
    explored_nodes: list[TreeNode] = Field(..., description="All explored nodes")
    collected_documents: list[Document] = Field(..., description="All collected documents")
    final_completeness: float = Field(..., ge=0.0, le=1.0, description="Final completeness")
    iterations: int = Field(..., ge=0, description="Number of iterations")

    # Metrics
    tools_used: list[str] = Field(default_factory=list, description="Tools used in search")
    total_time: float = Field(0.0, description="Total search time in seconds")
    llm_usage: dict[str, int] = Field(default_factory=dict, description="LLM calls by model")

    class Config:
        arbitrary_types_allowed = True


class MemoryContext(BaseModel):
    """Context loaded from memory for current request."""

    session_id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User ID")
    procedural_hints: str = Field(default="", description="Procedural memory hints")
    patterns: list[dict[str, Any]] = Field(default_factory=list, description="Loaded patterns")


class ThoughtCandidate(BaseModel):
    """Candidate thought from LLM."""

    reasoning: str = Field(..., description="Reasoning for this thought")
    tool_name: str = Field(..., description="Tool to use")
    tool_params: dict[str, Any] = Field(..., description="Tool parameters")
    explanation: str = Field(default="", description="Why this thought is promising")


class GeneratedThoughts(BaseModel):
    """Response from thought generation."""

    thoughts: list[ThoughtCandidate] = Field(..., description="Candidate thoughts")
    meta: dict[str, Any] = Field(default_factory=dict, description="Metadata")
