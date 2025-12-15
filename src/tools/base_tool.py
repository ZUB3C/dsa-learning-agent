"""
Base class for all tools used in Materials Agent v2.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class ToolStatus(StrEnum):
    """Tool execution status."""

    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class Document:
    """
    Document result from tools.
    """

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    relevance_score: float = 1.0

    def __hash__(self):
        return hash(self.page_content[:100])

    def __eq__(self, other):
        if not isinstance(other, Document):
            return False
        return self.page_content[:100] == other.page_content[:100]


@dataclass
class ToolResult:
    """
    Result from tool execution.
    """

    success: bool
    documents: list[Document] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def __repr__(self) -> str:
        return (
            f"ToolResult(success={self.success}, docs={len(self.documents)}, "
            f"time={self.execution_time_ms}ms)"
        )


class BaseTool(ABC):
    """
    Abstract base class for all tools.
    """

    name: str
    description: str

    def __init__(self) -> None:
        """Initialize tool."""
        self.name = self.__class__.__name__
        self.description = self.__class__.__doc__ or ""
        self.last_execution_time_ms = 0.0

    @abstractmethod
    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """
        Execute tool with given parameters.

        Args:
            params: Tool-specific parameters

        Returns:
            ToolResult with documents or error
        """
        raise NotImplementedError

    async def validate_params(self, params: dict[str, Any]) -> bool:
        """
        Validate parameters before execution.

        Args:
            params: Parameters to validate

        Returns:
            True if valid, False otherwise
        """
        return True

    def __call__(self, *args, **kwargs):
        """Make tool callable."""
        msg = "Use async execute() method"
        raise NotImplementedError(msg)
