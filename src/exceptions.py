"""
Custom exceptions for Materials Agent v2.
"""


class MaterialsAgentException(Exception):
    """Base exception for Materials Agent."""


class ConfigurationError(MaterialsAgentException):
    """Configuration-related errors."""


class LLMUnavailableError(MaterialsAgentException):
    """LLM service unavailable."""

    def __init__(self, model: str, message: str | None = None) -> None:
        self.model = model
        self.message = message or f"LLM model {model} is unavailable"
        super().__init__(self.message)


class ToolExecutionError(MaterialsAgentException):
    """Tool execution failed."""

    def __init__(self, tool_name: str, error: str, retry_count: int = 0) -> None:
        self.tool_name = tool_name
        self.error = error
        self.retry_count = retry_count
        super().__init__(f"Tool {tool_name} failed: {error} (retries: {retry_count})")


class ContentGuardFilteredError(MaterialsAgentException):
    """Content filtered by Content Guard."""

    def __init__(self, reason: str, documents_count: int) -> None:
        self.reason = reason
        self.documents_count = documents_count
        super().__init__(f"All {documents_count} documents filtered: {reason}")


class InvalidInputError(MaterialsAgentException):
    """Invalid user input."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(f"Invalid input: {message}")


class PromptInjectionError(InvalidInputError):
    """Prompt injection detected."""

    def __init__(self, detected_pattern: str) -> None:
        super().__init__(f"Prompt injection detected: {detected_pattern}")


class DatabaseError(MaterialsAgentException):
    """Database operation failed."""

    def __init__(self, operation: str, error: str) -> None:
        self.operation = operation
        self.error = error
        super().__init__(f"Database {operation} failed: {error}")


class ChromaDBUnavailableError(MaterialsAgentException):
    """ChromaDB service unavailable."""

    def __init__(self, message: str | None = None) -> None:
        self.message = message or "ChromaDB is unavailable, using fallback storage"
        super().__init__(self.message)


class ToTSearchError(MaterialsAgentException):
    """Tree-of-Thoughts search failed."""

    def __init__(self, reason: str, depth: int = 0) -> None:
        self.reason = reason
        self.depth = depth
        super().__init__(f"ToT search failed at depth {depth}: {reason}")


class WebSearchUnavailableError(MaterialsAgentException):
    """Web search service unavailable."""

    def __init__(self, service: str = "4get") -> None:
        self.service = service
        super().__init__(f"Web search service {service} unavailable")


class TimeoutError(MaterialsAgentException):
    """Operation timeout."""

    def __init__(self, operation: str, timeout_seconds: float) -> None:
        self.operation = operation
        self.timeout = timeout_seconds
        super().__init__(f"Operation '{operation}' timed out after {timeout_seconds}s")


class MemoryError(MaterialsAgentException):
    """Memory operation failed."""

    def __init__(self, operation: str, error: str) -> None:
        self.operation = operation
        self.error = error
        super().__init__(f"Memory {operation} failed: {error}")


class FallbackExhaustedError(MaterialsAgentException):
    """All fallback options exhausted."""

    def __init__(self, component: str, attempted_fallbacks: list) -> None:
        self.component = component
        self.fallbacks = attempted_fallbacks
        super().__init__(
            f"All fallback options exhausted for {component}: {', '.join(attempted_fallbacks)}"
        )
