"""
Database logging handler for storing application logs in PostgreSQL/SQLite.
"""

import logging
import traceback
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import JSON, Boolean, Column, DateTime, Integer, String, Text

from src.core.database import AsyncSessionLocal, Base

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class ApplicationLog(Base):
    """Application logs table."""

    __tablename__ = "application_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.now, nullable=False, index=True)
    level = Column(String(10), nullable=False, index=True)  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    logger_name = Column(String(200), nullable=False, index=True)
    module = Column(String(200), nullable=True)
    function_name = Column(String(200), nullable=True)
    line_number = Column(Integer, nullable=True)

    message = Column(Text, nullable=False)
    exception_type = Column(String(200), nullable=True)
    exception_message = Column(Text, nullable=True)
    stack_trace = Column(Text, nullable=True)

    # Context information
    user_id = Column(String(100), nullable=True, index=True)
    session_id = Column(String(100), nullable=True, index=True)
    request_id = Column(String(100), nullable=True, index=True)

    # Additional metadata
    extra_data = Column(JSON, nullable=True)

    # Performance tracking
    execution_time_ms = Column(Integer, nullable=True)
    memory_usage_mb = Column(Integer, nullable=True)


class LLMCallLog(Base):
    """LLM API calls logging."""

    __tablename__ = "llm_call_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.now, nullable=False, index=True)

    # Request info
    model_name = Column(String(100), nullable=False, index=True)
    task_type = Column(String(50), nullable=True, index=True)
    prompt = Column(Text, nullable=False)
    prompt_tokens = Column(Integer, nullable=True)

    # Response info
    response = Column(Text, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)

    # Performance
    latency_ms = Column(Integer, nullable=False)
    success = Column(Boolean, nullable=False, index=True)
    error_message = Column(Text, nullable=True)

    # Cost tracking
    estimated_cost_usd = Column(Integer, nullable=True)  # in cents

    # Context
    user_id = Column(String(100), nullable=True, index=True)
    session_id = Column(String(100), nullable=True, index=True)
    generation_id = Column(String(100), nullable=True, index=True)


class ToolExecutionLog(Base):
    """Tool execution logs."""

    __tablename__ = "tool_execution_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.now, nullable=False, index=True)

    tool_name = Column(String(100), nullable=False, index=True)
    tool_params = Column(JSON, nullable=True)

    execution_time_ms = Column(Integer, nullable=False)
    success = Column(Boolean, nullable=False, index=True)
    error_message = Column(Text, nullable=True)

    # Results
    documents_retrieved = Column(Integer, default=0)
    result_summary = Column(Text, nullable=True)

    # Context
    node_id = Column(String(100), nullable=True, index=True)
    generation_id = Column(String(100), nullable=True, index=True)
    session_id = Column(String(100), nullable=True, index=True)


class DatabaseHandler(logging.Handler):
    """Async logging handler that saves logs to database."""

    def __init__(self, level=logging.INFO) -> None:
        super().__init__(level)
        self._session: AsyncSession | None = None
        self._context = {}

    def set_context(self, **kwargs) -> None:
        """Set context variables (user_id, session_id, etc.)."""
        self._context.update(kwargs)

    def clear_context(self) -> None:
        """Clear context variables."""
        self._context.clear()

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a record to database.
        This method is called synchronously, so we queue it for async processing.
        """
        try:
            # Import here to avoid circular dependency
            import asyncio

            # Try to get current event loop
            try:
                loop = asyncio.get_running_loop()
                # Schedule the async task
                loop.create_task(self._async_emit(record))
            except RuntimeError:
                # No event loop running, create new one (fallback)
                asyncio.run(self._async_emit(record))

        except Exception:
            self.handleError(record)

    async def _async_emit(self, record: logging.LogRecord) -> None:
        """Async method to save log to database."""
        try:
            async with AsyncSessionLocal() as session:
                log_entry = ApplicationLog(
                    timestamp=datetime.fromtimestamp(record.created),
                    level=record.levelname,
                    logger_name=record.name,
                    module=record.module,
                    function_name=record.funcName,
                    line_number=record.lineno,
                    message=record.getMessage(),
                    user_id=self._context.get("user_id"),
                    session_id=self._context.get("session_id"),
                    request_id=self._context.get("request_id"),
                )

                # Add exception info if present
                if record.exc_info:
                    exc_type, exc_value, _exc_tb = record.exc_info
                    log_entry.exception_type = exc_type.__name__ if exc_type else None
                    log_entry.exception_message = str(exc_value) if exc_value else None
                    log_entry.stack_trace = "".join(traceback.format_exception(*record.exc_info))

                # Add extra data if present
                extra_data = {
                    key: value
                    for key, value in record.__dict__.items()
                    if key
                    not in {
                        "name",
                        "msg",
                        "args",
                        "created",
                        "filename",
                        "funcName",
                        "levelname",
                        "lineno",
                        "module",
                        "msecs",
                        "message",
                        "pathname",
                        "process",
                        "processName",
                        "relativeCreated",
                        "thread",
                        "threadName",
                        "exc_info",
                        "exc_text",
                        "stack_info",
                    }
                }

                if extra_data:
                    log_entry.extra_data = extra_data

                session.add(log_entry)
                await session.commit()

        except Exception as e:
            # Don't raise exceptions from logging handler
            print(f"Error saving log to database: {e}")


class LLMCallLogger:
    """Helper class for logging LLM calls."""

    @staticmethod
    async def log_call(
        model_name: str,
        task_type: str | None,
        prompt: str,
        response: str | None,
        latency_ms: int,
        success: bool,
        error_message: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        estimated_cost_usd: float | None = None,
        **context,
    ) -> None:
        """Log LLM API call to database."""
        try:
            async with AsyncSessionLocal() as session:
                log_entry = LLMCallLog(
                    model_name=model_name,
                    task_type=task_type,
                    prompt=prompt[:10000],  # Limit prompt size
                    response=response[:10000] if response else None,
                    latency_ms=latency_ms,
                    success=success,
                    error_message=error_message,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    estimated_cost_usd=int(estimated_cost_usd * 100)
                    if estimated_cost_usd
                    else None,
                    user_id=context.get("user_id"),
                    session_id=context.get("session_id"),
                    generation_id=context.get("generation_id"),
                )

                session.add(log_entry)
                await session.commit()

        except Exception as e:
            print(f"Error logging LLM call: {e}")


class ToolExecutionLogger:
    """Helper class for logging tool executions."""

    @staticmethod
    async def log_execution(
        tool_name: str,
        tool_params: dict,
        execution_time_ms: int,
        success: bool,
        error_message: str | None = None,
        documents_retrieved: int = 0,
        result_summary: str | None = None,
        **context,
    ) -> None:
        """Log tool execution to database."""
        try:
            async with AsyncSessionLocal() as session:
                log_entry = ToolExecutionLog(
                    tool_name=tool_name,
                    tool_params=tool_params,
                    execution_time_ms=execution_time_ms,
                    success=success,
                    error_message=error_message,
                    documents_retrieved=documents_retrieved,
                    result_summary=result_summary[:1000] if result_summary else None,
                    node_id=context.get("node_id"),
                    generation_id=context.get("generation_id"),
                    session_id=context.get("session_id"),
                )

                session.add(log_entry)
                await session.commit()

        except Exception as e:
            print(f"Error logging tool execution: {e}")


# Global database handler instance
_db_handler: DatabaseHandler | None = None


def setup_database_logging(level=logging.INFO):
    """Setup database logging for the application."""
    global _db_handler

    if _db_handler is None:
        _db_handler = DatabaseHandler(level=level)

        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(_db_handler)

    return _db_handler


def get_db_handler() -> DatabaseHandler | None:
    """Get the global database handler instance."""
    return _db_handler
