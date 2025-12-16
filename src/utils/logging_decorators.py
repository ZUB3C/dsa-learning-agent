"""
Decorators for automatic logging of functions and methods.
"""

import asyncio
import functools
import logging
import time
from collections.abc import Callable

from src.core.logging_handler import LLMCallLogger, ToolExecutionLogger

logger = logging.getLogger(__name__)


def log_function_call(logger_name: str | None = None):
    """
    Decorator to automatically log function calls with parameters and results.

    Usage:
        @log_function_call("my_module")
        async def my_function(arg1, arg2):
            return result
    """

    def decorator(func: Callable) -> Callable:
        func_logger = logging.getLogger(logger_name or func.__module__)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            func_name = func.__qualname__

            func_logger.info(
                f"Calling {func_name}",
                extra={
                    "function": func_name,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()),
                },
            )

            try:
                result = await func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000

                func_logger.info(
                    f"Completed {func_name} in {execution_time:.2f}ms",
                    extra={
                        "function": func_name,
                        "execution_time_ms": int(execution_time),
                        "success": True,
                    },
                )

                return result

            except Exception as e:
                execution_time = (time.time() - start_time) * 1000

                func_logger.error(
                    f"Error in {func_name}: {e!s}",
                    exc_info=True,
                    extra={
                        "function": func_name,
                        "execution_time_ms": int(execution_time),
                        "success": False,
                        "error_type": type(e).__name__,
                    },
                )
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            func_name = func.__qualname__

            func_logger.info(
                f"Calling {func_name}",
                extra={
                    "function": func_name,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()),
                },
            )

            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000

                func_logger.info(
                    f"Completed {func_name} in {execution_time:.2f}ms",
                    extra={
                        "function": func_name,
                        "execution_time_ms": int(execution_time),
                        "success": True,
                    },
                )

                return result

            except Exception as e:
                execution_time = (time.time() - start_time) * 1000

                func_logger.error(
                    f"Error in {func_name}: {e!s}",
                    exc_info=True,
                    extra={
                        "function": func_name,
                        "execution_time_ms": int(execution_time),
                        "success": False,
                        "error_type": type(e).__name__,
                    },
                )
                raise

        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def log_llm_call(task_type: str | None = None):
    """
    Decorator to automatically log LLM API calls.

    Usage:
        @log_llm_call("thought_generation")
        async def call_llm(prompt, model_name, **kwargs):
            # Should return response object with .content attribute
            return response
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()

            # Extract parameters
            prompt = kwargs.get("prompt", args[0] if args else "")
            model_name = kwargs.get("model_name", "unknown")
            context = kwargs.get("context", {})

            try:
                response = await func(*args, **kwargs)
                latency_ms = int((time.time() - start_time) * 1000)

                # Extract response content
                response_text = getattr(response, "content", str(response))

                # Log to database
                await LLMCallLogger.log_call(
                    model_name=model_name,
                    task_type=task_type,
                    prompt=str(prompt)[:10000],
                    response=str(response_text)[:10000],
                    latency_ms=latency_ms,
                    success=True,
                    **context,
                )

                return response

            except Exception as e:
                latency_ms = int((time.time() - start_time) * 1000)

                await LLMCallLogger.log_call(
                    model_name=model_name,
                    task_type=task_type,
                    prompt=str(prompt)[:10000],
                    response=None,
                    latency_ms=latency_ms,
                    success=False,
                    error_message=str(e),
                    **context,
                )
                raise

        return wrapper

    return decorator


def log_tool_execution():
    """
    Decorator to automatically log tool executions.

    Usage:
        @log_tool_execution()
        async def execute_tool(tool_name, tool_params, **context):
            # Should return ToolResult object
            return result
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()

            # Extract parameters
            tool_name = kwargs.get("tool_name", args[0] if args else "unknown")
            tool_params = kwargs.get("tool_params", {})
            context = {
                k: v for k, v in kwargs.items() if k in {"node_id", "generation_id", "session_id"}
            }

            try:
                result = await func(*args, **kwargs)
                execution_time_ms = int((time.time() - start_time) * 1000)

                # Extract result info
                documents_retrieved = len(getattr(result, "documents", []))
                success = getattr(result, "success", True)
                error_msg = getattr(result, "error", None)

                # Log to database
                await ToolExecutionLogger.log_execution(
                    tool_name=tool_name,
                    tool_params=tool_params,
                    execution_time_ms=execution_time_ms,
                    success=success,
                    error_message=error_msg,
                    documents_retrieved=documents_retrieved,
                    **context,
                )

                return result

            except Exception as e:
                execution_time_ms = int((time.time() - start_time) * 1000)

                await ToolExecutionLogger.log_execution(
                    tool_name=tool_name,
                    tool_params=tool_params,
                    execution_time_ms=execution_time_ms,
                    success=False,
                    error_message=str(e),
                    **context,
                )
                raise

        return wrapper

    return decorator
