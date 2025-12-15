"""
Real-time metrics collector.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class MetricsSnapshot:
    """Snapshot of current metrics."""

    timestamp: datetime = field(default_factory=datetime.now)

    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Performance metrics
    avg_response_time: float = 0.0
    min_response_time: float = float("inf")
    max_response_time: float = 0.0

    # LLM metrics
    gigachat2_calls: int = 0
    gigachat3_calls: int = 0
    total_cost: float = 0.0

    # Tool metrics
    tool_calls: dict[str, int] = field(default_factory=dict)

    # Content Guard metrics
    documents_filtered: int = 0
    toxicity_checks: int = 0


class MetricsCollector:
    """
    Collect real-time metrics.

    Thread-safe singleton for collecting metrics during runtime.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._initialized = True
        self.current = MetricsSnapshot()
        self.start_time = time.time()

        logger.info("📊 Metrics collector initialized")

    def record_request(self, success: bool, response_time: float) -> None:
        """
        Record a request.

        Args:
            success: Whether request was successful
            response_time: Response time in seconds
        """

        with self._lock:
            self.current.total_requests += 1

            if success:
                self.current.successful_requests += 1
            else:
                self.current.failed_requests += 1

            # Update response time stats
            self.current.min_response_time = min(self.current.min_response_time, response_time)
            self.current.max_response_time = max(self.current.max_response_time, response_time)

            # Update average
            total_time = (
                self.current.avg_response_time * (self.current.total_requests - 1) + response_time
            )
            self.current.avg_response_time = total_time / self.current.total_requests

    def record_llm_call(self, model: str) -> None:
        """
        Record LLM call.

        Args:
            model: Model name (gigachat2_max, gigachat3)
        """

        with self._lock:
            if model == "gigachat2_max":
                self.current.gigachat2_calls += 1
                self.current.total_cost += 0.002
            elif model == "gigachat3":
                self.current.gigachat3_calls += 1
                self.current.total_cost += 0.0005

    def record_tool_call(self, tool_name: str) -> None:
        """
        Record tool call.

        Args:
            tool_name: Tool name
        """

        with self._lock:
            self.current.tool_calls[tool_name] = self.current.tool_calls.get(tool_name, 0) + 1

    def record_content_guard(self, filtered: int, toxicity_checks: int) -> None:
        """
        Record Content Guard activity.

        Args:
            filtered: Number of documents filtered
            toxicity_checks: Number of toxicity checks performed
        """

        with self._lock:
            self.current.documents_filtered += filtered
            self.current.toxicity_checks += toxicity_checks

    def get_snapshot(self) -> MetricsSnapshot:
        """Get current metrics snapshot."""

        with self._lock:
            # Create a copy
            return MetricsSnapshot(
                timestamp=datetime.now(),
                total_requests=self.current.total_requests,
                successful_requests=self.current.successful_requests,
                failed_requests=self.current.failed_requests,
                avg_response_time=self.current.avg_response_time,
                min_response_time=self.current.min_response_time,
                max_response_time=self.current.max_response_time,
                gigachat2_calls=self.current.gigachat2_calls,
                gigachat3_calls=self.current.gigachat3_calls,
                total_cost=self.current.total_cost,
                tool_calls=self.current.tool_calls.copy(),
                documents_filtered=self.current.documents_filtered,
                toxicity_checks=self.current.toxicity_checks,
            )

    def reset(self) -> None:
        """Reset all metrics."""

        with self._lock:
            self.current = MetricsSnapshot()
            self.start_time = time.time()
            logger.info("🔄 Metrics reset")

    def get_uptime(self) -> float:
        """Get uptime in seconds."""
        return time.time() - self.start_time


# Global instance
metrics_collector = MetricsCollector()
