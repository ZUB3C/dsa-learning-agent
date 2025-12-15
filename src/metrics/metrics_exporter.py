"""
Metrics exporter (Prometheus format).
"""

import logging

from src.metrics.metrics_collector import metrics_collector

logger = logging.getLogger(__name__)


class PrometheusExporter:
    """
    Export metrics in Prometheus format.
    """

    @staticmethod
    def export() -> str:
        """
        Export metrics as Prometheus text format.

        Returns:
            Prometheus metrics string
        """

        snapshot = metrics_collector.get_snapshot()

        lines = []

        # ───────────────────────────────────────────────────────────
        # Request metrics
        # ───────────────────────────────────────────────────────────

        lines.append("# HELP materials_agent_requests_total Total number of requests")
        lines.append("# TYPE materials_agent_requests_total counter")
        lines.append(f"materials_agent_requests_total {snapshot.total_requests}")
        lines.append("")

        lines.append("# HELP materials_agent_requests_successful_total Successful requests")
        lines.append("# TYPE materials_agent_requests_successful_total counter")
        lines.append(f"materials_agent_requests_successful_total {snapshot.successful_requests}")
        lines.append("")

        lines.append("# HELP materials_agent_requests_failed_total Failed requests")
        lines.append("# TYPE materials_agent_requests_failed_total counter")
        lines.append(f"materials_agent_requests_failed_total {snapshot.failed_requests}")
        lines.append("")

        # ───────────────────────────────────────────────────────────
        # Response time metrics
        # ───────────────────────────────────────────────────────────

        lines.append("# HELP materials_agent_response_time_avg Average response time (seconds)")
        lines.append("# TYPE materials_agent_response_time_avg gauge")
        lines.append(f"materials_agent_response_time_avg {snapshot.avg_response_time:.3f}")
        lines.append("")

        lines.append("# HELP materials_agent_response_time_min Minimum response time (seconds)")
        lines.append("# TYPE materials_agent_response_time_min gauge")
        lines.append(f"materials_agent_response_time_min {snapshot.min_response_time:.3f}")
        lines.append("")

        lines.append("# HELP materials_agent_response_time_max Maximum response time (seconds)")
        lines.append("# TYPE materials_agent_response_time_max gauge")
        lines.append(f"materials_agent_response_time_max {snapshot.max_response_time:.3f}")
        lines.append("")

        # ───────────────────────────────────────────────────────────
        # LLM metrics
        # ───────────────────────────────────────────────────────────

        lines.append("# HELP materials_agent_llm_calls_total LLM calls by model")
        lines.append("# TYPE materials_agent_llm_calls_total counter")
        lines.append(
            f'materials_agent_llm_calls_total{{model="gigachat2_max"}} {snapshot.gigachat2_calls}'
        )
        lines.append(
            f'materials_agent_llm_calls_total{{model="gigachat3"}} {snapshot.gigachat3_calls}'
        )
        lines.append("")

        lines.append("# HELP materials_agent_cost_total Total cost (USD)")
        lines.append("# TYPE materials_agent_cost_total counter")
        lines.append(f"materials_agent_cost_total {snapshot.total_cost:.4f}")
        lines.append("")

        # ───────────────────────────────────────────────────────────
        # Tool metrics
        # ───────────────────────────────────────────────────────────

        lines.append("# HELP materials_agent_tool_calls_total Tool calls by name")
        lines.append("# TYPE materials_agent_tool_calls_total counter")
        for tool, count in snapshot.tool_calls.items():
            lines.append(f'materials_agent_tool_calls_total{{tool="{tool}"}} {count}')
        lines.append("")

        # ───────────────────────────────────────────────────────────
        # Content Guard metrics
        # ───────────────────────────────────────────────────────────

        lines.append(
            "# HELP materials_agent_documents_filtered_total Documents filtered by Content Guard"
        )
        lines.append("# TYPE materials_agent_documents_filtered_total counter")
        lines.append(f"materials_agent_documents_filtered_total {snapshot.documents_filtered}")
        lines.append("")

        # ───────────────────────────────────────────────────────────
        # Uptime
        # ───────────────────────────────────────────────────────────

        uptime = metrics_collector.get_uptime()
        lines.append("# HELP materials_agent_uptime_seconds Uptime in seconds")
        lines.append("# TYPE materials_agent_uptime_seconds counter")
        lines.append(f"materials_agent_uptime_seconds {uptime:.0f}")
        lines.append("")

        return "\n".join(lines)
