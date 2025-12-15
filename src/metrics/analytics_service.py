"""
Analytics service for ToT performance analysis.
"""

import logging
import operator
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import MaterialGeneration, ToTNodeLog

logger = logging.getLogger(__name__)


class AnalyticsService:
    """
    Service for analyzing ToT performance.
    """

    @staticmethod
    async def get_generation_statistics(
        db: AsyncSession, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> dict[str, Any]:
        """
        Get aggregated generation statistics.

        Args:
            db: Database session
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Dict with statistics
        """

        # Default to last 30 days
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()

        # Query generations
        query = select(MaterialGeneration).where(
            MaterialGeneration.created_at >= start_date, MaterialGeneration.created_at <= end_date
        )

        result = await db.execute(query)
        generations = result.scalars().all()

        if not generations:
            return {"total": 0, "message": "No data in date range"}

        # Calculate statistics
        total = len(generations)
        successful = sum(1 for g in generations if g.success)

        return {
            "total_generations": total,
            "successful_generations": successful,
            "failed_generations": total - successful,
            "success_rate": round(successful / total, 3) if total > 0 else 0,
            # Completeness
            "avg_completeness": round(
                sum(g.final_completeness_score for g in generations) / total, 3
            ),
            "min_completeness": min(g.final_completeness_score for g in generations),
            "max_completeness": max(g.final_completeness_score for g in generations),
            # Iterations
            "avg_iterations": round(sum(g.tot_iterations for g in generations) / total, 2),
            "min_iterations": min(g.tot_iterations for g in generations),
            "max_iterations": max(g.tot_iterations for g in generations),
            # Time
            "avg_generation_time": round(
                sum(g.generation_time_seconds for g in generations) / total, 2
            ),
            "min_generation_time": round(min(g.generation_time_seconds for g in generations), 2),
            "max_generation_time": round(max(g.generation_time_seconds for g in generations), 2),
            # Cost
            "total_cost_usd": round(sum(g.estimated_cost_usd for g in generations), 2),
            "avg_cost_per_generation": round(
                sum(g.estimated_cost_usd for g in generations) / total, 4
            ),
            # LLM usage
            "total_gigachat2_calls": sum(g.gigachat2_max_calls for g in generations),
            "total_gigachat3_calls": sum(g.gigachat3_calls for g in generations),
            # Documents
            "avg_documents_collected": round(
                sum(g.documents_collected for g in generations) / total, 1
            ),
        }

    @staticmethod
    async def get_tool_statistics(
        db: AsyncSession, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> dict[str, Any]:
        """
        Get tool usage statistics.

        Args:
            db: Database session
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Dict with tool statistics
        """

        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()

        query = select(MaterialGeneration).where(
            MaterialGeneration.created_at >= start_date, MaterialGeneration.created_at <= end_date
        )

        result = await db.execute(query)
        generations = result.scalars().all()

        # Aggregate tool usage
        tool_counts = {}
        tool_successes = {}

        for gen in generations:
            for tool, count in gen.tool_call_counts.items():
                tool_counts[tool] = tool_counts.get(tool, 0) + count
                if gen.success:
                    tool_successes[tool] = tool_successes.get(tool, 0) + count

        # Build statistics
        tool_stats = {}

        for tool, count in tool_counts.items():
            successes = tool_successes.get(tool, 0)
            tool_stats[tool] = {
                "total_calls": count,
                "successful_calls": successes,
                "success_rate": round(successes / count, 3) if count > 0 else 0,
            }

        return {
            "tools": tool_stats,
            "most_used_tool": max(tool_counts.items(), key=operator.itemgetter(1))[0]
            if tool_counts
            else None,
        }

    @staticmethod
    async def get_node_statistics(db: AsyncSession, generation_id: str) -> dict[str, Any]:
        """
        Get node-level statistics for a generation.

        Args:
            db: Database session
            generation_id: Generation ID

        Returns:
            Dict with node statistics
        """

        query = select(ToTNodeLog).where(ToTNodeLog.generation_id == generation_id)

        result = await db.execute(query)
        nodes = result.scalars().all()

        if not nodes:
            return {"message": "No nodes found"}

        # Calculate statistics
        total_nodes = len(nodes)

        status_counts = {}
        for node in nodes:
            status_counts[node.status] = status_counts.get(node.status, 0) + 1

        return {
            "total_nodes": total_nodes,
            "status_distribution": status_counts,
            "avg_promise_score": round(sum(n.promise_score for n in nodes) / total_nodes, 3),
            "avg_completeness": round(sum(n.completeness_score for n in nodes) / total_nodes, 3),
            "avg_relevance": round(sum(n.relevance_score for n in nodes) / total_nodes, 3),
            "avg_quality": round(sum(n.quality_score for n in nodes) / total_nodes, 3),
            "avg_execution_time_ms": round(
                sum(n.execution_time_ms for n in nodes) / total_nodes, 2
            ),
            "max_depth_reached": max(n.depth for n in nodes),
        }
