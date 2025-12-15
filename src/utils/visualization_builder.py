"""
Visualization builder for ToT analytics.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class VisualizationBuilder:
    """
    Build visualization data for ToT analytics.

    Generates JSON data for frontend charting libraries (Chart.js, Plotly, etc.)
    """

    @staticmethod
    def build_tot_tree_visualization(tot_result: Any) -> dict[str, Any]:
        """
        Build tree visualization data from ToT result.

        Args:
            tot_result: ToTResult object

        Returns:
            Dict with tree visualization data
        """

        nodes = []
        edges = []

        for node in tot_result.explored_nodes:
            nodes.append({
                "id": node.node_id,
                "label": node.thought[:50] + "..." if len(node.thought) > 50 else node.thought,
                "depth": node.depth,
                "completeness": node.completeness_score,
                "promise": node.promise_score,
                "status": node.status.value if hasattr(node.status, "value") else str(node.status),
                "tool": node.planned_action.get("tool_name") if node.planned_action else None,
            })

            if node.parent_id:
                edges.append({"from": node.parent_id, "to": node.node_id})

        return {
            "nodes": nodes,
            "edges": edges,
            "best_path": [n.node_id for n in tot_result.best_path],
        }

    @staticmethod
    def build_metrics_timeline(generations: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Build metrics timeline from generation logs.

        Args:
            generations: List of generation records

        Returns:
            Dict with timeline data
        """

        # Sort by date
        generations = sorted(generations, key=lambda g: g.get("created_at", ""))

        timeline = {"dates": [], "completeness": [], "iterations": [], "cost": [], "time": []}

        for gen in generations:
            timeline["dates"].append(gen.get("created_at", ""))
            timeline["completeness"].append(gen.get("final_completeness_score", 0))
            timeline["iterations"].append(gen.get("tot_iterations", 0))
            timeline["cost"].append(gen.get("estimated_cost_usd", 0))
            timeline["time"].append(gen.get("generation_time_seconds", 0))

        return timeline

    @staticmethod
    def build_tool_usage_chart(generations: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Build tool usage chart data.

        Args:
            generations: List of generation records

        Returns:
            Dict with chart data
        """

        tool_counts = {}

        for gen in generations:
            tools_used = gen.get("tools_used", [])
            for tool in tools_used:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1

        return {"labels": list(tool_counts.keys()), "values": list(tool_counts.values())}

    @staticmethod
    def build_success_rate_chart(generations: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Build success rate chart.

        Args:
            generations: List of generation records

        Returns:
            Dict with chart data
        """

        total = len(generations)
        successful = sum(1 for g in generations if g.get("success", False))
        failed = total - successful

        return {
            "labels": ["Successful", "Failed"],
            "values": [successful, failed],
            "percentages": [
                round(successful / total * 100, 1) if total > 0 else 0,
                round(failed / total * 100, 1) if total > 0 else 0,
            ],
        }

    @staticmethod
    def build_cost_breakdown(generations: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Build cost breakdown chart.

        Args:
            generations: List of generation records

        Returns:
            Dict with cost breakdown
        """

        total_gigachat2 = sum(g.get("gigachat2_max_calls", 0) for g in generations)
        total_gigachat3 = sum(g.get("gigachat3_calls", 0) for g in generations)

        # Calculate costs
        gigachat2_cost = total_gigachat2 * 0.002
        gigachat3_cost = total_gigachat3 * 0.0005

        return {
            "labels": ["GigaChat-2-Max", "GigaChat3"],
            "values": [round(gigachat2_cost, 2), round(gigachat3_cost, 2)],
            "calls": [total_gigachat2, total_gigachat3],
            "total_cost": round(gigachat2_cost + gigachat3_cost, 2),
        }

    @staticmethod
    def build_completeness_distribution(generations: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Build completeness score distribution.

        Args:
            generations: List of generation records

        Returns:
            Dict with histogram data
        """

        # Create bins
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bin_labels = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
        bin_counts = [0] * len(bin_labels)

        for gen in generations:
            score = gen.get("final_completeness_score", 0)
            for i in range(len(bins) - 1):
                if bins[i] <= score < bins[i + 1]:
                    bin_counts[i] += 1
                    break
            else:
                # Handle 1.0 edge case
                if score == 1.0:
                    bin_counts[-1] += 1

        return {"labels": bin_labels, "values": bin_counts}


class ToTPathVisualizer:
    """
    Visualize ToT search path as ASCII tree.
    """

    @staticmethod
    def visualize_path(tot_result: Any) -> str:
        """
        Create ASCII tree visualization of best path.

        Args:
            tot_result: ToTResult object

        Returns:
            ASCII tree string
        """

        lines = []
        lines.append("═" * 80)
        lines.append("TREE-OF-THOUGHTS BEST PATH")
        lines.append("═" * 80)
        lines.append("")

        for i, node in enumerate(tot_result.best_path):
            indent = "  " * node.depth

            # Node info
            lines.append(f"{indent}┌─ Node {i + 1} (depth={node.depth})")
            lines.append(f"{indent}│  Thought: {node.thought[:60]}...")
            lines.append(f"{indent}│  Tool: {node.planned_action.get('tool_name', 'None')}")
            lines.append(f"{indent}│  Completeness: {node.completeness_score:.2f}")
            lines.append(f"{indent}│  Promise: {node.promise_score:.2f}")
            lines.append(f"{indent}└─")

            if i < len(tot_result.best_path) - 1:
                lines.append(f"{indent}   ↓")

        lines.append("")
        lines.append("═" * 80)
        lines.append(f"Final Completeness: {tot_result.final_completeness:.2f}")
        lines.append(f"Total Iterations: {tot_result.iterations}")
        lines.append(f"Total Time: {tot_result.total_time:.1f}s")
        lines.append("═" * 80)

        return "\n".join(lines)
