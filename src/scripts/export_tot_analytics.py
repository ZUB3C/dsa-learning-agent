"""
Export ToT analytics to various formats (JSON, CSV, HTML report).
"""

import csv
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import select

from src.core.database import AsyncSessionLocal, MaterialGeneration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_to_json(
    generations: list[MaterialGeneration], output_file: str = "tot_analytics.json"
) -> None:
    """
    Export analytics to JSON.

    Args:
        generations: List of generation records
        output_file: Output file path
    """

    logger.info(f"📤 Exporting to JSON: {output_file}")

    data = {
        "exported_at": datetime.now().isoformat(),
        "total_generations": len(generations),
        "generations": [],
    }

    for gen in generations:
        data["generations"].append({
            "generation_id": gen.generation_id,
            "user_id": gen.user_id,
            "topic": gen.topic,
            "user_level": gen.user_level,
            "success": gen.success,
            "tot_iterations": gen.tot_iterations,
            "tot_explored_nodes": gen.tot_explored_nodes,
            "tot_dead_end_nodes": gen.tot_dead_end_nodes,
            "tot_best_path_depth": gen.tot_best_path_depth,
            "final_completeness_score": gen.final_completeness_score,
            "documents_collected": gen.documents_collected,
            "material_length": gen.material_length,
            "generation_time_seconds": gen.generation_time_seconds,
            "gigachat2_max_calls": gen.gigachat2_max_calls,
            "gigachat3_calls": gen.gigachat3_calls,
            "estimated_cost_usd": gen.estimated_cost_usd,
            "tools_used": gen.tools_used,
            "tool_call_counts": gen.tool_call_counts,
            "created_at": gen.created_at.isoformat(),
        })

    with Path(output_file).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Exported {len(generations)} generations to {output_file}")


def export_to_csv(
    generations: list[MaterialGeneration], output_file: str = "tot_analytics.csv"
) -> None:
    """
    Export analytics to CSV.

    Args:
        generations: List of generation records
        output_file: Output file path
    """

    logger.info(f"📤 Exporting to CSV: {output_file}")

    if not generations:
        logger.warning("⚠️ No data to export")
        return

    # Define CSV columns
    columns = [
        "generation_id",
        "user_id",
        "topic",
        "user_level",
        "success",
        "tot_iterations",
        "tot_explored_nodes",
        "tot_dead_end_nodes",
        "tot_best_path_depth",
        "final_completeness_score",
        "documents_collected",
        "material_length",
        "generation_time_seconds",
        "gigachat2_max_calls",
        "gigachat3_calls",
        "estimated_cost_usd",
        "created_at",
    ]

    with Path(output_file).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for gen in generations:
            writer.writerow({
                "generation_id": gen.generation_id,
                "user_id": gen.user_id,
                "topic": gen.topic,
                "user_level": gen.user_level,
                "success": gen.success,
                "tot_iterations": gen.tot_iterations,
                "tot_explored_nodes": gen.tot_explored_nodes,
                "tot_dead_end_nodes": gen.tot_dead_end_nodes,
                "tot_best_path_depth": gen.tot_best_path_depth,
                "final_completeness_score": gen.final_completeness_score,
                "documents_collected": gen.documents_collected,
                "material_length": gen.material_length,
                "generation_time_seconds": gen.generation_time_seconds,
                "gigachat2_max_calls": gen.gigachat2_max_calls,
                "gigachat3_calls": gen.gigachat3_calls,
                "estimated_cost_usd": gen.estimated_cost_usd,
                "created_at": gen.created_at.isoformat(),
            })

    logger.info(f"✅ Exported {len(generations)} generations to {output_file}")


def export_to_html_report(
    generations: list[MaterialGeneration], output_file: str = "tot_analytics_report.html"
) -> None:
    """
    Export analytics as HTML report.

    Args:
        generations: List of generation records
        output_file: Output file path
    """

    logger.info(f"📤 Generating HTML report: {output_file}")

    if not generations:
        logger.warning("⚠️ No data to export")
        return

    # Calculate statistics
    total = len(generations)
    successful = sum(1 for g in generations if g.success is True)

    avg_completeness = sum(g.final_completeness_score for g in generations) / total
    avg_iterations = sum(g.tot_iterations for g in generations) / total
    avg_time = sum(g.generation_time_seconds for g in generations) / total
    total_cost = sum(g.estimated_cost_usd for g in generations)

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ToT Analytics Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #4CAF50;
            margin: 10px 0;
        }}
        .stat-label {{
            color: #777;
            font-size: 0.9em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .success {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .failed {{
            color: #f44336;
            font-weight: bold;
        }}
        .footer {{
            margin-top: 40px;
            text-align: center;
            color: #777;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>🌳 Tree-of-Thoughts Analytics Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    <p><strong>Period:</strong> {generations[0].created_at.strftime("%Y-%m-%d")} to {generations[-1].created_at.strftime("%Y-%m-%d")}</p>

    <h2>📊 Summary Statistics</h2>
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-label">Total Generations</div>
            <div class="stat-value">{total}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Success Rate</div>
            <div class="stat-value">{successful / total * 100:.1f}%</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Avg Completeness</div>
            <div class="stat-value">{avg_completeness:.2f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Avg Iterations</div>
            <div class="stat-value">{avg_iterations:.1f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Avg Time</div>
            <div class="stat-value">{avg_time:.1f}s</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Total Cost</div>
            <div class="stat-value">${total_cost:.2f}</div>
        </div>
    </div>

    <h2>📋 Recent Generations</h2>
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>Topic</th>
                <th>Level</th>
                <th>Status</th>
                <th>Completeness</th>
                <th>Iterations</th>
                <th>Time (s)</th>
                <th>Cost ($)</th>
                <th>Created</th>
            </tr>
        </thead>
        <tbody>
"""

    # Add last 20 generations
    for gen in generations[-20:]:
        status_class = "success" if gen.success is True else "failed"
        status_text = "✅ Success" if gen.success is True else "❌ Failed"

        html += f"""
            <tr>
                <td>{gen.generation_id[:12]}...</td>
                <td>{gen.topic[:40]}</td>
                <td>{gen.user_level}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{gen.final_completeness_score:.2f}</td>
                <td>{gen.tot_iterations}</td>
                <td>{gen.generation_time_seconds:.1f}</td>
                <td>${gen.estimated_cost_usd:.4f}</td>
                <td>{gen.created_at.strftime("%Y-%m-%d %H:%M")}</td>
            </tr>
"""

    html += """
        </tbody>
    </table>

    <div class="footer">
        <p>Materials Agent v2 - Tree-of-Thoughts Analytics</p>
    </div>
</body>
</html>
"""

    with Path(output_file).open("w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"✅ Generated HTML report: {output_file}")


async def main() -> None:
    """Main export function."""

    import sys

    # Parse arguments
    format_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    logger.info("=" * 80)
    logger.info("EXPORT TOT ANALYTICS")
    logger.info("=" * 80)
    logger.info(f"Format: {format_arg}")
    logger.info(f"Period: Last {days} days")
    logger.info("")

    # Fetch data
    start_date = datetime.now() - timedelta(days=days)

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(MaterialGeneration)
            .where(MaterialGeneration.created_at >= start_date)
            .order_by(MaterialGeneration.created_at)
        )
        generations = list(result.scalars().all())

    if not generations:
        logger.warning("⚠️ No generations found in specified period")
        return

    logger.info(f"📊 Found {len(generations)} generations")
    logger.info("")

    # Create output directory
    output_dir = Path("exports")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Export
    if format_arg in {"json", "all"}:
        export_to_json(generations, str(output_dir / f"tot_analytics_{timestamp}.json"))

    if format_arg in {"csv", "all"}:
        export_to_csv(generations, str(output_dir / f"tot_analytics_{timestamp}.csv"))

    if format_arg in {"html", "all"}:
        export_to_html_report(
            generations, str(output_dir / f"tot_analytics_report_{timestamp}.html")
        )

    logger.info("")
    logger.info("=" * 80)
    logger.info("EXPORT COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
