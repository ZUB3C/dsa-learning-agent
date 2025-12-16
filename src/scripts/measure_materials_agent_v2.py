"""
Measurement script for Materials Agent v2 with DeepEval metrics.

This script evaluates the Materials Agent v2 by running test topics
through the API and collecting quality metrics (completeness, relevance, quality).

Usage:
    uv run python -m src.scripts.measure_materials_agent_v2 \
        --test-data material-examples/test_topics.json \
        --output material-examples/results \
        --base-url http://localhost:8000
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import aiohttp
from pydantic import BaseModel

from src.config import get_settings
from src.metrics.deepeval_metrics import DeepEvalMetrics

settings = get_settings()
logging.basicConfig(
    level=settings.logging.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════


class TestTopic(BaseModel):
    """Single test topic."""
    topic_id: str
    topic: str
    user_level: str
    language: str = "ru"
    description: str = ""


class APIResponse(BaseModel):
    """API response model."""
    generation_id: str
    success: bool
    material: str
    word_count: int
    documents_collected: int
    tot_metrics: dict[str, Any]
    warnings: list[str] = []
    fallbacks_used: list[str] = []


class TestResult(BaseModel):
    """Result for a single test."""
    topic_id: str
    topic: str
    user_level: str
    success: bool

    # Material info
    material_length: int
    word_count: int

    # ToT metrics
    tot_iterations: int
    explored_nodes: int
    final_completeness: float
    documents_collected: int

    # DeepEval metrics
    completeness_score: float
    relevance_score: float
    quality_score: float

    # Performance
    generation_time_seconds: float
    gigachat2_calls: int
    gigachat3_calls: int
    estimated_cost_usd: float

    # Tools
    tools_used: list[str]

    # API info
    generation_id: str
    warnings: list[str] = []
    fallbacks_used: list[str] = []
    error: str | None = None


class AggregatedMetrics(BaseModel):
    """Aggregated metrics across all tests."""
    total_tests: int
    successful_tests: int
    failed_tests: int
    success_rate: float

    # DeepEval metrics
    avg_completeness: float
    avg_relevance: float
    avg_quality: float
    min_completeness: float
    max_completeness: float
    min_relevance: float
    max_relevance: float
    min_quality: float
    max_quality: float

    # Performance
    avg_generation_time: float
    total_generation_time: float
    avg_tot_iterations: float
    avg_explored_nodes: float
    avg_documents_collected: float

    # Cost
    total_cost_usd: float
    avg_cost_per_test: float
    total_gigachat2_calls: int
    total_gigachat3_calls: int

    # Breakdown by level
    breakdown_by_level: dict[str, dict[str, Any]]


class MetricsReport(BaseModel):
    """Complete metrics report."""
    test_suite: str
    base_url: str
    overall_metrics: AggregatedMetrics
    individual_results: list[TestResult]


# ═══════════════════════════════════════════════════════════════════
# API CLIENT
# ═══════════════════════════════════════════════════════════════════


class MaterialsAgentClient:
    """Client for Materials Agent v2 API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/api/v2/materials/generate"

    async def generate_material(
        self, 
        topic: str, 
        user_level: str,
        language: str = "ru",
        user_id: str = "test_user"
    ) -> dict[str, Any]:
        """
        Call the Materials Agent v2 API.

        Args:
            topic: Topic to generate material for
            user_level: User knowledge level
            language: Language for material
            user_id: User ID

        Returns:
            API response dict
        """
        payload = {
            "topic": topic,
            "user_level": user_level,
            "user_id": user_id,
            "language": language
        }

        logger.info(f"📤 Calling API: {topic} ({user_level})")

        timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes timeout

        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(self.endpoint, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
                    logger.info(f"✅ API success: {topic}")
                    return data
            except aiohttp.ClientResponseError as e:
                logger.error(f"❌ API error {e.status}: {topic}")
                error_text = await e.response.text() if e.response else "Unknown error"  # pyright: ignore[reportAttributeAccessIssue]
                raise Exception(f"API error {e.status}: {error_text}")
            except asyncio.TimeoutError:
                logger.error(f"⏱️ API timeout: {topic}")
                raise Exception("API request timed out after 5 minutes")
            except Exception as e:
                logger.error(f"❌ Request failed: {topic} - {e}")
                raise


# ═══════════════════════════════════════════════════════════════════
# METRICS CALCULATION
# ═══════════════════════════════════════════════════════════════════


def calculate_deepeval_metrics(
    topic: str,
    material: str,
    context: str
) -> dict[str, float]:
    """
    Calculate DeepEval metrics for generated material.

    Args:
        topic: Original topic/query
        material: Generated material
        context: Source context (documents)

    Returns:
        Dict with completeness, relevance, quality scores
    """
    # Calculate individual metrics
    relevance = DeepEvalMetrics.calculate_answer_relevance(topic, material)
    quality = DeepEvalMetrics.calculate_coherence(material)

    # Calculate completeness (faithfulness to context)
    if context:
        completeness = DeepEvalMetrics.calculate_faithfulness(context, material)
    else:
        # If no context, estimate based on material structure
        completeness = estimate_completeness(material)

    return {
        "completeness": round(completeness, 3),
        "relevance": round(relevance, 3),
        "quality": round(quality, 3)
    }


def estimate_completeness(material: str) -> float:
    """
    Estimate completeness based on material structure.

    Args:
        material: Generated material

    Returns:
        Completeness score (0-1)
    """
    # Check for key sections
    has_intro = any(keyword in material.lower() for keyword in ["введение", "overview", "что такое"])
    has_examples = any(keyword in material.lower() for keyword in ["пример", "example", "рассмотрим"])
    has_code = "```" in material or "python" in material.lower()
    has_complexity = any(keyword in material.lower() for keyword in ["сложность", "o(", "complexity"])
    has_conclusion = any(keyword in material.lower() for keyword in ["заключение", "итог", "conclusion", "summary"])

    # Calculate score
    score = 0.0
    if has_intro:
        score += 0.2
    if has_examples:
        score += 0.25
    if has_code:
        score += 0.25
    if has_complexity:
        score += 0.15
    if has_conclusion:
        score += 0.15

    # Minimum score based on length
    word_count = len(material.split())
    if word_count >= 1000:
        score = max(score, 0.7)
    elif word_count >= 500:
        score = max(score, 0.5)

    return min(score, 1.0)


# ═══════════════════════════════════════════════════════════════════
# TEST EXECUTION
# ═══════════════════════════════════════════════════════════════════


async def run_single_test(
    client: MaterialsAgentClient,
    test_topic: TestTopic
) -> TestResult:
    """
    Run a single test case.

    Args:
        client: API client
        test_topic: Test topic

    Returns:
        Test result
    """
    logger.info("=" * 80)
    logger.info(f"🧪 Testing: {test_topic.topic} ({test_topic.user_level})")
    logger.info("=" * 80)

    start_time = time.time()

    try:
        # Call API
        response = await client.generate_material(
            topic=test_topic.topic,
            user_level=test_topic.user_level,
            language=test_topic.language
        )

        generation_time = time.time() - start_time

        # Extract metrics from response
        material = response["material"]
        tot_metrics = response["tot_metrics"]

        # Build context from sources (if available)
        context = ""
        if "sources" in response:
            for source in response["sources"][:5]:  # Top 5 sources
                if "content" in source:
                    context += source["content"] + "\n\n"

        # Calculate DeepEval metrics
        deepeval_metrics = calculate_deepeval_metrics(
            topic=test_topic.topic,
            material=material,
            context=context
        )

        # Build result
        result = TestResult(
            topic_id=test_topic.topic_id,
            topic=test_topic.topic,
            user_level=test_topic.user_level,
            success=True,
            material_length=len(material),
            word_count=response.get("word_count", len(material.split())),
            tot_iterations=tot_metrics["total_iterations"],
            explored_nodes=tot_metrics["explored_nodes"],
            final_completeness=tot_metrics["final_completeness"],
            documents_collected=response["documents_collected"],
            completeness_score=deepeval_metrics["completeness"],
            relevance_score=deepeval_metrics["relevance"],
            quality_score=deepeval_metrics["quality"],
            generation_time_seconds=generation_time,
            gigachat2_calls=tot_metrics["gigachat2_max_calls"],
            gigachat3_calls=tot_metrics["gigachat3_calls"],
            estimated_cost_usd=tot_metrics["estimated_cost_usd"],
            tools_used=tot_metrics["tools_used"],
            generation_id=response["generation_id"],
            warnings=response.get("warnings", []),
            fallbacks_used=response.get("fallbacks_used", [])
        )

        logger.info(f"✅ Test passed: {test_topic.topic}")
        logger.info(f"   Completeness: {result.completeness_score:.3f}")
        logger.info(f"   Relevance: {result.relevance_score:.3f}")
        logger.info(f"   Quality: {result.quality_score:.3f}")
        logger.info(f"   Time: {result.generation_time_seconds:.2f}s")

        return result

    except Exception as e:
        generation_time = time.time() - start_time
        logger.error(f"❌ Test failed: {test_topic.topic} - {e}")

        return TestResult(
            topic_id=test_topic.topic_id,
            topic=test_topic.topic,
            user_level=test_topic.user_level,
            success=False,
            material_length=0,
            word_count=0,
            tot_iterations=0,
            explored_nodes=0,
            final_completeness=0.0,
            documents_collected=0,
            completeness_score=0.0,
            relevance_score=0.0,
            quality_score=0.0,
            generation_time_seconds=generation_time,
            gigachat2_calls=0,
            gigachat3_calls=0,
            estimated_cost_usd=0.0,
            tools_used=[],
            generation_id="",
            error=str(e)
        )


async def run_all_tests(
    test_topics: list[TestTopic],
    base_url: str,
    delay_between_tests: float = 2.0
) -> list[TestResult]:
    """
    Run all test cases sequentially.

    Args:
        test_topics: List of test topics
        base_url: API base URL
        delay_between_tests: Delay between tests in seconds

    Returns:
        List of test results
    """
    client = MaterialsAgentClient(base_url)
    results = []

    for i, test_topic in enumerate(test_topics, 1):
        logger.info(f"\n📋 Test {i}/{len(test_topics)}")

        result = await run_single_test(client, test_topic)
        results.append(result)

        # Delay between tests to avoid overloading
        if i < len(test_topics):
            logger.info(f"⏳ Waiting {delay_between_tests}s before next test...")
            await asyncio.sleep(delay_between_tests)

    return results


# ═══════════════════════════════════════════════════════════════════
# METRICS AGGREGATION
# ═══════════════════════════════════════════════════════════════════


def aggregate_metrics(results: list[TestResult]) -> AggregatedMetrics:
    """
    Aggregate metrics from all test results.

    Args:
        results: List of test results

    Returns:
        Aggregated metrics
    """
    successful_results = [r for r in results if r.success]

    total_tests = len(results)
    successful_tests = len(successful_results)
    failed_tests = total_tests - successful_tests

    if not successful_results:
        # No successful tests
        return AggregatedMetrics(
            total_tests=total_tests,
            successful_tests=0,
            failed_tests=failed_tests,
            success_rate=0.0,
            avg_completeness=0.0,
            avg_relevance=0.0,
            avg_quality=0.0,
            min_completeness=0.0,
            max_completeness=0.0,
            min_relevance=0.0,
            max_relevance=0.0,
            min_quality=0.0,
            max_quality=0.0,
            avg_generation_time=0.0,
            total_generation_time=0.0,
            avg_tot_iterations=0.0,
            avg_explored_nodes=0.0,
            avg_documents_collected=0.0,
            total_cost_usd=0.0,
            avg_cost_per_test=0.0,
            total_gigachat2_calls=0,
            total_gigachat3_calls=0,
            breakdown_by_level={}
        )

    # Overall metrics
    completeness_scores = [r.completeness_score for r in successful_results]
    relevance_scores = [r.relevance_score for r in successful_results]
    quality_scores = [r.quality_score for r in successful_results]

    # Breakdown by level
    breakdown = {}
    for level in ["beginner", "intermediate", "advanced"]:
        level_results = [r for r in successful_results if r.user_level == level]
        if level_results:
            breakdown[level] = {
                "count": len(level_results),
                "avg_completeness": sum(r.completeness_score for r in level_results) / len(level_results),
                "avg_relevance": sum(r.relevance_score for r in level_results) / len(level_results),
                "avg_quality": sum(r.quality_score for r in level_results) / len(level_results),
                "avg_generation_time": sum(r.generation_time_seconds for r in level_results) / len(level_results),
                "avg_cost": sum(r.estimated_cost_usd for r in level_results) / len(level_results)
            }

    return AggregatedMetrics(
        total_tests=total_tests,
        successful_tests=successful_tests,
        failed_tests=failed_tests,
        success_rate=round(successful_tests / total_tests * 100, 2),

        avg_completeness=round(sum(completeness_scores) / len(completeness_scores), 3),
        avg_relevance=round(sum(relevance_scores) / len(relevance_scores), 3),
        avg_quality=round(sum(quality_scores) / len(quality_scores), 3),
        min_completeness=round(min(completeness_scores), 3),
        max_completeness=round(max(completeness_scores), 3),
        min_relevance=round(min(relevance_scores), 3),
        max_relevance=round(max(relevance_scores), 3),
        min_quality=round(min(quality_scores), 3),
        max_quality=round(max(quality_scores), 3),

        avg_generation_time=round(sum(r.generation_time_seconds for r in successful_results) / len(successful_results), 2),
        total_generation_time=round(sum(r.generation_time_seconds for r in results), 2),
        avg_tot_iterations=round(sum(r.tot_iterations for r in successful_results) / len(successful_results), 2),
        avg_explored_nodes=round(sum(r.explored_nodes for r in successful_results) / len(successful_results), 2),
        avg_documents_collected=round(sum(r.documents_collected for r in successful_results) / len(successful_results), 2),

        total_cost_usd=round(sum(r.estimated_cost_usd for r in results), 4),
        avg_cost_per_test=round(sum(r.estimated_cost_usd for r in successful_results) / len(successful_results), 4),
        total_gigachat2_calls=sum(r.gigachat2_calls for r in results),
        total_gigachat3_calls=sum(r.gigachat3_calls for r in results),

        breakdown_by_level=breakdown
    )


# ═══════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════


def generate_markdown_report(report: MetricsReport) -> str:
    """
    Generate Markdown report.

    Args:
        report: Metrics report

    Returns:
        Markdown string
    """
    md = []

    # Header
    md.append("# 📊 Materials Agent v2 - Metrics Report\n")
    md.append(f"**Test Suite:** {report.test_suite}\n")
    md.append(f"**API Base URL:** {report.base_url}\n")
    md.append("\n---\n")

    # Overall Summary
    m = report.overall_metrics
    md.append("## 📈 Overall Summary\n")
    md.append(f"- **Total Tests:** {m.total_tests}")
    md.append(f"- **Successful:** {m.successful_tests} ({m.success_rate}%)")
    md.append(f"- **Failed:** {m.failed_tests}\n")

    # DeepEval Metrics
    md.append("## 🎯 DeepEval Metrics\n")
    md.append("### Average Scores\n")
    md.append(f"- **Completeness:** {m.avg_completeness:.3f} (min: {m.min_completeness:.3f}, max: {m.max_completeness:.3f})")
    md.append(f"- **Relevance:** {m.avg_relevance:.3f} (min: {m.min_relevance:.3f}, max: {m.max_relevance:.3f})")
    md.append(f"- **Quality:** {m.avg_quality:.3f} (min: {m.min_quality:.3f}, max: {m.max_quality:.3f})\n")

    # Performance Metrics
    md.append("## ⚡ Performance Metrics\n")
    md.append(f"- **Average Generation Time:** {m.avg_generation_time:.2f}s")
    md.append(f"- **Total Generation Time:** {m.total_generation_time:.2f}s")
    md.append(f"- **Average ToT Iterations:** {m.avg_tot_iterations:.2f}")
    md.append(f"- **Average Explored Nodes:** {m.avg_explored_nodes:.2f}")
    md.append(f"- **Average Documents Collected:** {m.avg_documents_collected:.2f}\n")

    # Cost Metrics
    md.append("## 💰 Cost Metrics\n")
    md.append(f"- **Total Cost:** ${m.total_cost_usd:.4f}")
    md.append(f"- **Average Cost per Test:** ${m.avg_cost_per_test:.4f}")
    md.append(f"- **Total GigaChat-2-Max Calls:** {m.total_gigachat2_calls}")
    md.append(f"- **Total GigaChat3 Calls:** {m.total_gigachat3_calls}\n")

    # Breakdown by User Level
    md.append("## 📊 Breakdown by User Level\n")
    for level in ["beginner", "intermediate", "advanced"]:
        if level in m.breakdown_by_level:
            bd = m.breakdown_by_level[level]
            md.append(f"### {level.capitalize()}\n")
            md.append(f"- **Tests:** {bd['count']}")
            md.append(f"- **Avg Completeness:** {bd['avg_completeness']:.3f}")
            md.append(f"- **Avg Relevance:** {bd['avg_relevance']:.3f}")
            md.append(f"- **Avg Quality:** {bd['avg_quality']:.3f}")
            md.append(f"- **Avg Time:** {bd['avg_generation_time']:.2f}s")
            md.append(f"- **Avg Cost:** ${bd['avg_cost']:.4f}\n")

    # Detailed Results Table
    md.append("## 📋 Detailed Results\n")
    md.append("| ID | Topic | Level | ✅ | Completeness | Relevance | Quality | Time (s) | Cost ($) | ToT Iter | Docs |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")

    for r in report.individual_results:
        status = "✅" if r.success else "❌"
        md.append(
            f"| {r.topic_id} | {r.topic[:30]}... | {r.user_level} | {status} | "
            f"{r.completeness_score:.3f} | {r.relevance_score:.3f} | {r.quality_score:.3f} | "
            f"{r.generation_time_seconds:.2f} | ${r.estimated_cost_usd:.4f} | "
            f"{r.tot_iterations} | {r.documents_collected} |"
        )

    md.append("\n---\n")

    # Failed Tests (if any)
    failed_results = [r for r in report.individual_results if not r.success]
    if failed_results:
        md.append("## ❌ Failed Tests\n")
        for r in failed_results:
            md.append(f"### {r.topic_id}: {r.topic}\n")
            md.append(f"**Error:** {r.error}\n")

    # Footer
    md.append("\n---\n")
    md.append("_Generated by Materials Agent v2 Measurement Script_\n")

    return "\n".join(md)


def save_results(
    report: MetricsReport,
    output_dir: Path,
    test_suite_name: str
) -> tuple[Path, Path]:
    """
    Save results to files.

    Args:
        report: Metrics report
        output_dir: Output directory
        test_suite_name: Test suite name for filename

    Returns:
        Tuple of (json_path, markdown_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / f"{test_suite_name}_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, ensure_ascii=False, indent=2)

    logger.info(f"💾 Saved JSON results: {json_path}")

    # Save Markdown
    md_path = output_dir / f"{test_suite_name}_report.md"
    md_content = generate_markdown_report(report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"📄 Saved Markdown report: {md_path}")

    # Also save individual results by topic_id
    for result in report.individual_results:
        result_path = output_dir / f"{result.topic_id}_result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)

    logger.info(f"💾 Saved {len(report.individual_results)} individual result files")

    return json_path, md_path


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════


async def main(args: argparse.Namespace) -> None:
    """Main entry point."""
    logger.info("=" * 80)
    logger.info("📊 Materials Agent v2 - Metrics Measurement")
    logger.info("=" * 80)

    # Load test topics
    test_data_path = Path(args.test_data)
    if not test_data_path.exists():
        logger.error(f"❌ Test data file not found: {test_data_path}")
        return

    with open(test_data_path, encoding="utf-8") as f:
        test_data = json.load(f)

    test_topics = [TestTopic(**topic) for topic in test_data["topics"]]
    logger.info(f"📋 Loaded {len(test_topics)} test topics")

    # Run tests
    logger.info(f"🚀 Starting tests (API: {args.base_url})\n")
    start_time = time.time()

    logger.info("Testing only first test")
    test_topics = [test_topics[0]]

    results = await run_all_tests(
        test_topics=test_topics,
        base_url=args.base_url,
        delay_between_tests=args.delay
    )

    total_time = time.time() - start_time

    # Aggregate metrics
    logger.info("\n📊 Aggregating metrics...")
    aggregated = aggregate_metrics(results)

    # Build report
    report = MetricsReport(
        test_suite=test_data.get("test_suite", "unknown"),
        base_url=args.base_url,
        overall_metrics=aggregated,
        individual_results=results
    )

    # Save results
    output_dir = Path(args.output)
    json_path, md_path = save_results(
        report=report,
        output_dir=output_dir,
        test_suite_name=test_data.get("test_suite", "test")
    )

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ MEASUREMENT COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total Time: {total_time:.2f}s")
    logger.info(f"Success Rate: {aggregated.success_rate}%")
    logger.info(f"Avg Completeness: {aggregated.avg_completeness:.3f}")
    logger.info(f"Avg Relevance: {aggregated.avg_relevance:.3f}")
    logger.info(f"Avg Quality: {aggregated.avg_quality:.3f}")
    logger.info(f"Total Cost: ${aggregated.total_cost_usd:.4f}")
    logger.info("=" * 80)
    logger.info(f"📄 Markdown Report: {md_path}")
    logger.info(f"💾 JSON Results: {json_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure Materials Agent v2 metrics with DeepEval"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="material-examples/test_topics.json",
        help="Path to test topics JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="material-examples/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="API base URL"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between tests in seconds"
    )

    args = parser.parse_args()
    asyncio.run(main(args))
